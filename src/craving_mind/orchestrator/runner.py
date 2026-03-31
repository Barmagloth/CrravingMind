"""Epoch runner: drives the main training loop."""

import json
import logging
import os
from datetime import datetime, timezone


class EpochRunner:
    """Orchestrates agent epochs from start to convergence.

    One epoch:
      1. Start budget (venture multiplier, R&D carry-over), backup memory.
      2. Build system prompt; reset agent conversation.
      3. For each task: send → compress → evaluate → feedback → budget check.
      4. Finalize: compute success_rate, detect drift, export artifact.
      5. OOM → rollback bible.md via MemoryManager.restore().
    """

    def __init__(
        self,
        config: dict,
        agent_interface,
        judge_evaluator,
        benchmark_loader,
        budget_manager,
        phase_manager,
        memory_manager,
        scorer,
        dedup_filter,
        drift_monitor,
        smoke_test,
        token_counter,
        logger,
        run_dir: str = "runs",
        artifact_manager=None,
        checkpoint=None,
    ):
        self.config = config
        self.agent = agent_interface
        self.judge = judge_evaluator
        self.benchmark = benchmark_loader
        self.budget = budget_manager
        self.phase_manager = phase_manager
        self.memory = memory_manager
        self.scorer = scorer
        self.dedup = dedup_filter
        self.drift = drift_monitor
        self.smoke = smoke_test
        self.token_counter = token_counter
        self.logger = logger
        self.run_dir = run_dir
        self.artifact_manager = artifact_manager
        self.checkpoint = checkpoint
        self._prev_compress_code: str | None = None
        self._live_state_path = os.path.join(run_dir, "live_state.json")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_epoch(
        self,
        epoch: int,
        tasks: list,
        prev_success_rate: float = 0.0,
        prev_saved: int = 0,
        prev_oom: bool = False,
    ) -> dict:
        """Run a single epoch. Returns epoch result dict."""
        phase = self.phase_manager.get_phase(epoch)

        # 1. Initialise budget with venture multiplier and R&D carry-over.
        self.budget.start_epoch(epoch, prev_success_rate, prev_saved, prev_oom)
        self._write_live_state(epoch, tasks_completed=0, tasks_total=len(tasks))

        # 2. Backup memory for OOM rollback (Phase 2+).
        backup = self.memory.backup() if self.phase_manager.has_memory(epoch) else None

        # 3. Build system prompt and reset agent conversation.
        system_prompt = self._build_system_prompt(epoch, phase)
        self.agent.start_epoch(epoch, system_prompt)

        # 4. Run tasks.
        results = []
        for task in tasks:
            if self.budget.is_oom:
                break
            result = self._run_task(task, epoch)
            results.append(result)
            if self.checkpoint and not result.get("skipped"):
                self.checkpoint.save_task_log(epoch, result)
            self._write_live_state(epoch, tasks_completed=len(results), tasks_total=len(tasks))

        # 5. Finalise metrics, handle OOM rollback, export artifact.
        return self._finalize_epoch(epoch, results, backup)

    # ------------------------------------------------------------------
    # Per-task execution
    # ------------------------------------------------------------------

    def _run_task(self, task: dict, epoch: int) -> dict:
        """Run a single task within an epoch."""
        source_text = task["source_text"]
        target_ratio = task["target_ratio"]
        hidden_type = task.get("hidden_type", "discourse")
        is_dynamic = task.get("is_dynamic", False)

        # Dedup filter (Phase 3+).
        if self.phase_manager.has_duplicate_filter(epoch):
            if self.dedup.is_duplicate_task(source_text, target_ratio):
                self.logger.debug(
                    "Skipping duplicate task",
                    extra={"epoch": epoch, "hidden_type": hidden_type},
                )
                return {
                    "skipped": True,
                    "reason": "duplicate",
                    "hidden_type": hidden_type,
                    "is_dynamic": is_dynamic,
                }
            self.dedup.mark_task_seen(source_text, target_ratio)

        # Send task to agent.
        turn_result = self.agent.send_task(source_text, target_ratio)

        if turn_result["is_oom"]:
            return {"oom": True, "hidden_type": hidden_type, "is_dynamic": is_dynamic}

        # Circuit breaker: warn if a single task consumed too large a fraction.
        circuit_limit = self.budget.circuit_breaker_limit()
        if turn_result["tokens_spent"] > circuit_limit:
            self.logger.warning(
                "Circuit breaker: task exceeded single-task token limit",
                extra={
                    "epoch": epoch,
                    "tokens_spent": turn_result["tokens_spent"],
                    "circuit_limit": circuit_limit,
                },
            )

        # Extract compressed_text from run_compress tool result.
        compressed_text = ""
        for tr in turn_result.get("tool_results", []):
            if tr["name"] == "run_compress":
                compressed_text = tr["result"].get("output", "") or ""
                break

        # Evaluate with judge.
        eval_result = self.judge.evaluate_task(
            source_text=source_text,
            compressed_text=compressed_text,
            target_ratio=target_ratio,
            questions=task.get("questions", []),
            reference_answers=task.get("reference_answers", []),
            reference_entities=task.get("reference_entities", []),
            hidden_type=hidden_type,
        )

        # Send feedback to agent (hidden_type is operator-only).
        feedback = {k: v for k, v in eval_result.items() if k != "hidden_type"}
        self.agent.send_feedback(feedback)

        task_result = {
            "task_score": eval_result["task_score"],
            "passed": eval_result["pass"],
            "hidden_type": hidden_type,
            "is_dynamic": is_dynamic,
            "compression_ratio": eval_result["compression_ratio"],
            "semantic_score": eval_result["semantic_score"],
            "entity_score": eval_result["entity_score"],
            "tokens_spent": turn_result["tokens_spent"],
        }

        self.logger.info("Task complete", extra=task_result)
        return task_result

    # ------------------------------------------------------------------
    # Epoch finalisation
    # ------------------------------------------------------------------

    def _finalize_epoch(self, epoch: int, results: list, backup: dict | None) -> dict:
        """Calculate metrics, handle carry-over, export artifact."""
        # OOM rollback: restore memory to epoch-start state.
        if self.budget.is_oom and backup is not None:
            self.memory.restore(backup)
            self.logger.warning(
                "OOM: memory rolled back to epoch-start backup",
                extra={"epoch": epoch},
            )

        # Split completed tasks into frozen / dynamic.
        completed = [r for r in results if not r.get("skipped") and not r.get("oom")]
        frozen_results = [r for r in completed if not r.get("is_dynamic")]
        dynamic_results = [r for r in completed if r.get("is_dynamic")]

        def _by_type(task_list: list) -> dict:
            by_type: dict = {}
            for r in task_list:
                t = r.get("hidden_type", "discourse")
                by_type.setdefault(t, []).append(r["passed"])
            return by_type

        frozen_by_type = _by_type(frozen_results)
        dynamic_by_type = _by_type(dynamic_results)

        frozen_sr = self.scorer.epoch_success_rate(frozen_by_type) if frozen_by_type else 0.0
        dynamic_sr = (
            self.scorer.epoch_success_rate(dynamic_by_type)
            if dynamic_by_type
            else frozen_sr
        )

        combined = self.scorer.combined_success_rate(frozen_sr, dynamic_sr)

        # Overfit reaction: frozen >> dynamic → log (dynamic ratio increase handled externally).
        overfit_gap = frozen_sr - dynamic_sr if dynamic_by_type else 0.0
        if overfit_gap > 0.1:
            self.logger.warning(
                "Overfit gap detected — consider increasing dynamic task ratio",
                extra={"epoch": epoch, "overfit_gap": overfit_gap},
            )

        # Drift detection: CUSUM alert → R&D fund reset next epoch (via prev_oom=True signal).
        self.drift.update(combined)
        drift_detected = self.drift.is_drift()
        if drift_detected:
            self.logger.warning(
                "CUSUM drift detected — R&D fund will be suppressed next epoch",
                extra={"epoch": epoch, "success_rate": combined},
            )

        # Artifact export on successful epoch (not OOM).
        artifact_path = None
        pass_threshold = float(self.config.get("judge", {}).get("pass_threshold", 0.85))
        if combined >= pass_threshold and not self.budget.is_oom:
            extra = self._compute_export_metadata(completed)
            artifact_path = self._export_artifact(epoch, combined, extra)

        epoch_result = {
            "epoch": epoch,
            "success_rate": combined,
            "frozen_success_rate": frozen_sr,
            "dynamic_success_rate": dynamic_sr,
            "overfit_gap": overfit_gap,
            "tasks_completed": len(completed),
            "tasks_total": len(results),
            "saved_tokens": self.budget.saved_tokens,
            "is_oom": self.budget.is_oom,
            "artifact_path": artifact_path,
            "drift_detected": drift_detected,
        }

        self.logger.info(
            "Epoch complete",
            extra={"epoch": epoch, "success_rate": combined, "is_oom": self.budget.is_oom},
        )
        return epoch_result

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def _build_system_prompt(self, epoch: int, phase: int) -> str:
        """Build Crav's system prompt based on current phase."""
        phase_descriptions = {
            1: "Phase 1 — Venture: Generous token budget for exploration and experimentation.",
            2: "Phase 2 — Consolidation: Tighter budget. Refine your compress() algorithm.",
            3: "Phase 3 — Survival: Critical budget. Every token counts.",
        }

        return (
            "You are Crav, an adaptive text compression agent. "
            "Your goal is to compress text while preserving semantic meaning and key information.\n\n"
            f"Epoch {epoch} | {phase_descriptions.get(phase, '')}\n\n"
            "Available tools:\n"
            "- run_compress(text, target_ratio): Run your compress() function on the given text\n"
            "- read_file(filename): Read bible.md, graveyard.md, or compress.py\n"
            "- write_file(filename, content): Write to bible.md, graveyard.md, or compress.py\n"
            "- run_script(code): Execute a Python script in your workspace\n"
            "- audit_budget(): Check your remaining token budget\n\n"
            "For each task, call run_compress with the provided text and target_ratio."
        )

    # ------------------------------------------------------------------
    # Live state
    # ------------------------------------------------------------------

    def _write_live_state(self, epoch: int, tasks_completed: int, tasks_total: int) -> None:
        """Write live_state.json so the dashboard can show real-time budget/progress."""
        state = {
            "epoch": epoch,
            "budget_remaining": self.budget.remaining,
            "budget_initial": self.budget._initial_epoch_budget,
            "tasks_completed": tasks_completed,
            "tasks_total": tasks_total,
            "is_oom": self.budget.is_oom,
            "ts": datetime.now(tz=timezone.utc).isoformat(),
        }
        try:
            with open(self._live_state_path, "w", encoding="utf-8") as f:
                json.dump(state, f)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Artifact export
    # ------------------------------------------------------------------

    def _compute_export_metadata(self, completed: list) -> dict:
        """Compute aggregate metrics from completed tasks for artifact metadata."""
        if not completed:
            return {
                "semantic_score": 0.0,
                "entity_score": 0.0,
                "score_by_type": {},
                "mean_compression_ratio": 0.0,
            }
        n = len(completed)
        sem = sum(r.get("semantic_score", 0.0) for r in completed) / n
        ent = sum(r.get("entity_score", 0.0) for r in completed) / n
        cr = sum(r.get("compression_ratio", 0.0) for r in completed) / n

        by_type: dict = {}
        for r in completed:
            t = r.get("hidden_type", "discourse")
            by_type.setdefault(t, []).append(r.get("task_score", 0.0))
        score_by_type = {t: sum(v) / len(v) for t, v in by_type.items()}

        return {
            "semantic_score": sem,
            "entity_score": ent,
            "score_by_type": score_by_type,
            "mean_compression_ratio": cr,
        }

    def _export_artifact(
        self, epoch: int, score: float, extra: dict | None = None
    ) -> str | None:
        """Export compress.py as a versioned artifact (or plain copy if no ArtifactManager)."""
        compress_code = self.memory.read_file("compress.py")
        if not compress_code:
            return None

        if self.artifact_manager is not None:
            # Change detection: skip if compress.py hasn't changed since last export.
            if self._prev_compress_code is not None:
                if not self.artifact_manager.has_changed(
                    compress_code, self._prev_compress_code
                ):
                    self.logger.info(
                        "compress.py unchanged — skipping artifact export",
                        extra={"epoch": epoch},
                    )
                    return None
            self._prev_compress_code = compress_code

            crav_id = getattr(self.agent, "crav_id", "Crav-001")
            extra = extra or {}
            metadata = {
                "epoch": epoch,
                "crav_id": crav_id,
                "mean_score": score,
                "semantic_score": extra.get("semantic_score", 0.0),
                "entity_score": extra.get("entity_score", 0.0),
                "score_by_type": extra.get("score_by_type", {}),
                "mean_compression_ratio": extra.get("mean_compression_ratio", 0.0),
                "success_rate": score,
            }
            entry = self.artifact_manager.export(compress_code, metadata)
            self.logger.info(
                "Artifact exported (versioned)",
                extra={"artifact_path": entry["filepath"], "version": entry["version"]},
            )
            return entry["filepath"]

        # Legacy: basic file copy (no ArtifactManager).
        artifacts_dir = os.path.join(self.run_dir, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)
        filename = f"compress_epoch_{epoch}_{score:.3f}.py"
        artifact_path = os.path.join(artifacts_dir, filename)
        with open(artifact_path, "w", encoding="utf-8") as f:
            f.write(compress_code)
        self.logger.info("Artifact exported", extra={"artifact_path": artifact_path})
        return artifact_path
