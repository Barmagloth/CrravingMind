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

        # Assign a unique name to this epoch's agent.
        crav_name = f"Crav-{epoch + 1:03d}"
        self.agent.crav_id = crav_name

        # 1. Initialise budget with venture multiplier and R&D carry-over.
        self.budget.start_epoch(epoch, prev_success_rate, prev_saved, prev_oom)
        self._write_live_state(epoch, tasks_completed=0, tasks_total=len(tasks))

        # Clean up expired graveyard entries (TTL).
        self.memory.cleanup_graveyard(epoch)

        # 2. Backup memory for OOM rollback (Phase 2+).
        backup = self.memory.backup() if self.phase_manager.has_memory(epoch) else None

        # 3. Build system prompt and reset agent conversation.
        system_prompt = self._build_system_prompt(epoch, phase, crav_name)
        self.agent.start_epoch(epoch, system_prompt)

        self.logger.info(
            "[Epoch %d] Starting, budget: %d tokens",
            epoch, self.budget.remaining,
        )

        # 4. Run tasks.
        results = []
        tasks_total = len(tasks)
        for task_idx, task in enumerate(tasks):
            if self.budget.is_oom:
                break
            result = self._run_task(task, epoch, task_idx=task_idx, tasks_total=tasks_total)
            results.append(result)
            if self.checkpoint and not result.get("skipped"):
                self.checkpoint.save_task_log(epoch, result)
            self._write_live_state(epoch, tasks_completed=len(results), tasks_total=len(tasks))

        # 5. Finalise metrics, handle OOM rollback, export artifact.
        return self._finalize_epoch(epoch, results, backup)

    # ------------------------------------------------------------------
    # Per-task execution
    # ------------------------------------------------------------------

    def _run_task(
        self, task: dict, epoch: int, task_idx: int = 0, tasks_total: int = 0
    ) -> dict:
        """Run a single task within an epoch.

        Flow (agent never sees source text):
          1. Auto-run compress.py on source_text via sandbox.
          2. Judge evaluates compressed output.
          3. Send only numerical metrics to agent.
          4. Agent reacts: may read/write compress.py, run scripts, etc.
        """
        source_text = task["source_text"]
        target_ratio = task["target_ratio"]
        hidden_type = task.get("hidden_type", "discourse")
        is_dynamic = task.get("is_dynamic", False)

        task_id = task.get("task_id") or task.get("name") or (source_text[:20].replace("\n", " ") + "…")
        task_label = f"{hidden_type}/{task_id}"

        self.logger.info(
            "[Epoch %d][Task %d/%d] %s  ratio=%.2f",
            epoch, task_idx + 1, tasks_total, task_label, target_ratio,
        )

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

        # 1. Auto-run compress.py on source text (agent doesn't see this).
        compress_result = self.agent.tools.execute(
            "run_compress",
            {"text": source_text, "target_ratio": target_ratio},
        )
        compressed_text = compress_result.get("output", "") or ""

        if not compress_result.get("success"):
            self.logger.warning(
                "[Epoch %d][Task %d/%d] compress.py failed: %s",
                epoch, task_idx + 1, tasks_total,
                compress_result.get("error", "unknown error"),
            )

        # 2. Evaluate with judge.
        eval_result = self.judge.evaluate_task(
            source_text=source_text,
            compressed_text=compressed_text,
            target_ratio=target_ratio,
            questions=task.get("questions", []),
            reference_answers=task.get("reference_answers", []),
            reference_entities=task.get("reference_entities", []),
            hidden_type=hidden_type,
        )

        verdict = "PASS" if eval_result["pass"] else "FAIL"
        self.logger.info(
            "[Epoch %d][Task %d/%d] Judge: sem=%.2f ent=%.2f  %s",
            epoch, task_idx + 1, tasks_total,
            eval_result["semantic_score"], eval_result["entity_score"], verdict,
        )

        # 3. Send metrics to agent and let it react (improve compress.py).
        #    In starvation mode: skip agent turn entirely — just keep running
        #    tasks with existing compress.py, don't waste tokens on LLM calls.
        feedback = {k: v for k, v in eval_result.items() if k != "hidden_type"}
        tokens_spent = 0

        if self.budget.is_critical_starvation:
            self.logger.info(
                "[Epoch %d][Task %d/%d] Starvation mode — skipping agent turn",
                epoch, task_idx + 1, tasks_total,
            )
            # Still append feedback so agent sees it if budget recovers.
            self.agent.send_feedback(feedback)
        else:
            turn_result = self.agent.send_metrics(
                task_idx=task_idx + 1,
                tasks_total=tasks_total,
                feedback=feedback,
            )
            tokens_spent = turn_result["tokens_spent"]

            if turn_result["is_oom"]:
                self.logger.warning(
                    "[Epoch %d][Task %d/%d] OOM during agent reaction",
                    epoch, task_idx + 1, tasks_total,
                )

            # Circuit breaker: warn if agent's reaction consumed too many tokens.
            circuit_limit = self.budget.circuit_breaker_limit()
            if tokens_spent > circuit_limit:
                self.logger.warning(
                    "Circuit breaker: task exceeded single-task token limit",
                    extra={
                        "epoch": epoch,
                        "tokens_spent": tokens_spent,
                        "circuit_limit": circuit_limit,
                    },
                )

        task_result = {
            "task_score": eval_result["task_score"],
            "passed": eval_result["pass"],
            "hidden_type": hidden_type,
            "is_dynamic": is_dynamic,
            "compression_ratio": eval_result["compression_ratio"],
            "semantic_score": eval_result["semantic_score"],
            "entity_score": eval_result["entity_score"],
            "tokens_spent": tokens_spent,
            "task_idx": task_idx + 1,
            "tasks_total": tasks_total,
            "task_id": task_id,
            "target_ratio": target_ratio,
            "tool_calls": self._format_tool_calls_log(turn_result) if not self.budget.is_critical_starvation else [],
            "crav_text": (turn_result.get("content") or "")[:1000] if not self.budget.is_critical_starvation else "(starvation)",
        }
        return task_result

    def _format_tool_calls_log(self, turn_result: dict) -> list[dict]:
        """Summarise tool calls from a turn result for task_log storage."""
        tc_by_id = {tc["id"]: tc for tc in turn_result.get("tool_calls", [])}
        logged: list[dict] = []
        for tr in turn_result.get("tool_results", []):
            name = tr["name"]
            tc = tc_by_id.get(tr.get("tool_call_id", ""), {})
            raw_args = tc.get("arguments", {})
            result = tr["result"]

            if name == "write_file":
                args = {"filename": raw_args.get("filename", "?")}
            elif name == "run_compress":
                args = {
                    "text_len": len(raw_args.get("text", "")),
                    "ratio": raw_args.get("target_ratio"),
                }
            elif name == "read_file":
                args = {"filename": raw_args.get("filename", "?")}
            elif name == "run_script":
                code = raw_args.get("code", "")
                args = {"code": (code[:40] + "…") if len(code) > 40 else code}
            else:
                args = {}

            if isinstance(result, dict):
                if not result.get("success", True) or result.get("error"):
                    err = result.get("error") or "error"
                    result_str = f"FAIL: {err[:100]}"
                elif name == "run_compress":
                    output = result.get("output", "")
                    result_str = f"success ({len(output)} chars)"
                elif name == "write_file":
                    smoke = result.get("smoke_test", "")
                    result_str = "success" + (f" smoke={smoke}" if smoke else "")
                else:
                    result_str = "success"
            else:
                result_str = str(result)[:100]

            logged.append({"name": name, "args": args, "result": result_str})
        return logged

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

        # Artifact export: save compress.py every epoch (skip on OOM —
        # compress.py may be in a broken state after rollback).
        artifact_path = None
        if completed and not self.budget.is_oom:
            extra = self._compute_export_metadata(completed)
            artifact_path = self._export_artifact(epoch, combined, extra)

        # Write epitaph to graveyard (after OOM rollback, so it persists).
        self._write_epitaph(
            epoch, completed, results, combined, artifact_path is not None,
        )

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
            "[Epoch %d] Complete: SR=%.0f%%, frozen=%.0f%%, dynamic=%.0f%%, budget_remaining=%d",
            epoch, combined * 100, frozen_sr * 100, dynamic_sr * 100, self.budget.remaining,
        )
        return epoch_result

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def _build_system_prompt(self, epoch: int, phase: int, crav_name: str = "Crav") -> str:
        """Build Crav's system prompt based on current phase."""
        phase_descriptions = {
            1: "Phase 1 — Venture: Generous token budget for exploration and experimentation.",
            2: "Phase 2 — Consolidation: Tighter budget. Refine your compress() algorithm.",
            3: "Phase 3 — Survival: Critical budget. Every token counts.",
        }

        return (
            f"You are {crav_name}, an LLM agent whose sole job is to optimize a text compression function.\n\n"
            f"Epoch {epoch} | {phase_descriptions.get(phase, '')}\n\n"
            "## HOW IT WORKS\n"
            "You have a file compress.py with a function: compress(text, target_ratio) -> str.\n"
            "The system automatically runs your compress.py on benchmark texts and scores the output.\n"
            "You will receive ONLY the numerical metrics — you never see the source texts.\n"
            "Your job: read the metrics, improve compress.py, repeat.\n\n"
            "After each task you get: compression_ratio, semantic_score, entity_score, PASS/FAIL.\n"
            "  - semantic_score: cosine similarity of QA answers (compressed vs original)\n"
            "  - entity_score: named entity F1 (facts, names, numbers)\n"
            "  - PASS requires both scores ≥ 0.85 AND ratio within target\n\n"
            "## compress() RULES\n"
            "Signature: def compress(text: str, target_ratio: float) -> str\n"
            "MUST be pure Python — no LLM calls, no network, no API calls.\n"
            "ALLOWED: re, math, collections, itertools, functools, string, unicodedata,\n"
            "  json, heapq, pathlib, typing, dataclasses, abc, copy, os.path,\n"
            "  numpy, sklearn, spacy, nltk\n"
            "FORBIDDEN: anthropic, openai, requests, urllib, socket, subprocess, os\n\n"
            "## TOOLS\n"
            "- read_file(filename): Read compress.py, bible.md, or graveyard.md\n"
            "- write_file(filename, content): Write files (compress.py runs smoke test)\n"
            "- run_script(code): Run a Python script in your workspace to test ideas\n"
            "- run_compress(text, target_ratio): Test compress() on your own sample text\n"
            "- audit_budget(): Check remaining token budget\n\n"
            "## GRAVEYARD\n"
            "Read graveyard.md to see what happened to previous agents.\n"
            "Learn from their failures — don't repeat the same mistakes.\n\n"
            "## BUDGET\n"
            "Every token costs budget. When budget runs out you die (OOM).\n"
            "Be efficient — read metrics, make targeted improvements, minimize chatter."
        )

    def _write_epitaph(
        self, epoch: int, completed: list, all_results: list,
        success_rate: float, has_artifact: bool,
    ) -> None:
        """Write a brief epitaph for this epoch's agent to graveyard.md.

        Asks the agent for last words (1 line: what it tried, why it failed),
        then appends a structured epitaph block to graveyard.md.
        """
        crav_name = self.agent.crav_id
        n_passed = sum(1 for r in completed if r.get("passed"))
        n_completed = len(completed)
        n_total = len(all_results)

        # Cause of death.
        if self.budget.is_oom:
            cause = "OOM"
        elif success_rate >= float(self.config.get("judge", {}).get("pass_threshold", 0.85)):
            cause = "SURVIVED"
        elif n_completed == 0:
            cause = "instant OOM"
        else:
            cause = f"SR={success_rate:.0%}"

        # Per-type breakdown.
        by_type: dict[str, list[bool]] = {}
        for r in completed:
            t = r.get("hidden_type", "?")
            by_type.setdefault(t, []).append(r.get("passed", False))
        type_parts = []
        for t, results in sorted(by_type.items()):
            p = sum(results)
            type_parts.append(f"{t}={p}/{len(results)}")
        type_info = " ".join(type_parts) if type_parts else "no tasks"

        artifact_tag = " | artifact exported" if has_artifact else ""

        # Ask the agent for last words (what it tried, why it failed).
        last_words = self.agent.request_last_words()

        lines = [
            f"<!-- AMENDMENT:epoch={epoch} -->",
            f"{crav_name} | E{epoch} | {n_passed}/{n_completed} pass"
            f" ({n_total} queued) | {type_info} | died: {cause}{artifact_tag}",
        ]
        if last_words:
            lines.append(last_words)
        lines.append("<!-- /AMENDMENT -->")

        epitaph = "\n".join(lines) + "\n"

        # Append to existing graveyard.
        existing = self.memory.read_file("graveyard.md")
        self.memory.write_file("graveyard.md", existing + epitaph)

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
