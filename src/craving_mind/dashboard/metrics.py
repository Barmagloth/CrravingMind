"""Metrics aggregator for the CravingMind dashboard."""

from __future__ import annotations

import math
from typing import Any

from craving_mind.dashboard.storage import MetricsStorage


class MetricsCollector:
    """Aggregates run metrics into a dashboard state dict."""

    def __init__(self, storage: MetricsStorage, config: dict):
        self.storage = storage
        self.config = config

    # ------------------------------------------------------------------
    # Top-level
    # ------------------------------------------------------------------

    def get_dashboard_state(self) -> dict:
        """Full dashboard state pushed over WebSocket every tick."""
        epoch_history = self.storage.get_epoch_history()
        artifact_history = self.storage.get_artifact_history()
        checkpoint = self.storage.get_checkpoint()
        live_state = self.storage.get_live_state()

        return {
            "health": self._health_metrics(epoch_history, checkpoint, live_state),
            "efficiency": self._efficiency_metrics(epoch_history),
            "evolution": self._evolution_metrics(epoch_history, artifact_history),
            "artifact": self._artifact_metrics(artifact_history, epoch_history),
            "live": self._live_metrics(epoch_history, checkpoint, live_state),
            "epoch_history": self._epoch_history_for_charts(epoch_history),
            "console_lines": self.storage.get_console_lines(limit=200),
        }

    # ------------------------------------------------------------------
    # Sub-aggregators
    # ------------------------------------------------------------------

    def _health_metrics(
        self,
        epoch_history: list[dict],
        checkpoint: dict | None,
        live_state: dict | None = None,
    ) -> dict:
        base_tokens = self.config.get("budget", {}).get("base_tokens", 50000)
        bible_max_pct = self.config.get("memory", {}).get("bible_max_weight_pct", 0.20)

        latest = epoch_history[-1] if epoch_history else {}
        oom_epochs = [e for e in epoch_history if e.get("is_oom")]
        completed_tasks = sum(e.get("tasks_completed", 0) for e in epoch_history)
        total_tasks = sum(e.get("tasks_total", 0) for e in epoch_history)
        starved_tasks = sum(e.get("starved_tasks", 0) for e in epoch_history)
        starvation_rate = (
            0.0 if total_tasks == 0 else starved_tasks / max(total_tasks, 1)
        )

        # Budget remaining: prefer live_state (per-task granularity) over epoch log
        if live_state and live_state.get("budget_initial", 0) > 0:
            budget_remaining = live_state["budget_remaining"]
            budget_initial = live_state["budget_initial"]
            saved_tokens = budget_remaining
            budget_remaining_pct = min(1.0, budget_remaining / max(budget_initial, 1))
        else:
            saved_tokens = latest.get("saved_tokens", 0)
            effective_budget = base_tokens  # rough estimate
            budget_remaining_pct = min(1.0, saved_tokens / max(effective_budget, 1))

        # Bible weight — approximation from checkpoint
        bible_weight_pct = bible_max_pct  # placeholder until memory size is exposed

        phase = self._current_phase(epoch_history)

        return {
            "phase": phase,
            "bible_weight_pct": round(bible_weight_pct * 100, 1),
            "starvation_rate_pct": round(starvation_rate * 100, 1),
            "oom_count": len(oom_epochs),
            "budget_remaining_pct": round(budget_remaining_pct * 100, 1),
            "saved_tokens": saved_tokens,
            "base_tokens": base_tokens,
        }

    def _efficiency_metrics(self, epoch_history: list[dict]) -> dict:
        if not epoch_history:
            return {
                "cost_per_pass": 0.0,
                "success_rate_pct": 0.0,
                "frozen_success_rate_pct": 0.0,
                "dynamic_success_rate_pct": 0.0,
                "partial_fail_pct": 0.0,
                "mean_overfit_gap": 0.0,
            }

        recent = epoch_history[-10:]

        avg_sr = sum(e.get("success_rate", 0.0) for e in recent) / len(recent)
        avg_frozen = sum(e.get("frozen_success_rate", 0.0) for e in recent) / len(recent)
        avg_dynamic = sum(e.get("dynamic_success_rate", 0.0) for e in recent) / len(recent)
        avg_gap = sum(e.get("overfit_gap", 0.0) for e in recent) / len(recent)

        total_completed = sum(e.get("tasks_completed", 0) for e in epoch_history)
        total_tasks = sum(e.get("tasks_total", 0) for e in epoch_history)
        partial_fail_pct = (
            0.0 if total_tasks == 0
            else (total_tasks - total_completed) / max(total_tasks, 1) * 100
        )

        # cost_per_pass: tokens spent per passed task (approx from saved_tokens)
        total_saved = sum(e.get("saved_tokens", 0) for e in epoch_history)
        base_tokens = self.config.get("budget", {}).get("base_tokens", 50000)
        epochs_n = len(epoch_history)
        total_budget = base_tokens * epochs_n
        total_spent = max(0, total_budget - total_saved)
        passed_tasks = sum(
            int(e.get("tasks_completed", 0) * e.get("success_rate", 0.0))
            for e in epoch_history
        )
        cost_per_pass = total_spent / passed_tasks if passed_tasks > 0 else None

        return {
            "cost_per_pass": round(cost_per_pass, 1) if cost_per_pass is not None else None,
            "success_rate_pct": round(avg_sr * 100, 1),
            "frozen_success_rate_pct": round(avg_frozen * 100, 1),
            "dynamic_success_rate_pct": round(avg_dynamic * 100, 1),
            "partial_fail_pct": round(partial_fail_pct, 1),
            "mean_overfit_gap": round(avg_gap * 100, 1),
        }

    def _evolution_metrics(self, epoch_history: list[dict], artifact_history: list[dict]) -> dict:
        compressions = len(artifact_history)
        graveyard_size = 0  # not yet exposed in logs

        drift_epochs = [e for e in epoch_history if e.get("drift_detected")]

        return {
            "compressions_total": compressions,
            "graveyard_size": graveyard_size,
            "drift_events": len(drift_epochs),
            "duplicate_filter_count": 0,  # not yet in task logs
        }

    def _artifact_metrics(self, artifact_history: list[dict], epoch_history: list[dict]) -> dict:
        if not artifact_history:
            return {
                "latest_version": None,
                "best_mean_score": 0.0,
                "best_epoch": None,
                "score_by_type": {},
                "versions": [],
            }

        best = max(artifact_history, key=lambda a: a.get("mean_score", 0.0))
        latest = max(artifact_history, key=lambda a: a.get("version", 0))

        # Score-by-type from best artifact
        score_by_type = best.get("score_by_type", {})

        # Build compression vs score curve
        versions = [
            {
                "version": a.get("version"),
                "epoch": a.get("epoch"),
                "mean_score": round(a.get("mean_score", 0.0), 3),
                "semantic_score": round(a.get("semantic_score", 0.0), 3),
                "entity_score": round(a.get("entity_score", 0.0), 3),
            }
            for a in sorted(artifact_history, key=lambda a: a.get("version", 0))
        ]

        return {
            "latest_version": latest.get("version"),
            "latest_version_str": f"v{latest.get('version', 0):04d}",
            "best_mean_score": round(best.get("mean_score", 0.0), 3),
            "best_epoch": best.get("epoch"),
            "score_by_type": {k: round(v, 3) for k, v in score_by_type.items()},
            "versions": versions,
        }

    def _live_metrics(
        self,
        epoch_history: list[dict],
        checkpoint: dict | None,
        live_state: dict | None = None,
    ) -> dict:
        if live_state:
            current_epoch = live_state.get("epoch", 0)
        elif checkpoint:
            current_epoch = checkpoint.get("epoch", 0)
        elif epoch_history:
            current_epoch = max(e.get("epoch", 0) for e in epoch_history)
        else:
            current_epoch = 0

        latest = epoch_history[-1] if epoch_history else {}

        tasks_completed = live_state.get("tasks_completed", 0) if live_state else None
        tasks_total = live_state.get("tasks_total", 0) if live_state else None

        crav_id = live_state.get("crav_id") if live_state else None

        return {
            "current_epoch": current_epoch,
            "total_epochs": len(epoch_history),
            "latest_success_rate": round(latest.get("success_rate", 0.0) * 100, 1),
            "latest_is_oom": latest.get("is_oom", False),
            "latest_drift": latest.get("drift_detected", False),
            "phase": self._current_phase(epoch_history),
            "tasks_completed": tasks_completed,
            "tasks_total": tasks_total,
            "crav_id": crav_id,
        }

    def _epoch_history_for_charts(self, epoch_history: list[dict]) -> list[dict]:
        """Slim down epoch history to chart-relevant fields."""
        return [
            {
                "epoch": e.get("epoch"),
                "success_rate": round(e.get("success_rate", 0.0), 4),
                "frozen_success_rate": round(e.get("frozen_success_rate", 0.0), 4),
                "dynamic_success_rate": round(e.get("dynamic_success_rate", 0.0), 4),
                "overfit_gap": round(e.get("overfit_gap", 0.0), 4),
                "saved_tokens": e.get("saved_tokens", 0),
                "is_oom": e.get("is_oom", False),
                "drift_detected": e.get("drift_detected", False),
                "artifact_path": e.get("artifact_path"),
                "ts": e.get("ts", ""),
            }
            for e in epoch_history
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_phase(self, epoch_history: list[dict]) -> int:
        phases = self.config.get("phases", {})
        phase2_start = phases.get("phase2_start", 11)
        phase3_start = phases.get("phase3_start", 26)
        if not epoch_history:
            return 1
        current = max(e.get("epoch", 0) for e in epoch_history)
        if current >= phase3_start:
            return 3
        if current >= phase2_start:
            return 2
        return 1
