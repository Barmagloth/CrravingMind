"""Scoring formulas for the Judge pipeline (spec §6.2)."""

from __future__ import annotations

import math
from typing import Any


class Scorer:
    """Stateless scoring helpers for task and epoch evaluation."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        judge_cfg = cfg.get("judge", {})
        weights = judge_cfg.get("task_score_weights", {})
        self._semantic_weight: float = float(weights.get("semantic", 0.5))
        self._entity_weight: float = float(weights.get("entity", 0.5))
        self._pass_threshold: float = float(judge_cfg.get("pass_threshold", 0.85))
        epoch_cfg = judge_cfg.get("epoch", {})
        self._epsilon: float = float(epoch_cfg.get("epsilon", 0.01))
        self._type_weights: dict[str, float] = dict(epoch_cfg.get("type_weights", {}))
        self._dynamic_multiplier: float = float(
            judge_cfg.get("dynamic_multiplier", 1.3)
        )

    # ------------------------------------------------------------------
    # Per-task scoring
    # ------------------------------------------------------------------

    def task_score(
        self,
        semantic_score: float,
        entity_score: float,
        weights: dict[str, float] | None = None,
    ) -> float:
        """Weighted average of semantic and entity scores.

        Default: 0.5 * semantic + 0.5 * entity (spec §6.2, step 6).
        """
        if weights is not None:
            w_sem = float(weights.get("semantic", self._semantic_weight))
            w_ent = float(weights.get("entity", self._entity_weight))
        else:
            w_sem = self._semantic_weight
            w_ent = self._entity_weight
        total = w_sem + w_ent
        if total == 0:
            return 0.0
        return (w_sem * semantic_score + w_ent * entity_score) / total

    def is_pass(self, score: float, threshold: float | None = None) -> bool:
        """Return True iff score >= threshold (default 0.85)."""
        thr = threshold if threshold is not None else self._pass_threshold
        return score >= thr

    # ------------------------------------------------------------------
    # Epoch-level aggregation
    # ------------------------------------------------------------------

    def epoch_success_rate(
        self,
        scores_by_type: dict[str, list[bool]],
        type_weights: dict[str, float] | None = None,
        epsilon: float | None = None,
    ) -> float:
        """Weighted geometric mean of per-type pass rates (spec §6.3).

        Formula:
            success_rate = (prod(max(sr_t, ε)^w_t))^(1 / sum(w_t))

        Args:
            scores_by_type: mapping type → list of pass/fail booleans.
            type_weights: optional per-type weights; default all equal.
            epsilon: smoothing floor; default 0.01.
        """
        if not scores_by_type:
            return 0.0

        eps = epsilon if epsilon is not None else self._epsilon
        weights = type_weights if type_weights is not None else self._type_weights

        log_sum = 0.0
        weight_sum = 0.0
        for t, passes in scores_by_type.items():
            if not passes:
                sr_t = 0.0
            else:
                sr_t = sum(passes) / len(passes)
            w_t = weights.get(t, 1.0)
            log_sum += w_t * math.log(max(sr_t, eps))
            weight_sum += w_t

        if weight_sum == 0:
            return 0.0
        return math.exp(log_sum / weight_sum)

    # ------------------------------------------------------------------
    # Combined frozen + dynamic
    # ------------------------------------------------------------------

    def combined_success_rate(
        self,
        frozen_score: float,
        dynamic_score: float,
        dynamic_multiplier: float | None = None,
    ) -> float:
        """Weighted combination of frozen (working pool) and dynamic (holdout) scores.

        dynamic gets extra weight via the multiplier (default 1.3 from config).
        Formula: (frozen + multiplier * dynamic) / (1 + multiplier)
        """
        mult = (
            dynamic_multiplier
            if dynamic_multiplier is not None
            else self._dynamic_multiplier
        )
        return (frozen_score + mult * dynamic_score) / (1.0 + mult)
