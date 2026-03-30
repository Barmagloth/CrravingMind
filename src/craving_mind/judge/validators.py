"""Type-specific scoring validators for the Judge pipeline (spec §6.2, step 6)."""

from __future__ import annotations

from typing import Any


class TypeValidator:
    """Apply type-specific post-processing to raw semantic/entity scores.

    Hidden types (agent never sees them):
        - ``needle``   — every fact must survive; any entity_f1 < 1.0 zeroes that Q.
        - ``code``     — cosine is meaningless for exact-match logic; use entity_f1 only.
        - ``discourse``— base scoring, no adjustments.
    """

    SUPPORTED_TYPES = {"needle", "code", "discourse"}

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        # Reserved for future per-type weight overrides loaded from config.
        _ = config

    # ------------------------------------------------------------------
    # Per-type validators
    # ------------------------------------------------------------------

    def validate_needle(self, entity_f1_scores: list[float]) -> list[float]:
        """Zero out any question score where entity_f1 < 1.0.

        Needle tasks demand perfect fact retention: a single missing entity
        collapses that question's contribution to zero.
        """
        return [s if s >= 1.0 else 0.0 for s in entity_f1_scores]

    def validate_code(
        self,
        semantic_scores: list[float],
        entity_f1_scores: list[float],
    ) -> list[float]:
        """Ignore cosine similarity; return entity_f1 scores verbatim.

        Code tasks use exact-match normalised comparison, so cosine is
        unreliable and discarded entirely.
        """
        _ = semantic_scores  # explicitly unused
        return list(entity_f1_scores)

    def validate_discourse(
        self,
        semantic_scores: list[float],
        entity_f1_scores: list[float],
    ) -> list[float]:
        """Base scoring only: 0.5 * semantic + 0.5 * entity per question."""
        if len(semantic_scores) != len(entity_f1_scores):
            raise ValueError(
                f"Length mismatch: semantic={len(semantic_scores)}, "
                f"entity={len(entity_f1_scores)}"
            )
        return [
            0.5 * s + 0.5 * e
            for s, e in zip(semantic_scores, entity_f1_scores)
        ]

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def validate(
        self,
        hidden_type: str,
        semantic_scores: list[float],
        entity_f1_scores: list[float],
    ) -> float:
        """Dispatch to the correct validator and return the mean task score.

        Args:
            hidden_type: one of "needle", "code", "discourse".
            semantic_scores: per-question cosine similarity values.
            entity_f1_scores: per-question entity F1 values.

        Returns:
            Mean per-question score after type-specific adjustments.
        """
        if hidden_type == "needle":
            adjusted = self.validate_needle(entity_f1_scores)
            # Needle uses entity_f1 only (already zeroed where needed).
            per_question = adjusted
        elif hidden_type == "code":
            per_question = self.validate_code(semantic_scores, entity_f1_scores)
        elif hidden_type == "discourse":
            per_question = self.validate_discourse(semantic_scores, entity_f1_scores)
        else:
            # Unknown type: fall back to base scoring.
            per_question = self.validate_discourse(semantic_scores, entity_f1_scores)

        if not per_question:
            return 0.0
        return sum(per_question) / len(per_question)
