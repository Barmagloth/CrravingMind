"""Main Judge evaluation pipeline (spec §6.2, steps 4–7)."""

from __future__ import annotations

import abc
import logging
from typing import Any

from .embeddings import EmbeddingModel
from .entities import EntityExtractor
from .scoring import Scorer
from .validators import TypeValidator

logger = logging.getLogger(__name__)


class JudgeEvaluator(abc.ABC):
    """Orchestrates steps 4–7 of the Judge scoring pipeline.

    Wiring:
        - Step 4: compression ratio gate.
        - Step 5: LLM answers questions using compressed_text as context
                  (delegated to ``_query_llm``; inject concrete impl later).
        - Step 6: semantic + entity scoring with type-specific validator.
        - Step 7: return feedback dict to agent.

    Subclass and override ``_query_llm`` to connect a real LLM provider.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        scorer: Scorer | None = None,
        validator: TypeValidator | None = None,
        embedding_model: EmbeddingModel | None = None,
        entity_extractor: EntityExtractor | None = None,
        sandbox: Any = None,
    ) -> None:
        cfg = config or {}
        self._config = cfg
        self._scorer = scorer or Scorer(cfg)
        self._validator = validator or TypeValidator(cfg)
        self._embeddings = embedding_model or EmbeddingModel(cfg)
        self._entities = entity_extractor or EntityExtractor(cfg)
        self._sandbox = sandbox  # reserved for future sandbox integration
        judge_cfg = cfg.get("judge", {})
        self._ratio_tolerance: float = float(
            judge_cfg.get("ratio_tolerance", 1.05)
        )

    # ------------------------------------------------------------------
    # Abstract LLM stub — override in production
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _query_llm(self, context: str, question: str) -> str:
        """Send *question* to the LLM with *context* and return its answer.

        This method is intentionally abstract: production code wires a
        real LLM (e.g. Anthropic API) here; tests inject a mock.
        """

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def evaluate_task(
        self,
        source_text: str,
        compressed_text: str,
        target_ratio: float,
        questions: list[str],
        reference_answers: list[str],
        reference_entities: list[set[str]],
        hidden_type: str,
    ) -> dict[str, Any]:
        """Run the full Judge pipeline for one task.

        Returns a feedback dict shaped exactly as per spec §6.2, step 7:
            {
                "compression_ratio": float,
                "semantic_score": float,
                "entity_score": float,
                "pass": bool,
                "task_score": float,
                "hidden_type": str,      # operator-only; not sent to agent
            }
        """
        # Step 4 — compression ratio check
        actual_ratio = len(compressed_text) / len(source_text) if source_text else 0.0
        ratio_ok = actual_ratio <= target_ratio * self._ratio_tolerance
        if not ratio_ok:
            logger.debug(
                "Compression ratio gate failed: %.4f > %.4f",
                actual_ratio,
                target_ratio * self._ratio_tolerance,
            )
            return {
                "compression_ratio": actual_ratio,
                "semantic_score": 0.0,
                "entity_score": 0.0,
                "pass": False,
                "task_score": 0.0,
                "hidden_type": hidden_type,
            }

        # Step 5 — query LLM with compressed_text as context
        test_answers: list[str] = []
        test_entities: list[set[str]] = []
        for q in questions:
            answer = self._query_llm(compressed_text, q)
            test_answers.append(answer)
            test_entities.append(self._entities.extract(answer))

        # Step 6 — scoring
        semantic_scores = self._embeddings.batch_cosine_similarity(
            list(zip(reference_answers, test_answers))
        )
        entity_f1_scores = self._entities.batch_entity_f1(
            list(zip(reference_entities, test_entities))
        )

        # Type-specific validation → mean task score
        mean_score = self._validator.validate(
            hidden_type, semantic_scores, entity_f1_scores
        )
        # Also expose raw axis means for the feedback payload
        semantic_score = (
            sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0.0
        )
        entity_score = (
            sum(entity_f1_scores) / len(entity_f1_scores) if entity_f1_scores else 0.0
        )

        # For "code" type the task_score IS the entity score (cosine ignored).
        # For "needle" and "discourse" use the standard formula on validated scores.
        task_score = mean_score
        passed = self._scorer.is_pass(task_score)

        # Step 7 — feedback dict
        return {
            "compression_ratio": actual_ratio,
            "semantic_score": semantic_score,
            "entity_score": entity_score,
            "pass": passed,
            "task_score": task_score,
            "hidden_type": hidden_type,
        }
