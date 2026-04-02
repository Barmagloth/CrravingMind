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

    def _query_llm_batch(self, context: str, questions: list[str]) -> list[str]:
        """Answer multiple questions in one LLM call.

        Default implementation falls back to per-question calls.
        Override in subclass for batching (e.g. CLIProvider).
        """
        return [self._query_llm(context, q) for q in questions]

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

        # Step 5 — query LLM with compressed_text as context.
        # Batch all questions into a single LLM call to avoid N separate
        # CLI process launches (~5s each).
        test_answers = self._query_llm_batch(compressed_text, questions)
        test_entities = [self._entities.extract(a) for a in test_answers]

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


class ConcreteJudgeEvaluator(JudgeEvaluator):
    """JudgeEvaluator backed by a real or mock LLM provider.

    Injects a provider into the abstract ``_query_llm`` so the full
    scoring pipeline can run without subclassing for each LLM backend.
    """

    def __init__(self, provider, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        self._provider = provider

    def _reset_provider_session(self) -> None:
        """Drop CLI session so judge QA calls are independent (no context bleed)."""
        if hasattr(self._provider, "new_session"):
            self._provider.new_session()

    def _query_llm(self, context: str, question: str) -> str:
        self._reset_provider_session()
        messages = [
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
        response = self._provider.chat(messages, max_tokens=256)
        return response.content

    def _query_llm_batch(self, context: str, questions: list[str]) -> list[str]:
        """Answer all questions in a single LLM call.

        Sends numbered questions, expects numbered answers.
        If parsing yields fewer answers than expected, pads with the
        raw response text (still better than N individual CLI calls).
        Resets CLI session first so no previous context bleeds in.
        """
        if not questions:
            return []
        if len(questions) == 1:
            return [self._query_llm(context, questions[0])]

        self._reset_provider_session()
        numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        prompt = (
            f"Context:\n{context}\n\n"
            f"Answer each question below in 1-2 sentences. "
            f"Format: number followed by period, then answer. "
            f"Example:\n1. First answer here.\n2. Second answer here.\n\n{numbered}"
        )
        messages = [{"role": "user", "content": prompt}]
        response = self._provider.chat(messages, max_tokens=120 * len(questions))
        raw = response.content.strip()

        # Strip JSON wrapper if CLI provider wrapped the response.
        if raw.startswith("{"):
            try:
                import json
                data = json.loads(raw)
                raw = data.get("content", raw)
            except (ValueError, AttributeError):
                pass

        # Parse numbered answers: "1. answer\n2. answer\n..."
        import re
        answers = re.split(r"\n\s*\d+[\.\)]\s*", "\n" + raw)
        answers = [a.strip() for a in answers if a.strip()]

        if len(answers) >= len(questions):
            return answers[: len(questions)]

        if answers:
            # Partial match: pad missing answers with the full raw response.
            # This avoids N separate CLI calls (~5s each) while still giving
            # the scorer something to work with for unparsed questions.
            logger.warning(
                "Batch QA partial parse: expected %d, got %d — padding remainder",
                len(questions), len(answers),
            )
            while len(answers) < len(questions):
                answers.append(raw)
            return answers

        # No numbered answers found at all — use raw text for every question.
        logger.warning(
            "Batch QA: no numbered answers parsed — using raw response for all %d questions",
            len(questions),
        )
        return [raw] * len(questions)
