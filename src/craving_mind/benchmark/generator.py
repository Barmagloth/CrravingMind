"""Dynamic benchmark task generation."""

from __future__ import annotations

import abc
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd


_HIDDEN_TYPES = ["discourse", "needle", "code"]

_QUESTION_SYSTEM = (
    "You are a rigorous QA engineer. Generate exactly 10 questions about the given text."
    " Cover: general meaning, specific numbers, names, conditions, and logical connections."
    " Return a JSON array of 10 question strings, nothing else."
)

_ANSWER_SYSTEM = (
    "You are a precise reading comprehension assistant."
    " Answer each question using only the provided context."
    " Return a JSON array of 10 answer strings matching the question order, nothing else."
)

_SOURCE_GEN_SYSTEM = (
    "You are a content generator. Generate a realistic text of the requested type."
    " Return only the text, no preamble."
)


class BenchmarkGenerator(abc.ABC):
    """Generates benchmark task records for the frozen and dynamic pools.

    Subclass and override ``_call_llm`` to connect a real LLM provider.
    For offline testing use ``MockBenchmarkGenerator``.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        benchmark_cfg = cfg.get("benchmark", {})
        judge_cfg = cfg.get("judge", {})
        self._config = cfg
        self._ratio_min: float = float(benchmark_cfg.get("target_ratio_min", 0.2))
        self._ratio_max: float = float(benchmark_cfg.get("target_ratio_max", 0.6))
        self._n_questions: int = int(benchmark_cfg.get("n_questions", 10))
        self._llm_config = judge_cfg.get("llm", {})

    # ------------------------------------------------------------------
    # Abstract LLM stub — override in production
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _call_llm(self, prompt: str, system: str) -> str:
        """Call the LLM and return the raw response string."""

    # ------------------------------------------------------------------
    # Public pipeline
    # ------------------------------------------------------------------

    def generate_record(
        self, source_text: str, hidden_type: str
    ) -> dict[str, Any]:
        """Run the full pipeline for one source_text + hidden_type.

        Returns a record dict ready to be written to Parquet (lists serialised as JSON).
        """
        from craving_mind.judge.entities import EntityExtractor

        extractor = EntityExtractor(self._config)

        questions = self._generate_questions(source_text)
        reference_answers = self._generate_answers(source_text, questions)
        reference_entities = [
            sorted(extractor.extract(ans)) for ans in reference_answers
        ]
        target_ratio = round(random.uniform(self._ratio_min, self._ratio_max), 4)

        return {
            "source_text": source_text,
            "hidden_type": hidden_type,
            "questions": json.dumps(questions),
            "reference_answers": json.dumps(reference_answers),
            "reference_entities": json.dumps(reference_entities),
            "target_ratio": target_ratio,
        }

    def generate_benchmark(
        self,
        source_records: list[dict[str, str]],
        output_path: str,
    ) -> None:
        """Generate and write a benchmark Parquet from source records.

        Args:
            source_records: list of dicts with keys ``source_text`` and ``hidden_type``.
            output_path: destination .parquet file path.
        """
        rows = [self.generate_record(r["source_text"], r["hidden_type"]) for r in source_records]
        df = pd.DataFrame(rows)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)

    def generate_dynamic_batch(
        self, count: int, hidden_types: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Create *count* fresh task records with LLM-generated source texts."""
        types = hidden_types or _HIDDEN_TYPES
        return [
            self.generate_record(
                self._generate_source_text(t := random.choice(types)), t
            )
            for _ in range(count)
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_questions(self, source_text: str) -> list[str]:
        prompt = f"Text:\n{source_text}\n\nGenerate 10 questions as a JSON array."
        raw = self._call_llm(prompt, _QUESTION_SYSTEM)
        return self._parse_json_list(raw, expected_length=self._n_questions)

    def _generate_answers(self, source_text: str, questions: list[str]) -> list[str]:
        q_block = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        prompt = (
            f"Context:\n{source_text}\n\n"
            f"Questions:\n{q_block}\n\n"
            "Return answers as a JSON array."
        )
        raw = self._call_llm(prompt, _ANSWER_SYSTEM)
        return self._parse_json_list(raw, expected_length=len(questions))

    def _generate_source_text(self, hidden_type: str) -> str:
        prompt = f"Generate a realistic {hidden_type} text of approximately 300-500 words."
        return self._call_llm(prompt, _SOURCE_GEN_SYSTEM).strip()

    @staticmethod
    def _parse_json_list(raw: str, expected_length: int) -> list[str]:
        """Parse a JSON array from LLM output, padding/truncating to expected_length."""
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(ln for ln in lines if not ln.startswith("```")).strip()
        try:
            result = json.loads(raw)
            if isinstance(result, list):
                items = [str(x) for x in result]
                while len(items) < expected_length:
                    items.append("")
                return items[:expected_length]
        except (json.JSONDecodeError, ValueError):
            pass
        lines = [ln.strip().lstrip("-•0123456789. ") for ln in raw.splitlines() if ln.strip()]
        while len(lines) < expected_length:
            lines.append("")
        return lines[:expected_length]


# ------------------------------------------------------------------
# Mock implementation for testing / offline use
# ------------------------------------------------------------------

_MOCK_QUESTIONS = [
    "What is the main topic discussed in the text?",
    "What specific numbers or quantities are mentioned?",
    "Who are the key entities or people mentioned?",
    "What conditions or requirements are stated?",
    "What is the logical structure of the argument?",
    "What conclusion is reached in the text?",
    "What cause-and-effect relationships are described?",
    "What time periods or dates are referenced?",
    "What locations or places are mentioned?",
    "What actions or processes are described?",
]

_MOCK_ANSWERS = [
    "The text discusses a general subject matter.",
    "Several numerical values are mentioned including specific figures.",
    "Key entities include named individuals and organizations.",
    "The conditions state that certain criteria must be met.",
    "The argument proceeds from premise to conclusion logically.",
    "The conclusion summarizes the main findings.",
    "The cause leads directly to the described effect.",
    "The time period referenced is within the scope of the text.",
    "The location mentioned serves as the primary setting.",
    "The process involves multiple sequential steps.",
]

_MOCK_SOURCE = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
    "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. "
    "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum. "
    "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui "
    "officia deserunt mollit anim id est laborum."
)


class MockBenchmarkGenerator(BenchmarkGenerator):
    """Deterministic mock generator for testing and the --mock CLI flag."""

    def __init__(self, config: dict[str, Any] | None = None, seed: int = 42) -> None:
        super().__init__(config)
        self._rng = random.Random(seed)

    def _call_llm(self, prompt: str, system: str) -> str:
        if "questions" in system.lower() or "Generate 10 questions" in prompt:
            return json.dumps(_MOCK_QUESTIONS)
        if "answers" in system.lower() or "Return answers" in prompt:
            return json.dumps(_MOCK_ANSWERS)
        return _MOCK_SOURCE

    def generate_record(self, source_text: str, hidden_type: str) -> dict[str, Any]:
        """Override to use seeded RNG for deterministic target_ratio."""
        from craving_mind.judge.entities import EntityExtractor

        extractor = EntityExtractor(self._config)
        questions = self._generate_questions(source_text)
        reference_answers = self._generate_answers(source_text, questions)
        reference_entities = [
            sorted(extractor.extract(ans)) for ans in reference_answers
        ]
        target_ratio = round(self._rng.uniform(self._ratio_min, self._ratio_max), 4)
        return {
            "source_text": source_text,
            "hidden_type": hidden_type,
            "questions": json.dumps(questions),
            "reference_answers": json.dumps(reference_answers),
            "reference_entities": json.dumps(reference_entities),
            "target_ratio": target_ratio,
        }
