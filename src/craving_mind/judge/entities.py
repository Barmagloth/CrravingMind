"""NER pipeline with spaCy, lazy loading, and regex fallback (spec §6.2)."""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "en_core_web_lg"


class EntityExtractor:
    """Named-entity extraction backed by spaCy with a regex fallback.

    If the spaCy model is not installed, falls back to extracting
    numbers and capitalised words via regex and logs a warning.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        ent_cfg = cfg.get("judge", {}).get("entities", {})
        self.model_name: str = ent_cfg.get("model_name", _DEFAULT_MODEL)
        self._nlp: Any = None
        self._use_fallback: bool = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Eagerly load the spaCy model."""
        if self._nlp is not None:
            return
        try:
            import spacy  # type: ignore

            self._nlp = spacy.load(self.model_name)
            logger.info("Loaded spaCy model: %s", self.model_name)
        except (ImportError, OSError) as exc:
            logger.warning(
                "spaCy model '%s' unavailable (%s); "
                "falling back to regex entity extraction.",
                self.model_name,
                exc,
            )
            self._use_fallback = True

    def _ensure_loaded(self) -> None:
        if self._nlp is None and not self._use_fallback:
            self.load()

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract(self, text: str) -> set[str]:
        """Return a set of lowercased, stripped named entities from *text*."""
        self._ensure_loaded()
        if self._use_fallback:
            return self._fallback_extract(text)
        doc = self._nlp(text)
        return {ent.text.strip().lower() for ent in doc.ents if ent.text.strip()}

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def entity_f1(self, reference: set[str], test: set[str]) -> float:
        """Compute token-level F1 between two entity sets.

        F1 = 2*P*R / (P+R), where:
            P = |reference ∩ test| / |test|
            R = |reference ∩ test| / |reference|

        Both empty sets → 1.0 (perfect trivial match).
        One empty, one non-empty → 0.0.
        """
        if not reference and not test:
            return 1.0
        if not reference or not test:
            return 0.0
        common = len(reference & test)
        precision = common / len(test)
        recall = common / len(reference)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def batch_entity_f1(
        self, pairs: list[tuple[set[str], set[str]]]
    ) -> list[float]:
        """Compute entity F1 for multiple (reference, test) pairs."""
        return [self.entity_f1(ref, tst) for ref, tst in pairs]

    # ------------------------------------------------------------------
    # Regex fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_extract(text: str) -> set[str]:
        """Extract numbers and capitalised multi-char words via regex."""
        entities: set[str] = set()
        # Numbers (integers, decimals, percentages)
        for m in re.finditer(r"\b\d+(?:[.,]\d+)?%?\b", text):
            entities.add(m.group().strip().lower())
        # Capitalised words (likely proper nouns / names)
        for m in re.finditer(r"\b[A-Z][a-z]{1,}\b", text):
            entities.add(m.group().strip().lower())
        return entities
