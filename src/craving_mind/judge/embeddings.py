"""Sentence-embedding wrapper with lazy loading and graceful fallback (spec §6.2)."""

from __future__ import annotations

import logging
import math
import os
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L12-v2"


class EmbeddingModel:
    """Thin wrapper around sentence-transformers with lazy model loading.

    If *sentence-transformers* is not installed, falls back to a simple
    word-overlap (Jaccard) heuristic and logs a warning.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        emb_cfg = cfg.get("judge", {}).get("embeddings", {})
        self.model_name: str = emb_cfg.get("model_name", _DEFAULT_MODEL)
        self._model: Any = None
        self._use_fallback: bool = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Eagerly load the sentence-transformers model into memory."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            # Use local cache if available (avoids re-downloading on every run)
            _pkg_root = os.path.dirname(os.path.abspath(__file__))
            _local_cache = os.path.normpath(
                os.path.join(_pkg_root, "..", "..", "..", "models", "sentence-transformers")
            )
            if os.path.isdir(_local_cache):
                # Force offline mode so HuggingFace Hub doesn't attempt network
                # access when the model is already cached locally.
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                self._model = SentenceTransformer(self.model_name, cache_folder=_local_cache)
            else:
                self._model = SentenceTransformer(self.model_name)
            logger.info("Loaded embedding model: %s", self.model_name)
        except ImportError:
            logger.warning(
                "sentence-transformers not installed; "
                "falling back to word-overlap heuristic for cosine similarity."
            )
            self._use_fallback = True

    def _ensure_loaded(self) -> None:
        if self._model is None and not self._use_fallback:
            self.load()

    # ------------------------------------------------------------------
    # Similarity
    # ------------------------------------------------------------------

    def cosine_similarity(self, text_a: str, text_b: str) -> float:
        """Return cosine similarity in [0, 1] between two texts."""
        self._ensure_loaded()
        if self._use_fallback:
            return self._fallback_similarity(text_a, text_b)
        return float(self._model.similarity(
            self._model.encode([text_a]),
            self._model.encode([text_b]),
        )[0][0])

    def batch_cosine_similarity(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Compute cosine similarity for multiple text pairs in one batch."""
        if not pairs:
            return []
        self._ensure_loaded()
        if self._use_fallback:
            return [self._fallback_similarity(a, b) for a, b in pairs]

        texts_a = [a for a, _ in pairs]
        texts_b = [b for _, b in pairs]
        emb_a = self._model.encode(texts_a, convert_to_tensor=True)
        emb_b = self._model.encode(texts_b, convert_to_tensor=True)
        sims = self._model.similarity(emb_a, emb_b)
        # sims is (N, N); we want the diagonal
        return [float(sims[i][i]) for i in range(len(pairs))]

    # ------------------------------------------------------------------
    # Fallback: word-overlap Jaccard similarity
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_similarity(text_a: str, text_b: str) -> float:
        """Word-overlap Jaccard as a cosine-similarity stand-in."""
        tokens_a = set(text_a.lower().split())
        tokens_b = set(text_b.lower().split())
        if not tokens_a and not tokens_b:
            return 1.0
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union)
