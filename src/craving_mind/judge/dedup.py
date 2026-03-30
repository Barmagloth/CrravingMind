"""Cheap deduplication heuristics for tasks and amendment patches (spec §3.2, §4.4)."""

from __future__ import annotations

import hashlib
from typing import Any


class DedupFilter:
    """Detect duplicate tasks and blacklist failed amendment patches.

    Task dedup (spec §3.2):
        SHA-256 of ``lowercase(strip(source_text[:prefix_length])) + str(target_ratio)``.
        This is intentionally cheap — identical text prefixes with different
        bodies collide.  Acceptable for the hand-curated MVP benchmark.

    Amendment dedup (spec §4.4):
        SHA-256 of the raw patch text.  Only exact duplicates are caught;
        a single-character change creates a new hash.  Deliberate trade-off:
        semantic comparison would be too expensive for the deterministic Judge.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        dedup_cfg = cfg.get("judge", {}).get("dedup", {})
        self._prefix_length: int = int(dedup_cfg.get("task_prefix_length", 500))
        self._seen_tasks: set[str] = set()
        self._amendment_blacklist: set[str] = set()

    # ------------------------------------------------------------------
    # Task dedup
    # ------------------------------------------------------------------

    def task_hash(
        self,
        source_text: str,
        target_ratio: float,
        prefix_length: int | None = None,
    ) -> str:
        """Compute the dedup hash for a task.

        Hash input: ``lowercase(strip(source_text[:prefix_length])) + str(target_ratio)``
        """
        length = prefix_length if prefix_length is not None else self._prefix_length
        normalised = source_text[:length].strip().lower()
        payload = normalised + str(target_ratio)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def is_duplicate_task(self, source_text: str, target_ratio: float) -> bool:
        """Return True if this (source_text, target_ratio) was seen before."""
        return self.task_hash(source_text, target_ratio) in self._seen_tasks

    def mark_task_seen(self, source_text: str, target_ratio: float) -> None:
        """Record the task hash so future calls detect it as a duplicate."""
        self._seen_tasks.add(self.task_hash(source_text, target_ratio))

    # ------------------------------------------------------------------
    # Amendment blacklist
    # ------------------------------------------------------------------

    def amendment_hash(self, patch_text: str) -> str:
        """SHA-256 of the raw amendment patch text."""
        return hashlib.sha256(patch_text.encode("utf-8")).hexdigest()

    def is_duplicate_amendment(self, patch_text: str) -> bool:
        """Return True if this exact patch was previously blacklisted."""
        return self.amendment_hash(patch_text) in self._amendment_blacklist

    def blacklist_amendment(self, patch_text: str) -> None:
        """Add the patch to the blacklist (called when a patch causes degradation)."""
        self._amendment_blacklist.add(self.amendment_hash(patch_text))
