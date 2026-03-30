"""Metrics storage: reads run-dir log files for the dashboard."""

from __future__ import annotations

import json
import os
from typing import Any


class MetricsStorage:
    """Reads epoch/task/artifact logs from a run directory."""

    def __init__(self, run_dir: str):
        self.run_dir = run_dir

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_jsonl(self, path: str) -> list[dict]:
        if not os.path.exists(path):
            return []
        entries: list[dict] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except OSError:
            pass
        return entries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_epoch_history(self) -> list[dict]:
        """Read epoch_log.jsonl, return all epoch results."""
        path = os.path.join(self.run_dir, "epoch_log.jsonl")
        return self._read_jsonl(path)

    def get_task_history(self, epoch: int | None = None) -> list[dict]:
        """Read task_log.jsonl, optionally filtered by epoch."""
        path = os.path.join(self.run_dir, "task_log.jsonl")
        entries = self._read_jsonl(path)
        if epoch is not None:
            entries = [e for e in entries if e.get("epoch") == epoch]
        return entries

    def get_artifact_history(self) -> list[dict]:
        """Read artifacts/manifest.jsonl."""
        path = os.path.join(self.run_dir, "artifacts", "manifest.jsonl")
        return self._read_jsonl(path)

    def get_checkpoint(self) -> dict | None:
        """Read current checkpoint.json."""
        path = os.path.join(self.run_dir, "checkpoint.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def get_latest_epoch(self) -> dict | None:
        """Get the most recent epoch result."""
        history = self.get_epoch_history()
        if not history:
            return None
        return max(history, key=lambda e: e.get("epoch", -1))
