"""Versioned artifact management for compress.py exports."""

import json
import os
from datetime import datetime


class ArtifactManager:
    """Manages versioned compress.py artifacts with metadata."""

    def __init__(self, artifacts_dir: str, manifest_path: str = None):
        self.artifacts_dir = artifacts_dir
        self.manifest_path = manifest_path or os.path.join(artifacts_dir, "manifest.jsonl")
        self._current_version = 0
        os.makedirs(artifacts_dir, exist_ok=True)
        self._load_manifest()

    def _load_manifest(self):
        """Load existing manifest to determine current version."""
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        self._current_version = max(
                            self._current_version, entry.get("version", 0)
                        )

    @property
    def next_version(self) -> int:
        return self._current_version + 1

    def export(self, compress_code: str, metadata: dict) -> dict:
        """Export a new version of compress.py.

        metadata should include:
          - epoch: int
          - crav_id: str
          - mean_score: float
          - semantic_score: float
          - entity_score: float
          - score_by_type: dict
          - mean_compression_ratio: float
          - success_rate: float

        Returns the full versioned entry dict.
        """
        version = self.next_version
        self._current_version = version

        filename = (
            f"compress_v{version:04d}"
            f"_epoch{metadata['epoch']:04d}"
            f"_{metadata['mean_score']:.3f}.py"
        )
        filepath = os.path.join(self.artifacts_dir, filename)

        header = (
            f"# CravingMind Artifact v{version}\n"
            f"# Epoch: {metadata['epoch']}, Score: {metadata['mean_score']:.3f}\n"
            f"# Crav: {metadata['crav_id']}\n\n"
        )
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(header + compress_code)

        entry = {
            "version": version,
            "filename": filename,
            "filepath": filepath,
            "timestamp": datetime.now().isoformat(),
            **metadata,
        }

        with open(self.manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        return entry

    def get_best(self, metric: str = "mean_score") -> dict | None:
        """Get the best artifact by a given metric."""
        history = self.get_history()
        if not history:
            return None
        return max(history, key=lambda e: e.get(metric, 0.0))

    def get_latest(self) -> dict | None:
        """Get the latest version."""
        history = self.get_history()
        if not history:
            return None
        return max(history, key=lambda e: e.get("version", 0))

    def get_history(self) -> list[dict]:
        """Get full version history from manifest."""
        if not os.path.exists(self.manifest_path):
            return []
        entries = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def has_changed(self, new_code: str, prev_code: str) -> bool:
        """Check if compress.py actually changed (ignores leading/trailing whitespace)."""
        return new_code.strip() != prev_code.strip()
