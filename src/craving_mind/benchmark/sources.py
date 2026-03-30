"""Source loading helpers for discourse, needle, and code datasets."""

from __future__ import annotations

import os
from pathlib import Path


_TEXT_EXTENSIONS = {".txt", ".md"}


def load_texts_from_dir(directory: str, hidden_type: str) -> list[dict[str, str]]:
    """Load all .txt/.md files from ``directory`` as source records.

    Args:
        directory: path to a directory containing text files
                   (e.g. ``data/sources/discourse/``).
        hidden_type: the task type label to assign each record.

    Returns:
        List of dicts with keys ``source_text`` and ``hidden_type``.
        Empty list if directory does not exist or contains no valid files.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return []
    records = []
    for entry in sorted(dir_path.iterdir()):
        if entry.is_file() and entry.suffix.lower() in _TEXT_EXTENSIONS:
            text = entry.read_text(encoding="utf-8").strip()
            if text:
                records.append({"source_text": text, "hidden_type": hidden_type})
    return records


def list_available_sources(data_dir: str) -> dict[str, int]:
    """Count available source texts per type under ``data_dir``.

    Expects the layout: ``data_dir/{hidden_type}/*.txt`` or ``*.md``.

    Args:
        data_dir: root directory containing per-type subdirectories.

    Returns:
        Dict mapping type name → number of text files found.
    """
    root = Path(data_dir)
    counts: dict[str, int] = {}
    if not root.is_dir():
        return counts
    for sub in sorted(root.iterdir()):
        if sub.is_dir():
            n = sum(
                1
                for f in sub.iterdir()
                if f.is_file() and f.suffix.lower() in _TEXT_EXTENSIONS
            )
            counts[sub.name] = n
    return counts
