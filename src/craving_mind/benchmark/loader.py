"""Benchmark dataset loader (Parquet + frozen split)."""

from __future__ import annotations

import json
import math
import random
from typing import Any

import pandas as pd


class BenchmarkLoader:
    """Loads and splits the benchmark dataset into frozen and dynamic subsets."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        benchmark_cfg = cfg.get("benchmark", {})
        judge_cfg = cfg.get("judge", {})
        self._config = cfg
        self._frozen_ratio: float = float(benchmark_cfg.get("frozen_ratio", 0.7))
        self._dynamic_multiplier: float = float(
            judge_cfg.get("dynamic_multiplier", 1.3)
        )
        self._ratio_min: float = float(benchmark_cfg.get("target_ratio_min", 0.2))
        self._ratio_max: float = float(benchmark_cfg.get("target_ratio_max", 0.6))

    def load_frozen(self, parquet_path: str) -> list[dict[str, Any]]:
        """Load frozen Parquet and deserialise JSON list fields.

        Returns a list of task dicts with Python lists for
        ``questions``, ``reference_answers``, and ``reference_entities``.
        """
        df = pd.read_parquet(parquet_path)
        tasks = []
        for _, row in df.iterrows():
            task = row.to_dict()
            for field in ("questions", "reference_answers", "reference_entities"):
                raw = task.get(field)
                if isinstance(raw, str):
                    task[field] = json.loads(raw)
            # Ensure target_ratio is present; randomise if missing or NaN
            tr = task.get("target_ratio")
            if tr is None or (isinstance(tr, float) and math.isnan(tr)):
                task["target_ratio"] = round(
                    random.uniform(self._ratio_min, self._ratio_max), 4
                )
            tasks.append(task)
        return tasks

    def get_epoch_tasks(
        self,
        frozen_tasks: list[dict[str, Any]],
        dynamic_tasks: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Combine frozen + dynamic tasks, shuffle, and tag with ``is_dynamic``.

        Args:
            frozen_tasks: tasks from the frozen pool (already loaded/sampled).
            dynamic_tasks: optional freshly-generated dynamic tasks.

        Returns:
            Shuffled combined list; each task has an ``is_dynamic`` bool flag.
        """
        tagged: list[dict[str, Any]] = []
        for t in frozen_tasks:
            tagged.append({**t, "is_dynamic": False})
        for t in (dynamic_tasks or []):
            tagged.append({**t, "is_dynamic": True})
        random.shuffle(tagged)
        return tagged

    def select_frozen_subset(
        self,
        all_frozen: list[dict[str, Any]],
        tasks_per_epoch: int,
    ) -> list[dict[str, Any]]:
        """Randomly sample *tasks_per_epoch* tasks from the frozen pool.

        If the pool is smaller than requested, returns all tasks (no replacement).
        """
        if tasks_per_epoch >= len(all_frozen):
            return list(all_frozen)
        return random.sample(all_frozen, tasks_per_epoch)
