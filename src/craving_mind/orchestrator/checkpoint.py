"""Checkpoint serialisation and restoration."""

import json
import os
from datetime import datetime, timezone


class CheckpointManager:
    """Saves and loads run checkpoints for fault tolerance."""

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(run_dir, "checkpoint.json")
        self._epoch_log_path = os.path.join(run_dir, "epoch_log.jsonl")
        self._task_log_path = os.path.join(run_dir, "task_log.jsonl")

    def save(self, state: dict):
        """Save checkpoint: epoch number, success_rate history, R&D fund, best score, crav_id, etc."""
        state_with_ts = {
            **state,
            "saved_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(state_with_ts, f, indent=2)

    def load(self) -> dict | None:
        """Load checkpoint. Returns None if no checkpoint exists."""
        if not os.path.exists(self.checkpoint_path):
            return None
        with open(self.checkpoint_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_epoch_log(self, epoch: int, result: dict):
        """Append epoch result to epoch_log.jsonl."""
        entry = {
            "epoch": epoch,
            **result,
            "ts": datetime.now(tz=timezone.utc).isoformat(),
        }
        with open(self._epoch_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def save_task_log(self, epoch: int, task_result: dict):
        """Append task result to task_log.jsonl."""
        entry = {
            "epoch": epoch,
            **task_result,
            "ts": datetime.now(tz=timezone.utc).isoformat(),
        }
        with open(self._task_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
