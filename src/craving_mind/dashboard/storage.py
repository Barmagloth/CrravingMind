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

    def get_console_lines(self, limit: int = 200) -> list[str]:
        """Read task_log.jsonl and format each entry as human-readable log lines.

        Each task entry expands into a header line, optional tool-call sub-lines,
        an optional Crav-text line, and a Judge verdict line.
        """
        path = os.path.join(self.run_dir, "task_log.jsonl")
        entries = self._read_jsonl(path)
        lines: list[str] = []
        for e in entries[-limit:]:
            epoch = e.get("epoch", "?")
            task_idx = e.get("task_idx", "?")
            tasks_total = e.get("tasks_total", "?")
            task_id = e.get("task_id", "")
            hidden_type = e.get("hidden_type", "")
            target_ratio = e.get("target_ratio")
            sem = e.get("semantic_score")
            ent = e.get("entity_score")
            toks = e.get("tokens_spent")
            passed = e.get("passed")

            prefix = f"[E{epoch}][T{task_idx}/{tasks_total}]"
            label = f"{hidden_type}/{task_id}" if hidden_type and task_id else task_id or hidden_type
            ratio_str = f" ratio={target_ratio:.2f}" if target_ratio is not None else ""

            # Header line
            lines.append(f"{prefix} {label}{ratio_str}")

            # Tool-call sub-lines
            for tc in e.get("tool_calls", []):
                name = tc.get("name", "?")
                args = tc.get("args", {})
                result = tc.get("result", "")

                if name == "write_file":
                    args_str = f"({args.get('filename', '?')})"
                elif name == "run_compress":
                    text_len = args.get("text_len", "?")
                    ratio = args.get("ratio")
                    args_str = f"({text_len} chars, ratio={ratio:.2f})" if ratio is not None else f"({text_len} chars)"
                elif name == "read_file":
                    args_str = f"({args.get('filename', '?')})"
                elif name == "run_script":
                    args_str = "(script)"
                else:
                    args_str = ""

                ok = "OK" if not result.startswith("FAIL") else result
                lines.append(f"  \u2192 {name}{args_str} {ok}")

            # Crav thinking (abbreviated)
            crav_text = e.get("crav_text", "")
            if crav_text:
                lines.append(f'  Crav: "{crav_text}"')

            # Judge verdict line
            toks_str = f" ({toks} tok)" if toks is not None else ""
            verdict = "PASS" if passed else "FAIL"
            sem_str = f"sem={sem:.2f}" if sem is not None else "sem=?"
            ent_str = f"ent={ent:.2f}" if ent is not None else "ent=?"
            lines.append(f"  Judge: {sem_str} {ent_str} {verdict}{toks_str}")

        return lines

    def get_live_state(self) -> dict | None:
        """Read live_state.json written by the runner per-task."""
        path = os.path.join(self.run_dir, "live_state.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
