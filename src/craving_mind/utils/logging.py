"""Logging setup: JSONL file handler + human-readable console handler."""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path


class _JsonlFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Merge any extra fields passed via `extra={...}`
        skip = {
            "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
            "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "created", "msecs", "relativeCreated", "thread", "threadName",
            "processName", "process", "message", "taskName",
        }
        for key, val in record.__dict__.items():
            if key not in skip:
                payload[key] = val
        return json.dumps(payload, ensure_ascii=False, default=str)


class _ConsoleFormatter(logging.Formatter):
    """Human-readable console format with timestamps."""

    _FMT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    _DATE = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self._FMT, datefmt=self._DATE)


def setup_logging(run_dir: str = "runs", level: str = "INFO") -> None:
    """Configure root logger with a JSONL file handler and a console handler.

    Args:
        run_dir: Directory where the JSONL log file will be written.
        level:   Log level string, e.g. "INFO", "DEBUG".
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    if root.handlers:
        # Already configured — skip to avoid duplicate handlers on re-import.
        return

    root.setLevel(numeric_level)

    # --- Console handler ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(_ConsoleFormatter())
    root.addHandler(console_handler)

    # --- JSONL file handler ---
    log_path = Path(run_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_file = log_path / f"craving_mind_{timestamp}.jsonl"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)  # capture everything to file
    file_handler.setFormatter(_JsonlFormatter())
    root.addHandler(file_handler)

    # Suppress noisy third-party loggers that flood the console.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
