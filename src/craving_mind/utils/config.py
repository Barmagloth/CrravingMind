"""Config loader: YAML → dict with required-key validation."""

from pathlib import Path
from typing import Any

import yaml

_REQUIRED_TOP_LEVEL_KEYS = [
    "agent",
    "judge",
    "budget",
    "phases",
    "benchmark",
    "memory",
    "sandbox",
    "dashboard",
    "scoring",
    "drift",
    "inheritance",
    "logging",
    "dedup",
]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and validate required top-level keys.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed config as a plain dict.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If required top-level keys are missing.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        config: dict[str, Any] = yaml.safe_load(fh) or {}

    missing = [k for k in _REQUIRED_TOP_LEVEL_KEYS if k not in config]
    if missing:
        raise ValueError(f"Config missing required top-level keys: {missing}")

    return config
