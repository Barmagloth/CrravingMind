"""CUSUM-based success_rate drift monitor for the operator dashboard (spec §3.3)."""

from __future__ import annotations

import math
import statistics
from collections import deque
from typing import Any


class CUSUMMonitor:
    """Sliding-window CUSUM monitor for success_rate drift detection.

    Algorithm (spec §3.3):
        - Maintain a sliding window of the last ``window`` epoch success_rates.
        - target_mean  = mean of values in the window.
        - sigma        = std-dev of values in the window (population std).
        - threshold    = sigma_multiplier * sigma  (default 2σ).
        - cusum        = cumulative sum of (x_i - target_mean) over the window.
        - is_drift()   = True  iff  |cusum| > threshold.

    When fewer values than the window size have been seen, all available
    values are used and is_drift() conservatively returns False until the
    window is full (insufficient data to establish a baseline).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        drift_cfg = cfg.get("judge", {}).get("drift", {})
        self._window_size: int = int(drift_cfg.get("window", 10))
        self._sigma_multiplier: float = float(drift_cfg.get("sigma_multiplier", 2.0))
        self._values: deque[float] = deque(maxlen=self._window_size)

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def update(self, success_rate: float) -> None:
        """Record the success_rate for the most recently completed epoch."""
        self._values.append(success_rate)

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    def is_drift(self) -> bool:
        """Return True if CUSUM exceeds the 2σ threshold.

        Returns False when fewer than 2 values have been observed
        (can't compute standard deviation).
        """
        if len(self._values) < 2:
            return False
        stats = self.get_stats()
        return abs(stats["current_cusum"]) > stats["threshold"]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return current monitor state for the operator dashboard.

        Keys:
            target_mean     — mean of window values.
            current_cusum   — cumulative deviation from target_mean.
            threshold       — sigma_multiplier * sigma.
            window_values   — list of values currently in the window.
        """
        values = list(self._values)
        if not values:
            return {
                "target_mean": 0.0,
                "current_cusum": 0.0,
                "threshold": 0.0,
                "window_values": [],
            }

        target_mean = sum(values) / len(values)

        if len(values) < 2:
            sigma = 0.0
        else:
            variance = sum((v - target_mean) ** 2 for v in values) / len(values)
            sigma = math.sqrt(variance)

        threshold = self._sigma_multiplier * sigma
        cusum = sum(v - target_mean for v in values)

        return {
            "target_mean": target_mean,
            "current_cusum": cusum,
            "threshold": threshold,
            "window_values": values,
        }
