"""
Bollinger Bands indicator with incremental computation.

Returns SMA(period) +/- num_std * StdDev, plus %B and bandwidth.
Uses a rolling window internally — no look-ahead.
"""

from collections import deque
from typing import Dict, Optional


class BollingerBands:
    """Incremental Bollinger Bands.

    Usage:
        bb = BollingerBands(period=20, num_std=2.0)
        result = bb.update(close_price)
        # result = {'middle': ..., 'upper': ..., 'lower': ..., 'pct_b': ..., 'bandwidth': ...}
    """

    def __init__(self, period: int = 20, num_std: float = 2.0) -> None:
        if period < 2:
            raise ValueError("period must be >= 2")
        self.period = period
        self.num_std = num_std
        self._window: deque = deque(maxlen=period)
        self._sum: float = 0.0
        self._sum_sq: float = 0.0

    def reset(self) -> None:
        """Clear the rolling window."""
        self._window.clear()
        self._sum = 0.0
        self._sum_sq = 0.0

    def update(self, close: float) -> Optional[Dict[str, float]]:
        """Add a new close price and return Bollinger Band values.

        Returns None until the window is full (period bars received).

        Returns:
            Dict with keys: middle, upper, lower, pct_b, bandwidth
            or None if not enough data yet.
        """
        # If window is full, remove the oldest value from running sums
        if len(self._window) == self.period:
            old = self._window[0]
            self._sum -= old
            self._sum_sq -= old * old

        self._window.append(close)
        self._sum += close
        self._sum_sq += close * close

        if len(self._window) < self.period:
            return None

        n = self.period
        mean = self._sum / n
        # Population std (matching typical TA implementations)
        variance = (self._sum_sq / n) - (mean * mean)
        # Guard against floating point issues
        variance = max(0.0, variance)
        std = variance ** 0.5

        upper = mean + self.num_std * std
        lower = mean - self.num_std * std

        # %B = (close - lower) / (upper - lower)
        band_width_abs = upper - lower
        if band_width_abs > 0:
            pct_b = (close - lower) / band_width_abs
        else:
            pct_b = 0.5  # Bands collapsed to a point

        # Bandwidth = (upper - lower) / middle
        bandwidth = band_width_abs / mean if mean != 0 else 0.0

        return {
            "middle": mean,
            "upper": upper,
            "lower": lower,
            "pct_b": pct_b,
            "bandwidth": bandwidth,
        }

    @property
    def is_ready(self) -> bool:
        """True when enough data has been received to compute bands."""
        return len(self._window) >= self.period

    @property
    def current_values(self) -> Optional[Dict[str, float]]:
        """Return current Bollinger Band values without adding new data."""
        if not self.is_ready:
            return None
        n = self.period
        mean = self._sum / n
        variance = max(0.0, (self._sum_sq / n) - (mean * mean))
        std = variance ** 0.5
        upper = mean + self.num_std * std
        lower = mean - self.num_std * std
        last_close = self._window[-1]
        band_width_abs = upper - lower
        pct_b = (last_close - lower) / band_width_abs if band_width_abs > 0 else 0.5
        bandwidth = band_width_abs / mean if mean != 0 else 0.0
        return {
            "middle": mean,
            "upper": upper,
            "lower": lower,
            "pct_b": pct_b,
            "bandwidth": bandwidth,
        }
