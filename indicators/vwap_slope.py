"""
VWAP Slope indicator using linear regression over a lookback window.

Captures short-term institutional money flow direction.
Positive slope = net buying pressure, negative = net selling pressure.
"""

from collections import deque
from typing import Optional


class VWAPSlope:
    """Compute the slope of VWAP over the last N bars using OLS.

    Usage:
        slope_ind = VWAPSlope(lookback=5)
        slope = slope_ind.update(vwap_value)
    """

    def __init__(self, lookback: int = 5) -> None:
        if lookback < 2:
            raise ValueError("lookback must be >= 2")
        self.lookback = lookback
        self._buffer: deque = deque(maxlen=lookback)
        # Precompute x-related sums for OLS: x = [0, 1, ..., N-1]
        n = lookback
        self._sum_x = n * (n - 1) / 2.0
        self._sum_x2 = n * (n - 1) * (2 * n - 1) / 6.0
        self._n = float(n)
        self._slope: Optional[float] = None

    def reset(self) -> None:
        """Reset all state."""
        self._buffer.clear()
        self._slope = None

    def update(self, vwap_value: float) -> Optional[float]:
        """Feed new VWAP value. Returns slope when buffer is full, else None.

        Slope via OLS:
            slope = (N * sum(x*y) - sum(x) * sum(y)) / (N * sum(x^2) - sum(x)^2)
        """
        self._buffer.append(vwap_value)
        if len(self._buffer) < self.lookback:
            self._slope = None
            return None

        sum_y = 0.0
        sum_xy = 0.0
        for i, y in enumerate(self._buffer):
            sum_y += y
            sum_xy += i * y

        denom = self._n * self._sum_x2 - self._sum_x ** 2
        if denom == 0:
            self._slope = 0.0
            return 0.0

        self._slope = (self._n * sum_xy - self._sum_x * sum_y) / denom
        return self._slope

    def get_normalized_slope(self, atr: float) -> Optional[float]:
        """Return slope normalized by ATR. Comparable across regimes."""
        if self._slope is None or atr <= 0:
            return None
        return self._slope / atr

    @property
    def current_slope(self) -> Optional[float]:
        """Return current slope without adding new data."""
        return self._slope

    @property
    def is_ready(self) -> bool:
        """True when buffer is full."""
        return len(self._buffer) >= self.lookback
