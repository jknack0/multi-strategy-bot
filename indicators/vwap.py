"""
Session VWAP (Volume Weighted Average Price) with standard deviation bands.

Resets at 9:30 AM ET daily (RTH open) for equity index futures.
Computed incrementally — each new bar updates the running totals.
"""

from datetime import datetime, time as dt_time
from typing import Dict, Optional

import pytz

from config.constants import VWAP_RESET_HOUR, VWAP_RESET_MINUTE

ET = pytz.timezone("US/Eastern")


class VWAP:
    """Incremental session VWAP with standard deviation bands.

    Usage:
        vwap = VWAP()
        result = vwap.update(timestamp, high, low, close, volume)
        # result = {'vwap': ..., 'std': ..., 'upper_1': ..., 'lower_1': ..., ...}
    """

    def __init__(self, num_std_bands: int = 2) -> None:
        self.num_std_bands = num_std_bands
        self._cumulative_pv: float = 0.0   # sum(typical_price * volume)
        self._cumulative_v: float = 0.0    # sum(volume)
        self._cumulative_pv2: float = 0.0  # sum(typical_price^2 * volume)
        self._bar_count: int = 0
        self._last_timestamp: Optional[datetime] = None

    def reset(self) -> None:
        """Reset VWAP accumulators for a new session."""
        self._cumulative_pv = 0.0
        self._cumulative_v = 0.0
        self._cumulative_pv2 = 0.0
        self._bar_count = 0

    def _should_reset(self, timestamp: datetime) -> bool:
        """Check if we should reset VWAP (crossed 9:30 ET boundary)."""
        if self._last_timestamp is None:
            return True
        curr_et = timestamp.astimezone(ET) if timestamp.tzinfo else ET.localize(timestamp)
        prev_et = (
            self._last_timestamp.astimezone(ET)
            if self._last_timestamp.tzinfo
            else ET.localize(self._last_timestamp)
        )
        reset_time = dt_time(VWAP_RESET_HOUR, VWAP_RESET_MINUTE)
        return prev_et.time() < reset_time <= curr_et.time()

    def update(
        self,
        timestamp: datetime,
        high: float,
        low: float,
        close: float,
        volume: int,
    ) -> Dict[str, float]:
        """Update VWAP with a new bar and return current values.

        Args:
            timestamp: Bar timestamp
            high: Bar high price
            low: Bar low price
            close: Bar close price
            volume: Bar volume

        Returns:
            Dict with keys: vwap, std, upper_1, lower_1, upper_2, lower_2, ...
        """
        if self._should_reset(timestamp):
            self.reset()

        typical_price = (high + low + close) / 3.0

        self._cumulative_pv += typical_price * volume
        self._cumulative_v += volume
        self._cumulative_pv2 += (typical_price ** 2) * volume
        self._bar_count += 1
        self._last_timestamp = timestamp

        if self._cumulative_v == 0:
            return self._empty_result()

        vwap_val = self._cumulative_pv / self._cumulative_v

        # Compute VWAP standard deviation
        # Var = E[X^2] - E[X]^2 where X is weighted by volume
        mean_p2 = self._cumulative_pv2 / self._cumulative_v
        variance = max(0.0, mean_p2 - vwap_val ** 2)
        std = variance ** 0.5

        result: Dict[str, float] = {
            "vwap": vwap_val,
            "std": std,
        }

        for i in range(1, self.num_std_bands + 1):
            result[f"upper_{i}"] = vwap_val + i * std
            result[f"lower_{i}"] = vwap_val - i * std

        return result

    def _empty_result(self) -> Dict[str, float]:
        result: Dict[str, float] = {"vwap": 0.0, "std": 0.0}
        for i in range(1, self.num_std_bands + 1):
            result[f"upper_{i}"] = 0.0
            result[f"lower_{i}"] = 0.0
        return result

    @property
    def current_vwap(self) -> float:
        if self._cumulative_v == 0:
            return 0.0
        return self._cumulative_pv / self._cumulative_v

    @property
    def bar_count(self) -> int:
        return self._bar_count
