"""
Average True Range (ATR) using Wilder smoothing.

Computed incrementally — handles gaps between bars via true range formula.
"""

from typing import Optional


class ATR:
    """Incremental ATR with Wilder smoothing.

    true_range = max(H - L, |H - prev_close|, |L - prev_close|)
    ATR = prev_ATR * (period-1)/period + TR / period  (Wilder smoothing)

    Usage:
        atr = ATR(period=14)
        value = atr.update(high, low, close)
    """

    def __init__(self, period: int = 14) -> None:
        if period < 1:
            raise ValueError("period must be >= 1")
        self.period = period
        self._prev_close: Optional[float] = None
        self._atr: Optional[float] = None
        self._tr_buffer: list = []
        self._initialized: bool = False

    def reset(self) -> None:
        """Reset the ATR state."""
        self._prev_close = None
        self._atr = None
        self._tr_buffer = []
        self._initialized = False

    @staticmethod
    def true_range(high: float, low: float, prev_close: Optional[float]) -> float:
        """Compute true range for a single bar.

        Args:
            high: Current bar high
            low: Current bar low
            prev_close: Previous bar close (None for first bar)

        Returns:
            True range value
        """
        if prev_close is None:
            return high - low
        return max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close),
        )

    def update(self, high: float, low: float, close: float) -> Optional[float]:
        """Update ATR with a new bar.

        Args:
            high: Bar high price
            low: Bar low price
            close: Bar close price

        Returns:
            Current ATR value, or None if not yet initialized (need `period` bars).
        """
        tr = self.true_range(high, low, self._prev_close)
        self._prev_close = close

        if not self._initialized:
            self._tr_buffer.append(tr)
            if len(self._tr_buffer) >= self.period:
                # Initial ATR is SMA of first `period` true ranges
                self._atr = sum(self._tr_buffer) / self.period
                self._initialized = True
                self._tr_buffer = []
                return self._atr
            return None

        # Wilder smoothing
        self._atr = (self._atr * (self.period - 1) + tr) / self.period
        return self._atr

    @property
    def current_atr(self) -> Optional[float]:
        """Return current ATR value without adding new data."""
        return self._atr

    @property
    def is_ready(self) -> bool:
        """True when ATR has been initialized."""
        return self._initialized
