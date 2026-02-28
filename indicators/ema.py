"""
Exponential Moving Average (EMA) with incremental computation.

Seeded with SMA of the first ``period`` values, then applies standard
EMA smoothing: EMA = close * k + prev_EMA * (1 - k), where k = 2 / (period + 1).
"""

from typing import Optional


class EMA:
    """Incremental EMA indicator.

    Usage:
        ema = EMA(period=50)
        value = ema.update(close)
    """

    def __init__(self, period: int = 50) -> None:
        if period < 1:
            raise ValueError("period must be >= 1")
        self.period = period
        self._k: float = 2.0 / (period + 1.0)
        self._ema: Optional[float] = None
        self._buffer: list = []
        self._initialized: bool = False

    def reset(self) -> None:
        """Reset the EMA state."""
        self._ema = None
        self._buffer = []
        self._initialized = False

    def update(self, close: float) -> Optional[float]:
        """Update EMA with a new close price.

        Args:
            close: Bar close price

        Returns:
            Current EMA value, or None if not yet initialized.
        """
        if not self._initialized:
            self._buffer.append(close)
            if len(self._buffer) >= self.period:
                self._ema = sum(self._buffer) / self.period
                self._initialized = True
                self._buffer = []
                return self._ema
            return None

        self._ema = close * self._k + self._ema * (1.0 - self._k)
        return self._ema

    @property
    def current_value(self) -> Optional[float]:
        """Return current EMA value without adding new data."""
        return self._ema

    @property
    def is_ready(self) -> bool:
        """True when EMA has been initialized."""
        return self._initialized
