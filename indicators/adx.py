"""
Average Directional Index (ADX) using Wilder smoothing.

Measures trend strength regardless of direction.
High ADX (>25) = strong trend, low ADX (<20) = ranging/choppy market.
Used as a regime filter for mean reversion strategies.
"""

from typing import Optional


class ADX:
    """Incremental ADX with Wilder smoothing.

    Computes +DI, -DI, and ADX from directional movement and true range.
    Requires 2 * period bars to fully initialize (period for smoothed DM/TR,
    then period more for smoothed DX -> ADX).

    Usage:
        adx = ADX(period=14)
        value = adx.update(high, low, close)  # Returns None until ready
    """

    def __init__(self, period: int = 14) -> None:
        if period < 1:
            raise ValueError("period must be >= 1")
        self.period = period

        self._prev_high: Optional[float] = None
        self._prev_low: Optional[float] = None
        self._prev_close: Optional[float] = None

        # Phase 1: collect period bars of +DM, -DM, TR for initial SMA
        self._dm_plus_buf: list = []
        self._dm_minus_buf: list = []
        self._tr_buf: list = []

        # Smoothed values (after phase 1)
        self._smooth_dm_plus: Optional[float] = None
        self._smooth_dm_minus: Optional[float] = None
        self._smooth_tr: Optional[float] = None

        # Phase 2: collect period bars of DX for initial ADX SMA
        self._dx_buf: list = []
        self._adx: Optional[float] = None

        self._phase: int = 0  # 0=collecting DM/TR, 1=collecting DX, 2=ready

    def reset(self) -> None:
        self._prev_high = None
        self._prev_low = None
        self._prev_close = None
        self._dm_plus_buf = []
        self._dm_minus_buf = []
        self._tr_buf = []
        self._smooth_dm_plus = None
        self._smooth_dm_minus = None
        self._smooth_tr = None
        self._dx_buf = []
        self._adx = None
        self._phase = 0

    def update(self, high: float, low: float, close: float) -> Optional[float]:
        """Update ADX with a new bar.

        Returns:
            Current ADX value (0-100), or None if not yet initialized.
        """
        if self._prev_high is None:
            self._prev_high = high
            self._prev_low = low
            self._prev_close = close
            return None

        # Directional movement
        up_move = high - self._prev_high
        down_move = self._prev_low - low

        dm_plus = up_move if (up_move > down_move and up_move > 0) else 0.0
        dm_minus = down_move if (down_move > up_move and down_move > 0) else 0.0

        # True range
        tr = max(
            high - low,
            abs(high - self._prev_close),
            abs(low - self._prev_close),
        )

        self._prev_high = high
        self._prev_low = low
        self._prev_close = close

        if self._phase == 0:
            # Collecting initial period bars
            self._dm_plus_buf.append(dm_plus)
            self._dm_minus_buf.append(dm_minus)
            self._tr_buf.append(tr)

            if len(self._dm_plus_buf) >= self.period:
                self._smooth_dm_plus = sum(self._dm_plus_buf)
                self._smooth_dm_minus = sum(self._dm_minus_buf)
                self._smooth_tr = sum(self._tr_buf)
                self._dm_plus_buf = []
                self._dm_minus_buf = []
                self._tr_buf = []
                self._phase = 1

                # Compute first DX
                dx = self._compute_dx()
                if dx is not None:
                    self._dx_buf.append(dx)
            return None

        # Wilder smoothing for DM and TR
        p = self.period
        self._smooth_dm_plus = self._smooth_dm_plus - (self._smooth_dm_plus / p) + dm_plus
        self._smooth_dm_minus = self._smooth_dm_minus - (self._smooth_dm_minus / p) + dm_minus
        self._smooth_tr = self._smooth_tr - (self._smooth_tr / p) + tr

        dx = self._compute_dx()
        if dx is None:
            return self._adx

        if self._phase == 1:
            # Collecting DX values for initial ADX
            self._dx_buf.append(dx)
            if len(self._dx_buf) >= self.period:
                self._adx = sum(self._dx_buf) / self.period
                self._dx_buf = []
                self._phase = 2
                return self._adx
            return None

        # Phase 2: Wilder smooth ADX
        self._adx = (self._adx * (p - 1) + dx) / p
        return self._adx

    def _compute_dx(self) -> Optional[float]:
        """Compute DX from smoothed DM and TR."""
        if self._smooth_tr == 0 or self._smooth_tr is None:
            return None

        di_plus = 100.0 * self._smooth_dm_plus / self._smooth_tr
        di_minus = 100.0 * self._smooth_dm_minus / self._smooth_tr
        di_sum = di_plus + di_minus

        if di_sum == 0:
            return 0.0

        return 100.0 * abs(di_plus - di_minus) / di_sum

    @property
    def current_adx(self) -> Optional[float]:
        return self._adx

    @property
    def is_ready(self) -> bool:
        return self._phase == 2
