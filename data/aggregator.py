"""
Real-time bar aggregator: converts raw 5-second bars into 1-min and 5-min OHLCV bars.

Handles session boundaries (futures session reset at 18:00 ET, VWAP reset at 9:30 ET).
Uses a callback pattern so downstream consumers can subscribe to completed bars.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time
from typing import Callable, List, Optional

import pytz

from config.constants import (
    FUTURES_SESSION_OPEN_HOUR,
    FUTURES_SESSION_OPEN_MINUTE,
    VWAP_RESET_HOUR,
    VWAP_RESET_MINUTE,
)

logger = logging.getLogger(__name__)

ET = pytz.timezone("US/Eastern")


@dataclass
class OHLCVBar:
    """Completed OHLCV bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    bar_size: str
    symbol: str = ""
    vwap: float = 0.0


@dataclass
class _BarAccumulator:
    """Internal accumulator for building a bar from ticks/sub-bars."""
    open: Optional[float] = None
    high: float = float("-inf")
    low: float = float("inf")
    close: float = 0.0
    volume: int = 0
    first_timestamp: Optional[datetime] = None
    bar_count: int = 0

    def update(self, timestamp: datetime, open_: float, high: float, low: float, close: float, volume: int) -> None:
        if self.open is None:
            self.open = open_
            self.first_timestamp = timestamp
        self.high = max(self.high, high)
        self.low = min(self.low, low)
        self.close = close
        self.volume += volume
        self.bar_count += 1

    def reset(self) -> None:
        self.open = None
        self.high = float("-inf")
        self.low = float("inf")
        self.close = 0.0
        self.volume = 0
        self.first_timestamp = None
        self.bar_count = 0

    @property
    def is_empty(self) -> bool:
        return self.open is None


class BarAggregator:
    """Aggregates raw 5-second bars into 1-minute and 5-minute OHLCV bars.

    Usage:
        agg = BarAggregator(symbol="MES")
        agg.on_bar_complete = my_callback  # receives OHLCVBar
        agg.process_bar(timestamp, o, h, l, c, v)
    """

    # 5-sec bars per interval
    BARS_PER_1MIN: int = 12   # 60 / 5
    BARS_PER_5MIN: int = 60   # 300 / 5

    def __init__(self, symbol: str = "MES") -> None:
        self.symbol = symbol
        self._1min_acc = _BarAccumulator()
        self._5min_acc = _BarAccumulator()
        self._callbacks: List[Callable[[OHLCVBar], None]] = []
        self._last_bar_time: Optional[datetime] = None

    @property
    def on_bar_complete(self) -> Optional[Callable[[OHLCVBar], None]]:
        return self._callbacks[0] if self._callbacks else None

    @on_bar_complete.setter
    def on_bar_complete(self, callback: Callable[[OHLCVBar], None]) -> None:
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def add_callback(self, callback: Callable[[OHLCVBar], None]) -> None:
        """Register an additional bar-complete callback."""
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def _emit(self, bar: OHLCVBar) -> None:
        """Notify all registered callbacks."""
        for cb in self._callbacks:
            try:
                cb(bar)
            except Exception as exc:
                logger.error("Callback error: %s", exc)

    @staticmethod
    def _is_session_boundary(current: datetime, previous: Optional[datetime]) -> bool:
        """Check if we crossed the futures session boundary (18:00 ET)."""
        if previous is None:
            return False
        curr_et = current.astimezone(ET) if current.tzinfo else ET.localize(current)
        prev_et = previous.astimezone(ET) if previous.tzinfo else ET.localize(previous)
        boundary = dt_time(FUTURES_SESSION_OPEN_HOUR, FUTURES_SESSION_OPEN_MINUTE)
        return prev_et.time() < boundary <= curr_et.time()

    @staticmethod
    def is_vwap_reset(current: datetime, previous: Optional[datetime]) -> bool:
        """Check if we crossed the VWAP reset boundary (9:30 ET)."""
        if previous is None:
            return False
        curr_et = current.astimezone(ET) if current.tzinfo else ET.localize(current)
        prev_et = previous.astimezone(ET) if previous.tzinfo else ET.localize(previous)
        reset = dt_time(VWAP_RESET_HOUR, VWAP_RESET_MINUTE)
        return prev_et.time() < reset <= curr_et.time()

    def _flush_accumulators(self, timestamp: datetime) -> None:
        """Flush any in-progress bars (used at session boundaries)."""
        if not self._1min_acc.is_empty:
            bar = OHLCVBar(
                timestamp=self._1min_acc.first_timestamp or timestamp,
                open=self._1min_acc.open,
                high=self._1min_acc.high,
                low=self._1min_acc.low,
                close=self._1min_acc.close,
                volume=self._1min_acc.volume,
                bar_size="1min",
                symbol=self.symbol,
            )
            self._emit(bar)
            self._1min_acc.reset()

        if not self._5min_acc.is_empty:
            bar = OHLCVBar(
                timestamp=self._5min_acc.first_timestamp or timestamp,
                open=self._5min_acc.open,
                high=self._5min_acc.high,
                low=self._5min_acc.low,
                close=self._5min_acc.close,
                volume=self._5min_acc.volume,
                bar_size="5min",
                symbol=self.symbol,
            )
            self._emit(bar)
            self._5min_acc.reset()

    def process_bar(
        self,
        timestamp: datetime,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: int,
    ) -> None:
        """Process one raw 5-second bar.

        Accumulates into 1-min and 5-min bars and emits them when complete.
        Resets accumulators at session boundaries.
        """
        # Check for session boundary
        if self._is_session_boundary(timestamp, self._last_bar_time):
            logger.info("Session boundary detected at %s, flushing bars", timestamp)
            self._flush_accumulators(timestamp)

        # Update 1-min accumulator
        self._1min_acc.update(timestamp, open_, high, low, close, volume)
        if self._1min_acc.bar_count >= self.BARS_PER_1MIN:
            bar = OHLCVBar(
                timestamp=self._1min_acc.first_timestamp or timestamp,
                open=self._1min_acc.open,
                high=self._1min_acc.high,
                low=self._1min_acc.low,
                close=self._1min_acc.close,
                volume=self._1min_acc.volume,
                bar_size="1min",
                symbol=self.symbol,
            )
            self._emit(bar)
            self._1min_acc.reset()

        # Update 5-min accumulator
        self._5min_acc.update(timestamp, open_, high, low, close, volume)
        if self._5min_acc.bar_count >= self.BARS_PER_5MIN:
            bar = OHLCVBar(
                timestamp=self._5min_acc.first_timestamp or timestamp,
                open=self._5min_acc.open,
                high=self._5min_acc.high,
                low=self._5min_acc.low,
                close=self._5min_acc.close,
                volume=self._5min_acc.volume,
                bar_size="5min",
                symbol=self.symbol,
            )
            self._emit(bar)
            self._5min_acc.reset()

        self._last_bar_time = timestamp
