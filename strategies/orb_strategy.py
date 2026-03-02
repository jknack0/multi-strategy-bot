"""
Strategy B: VIX-Adaptive Opening Range Breakout.

Trades MES during RTH. Adapts the Opening Range duration to the current
VIX regime, then enters on confirmed breakouts with volume and VWAP
slope alignment. Exits via measured-move target, OR-width stop, or time stop.
"""

import logging
import math
from dataclasses import dataclass
from datetime import datetime, time as dt_time, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz

from config.constants import INSTRUMENTS, InstrumentSpec, VIXRegime
from indicators.atr import ATR
from indicators.relative_volume import RelativeVolume
from indicators.vwap import VWAP
from indicators.vwap_slope import VWAPSlope

logger = logging.getLogger(__name__)
ET = pytz.timezone("US/Eastern")


class Side:
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class ORBPosition:
    """Tracks an open ORB position."""
    side: str
    entry_price: float
    entry_time: datetime
    contracts: int
    stop_price: float
    target_price: float
    or_width: float
    max_favorable: float = 0.0
    breakeven_moved: bool = False


@dataclass
class ORBTradeRecord:
    """Completed ORB trade for analysis."""
    side: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    contracts: int
    pnl: float
    exit_reason: str
    or_width: float
    or_duration_minutes: int


class VIXAdaptiveORBStrategy:
    """VIX-Adaptive Opening Range Breakout Strategy.

    Phase 1: Build OR (9:30 to 9:30 + adaptive duration)
    Phase 2: Detect breakouts with filters (volume, VWAP slope, OR width)
    Phase 3: Manage exits (target, stop, trailing, time stop)
    """

    DEFAULT_PARAMS: Dict = {
        # Opening Range
        "vix_or_durations": {
            "LOW": 5,
            "NORMAL": 15,
            "HIGH": 30,
            "EXTREME": 30,
        },
        # Entry filters
        "or_width_min_atr_fraction": 0.3,
        "volume_threshold": 1.5,
        "vwap_slope_lookback": 5,
        # Exit parameters
        "target_or_multiplier": 1.0,
        "stop_or_multiplier": 0.5,
        "trail_trigger_or": 0.75,
        # Time filters
        "max_entry_hour": 14,
        "max_entry_minute": 0,
        "flatten_hour": 15,
        "flatten_minute": 55,
        # Position sizing
        "symbol": "MES",
        "risk_per_trade": 0.01,
        "max_positions": 1,
        "max_margin_pct": 0.50,
    }

    def __init__(
        self,
        params: Optional[Dict] = None,
        capital: float = 20_000.0,
    ) -> None:
        self.params = {**self.DEFAULT_PARAMS}
        if params:
            self.params.update(params)
        self.capital = capital
        self.spec: InstrumentSpec = INSTRUMENTS[self.params["symbol"]]

        # Indicators
        self._vwap = VWAP(num_std_bands=2)
        self._vwap_slope = VWAPSlope(lookback=self.params["vwap_slope_lookback"])
        self._atr = ATR(14)
        self._rvol: Optional[RelativeVolume] = None

        # OR state — resets each trading day
        self._or_high: Optional[float] = None
        self._or_low: Optional[float] = None
        self._or_width: Optional[float] = None
        self._or_complete: bool = False
        self._or_end_time: Optional[datetime] = None
        self._or_duration_minutes: int = 15
        self._current_vix: float = 20.0
        self._current_date = None
        self._traded_today: bool = False

        # Position state
        self.position: Optional[ORBPosition] = None
        self.trades: List[ORBTradeRecord] = []

    def reset(self) -> None:
        """Reset all state for a new backtest run."""
        self._vwap = VWAP(num_std_bands=2)
        self._vwap_slope = VWAPSlope(lookback=self.params["vwap_slope_lookback"])
        self._atr = ATR(14)
        self._or_high = None
        self._or_low = None
        self._or_width = None
        self._or_complete = False
        self._or_end_time = None
        self._or_duration_minutes = 15
        self._current_vix = 20.0
        self._current_date = None
        self._traded_today = False
        self.position = None
        self.trades = []

    def set_vix(self, vix_value: float) -> None:
        """Update current VIX level."""
        self._current_vix = vix_value

    def set_relative_volume(self, rvol: RelativeVolume) -> None:
        """Set the relative volume indicator (built from historical data)."""
        self._rvol = rvol

    # ── Time helpers ──────────────────────────────────────────────────

    def _to_et(self, ts: datetime) -> datetime:
        if ts.tzinfo is None:
            return ET.localize(ts)
        return ts.astimezone(ET)

    def _get_or_duration(self) -> int:
        """Get OR duration in minutes based on current VIX."""
        durations = self.params["vix_or_durations"]
        vix = self._current_vix
        if vix < 15:
            return durations.get("LOW", 5)
        if vix < 25:
            return durations.get("NORMAL", 15)
        if vix < 35:
            return durations.get("HIGH", 30)
        return durations.get("EXTREME", 30)

    def _is_flatten_time(self, ts: datetime) -> bool:
        t = self._to_et(ts).time()
        return t >= dt_time(self.params["flatten_hour"], self.params["flatten_minute"])

    def _past_max_entry_time(self, ts: datetime) -> bool:
        t = self._to_et(ts).time()
        return t >= dt_time(self.params["max_entry_hour"], self.params["max_entry_minute"])

    # ── Session management ────────────────────────────────────────────

    def _on_session_open(self, ts: datetime) -> None:
        """Called when a new trading day is detected. Resets OR state."""
        self._or_high = None
        self._or_low = None
        self._or_width = None
        self._or_complete = False
        self._traded_today = False
        self._or_duration_minutes = self._get_or_duration()
        # OR end time = 9:30 AM ET + duration
        et_ts = self._to_et(ts)
        session_open = et_ts.replace(hour=9, minute=30, second=0, microsecond=0)
        self._or_end_time = session_open + timedelta(minutes=self._or_duration_minutes)
        self._vwap_slope.reset()

    # ── Position sizing ───────────────────────────────────────────────

    def compute_position_size(
        self,
        stop_distance_points: float,
        scale: float = 1.0,
    ) -> int:
        """Compute contracts from OR-width-based stop."""
        if stop_distance_points <= 0:
            return 0
        dollar_risk = self.capital * self.params["risk_per_trade"]
        raw = dollar_risk / (stop_distance_points * self.spec.point_value)
        scaled = math.floor(raw * scale)
        margin_cap = int(self.capital * self.params["max_margin_pct"] / self.spec.margin)
        if margin_cap <= 0 or scaled <= 0:
            return 0
        return min(scaled, margin_cap)

    # ── Core bar processing ───────────────────────────────────────────

    def on_bar(
        self,
        timestamp: datetime,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: int,
    ) -> Optional[ORBTradeRecord]:
        """Process a single bar. Returns ORBTradeRecord if a trade closed."""
        et_ts = self._to_et(timestamp)
        current_date = et_ts.date()

        # New day detection
        if self._current_date != current_date:
            # Flatten any overnight position (shouldn't happen, but safety)
            record = None
            if self.position is not None:
                record = self._close_position(close, timestamp, "end_of_day")
            self._current_date = current_date
            self._on_session_open(timestamp)
            if record is not None:
                return record

        # Update indicators
        vwap_data = self._vwap.update(timestamp, high, low, close, volume)
        self._vwap_slope.update(vwap_data["vwap"])
        atr_val = self._atr.update(high, low, close)

        # Time stop: flatten at 3:55 PM ET
        if self._is_flatten_time(timestamp) and self.position is not None:
            return self._close_position(close, timestamp, "time_stop")

        # Manage existing position
        if self.position is not None:
            return self._manage_exits(close, timestamp)

        # Only process entries if before 9:30 AM skipped (OR build handles this)
        # Skip if before RTH
        if et_ts.time() < dt_time(9, 30):
            return None

        # Phase 1: Build Opening Range
        if not self._or_complete:
            if self._or_high is None:
                self._or_high = high
                self._or_low = low
            else:
                self._or_high = max(self._or_high, high)
                self._or_low = min(self._or_low, low)

            if self._or_end_time is not None and et_ts >= self._or_end_time:
                self._or_complete = True
                self._or_width = self._or_high - self._or_low
            return None

        # Phase 2: Breakout detection
        if self._traded_today:
            return None  # Only one ORB trade per day
        if self._past_max_entry_time(timestamp):
            return None
        if atr_val is None or not self._vwap_slope.is_ready:
            return None

        # Filter: OR width >= min_atr_fraction * ATR
        min_width = self.params["or_width_min_atr_fraction"] * atr_val
        if self._or_width < min_width:
            return None

        # Filter: relative volume
        if self._rvol is not None:
            rvol = self._rvol.get_relative_volume(timestamp, volume)
        else:
            rvol = 1.5  # Pass if no rvol data

        if rvol < self.params["volume_threshold"]:
            return None

        slope = self._vwap_slope.current_slope
        stop_mult = self.params["stop_or_multiplier"]
        target_mult = self.params["target_or_multiplier"]

        # VIX scaling for extreme regimes
        pos_scale = 1.0
        if self._current_vix > 35:
            pos_scale = 0.3
        elif self._current_vix > 25:
            pos_scale = 0.6

        # LONG breakout
        if close > self._or_high and slope is not None and slope > 0:
            stop_dist = stop_mult * self._or_width
            target_dist = target_mult * self._or_width
            contracts = self.compute_position_size(stop_dist, pos_scale)
            if contracts > 0:
                self.position = ORBPosition(
                    side=Side.LONG,
                    entry_price=close,
                    entry_time=timestamp,
                    contracts=contracts,
                    stop_price=close - stop_dist,
                    target_price=close + target_dist,
                    or_width=self._or_width,
                )
                self._traded_today = True
            return None

        # SHORT breakout
        if close < self._or_low and slope is not None and slope < 0:
            stop_dist = stop_mult * self._or_width
            target_dist = target_mult * self._or_width
            contracts = self.compute_position_size(stop_dist, pos_scale)
            if contracts > 0:
                self.position = ORBPosition(
                    side=Side.SHORT,
                    entry_price=close,
                    entry_time=timestamp,
                    contracts=contracts,
                    stop_price=close + stop_dist,
                    target_price=close - target_dist,
                    or_width=self._or_width,
                )
                self._traded_today = True
            return None

        return None

    def _manage_exits(
        self,
        close: float,
        timestamp: datetime,
    ) -> Optional[ORBTradeRecord]:
        """Exit management for open positions."""
        pos = self.position
        trail_trigger = self.params["trail_trigger_or"]

        if pos.side == Side.LONG:
            unrealized = close - pos.entry_price

            # Stop loss
            if close <= pos.stop_price:
                return self._close_position(close, timestamp, "stop_loss")
            # Target hit
            if close >= pos.target_price:
                return self._close_position(close, timestamp, "target_hit")
            # Track max favorable excursion
            if unrealized > pos.max_favorable:
                pos.max_favorable = unrealized
            # Trailing: move to breakeven after trail_trigger * OR_width profit
            if not pos.breakeven_moved and pos.max_favorable >= trail_trigger * pos.or_width:
                pos.stop_price = pos.entry_price
                pos.breakeven_moved = True

        else:  # SHORT
            unrealized = pos.entry_price - close

            if close >= pos.stop_price:
                return self._close_position(close, timestamp, "stop_loss")
            if close <= pos.target_price:
                return self._close_position(close, timestamp, "target_hit")
            if unrealized > pos.max_favorable:
                pos.max_favorable = unrealized
            if not pos.breakeven_moved and pos.max_favorable >= trail_trigger * pos.or_width:
                pos.stop_price = pos.entry_price
                pos.breakeven_moved = True

        return None

    def _close_position(
        self, exit_price: float, exit_time: datetime, reason: str,
    ) -> ORBTradeRecord:
        """Close position and record the trade."""
        pos = self.position
        if pos.side == Side.LONG:
            pnl_per = (exit_price - pos.entry_price) * self.spec.point_value
        else:
            pnl_per = (pos.entry_price - exit_price) * self.spec.point_value
        total_pnl = pnl_per * pos.contracts

        record = ORBTradeRecord(
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            entry_time=pos.entry_time,
            exit_time=exit_time,
            contracts=pos.contracts,
            pnl=total_pnl,
            exit_reason=reason,
            or_width=pos.or_width,
            or_duration_minutes=self._or_duration_minutes,
        )
        self.trades.append(record)
        self.capital += total_pnl
        self.position = None
        return record

    # ── Signal generation for backtesting ──────────────────────────────

    def generate_signals(
        self,
        df: pd.DataFrame,
        vix_series: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Run bar-by-bar backtest and return signal DataFrame.

        Args:
            df: OHLCV DataFrame with DatetimeIndex
            vix_series: Daily VIX values indexed by date

        Returns:
            DataFrame with columns: signal, or_high, or_low, or_width, or_duration
        """
        return self.generate_signals_fast(df, vix_series)

    def generate_signals_fast(
        self,
        df: pd.DataFrame,
        vix_series: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Fast bar-by-bar backtest using pre-extracted numpy arrays.

        Same semantics as generate_signals but avoids df.iloc[] per bar.
        """
        self.reset()
        n = len(df)

        # Pre-extract to numpy arrays (avoids iloc overhead)
        opens = df["open"].values.astype(np.float64)
        highs = df["high"].values.astype(np.float64)
        lows = df["low"].values.astype(np.float64)
        closes = df["close"].values.astype(np.float64)
        volumes = df["volume"].values.astype(np.int64)

        # Pre-convert timestamps to tz-aware datetimes
        raw_index = df.index
        timestamps: List[datetime] = []
        for ts in raw_index:
            if not isinstance(ts, datetime):
                ts = pd.Timestamp(ts).to_pydatetime()
            if ts.tzinfo is None:
                ts = ET.localize(ts)
            timestamps.append(ts)

        # Pre-build VIX lookup by date
        vix_dict = vix_series.to_dict() if vix_series is not None else {}

        signals = np.zeros(n, dtype=np.int8)
        or_highs = np.full(n, np.nan)
        or_lows = np.full(n, np.nan)
        or_widths = np.full(n, np.nan)
        or_durations = np.zeros(n, dtype=np.int16)

        prev_date = None
        for i in range(n):
            ts = timestamps[i]

            # Update VIX only on date change
            d = ts.date()
            if d != prev_date:
                prev_date = d
                if d in vix_dict:
                    self.set_vix(float(vix_dict[d]))

            had_position = self.position is not None

            result = self.on_bar(
                timestamp=ts,
                open_=opens[i],
                high=highs[i],
                low=lows[i],
                close=closes[i],
                volume=int(volumes[i]),
            )

            if not had_position and self.position is not None:
                signals[i] = 1 if self.position.side == Side.LONG else -1

            if self._or_high is not None:
                or_highs[i] = self._or_high
            if self._or_low is not None:
                or_lows[i] = self._or_low
            if self._or_width is not None:
                or_widths[i] = self._or_width
            or_durations[i] = self._or_duration_minutes

        return pd.DataFrame({
            "signal": signals,
            "or_high": or_highs,
            "or_low": or_lows,
            "or_width": or_widths,
            "or_duration": or_durations,
        }, index=df.index)
