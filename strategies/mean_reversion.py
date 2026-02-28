"""
Strategy A: MES Bollinger/VWAP Mean Reversion.

Trades MES 5-min bars during RTH, entering when price touches Bollinger
lower/upper band AND deviates from session VWAP.  Exits at VWAP cross,
ATR-based stop, or time stop at 3:55 PM ET.
"""

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz

from config.constants import INSTRUMENTS, InstrumentSpec, VIXRegime
from indicators.atr import ATR
from indicators.bollinger import BollingerBands
from indicators.vix_regime import VIXRegimeClassifier
from indicators.vwap import VWAP

logger = logging.getLogger(__name__)
ET = pytz.timezone("US/Eastern")


class Side(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Position:
    """Tracks an open position."""
    side: Side
    entry_price: float
    entry_time: datetime
    contracts: int
    stop_price: float
    atr_at_entry: float
    target_price: float = 0.0
    max_favorable: float = 0.0
    trailing_active: bool = False
    breakeven_moved: bool = False


@dataclass
class TradeRecord:
    """Completed trade for analysis."""
    side: Side
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    contracts: int
    pnl: float
    exit_reason: str


class MESMeanReversionStrategy:
    """MES Bollinger/VWAP mean-reversion strategy.

    Parameters are fully configurable and can be overridden at init
    or loaded from a JSON file.
    """

    DEFAULT_PARAMS: Dict = {
        "symbol": "MES",
        "bar_size": "5min",
        "bb_period": 20,
        "bb_sigma": 2.0,
        "atr_period": 14,
        "atr_stop_multiplier": 3.5,
        "vwap_deviation_entry": 1.0,
        "trade_start_hour": 10,
        "trade_start_minute": 0,
        "trade_end_hour": 14,
        "trade_end_minute": 0,
        "flatten_hour": 15,
        "flatten_minute": 55,
        "max_positions": 1,
        "risk_per_trade": 0.01,
        "max_margin_pct": 0.50,
        "breakeven_atr_mult": 0.7,
        # Trend filter: 0 = disabled, >0 = EMA period
        "trend_ema_period": 0,
        # Partial TP: fraction of entry-to-VWAP distance (1.0 = full VWAP)
        "take_profit_pct": 1.0,
        # Trailing stop: 0.0 = legacy breakeven, >0 = trail at factor * max_favorable
        "trailing_stop_factor": 0.6,
        # R:R filter: skip entries with reward/risk below this (0.0 = no filter)
        "min_rr_ratio": 0.0,
        # Daily risk limits: 0 = unlimited
        "max_daily_losses": 0,
        "max_daily_loss_dollars": 0.0,
        # Long only mode: skip all short entries
        "long_only": False,
    }

    def __init__(self, params: Optional[Dict] = None, capital: float = 10000.0) -> None:
        self.params = {**self.DEFAULT_PARAMS}
        if params:
            self.params.update(params)

        self.capital = capital
        self.spec: InstrumentSpec = INSTRUMENTS[self.params["symbol"]]

        # Indicators
        self._bb = BollingerBands(
            period=self.params["bb_period"],
            num_std=self.params["bb_sigma"],
        )
        self._atr = ATR(period=self.params["atr_period"])
        self._vwap = VWAP(num_std_bands=2)
        self._vix_clf = VIXRegimeClassifier()

        # EMA for trend filter
        self._ema = None
        if self.params["trend_ema_period"] > 0:
            from indicators.ema import EMA
            self._ema = EMA(period=self.params["trend_ema_period"])

        # State
        self.position: Optional[Position] = None
        self.trades: List[TradeRecord] = []
        self._current_vix: float = 20.0  # Default normal

        # Daily risk state
        self._daily_losses: int = 0
        self._daily_loss_dollars: float = 0.0
        self._last_trade_date = None
        self._daily_halted: bool = False

    # ── Configuration ────────────────────────────────────────────────────

    @classmethod
    def from_json(cls, path: str, capital: float = 10000.0) -> "MESMeanReversionStrategy":
        """Load strategy parameters from a JSON file."""
        with open(path) as f:
            params = json.load(f)
        return cls(params=params, capital=capital)

    def save_params(self, path: str) -> None:
        """Save current parameters to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.params, f, indent=2)

    def reset(self) -> None:
        """Reset all state for a new backtest run."""
        self._bb = BollingerBands(
            period=self.params["bb_period"],
            num_std=self.params["bb_sigma"],
        )
        self._atr = ATR(period=self.params["atr_period"])
        self._vwap = VWAP(num_std_bands=2)
        self._vix_clf = VIXRegimeClassifier()
        if self.params["trend_ema_period"] > 0:
            from indicators.ema import EMA
            self._ema = EMA(period=self.params["trend_ema_period"])
        else:
            self._ema = None
        self.position = None
        self.trades = []
        self._current_vix = 20.0
        self._daily_losses = 0
        self._daily_loss_dollars = 0.0
        self._last_trade_date = None
        self._daily_halted = False

    # ── Time Filters ─────────────────────────────────────────────────────

    def _to_et(self, ts: datetime) -> datetime:
        if ts.tzinfo is None:
            return ET.localize(ts)
        return ts.astimezone(ET)

    def _in_trading_window(self, ts: datetime) -> bool:
        t = self._to_et(ts).time()
        start = dt_time(self.params["trade_start_hour"], self.params["trade_start_minute"])
        end = dt_time(self.params["trade_end_hour"], self.params["trade_end_minute"])
        return start <= t < end

    def _is_flatten_time(self, ts: datetime) -> bool:
        t = self._to_et(ts).time()
        flatten = dt_time(self.params["flatten_hour"], self.params["flatten_minute"])
        return t >= flatten

    # ── VIX Regime Adjustments ───────────────────────────────────────────

    def set_vix(self, vix_value: float) -> None:
        """Update the current VIX level."""
        self._current_vix = vix_value

    def _get_vix_adjustments(self) -> Tuple[float, float, bool]:
        """Return (effective_bb_sigma, position_scale, trade_allowed)."""
        vix = self._current_vix
        if vix > 35:
            return self.params["bb_sigma"], 0.0, False
        if vix > 25:
            return 2.5, 0.5, True
        return self.params["bb_sigma"], 1.0, True

    # ── Position Sizing ──────────────────────────────────────────────────

    def compute_position_size(
        self,
        stop_distance_points: float,
        position_scale: float,
    ) -> int:
        """Compute number of contracts based on risk budget.

        dollar_risk = capital * risk_per_trade
        contracts = floor(dollar_risk / (stop_distance * point_value)) * position_scale
        Capped at max_margin_pct of margin capacity.
        """
        if stop_distance_points <= 0:
            return 0

        dollar_risk = self.capital * self.params["risk_per_trade"]
        raw_contracts = dollar_risk / (stop_distance_points * self.spec.point_value)
        scaled = math.floor(raw_contracts * position_scale)

        # Cap at margin capacity
        margin_capacity = (self.capital * self.params["max_margin_pct"]) / self.spec.margin
        max_contracts = int(margin_capacity)

        if max_contracts <= 0 or scaled <= 0:
            return 0

        return min(scaled, max_contracts)

    # ── Signal Generation ────────────────────────────────────────────────

    def on_bar(
        self,
        timestamp: datetime,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: int,
    ) -> Optional[TradeRecord]:
        """Process a single bar. Returns a TradeRecord if a trade closed.

        This is the core method called for each bar during live trading
        or event-driven backtesting.
        """
        # Update indicators
        vwap_data = self._vwap.update(timestamp, high, low, close, volume)
        bb_data = self._bb.update(close)
        atr_val = self._atr.update(high, low, close)
        ema_val = self._ema.update(close) if self._ema is not None else None

        # Daily risk reset
        current_date = self._to_et(timestamp).date()
        if self._last_trade_date is not None and current_date != self._last_trade_date:
            self._daily_losses = 0
            self._daily_loss_dollars = 0.0
            self._daily_halted = False
        self._last_trade_date = current_date

        # Time stop: flatten at 3:55 PM ET (only losing/flat positions)
        if self._is_flatten_time(timestamp) and self.position is not None:
            pos = self.position
            if pos.side == Side.LONG:
                unrealized = close - pos.entry_price
            else:
                unrealized = pos.entry_price - close
            if unrealized <= 0:
                return self._close_position(close, timestamp, "time_stop")

        # Manage existing position
        if self.position is not None:
            return self._manage_position(close, timestamp, vwap_data, atr_val)

        # Daily halt check
        if self._daily_halted:
            return None

        # Need all indicators ready before entering
        if bb_data is None or atr_val is None:
            return None

        # Time filter
        if not self._in_trading_window(timestamp):
            return None

        # VIX filter
        eff_sigma, pos_scale, allowed = self._get_vix_adjustments()
        if not allowed:
            return None

        # Recompute BB with effective sigma if VIX widened it
        if eff_sigma != self.params["bb_sigma"]:
            bb_adj = self._compute_adjusted_bb(close, eff_sigma)
            if bb_adj is None:
                return None
            bb_upper = bb_adj["upper"]
            bb_lower = bb_adj["lower"]
        else:
            bb_upper = bb_data["upper"]
            bb_lower = bb_data["lower"]

        vwap_val = vwap_data["vwap"]
        vwap_std = vwap_data["std"]
        vwap_dev = self.params["vwap_deviation_entry"]
        stop_mult = self.params["atr_stop_multiplier"]
        stop_dist = stop_mult * atr_val
        tp_pct = self.params["take_profit_pct"]
        min_rr = self.params["min_rr_ratio"]
        trend_period = self.params["trend_ema_period"]

        # LONG signal
        if close <= bb_lower and close < vwap_val - vwap_dev * vwap_std:
            # Trend filter: skip longs in downtrend
            if trend_period > 0 and ema_val is not None and close < ema_val:
                return None
            # Compute partial TP target and R:R
            target_price = close + tp_pct * (vwap_val - close)
            if min_rr > 0 and stop_dist > 0:
                rr = (target_price - close) / stop_dist
                if rr < min_rr:
                    return None
            contracts = self.compute_position_size(stop_dist, pos_scale)
            if contracts > 0 and self.position is None:
                self.position = Position(
                    side=Side.LONG,
                    entry_price=close,
                    entry_time=timestamp,
                    contracts=contracts,
                    stop_price=close - stop_dist,
                    atr_at_entry=atr_val,
                    target_price=target_price,
                )
            return None

        # SHORT signal
        if not self.params.get("long_only", False):
            if close >= bb_upper and close > vwap_val + vwap_dev * vwap_std:
                # Trend filter: skip shorts in uptrend
                if trend_period > 0 and ema_val is not None and close > ema_val:
                    return None
                target_price = close - tp_pct * (close - vwap_val)
                if min_rr > 0 and stop_dist > 0:
                    rr = (close - target_price) / stop_dist
                    if rr < min_rr:
                        return None
                contracts = self.compute_position_size(stop_dist, pos_scale)
                if contracts > 0 and self.position is None:
                    self.position = Position(
                        side=Side.SHORT,
                        entry_price=close,
                        entry_time=timestamp,
                        contracts=contracts,
                        stop_price=close + stop_dist,
                        atr_at_entry=atr_val,
                        target_price=target_price,
                    )
                return None

        return None

    def _compute_adjusted_bb(self, close: float, sigma: float) -> Optional[Dict[str, float]]:
        """Recompute Bollinger Bands with a different sigma using the existing window."""
        vals = self._bb.current_values
        if vals is None:
            return None
        middle = vals["middle"]
        # Recover std from existing bandwidth: bandwidth = (upper - lower) / middle
        # (upper - lower) = bb_sigma * 2 * std => std = (upper - lower) / (2 * bb_sigma)
        if self.params["bb_sigma"] != 0:
            orig_half = (vals["upper"] - vals["lower"]) / 2.0
            std = orig_half / self.params["bb_sigma"]
        else:
            std = 0.0
        upper = middle + sigma * std
        lower = middle - sigma * std
        band_width = upper - lower
        pct_b = (close - lower) / band_width if band_width > 0 else 0.5
        return {
            "middle": middle,
            "upper": upper,
            "lower": lower,
            "pct_b": pct_b,
            "bandwidth": band_width / middle if middle != 0 else 0.0,
        }

    # ── Position Management ──────────────────────────────────────────────

    def _manage_position(
        self,
        close: float,
        timestamp: datetime,
        vwap_data: Dict[str, float],
        atr_val: Optional[float],
    ) -> Optional[TradeRecord]:
        pos = self.position
        trailing_factor = self.params["trailing_stop_factor"]

        if pos.side == Side.LONG:
            unrealized = close - pos.entry_price

            # Stop loss
            if close <= pos.stop_price:
                return self._close_position(close, timestamp, "stop_loss")
            # Take profit at target
            if pos.target_price > 0 and close >= pos.target_price:
                return self._close_position(close, timestamp, "take_profit")

            # Track max favorable excursion
            if unrealized > pos.max_favorable:
                pos.max_favorable = unrealized

            # Trailing stop logic
            if trailing_factor > 0.0 and atr_val is not None:
                activation = self.params["breakeven_atr_mult"] * pos.atr_at_entry
                if pos.max_favorable >= activation:
                    pos.trailing_active = True
                    new_stop = pos.entry_price + trailing_factor * pos.max_favorable
                    if new_stop > pos.stop_price:
                        pos.stop_price = new_stop
            elif not pos.breakeven_moved and atr_val is not None:
                # Legacy breakeven logic
                be_threshold = self.params["breakeven_atr_mult"] * pos.atr_at_entry
                if unrealized >= be_threshold:
                    pos.stop_price = pos.entry_price
                    pos.breakeven_moved = True

        else:  # SHORT
            unrealized = pos.entry_price - close

            if close >= pos.stop_price:
                return self._close_position(close, timestamp, "stop_loss")
            if pos.target_price > 0 and close <= pos.target_price:
                return self._close_position(close, timestamp, "take_profit")

            if unrealized > pos.max_favorable:
                pos.max_favorable = unrealized

            if trailing_factor > 0.0 and atr_val is not None:
                activation = self.params["breakeven_atr_mult"] * pos.atr_at_entry
                if pos.max_favorable >= activation:
                    pos.trailing_active = True
                    new_stop = pos.entry_price - trailing_factor * pos.max_favorable
                    if new_stop < pos.stop_price:
                        pos.stop_price = new_stop
            elif not pos.breakeven_moved and atr_val is not None:
                be_threshold = self.params["breakeven_atr_mult"] * pos.atr_at_entry
                if unrealized >= be_threshold:
                    pos.stop_price = pos.entry_price
                    pos.breakeven_moved = True

        return None

    def _close_position(
        self, exit_price: float, exit_time: datetime, reason: str
    ) -> TradeRecord:
        pos = self.position
        if pos.side == Side.LONG:
            pnl_per_contract = (exit_price - pos.entry_price) * self.spec.point_value
        else:
            pnl_per_contract = (pos.entry_price - exit_price) * self.spec.point_value
        total_pnl = pnl_per_contract * pos.contracts

        record = TradeRecord(
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            entry_time=pos.entry_time,
            exit_time=exit_time,
            contracts=pos.contracts,
            pnl=total_pnl,
            exit_reason=reason,
        )
        self.trades.append(record)
        self.capital += total_pnl
        self.position = None

        # Daily loss tracking
        if total_pnl < 0:
            self._daily_losses += 1
            self._daily_loss_dollars += abs(total_pnl)
            max_losses = self.params["max_daily_losses"]
            max_loss_dollars = self.params["max_daily_loss_dollars"]
            if max_losses > 0 and self._daily_losses >= max_losses:
                self._daily_halted = True
            if max_loss_dollars > 0 and self._daily_loss_dollars >= max_loss_dollars:
                self._daily_halted = True

        return record

    # ── Optimized Signal Generation ─────────────────────────────────────

    def _process_bar_precomputed(
        self,
        timestamp: datetime,
        close: float,
        vwap_val: float,
        vwap_std: float,
        bb_mean: float,
        bb_std: float,
        atr_val: float,
        in_window: bool,
        is_flatten: bool,
        ema_val: float = float("nan"),
    ) -> Optional[TradeRecord]:
        """Process a bar using pre-computed indicator values.

        Same logic as on_bar() but skips indicator computation.
        Used by generate_signals_fast() with precomputed arrays.
        """
        # Daily risk reset
        current_date = timestamp.date() if hasattr(timestamp, "date") else timestamp
        if self._last_trade_date is not None and current_date != self._last_trade_date:
            self._daily_losses = 0
            self._daily_loss_dollars = 0.0
            self._daily_halted = False
        self._last_trade_date = current_date

        # Time stop: flatten losing/flat positions at 3:55 PM ET
        if is_flatten and self.position is not None:
            pos = self.position
            if pos.side == Side.LONG:
                unrealized = close - pos.entry_price
            else:
                unrealized = pos.entry_price - close
            if unrealized <= 0:
                return self._close_position(close, timestamp, "time_stop")

        # Manage existing position
        if self.position is not None:
            vwap_data = {"vwap": vwap_val, "std": vwap_std}
            return self._manage_position(close, timestamp, vwap_data, atr_val)

        # Daily halt check
        if self._daily_halted:
            return None

        # Need all indicators ready
        if np.isnan(bb_mean) or np.isnan(atr_val):
            return None

        # Time filter
        if not in_window:
            return None

        # VIX filter
        eff_sigma, pos_scale, allowed = self._get_vix_adjustments()
        if not allowed:
            return None

        # Compute effective BB bands
        if eff_sigma != self.params["bb_sigma"]:
            bb_upper = bb_mean + eff_sigma * bb_std
            bb_lower = bb_mean - eff_sigma * bb_std
        else:
            bb_upper = bb_mean + self.params["bb_sigma"] * bb_std
            bb_lower = bb_mean - self.params["bb_sigma"] * bb_std

        vwap_dev = self.params["vwap_deviation_entry"]
        stop_mult = self.params["atr_stop_multiplier"]
        stop_dist = stop_mult * atr_val
        tp_pct = self.params["take_profit_pct"]
        min_rr = self.params["min_rr_ratio"]
        trend_period = self.params["trend_ema_period"]

        # LONG signal
        if close <= bb_lower and close < vwap_val - vwap_dev * vwap_std:
            # Trend filter: skip longs in downtrend
            if trend_period > 0 and not np.isnan(ema_val) and close < ema_val:
                return None
            target_price = close + tp_pct * (vwap_val - close)
            if min_rr > 0 and stop_dist > 0:
                rr = (target_price - close) / stop_dist
                if rr < min_rr:
                    return None
            contracts = self.compute_position_size(stop_dist, pos_scale)
            if contracts > 0 and self.position is None:
                self.position = Position(
                    side=Side.LONG,
                    entry_price=close,
                    entry_time=timestamp,
                    contracts=contracts,
                    stop_price=close - stop_dist,
                    atr_at_entry=atr_val,
                    target_price=target_price,
                )
            return None

        # SHORT signal
        if not self.params.get("long_only", False):
            if close >= bb_upper and close > vwap_val + vwap_dev * vwap_std:
                if trend_period > 0 and not np.isnan(ema_val) and close > ema_val:
                    return None
                target_price = close - tp_pct * (close - vwap_val)
                if min_rr > 0 and stop_dist > 0:
                    rr = (close - target_price) / stop_dist
                    if rr < min_rr:
                        return None
                contracts = self.compute_position_size(stop_dist, pos_scale)
                if contracts > 0 and self.position is None:
                    self.position = Position(
                        side=Side.SHORT,
                        entry_price=close,
                        entry_time=timestamp,
                        contracts=contracts,
                        stop_price=close + stop_dist,
                        atr_at_entry=atr_val,
                        target_price=target_price,
                    )
                return None

        return None

    def generate_signals_fast(
        self,
        df: pd.DataFrame,
        vix_series: Optional[pd.Series] = None,
        precomputed: Optional[Dict] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """Optimized generate_signals() — same logic, no iterrows().

        Pre-extracts numpy arrays and pre-computes time masks to avoid
        per-bar overhead. When precomputed indicators are provided, skips
        redundant indicator computation entirely.

        Args:
            df: OHLCV DataFrame with DatetimeIndex
            vix_series: Optional daily VIX values indexed by date
            precomputed: Optional dict from precompute_indicators()

        Returns:
            (entries, exits) boolean Series aligned with df.index
        """
        self.reset()
        n = len(df)

        if precomputed is not None:
            return self._generate_signals_precomputed(df, precomputed)

        # Pre-extract OHLCV as numpy arrays
        opens = df["open"].values.astype(np.float64)
        highs = df["high"].values.astype(np.float64)
        lows = df["low"].values.astype(np.float64)
        closes = df["close"].values.astype(np.float64)
        volumes = df["volume"].values.astype(np.int64)

        # Pre-compute timestamps as ET-localized datetimes
        timestamps = np.empty(n, dtype=object)
        for i in range(n):
            ts = df.index[i]
            if not isinstance(ts, datetime):
                ts = pd.Timestamp(ts).to_pydatetime()
            if ts.tzinfo is None:
                ts = ET.localize(ts)
            timestamps[i] = ts

        # Pre-compute VIX per bar
        if vix_series is not None:
            vix_dict = vix_series.to_dict()
            current_vix = 20.0
            for i in range(n):
                d = timestamps[i].date()
                if d in vix_dict:
                    current_vix = float(vix_dict[d])
                    self._current_vix = current_vix

        # Main loop — direct array indexing instead of iterrows
        entries_arr = np.zeros(n, dtype=bool)
        exits_arr = np.zeros(n, dtype=bool)

        for i in range(n):
            if vix_series is not None:
                d = timestamps[i].date()
                if d in vix_dict:
                    self._current_vix = float(vix_dict[d])

            had_position = self.position is not None

            result = self.on_bar(
                timestamp=timestamps[i],
                open_=opens[i],
                high=highs[i],
                low=lows[i],
                close=closes[i],
                volume=int(volumes[i]),
            )

            if not had_position and self.position is not None:
                entries_arr[i] = True
            if result is not None:
                exits_arr[i] = True

        return (
            pd.Series(entries_arr, index=df.index),
            pd.Series(exits_arr, index=df.index),
        )

    def _generate_signals_precomputed(
        self,
        df: pd.DataFrame,
        precomputed: Dict,
    ) -> Tuple[pd.Series, pd.Series]:
        """Fast path using precomputed indicators — no indicator recomputation."""
        n = len(df)
        bb_period = self.params["bb_period"]
        trend_period = self.params["trend_ema_period"]

        timestamps = precomputed["timestamps"]
        closes = precomputed["closes"]
        vix_per_bar = precomputed["vix_per_bar"]
        vwap_vals = precomputed["vwap_vals"]
        vwap_stds = precomputed["vwap_stds"]
        bb_means = precomputed["bb_means"][bb_period]
        bb_stds = precomputed["bb_stds"][bb_period]
        atr_vals = precomputed["atr_vals"]
        in_window = precomputed["in_window"]
        is_flatten = precomputed["is_flatten"]

        # EMA values for trend filter
        ema_vals_dict = precomputed.get("ema_vals", {})
        if trend_period > 0 and trend_period in ema_vals_dict:
            ema_arr = ema_vals_dict[trend_period]
        else:
            ema_arr = np.full(n, np.nan)

        entries_arr = np.zeros(n, dtype=bool)
        exits_arr = np.zeros(n, dtype=bool)

        for i in range(n):
            self._current_vix = vix_per_bar[i]
            had_position = self.position is not None

            result = self._process_bar_precomputed(
                timestamp=timestamps[i],
                close=closes[i],
                vwap_val=vwap_vals[i],
                vwap_std=vwap_stds[i],
                bb_mean=bb_means[i],
                bb_std=bb_stds[i],
                atr_val=atr_vals[i],
                in_window=in_window[i],
                is_flatten=is_flatten[i],
                ema_val=ema_arr[i],
            )

            if not had_position and self.position is not None:
                entries_arr[i] = True
            if result is not None:
                exits_arr[i] = True

        return (
            pd.Series(entries_arr, index=df.index),
            pd.Series(exits_arr, index=df.index),
        )

    # ── Vectorized Signal Generation (for VectorBT backtesting) ──────────

    def generate_signals(
        self,
        df: pd.DataFrame,
        vix_series: Optional[pd.Series] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """Generate entry/exit boolean signals for a full DataFrame.

        This produces signals compatible with BacktestEngine.set_signals().
        Uses the event-driven on_bar logic internally for consistency.

        Args:
            df: OHLCV DataFrame with DatetimeIndex
            vix_series: Optional daily VIX values indexed by date

        Returns:
            (entries, exits) boolean Series aligned with df.index
        """
        self.reset()

        entries = pd.Series(False, index=df.index)
        exits = pd.Series(False, index=df.index)

        prev_has_position = False

        for i, (idx, row) in enumerate(df.iterrows()):
            ts = idx if isinstance(idx, datetime) else pd.Timestamp(idx).to_pydatetime()
            if ts.tzinfo is None:
                ts = ET.localize(ts)

            # Update VIX if series provided
            if vix_series is not None:
                ts_date = ts.date() if hasattr(ts, 'date') else ts
                if ts_date in vix_series.index:
                    self.set_vix(float(vix_series[ts_date]))

            had_position = self.position is not None

            result = self.on_bar(
                timestamp=ts,
                open_=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row["volume"]),
            )

            # Detect new entry
            if not had_position and self.position is not None:
                entries.iloc[i] = True

            # Detect exit
            if result is not None:
                exits.iloc[i] = True

        return entries, exits

    def generate_signals_vectorized(
        self,
        df: pd.DataFrame,
        vix_series: Optional[pd.Series] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """Pure vectorized signal generation for parameter sweeps.

        Faster than generate_signals() but does not model position management
        (trailing stops, breakeven moves). Use for initial screening only.
        CPCV validation should use generate_signals() for accuracy.

        Returns:
            (long_entries, short_entries) boolean Series
        """
        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        volumes = df["volume"].values
        n = len(df)

        # Compute indicators
        vwap_vals = np.full(n, np.nan)
        vwap_stds = np.full(n, np.nan)
        bb_uppers = np.full(n, np.nan)
        bb_lowers = np.full(n, np.nan)
        atr_vals = np.full(n, np.nan)

        vwap = VWAP(num_std_bands=2)
        bb = BollingerBands(period=self.params["bb_period"], num_std=self.params["bb_sigma"])
        atr = ATR(period=self.params["atr_period"])

        for i in range(n):
            ts = df.index[i]
            if not isinstance(ts, datetime):
                ts = pd.Timestamp(ts).to_pydatetime()
            if ts.tzinfo is None:
                ts = ET.localize(ts)

            v = vwap.update(ts, highs[i], lows[i], closes[i], int(volumes[i]))
            vwap_vals[i] = v["vwap"]
            vwap_stds[i] = v["std"]

            b = bb.update(closes[i])
            if b is not None:
                bb_uppers[i] = b["upper"]
                bb_lowers[i] = b["lower"]

            a = atr.update(highs[i], lows[i], closes[i])
            if a is not None:
                atr_vals[i] = a

        # Time filter
        time_mask = np.zeros(n, dtype=bool)
        start = dt_time(self.params["trade_start_hour"], self.params["trade_start_minute"])
        end = dt_time(self.params["trade_end_hour"], self.params["trade_end_minute"])
        for i in range(n):
            ts = df.index[i]
            if not isinstance(ts, datetime):
                ts = pd.Timestamp(ts).to_pydatetime()
            if ts.tzinfo is None:
                ts = ET.localize(ts)
            else:
                ts = ts.astimezone(ET)
            t = ts.time()
            time_mask[i] = start <= t < end

        # VIX filter
        vix_mask = np.ones(n, dtype=bool)
        eff_sigma = np.full(n, self.params["bb_sigma"])
        pos_scales = np.ones(n)
        if vix_series is not None:
            for i in range(n):
                ts = df.index[i]
                if not isinstance(ts, datetime):
                    ts = pd.Timestamp(ts).to_pydatetime()
                d = ts.date() if hasattr(ts, 'date') else ts
                if d in vix_series.index:
                    vix_val = float(vix_series[d])
                    if vix_val > 35:
                        vix_mask[i] = False
                    elif vix_val > 25:
                        eff_sigma[i] = 2.5
                        pos_scales[i] = 0.5

        # Adjust BB for VIX widening
        # Recover std from existing: std = (upper - lower) / (2 * bb_sigma)
        orig_sigma = self.params["bb_sigma"]
        if orig_sigma > 0:
            bb_std = (bb_uppers - bb_lowers) / (2.0 * orig_sigma)
        else:
            bb_std = np.zeros(n)
        bb_middle = (bb_uppers + bb_lowers) / 2.0
        adj_upper = bb_middle + eff_sigma * bb_std
        adj_lower = bb_middle - eff_sigma * bb_std

        vwap_dev = self.params["vwap_deviation_entry"]
        ready = ~np.isnan(bb_uppers) & ~np.isnan(atr_vals)

        long_entries = (
            ready
            & time_mask
            & vix_mask
            & (closes <= adj_lower)
            & (closes < vwap_vals - vwap_dev * vwap_stds)
        )

        short_entries = (
            ready
            & time_mask
            & vix_mask
            & (closes >= adj_upper)
            & (closes > vwap_vals + vwap_dev * vwap_stds)
        )

        return pd.Series(long_entries, index=df.index), pd.Series(short_entries, index=df.index)


# ── Precomputation for Grid Search ──────────────────────────────────────


def precompute_indicators(
    df: pd.DataFrame,
    vix_series: Optional[pd.Series] = None,
    bb_periods: Optional[List[int]] = None,
    ema_periods: Optional[List[int]] = None,
    atr_period: int = 14,
    trade_start: Tuple[int, int] = (10, 0),
    trade_end: Tuple[int, int] = (14, 0),
    flatten_time: Tuple[int, int] = (15, 55),
) -> Dict:
    """Pre-compute all indicators shared across a parameter grid.

    Computes VWAP, ATR, BB (for each period), EMA (for each period),
    time masks, and VIX arrays once, avoiding redundant work across
    grid search combos.

    Args:
        df: OHLCV DataFrame with DatetimeIndex
        vix_series: Optional daily VIX values indexed by date
        bb_periods: List of BB periods to precompute (default [20])
        ema_periods: List of EMA periods to precompute for trend filter
        atr_period: ATR period (default 14)
        trade_start: (hour, minute) trading window start
        trade_end: (hour, minute) trading window end
        flatten_time: (hour, minute) flatten time

    Returns:
        Dict of precomputed arrays for generate_signals_fast()
    """
    if bb_periods is None:
        bb_periods = [20]
    if ema_periods is None:
        ema_periods = []

    n = len(df)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    volumes = df["volume"].values.astype(np.int64)

    # Vectorized timestamp handling via pandas
    idx = df.index
    if idx.tz is None:
        et_index = idx.tz_localize(ET)
    else:
        et_index = idx.tz_convert(ET)

    # Time masks — vectorized
    hours = et_index.hour
    minutes = et_index.minute
    bar_minutes = hours * 60 + minutes
    start_minutes = trade_start[0] * 60 + trade_start[1]
    end_minutes = trade_end[0] * 60 + trade_end[1]
    flatten_minutes = flatten_time[0] * 60 + flatten_time[1]
    in_window = np.asarray((bar_minutes >= start_minutes) & (bar_minutes < end_minutes))
    is_flatten = np.asarray(bar_minutes >= flatten_minutes)

    # Convert to Python datetimes for VWAP session reset logic
    timestamps = et_index.to_pydatetime()

    # VIX per bar — vectorized via date mapping
    vix_per_bar = np.full(n, 20.0)
    if vix_series is not None:
        bar_dates = et_index.date
        vix_dict = vix_series.to_dict()
        current_vix = 20.0
        for i in range(n):
            d = bar_dates[i]
            if d in vix_dict:
                current_vix = float(vix_dict[d])
            vix_per_bar[i] = current_vix

    # VWAP (session-resetting, must be incremental)
    vwap_vals = np.full(n, np.nan)
    vwap_stds = np.full(n, np.nan)
    vwap_ind = VWAP(num_std_bands=2)
    for i in range(n):
        v = vwap_ind.update(timestamps[i], highs[i], lows[i], closes[i], int(volumes[i]))
        vwap_vals[i] = v["vwap"]
        vwap_stds[i] = v["std"]

    # ATR
    atr_vals = np.full(n, np.nan)
    atr_ind = ATR(period=atr_period)
    for i in range(n):
        a = atr_ind.update(highs[i], lows[i], closes[i])
        if a is not None:
            atr_vals[i] = a

    # BB for each period (mean and std separately)
    bb_means: Dict[int, np.ndarray] = {}
    bb_stds: Dict[int, np.ndarray] = {}
    for period in bb_periods:
        means = np.full(n, np.nan)
        stds = np.full(n, np.nan)
        bb_ind = BollingerBands(period=period, num_std=1.0)
        for i in range(n):
            result = bb_ind.update(closes[i])
            if result is not None:
                means[i] = result["middle"]
                stds[i] = result["upper"] - result["middle"]
        bb_means[period] = means
        bb_stds[period] = stds

    # EMA for each period (trend filter)
    ema_vals: Dict[int, np.ndarray] = {}
    if ema_periods:
        from indicators.ema import EMA as EMAIndicator
        for period in ema_periods:
            if period <= 0:
                continue
            vals = np.full(n, np.nan)
            ema_ind = EMAIndicator(period=period)
            for i in range(n):
                result = ema_ind.update(closes[i])
                if result is not None:
                    vals[i] = result
            ema_vals[period] = vals

    return {
        "timestamps": timestamps,
        "closes": closes,
        "vix_per_bar": vix_per_bar,
        "vwap_vals": vwap_vals,
        "vwap_stds": vwap_stds,
        "bb_means": bb_means,
        "bb_stds": bb_stds,
        "atr_vals": atr_vals,
        "in_window": in_window,
        "is_flatten": is_flatten,
        "ema_vals": ema_vals,
    }
