"""
Strategy A ported to NautilusTrader framework.

Implements identical entry/exit/sizing logic to MESMeanReversionStrategy
but using NautilusTrader's Strategy base class, indicator system, and
order management.

NOTE: NautilusTrader uses its own data types and event model.  This port
mirrors the VectorBT version's logic so that both produce matching signals
on the same data (within 1% tolerance due to float precision).
"""

import math
from collections import deque
from datetime import time as dt_time
from decimal import Decimal
from typing import Dict, Optional

import pytz

try:
    from nautilus_trader.config import StrategyConfig
    from nautilus_trader.core.datetime import dt_to_unix_nanos
    from nautilus_trader.indicators.average.moving_average import MovingAverageType
    from nautilus_trader.indicators.bollinger_bands import BollingerBands as NTBollingerBands
    from nautilus_trader.indicators.atr import AverageTrueRange
    from nautilus_trader.model.data import Bar, BarType
    from nautilus_trader.model.enums import OrderSide, TimeInForce
    from nautilus_trader.model.identifiers import InstrumentId
    from nautilus_trader.model.instruments import Instrument
    from nautilus_trader.model.objects import Price, Quantity
    from nautilus_trader.trading.strategy import Strategy

    _HAS_NAUTILUS = True
except ImportError:
    _HAS_NAUTILUS = False


ET = pytz.timezone("US/Eastern")


# ── Custom Session VWAP Indicator ────────────────────────────────────────

class SessionVWAP:
    """Session VWAP that resets at 9:30 AM ET.

    NautilusTrader doesn't ship a session-resetting VWAP, so we
    implement one with the same logic as indicators/vwap.py.
    """

    def __init__(self) -> None:
        self._cumulative_pv: float = 0.0
        self._cumulative_v: float = 0.0
        self._cumulative_pv2: float = 0.0
        self._last_ts_ns: int = 0
        self.value: float = 0.0
        self.std: float = 0.0

    def reset_session(self) -> None:
        self._cumulative_pv = 0.0
        self._cumulative_v = 0.0
        self._cumulative_pv2 = 0.0

    def _should_reset(self, ts_ns: int) -> bool:
        """Check if we crossed the 9:30 ET boundary."""
        if self._last_ts_ns == 0:
            return True
        from datetime import datetime
        curr = datetime.utcfromtimestamp(ts_ns / 1e9).replace(tzinfo=pytz.utc).astimezone(ET)
        prev = datetime.utcfromtimestamp(self._last_ts_ns / 1e9).replace(tzinfo=pytz.utc).astimezone(ET)
        reset_time = dt_time(9, 30)
        return prev.time() < reset_time <= curr.time()

    def update(self, ts_ns: int, high: float, low: float, close: float, volume: float) -> None:
        if self._should_reset(ts_ns):
            self.reset_session()

        typical = (high + low + close) / 3.0
        self._cumulative_pv += typical * volume
        self._cumulative_v += volume
        self._cumulative_pv2 += (typical ** 2) * volume
        self._last_ts_ns = ts_ns

        if self._cumulative_v == 0:
            self.value = 0.0
            self.std = 0.0
            return

        self.value = self._cumulative_pv / self._cumulative_v
        mean_p2 = self._cumulative_pv2 / self._cumulative_v
        variance = max(0.0, mean_p2 - self.value ** 2)
        self.std = variance ** 0.5


# ── Strategy Config ──────────────────────────────────────────────────────

if _HAS_NAUTILUS:

    class MeanReversionConfig(StrategyConfig, frozen=True):
        """Configuration for the MES Mean Reversion strategy."""
        instrument_id: str = "MES.CME"
        bar_type: str = "MES.CME-5-MINUTE-LAST"
        bb_period: int = 20
        bb_sigma: float = 2.0
        atr_period: int = 14
        atr_stop_multiplier: float = 2.0
        vwap_deviation_entry: float = 1.0
        trade_start_hour: int = 10
        trade_start_minute: int = 0
        trade_end_hour: int = 14
        trade_end_minute: int = 0
        flatten_hour: int = 15
        flatten_minute: int = 55
        risk_per_trade: float = 0.01
        max_margin_pct: float = 0.50
        breakeven_atr_mult: float = 1.0
        point_value: float = 5.0
        tick_size: float = 0.25
        margin: float = 50.0

    class NautilusMeanReversionStrategy(Strategy):
        """NautilusTrader port of MES Bollinger/VWAP Mean Reversion.

        Produces identical signals to MESMeanReversionStrategy on the same data.
        """

        def __init__(self, config: MeanReversionConfig) -> None:
            super().__init__(config)
            self.config = config

            # Indicators
            self._bb = NTBollingerBands(
                period=config.bb_period,
                k=config.bb_sigma,
                ma_type=MovingAverageType.SIMPLE,
            )
            self._atr = AverageTrueRange(
                period=config.atr_period,
                ma_type=MovingAverageType.WILDER,
            )
            self._vwap = SessionVWAP()

            # State
            self._entry_price: float = 0.0
            self._stop_price: float = 0.0
            self._atr_at_entry: float = 0.0
            self._side: Optional[str] = None  # 'LONG' or 'SHORT'
            self._breakeven_moved: bool = False
            self._current_vix: float = 20.0
            self._instrument: Optional[Instrument] = None

        def on_start(self) -> None:
            instrument_id = InstrumentId.from_str(self.config.instrument_id)
            self._instrument = self.cache.instrument(instrument_id)
            bar_type = BarType.from_str(self.config.bar_type)
            self.subscribe_bars(bar_type)
            self.register_indicator_for_bars(bar_type, self._bb)
            self.register_indicator_for_bars(bar_type, self._atr)

        def set_vix(self, vix_value: float) -> None:
            self._current_vix = vix_value

        def _in_trading_window(self, ts_ns: int) -> bool:
            from datetime import datetime
            ts = datetime.utcfromtimestamp(ts_ns / 1e9).replace(tzinfo=pytz.utc).astimezone(ET)
            t = ts.time()
            start = dt_time(self.config.trade_start_hour, self.config.trade_start_minute)
            end = dt_time(self.config.trade_end_hour, self.config.trade_end_minute)
            return start <= t < end

        def _is_flatten_time(self, ts_ns: int) -> bool:
            from datetime import datetime
            ts = datetime.utcfromtimestamp(ts_ns / 1e9).replace(tzinfo=pytz.utc).astimezone(ET)
            t = ts.time()
            flatten = dt_time(self.config.flatten_hour, self.config.flatten_minute)
            return t >= flatten

        def _get_vix_adjustments(self) -> tuple:
            vix = self._current_vix
            if vix > 35:
                return self.config.bb_sigma, 0.0, False
            if vix > 25:
                return 2.5, 0.5, True
            return self.config.bb_sigma, 1.0, True

        def _compute_position_size(
            self, stop_distance: float, position_scale: float
        ) -> int:
            if stop_distance <= 0:
                return 0
            account = self.portfolio.account(self.cache.account_for_venue(
                InstrumentId.from_str(self.config.instrument_id).venue
            ).id) if self._instrument else None
            capital = 10000.0  # Default fallback
            if account is not None:
                capital = float(account.balance_total().as_double())

            dollar_risk = capital * self.config.risk_per_trade
            raw = dollar_risk / (stop_distance * self.config.point_value)
            scaled = math.floor(raw * position_scale)
            margin_cap = int((capital * self.config.max_margin_pct) / self.config.margin)
            return max(1, min(scaled, margin_cap))

        def on_bar(self, bar: Bar) -> None:
            ts_ns = bar.ts_event
            high = float(bar.high)
            low = float(bar.low)
            close = float(bar.close)
            volume = float(bar.volume)

            # Update session VWAP
            self._vwap.update(ts_ns, high, low, close, volume)

            # Flatten at time stop
            if self._is_flatten_time(ts_ns) and self._side is not None:
                self._close_position(close, "time_stop")
                return

            # Manage existing position
            if self._side is not None:
                self._manage_position(close)
                return

            # Need indicators ready
            if not self._bb.initialized or not self._atr.initialized:
                return

            if not self._in_trading_window(ts_ns):
                return

            eff_sigma, pos_scale, allowed = self._get_vix_adjustments()
            if not allowed:
                return

            bb_upper = float(self._bb.upper)
            bb_lower = float(self._bb.lower)
            bb_middle = float(self._bb.middle)
            atr_val = float(self._atr.value)

            # Adjust BB for VIX widening
            if eff_sigma != self.config.bb_sigma and self.config.bb_sigma > 0:
                bb_std = (bb_upper - bb_lower) / (2.0 * self.config.bb_sigma)
                bb_upper = bb_middle + eff_sigma * bb_std
                bb_lower = bb_middle - eff_sigma * bb_std

            vwap_val = self._vwap.value
            vwap_std = self._vwap.std
            vwap_dev = self.config.vwap_deviation_entry
            stop_mult = self.config.atr_stop_multiplier
            stop_dist = stop_mult * atr_val

            # LONG signal
            if close <= bb_lower and close < vwap_val - vwap_dev * vwap_std:
                contracts = self._compute_position_size(stop_dist, pos_scale)
                if contracts > 0:
                    self._enter_position("LONG", close, stop_dist, atr_val, contracts)
                return

            # SHORT signal
            if close >= bb_upper and close > vwap_val + vwap_dev * vwap_std:
                contracts = self._compute_position_size(stop_dist, pos_scale)
                if contracts > 0:
                    self._enter_position("SHORT", close, stop_dist, atr_val, contracts)
                return

        def _enter_position(
            self, side: str, price: float, stop_dist: float,
            atr_val: float, contracts: int
        ) -> None:
            self._side = side
            self._entry_price = price
            self._atr_at_entry = atr_val
            self._breakeven_moved = False

            if side == "LONG":
                self._stop_price = price - stop_dist
                order_side = OrderSide.BUY
            else:
                self._stop_price = price + stop_dist
                order_side = OrderSide.SELL

            if self._instrument:
                self.submit_order(
                    self.order_factory.market(
                        instrument_id=self._instrument.id,
                        order_side=order_side,
                        quantity=Quantity.from_int(contracts),
                        time_in_force=TimeInForce.IOC,
                    )
                )

        def _manage_position(self, close: float) -> None:
            vwap_val = self._vwap.value
            atr_val = float(self._atr.value) if self._atr.initialized else self._atr_at_entry

            if self._side == "LONG":
                if close <= self._stop_price:
                    self._close_position(close, "stop_loss")
                    return
                if close >= vwap_val:
                    self._close_position(close, "take_profit")
                    return
                if not self._breakeven_moved:
                    be_thresh = self.config.breakeven_atr_mult * self._atr_at_entry
                    if close - self._entry_price >= be_thresh:
                        self._stop_price = self._entry_price
                        self._breakeven_moved = True
            else:  # SHORT
                if close >= self._stop_price:
                    self._close_position(close, "stop_loss")
                    return
                if close <= vwap_val:
                    self._close_position(close, "take_profit")
                    return
                if not self._breakeven_moved:
                    be_thresh = self.config.breakeven_atr_mult * self._atr_at_entry
                    if self._entry_price - close >= be_thresh:
                        self._stop_price = self._entry_price
                        self._breakeven_moved = True

        def _close_position(self, price: float, reason: str) -> None:
            if self._instrument and self._side:
                order_side = OrderSide.SELL if self._side == "LONG" else OrderSide.BUY
                self.close_all_positions(self._instrument.id)
            self._side = None
            self._entry_price = 0.0
            self._stop_price = 0.0
            self._breakeven_moved = False

        def on_stop(self) -> None:
            self.cancel_all_orders(InstrumentId.from_str(self.config.instrument_id))
            self.close_all_positions(InstrumentId.from_str(self.config.instrument_id))

else:
    # Stub when nautilus_trader is not installed
    class SessionVWAP:
        """Session VWAP stub (nautilus_trader not installed)."""

        def __init__(self) -> None:
            self._cumulative_pv: float = 0.0
            self._cumulative_v: float = 0.0
            self._cumulative_pv2: float = 0.0
            self._last_ts_ns: int = 0
            self.value: float = 0.0
            self.std: float = 0.0

        def reset_session(self) -> None:
            self._cumulative_pv = 0.0
            self._cumulative_v = 0.0
            self._cumulative_pv2 = 0.0

        def update(self, ts_ns: int, high: float, low: float, close: float, volume: float) -> None:
            typical = (high + low + close) / 3.0
            self._cumulative_pv += typical * volume
            self._cumulative_v += volume
            self._cumulative_pv2 += (typical ** 2) * volume
            self._last_ts_ns = ts_ns

            if self._cumulative_v == 0:
                self.value = 0.0
                self.std = 0.0
                return

            self.value = self._cumulative_pv / self._cumulative_v
            mean_p2 = self._cumulative_pv2 / self._cumulative_v
            variance = max(0.0, mean_p2 - self.value ** 2)
            self.std = variance ** 0.5

    class NautilusMeanReversionStrategy:
        """Stub: nautilus_trader not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "nautilus_trader is not installed. "
                "Install with: pip install nautilus_trader"
            )
