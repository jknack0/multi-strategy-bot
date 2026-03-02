"""
Tests for MES Mean Reversion Strategy (Strategy A).

Covers signal generation, VIX filtering, position sizing,
time filtering, and stop/exit logic using synthetic data
with known outcomes.
"""

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz
import pytest

from config.constants import INSTRUMENTS, VIXRegime
from strategies.mean_reversion import (
    MESMeanReversionStrategy,
    Side,
    Position,
    TradeRecord,
)

ET = pytz.timezone("US/Eastern")


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def _make_bar_df(
    n: int = 100,
    base_price: float = 4500.0,
    start: datetime = None,
    sigma: float = 3.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV bars during RTH."""
    rng = np.random.RandomState(seed)
    if start is None:
        start = ET.localize(datetime(2024, 3, 1, 9, 30))

    bars_per_day = 78
    timestamps = []
    day = start
    while len(timestamps) < n:
        day_start = day.replace(hour=9, minute=30, second=0, microsecond=0)
        for j in range(bars_per_day):
            timestamps.append(day_start + timedelta(minutes=5 * j))
            if len(timestamps) >= n:
                break
        day += timedelta(days=1)
        while day.weekday() >= 5:
            day += timedelta(days=1)

    closes = np.empty(n)
    closes[0] = base_price
    for i in range(1, n):
        closes[i] = closes[i - 1] + sigma * rng.randn()

    noise = rng.uniform(0.5, 2.0, n)
    opens = closes + rng.randn(n) * 0.3
    highs = np.maximum(opens, closes) + noise
    lows = np.minimum(opens, closes) - noise
    volumes = rng.randint(500, 3000, n)

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows,
         "close": closes, "volume": volumes},
        index=pd.DatetimeIndex(timestamps[:n], name="timestamp"),
    )


def _make_vix_series(df: pd.DataFrame, value: float = 20.0) -> pd.Series:
    """Constant VIX series."""
    dates = sorted(set(df.index.date))
    return pd.Series(value, index=dates, name="vix")


def _make_long_entry_scenario() -> pd.DataFrame:
    """Construct bars that guarantee a LONG entry signal.

    We need: close <= BB_lower AND close < VWAP - dev * vwap_std
    Strategy: create 20 bars of rising prices, then one huge drop.
    The drop bar will be below the lower Bollinger Band and below VWAP.
    """
    n_warmup = 25  # Enough for BB(20) + ATR(14) to initialize
    start = ET.localize(datetime(2024, 3, 1, 10, 0))
    timestamps = [start + timedelta(minutes=5 * i) for i in range(n_warmup + 1)]

    # Stable prices for warmup
    base = 4500.0
    closes = [base + i * 0.1 for i in range(n_warmup)]
    opens = [c - 0.05 for c in closes]
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.0 for c in closes]
    volumes = [1000] * n_warmup

    # Drop bar: way below BB lower and VWAP
    drop_close = base - 30.0
    closes.append(drop_close)
    opens.append(base)
    highs.append(base + 0.5)
    lows.append(drop_close - 1.0)
    volumes.append(1000)

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows,
         "close": closes, "volume": volumes},
        index=pd.DatetimeIndex(timestamps, name="timestamp"),
    )


def _make_short_entry_scenario() -> pd.DataFrame:
    """Construct bars that guarantee a SHORT entry signal."""
    n_warmup = 25
    start = ET.localize(datetime(2024, 3, 1, 10, 0))
    timestamps = [start + timedelta(minutes=5 * i) for i in range(n_warmup + 1)]

    base = 4500.0
    closes = [base - i * 0.1 for i in range(n_warmup)]
    opens = [c + 0.05 for c in closes]
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.0 for c in closes]
    volumes = [1000] * n_warmup

    # Spike bar: way above BB upper and VWAP
    spike_close = base + 30.0
    closes.append(spike_close)
    opens.append(base)
    highs.append(spike_close + 1.0)
    lows.append(base - 0.5)
    volumes.append(1000)

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows,
         "close": closes, "volume": volumes},
        index=pd.DatetimeIndex(timestamps, name="timestamp"),
    )


# ══════════════════════════════════════════════════════════════════════════
# Signal Generation Tests
# ══════════════════════════════════════════════════════════════════════════

class TestSignalGeneration:
    """Test signal generation on synthetic data with known outcomes."""

    def test_long_entry_generated(self):
        """A large price drop should trigger a LONG entry."""
        df = _make_long_entry_scenario()
        vix = _make_vix_series(df, value=18.0)
        strat = MESMeanReversionStrategy(capital=10_000.0)
        entries, exits = strat.generate_signals(df, vix_series=vix)
        assert entries.sum() >= 1, "Expected at least one LONG entry signal"

    def test_short_entry_generated(self):
        """A large price spike should trigger a SHORT entry."""
        df = _make_short_entry_scenario()
        vix = _make_vix_series(df, value=18.0)
        strat = MESMeanReversionStrategy(capital=10_000.0)
        entries, exits = strat.generate_signals(df, vix_series=vix)
        assert entries.sum() >= 1, "Expected at least one SHORT entry signal"

    def test_no_signal_during_flat_market(self):
        """Nearly flat prices should not trigger entries."""
        start = ET.localize(datetime(2024, 3, 1, 10, 0))
        n = 50
        timestamps = [start + timedelta(minutes=5 * i) for i in range(n)]
        # Very tight range → price stays inside BB bands
        closes = [4500.0 + 0.01 * i for i in range(n)]
        df = pd.DataFrame({
            "open": closes, "high": [c + 0.1 for c in closes],
            "low": [c - 0.1 for c in closes], "close": closes,
            "volume": [1000] * n,
        }, index=pd.DatetimeIndex(timestamps, name="timestamp"))

        vix = _make_vix_series(df, value=18.0)
        strat = MESMeanReversionStrategy(capital=10_000.0)
        entries, _ = strat.generate_signals(df, vix_series=vix)
        assert entries.sum() == 0, "Flat market should generate zero entries"

    def test_entries_and_exits_same_length_as_data(self):
        df = _make_bar_df(200)
        vix = _make_vix_series(df)
        strat = MESMeanReversionStrategy()
        entries, exits = strat.generate_signals(df, vix_series=vix)
        assert len(entries) == len(df)
        assert len(exits) == len(df)


# ══════════════════════════════════════════════════════════════════════════
# VIX Filter Tests
# ══════════════════════════════════════════════════════════════════════════

class TestVIXFilter:
    """Verify VIX filter correctly modifies parameters and blocks trades."""

    def test_extreme_vix_blocks_trades(self):
        """VIX > 35 should prevent all entries."""
        df = _make_long_entry_scenario()
        vix = _make_vix_series(df, value=40.0)  # EXTREME
        strat = MESMeanReversionStrategy(capital=10_000.0)
        entries, _ = strat.generate_signals(df, vix_series=vix)
        assert entries.sum() == 0, "VIX > 35 should block all trades"

    def test_high_vix_widens_bb(self):
        """VIX 25-35 should use bb_sigma=2.5 and position_scale=0.5."""
        strat = MESMeanReversionStrategy()
        strat.set_vix(30.0)
        eff_sigma, pos_scale, allowed = strat._get_vix_adjustments()
        assert allowed is True
        assert eff_sigma == 2.5
        assert pos_scale == 0.5

    def test_normal_vix_standard_params(self):
        """VIX <= 25 should use standard parameters."""
        strat = MESMeanReversionStrategy()
        strat.set_vix(20.0)
        eff_sigma, pos_scale, allowed = strat._get_vix_adjustments()
        assert allowed is True
        assert eff_sigma == strat.params["bb_sigma"]
        assert pos_scale == 1.0

    def test_low_vix_same_as_normal(self):
        strat = MESMeanReversionStrategy()
        strat.set_vix(12.0)
        eff_sigma, pos_scale, allowed = strat._get_vix_adjustments()
        assert allowed is True
        assert pos_scale == 1.0

    def test_vix_boundary_35(self):
        """VIX exactly 35 is still in HIGH regime (not EXTREME)."""
        strat = MESMeanReversionStrategy()
        strat.set_vix(35.0)
        _, _, allowed = strat._get_vix_adjustments()
        assert allowed is True  # 35 is not > 35

    def test_vix_boundary_above_35(self):
        strat = MESMeanReversionStrategy()
        strat.set_vix(35.01)
        _, _, allowed = strat._get_vix_adjustments()
        assert allowed is False


# ══════════════════════════════════════════════════════════════════════════
# Position Sizing Tests
# ══════════════════════════════════════════════════════════════════════════

class TestPositionSizing:
    """Verify position sizing math."""

    def test_basic_sizing(self):
        """contracts = floor(dollar_risk / (stop_dist * point_value)) * scale"""
        strat = MESMeanReversionStrategy(capital=10_000.0)
        # dollar_risk = 10000 * 0.01 = 100
        # stop_dist = 5.0 points, point_value = 5.0
        # raw = 100 / (5.0 * 5.0) = 4.0
        # scaled = floor(4.0 * 1.0) = 4
        # margin cap = 10000 * 0.50 / 50 = 100 (not binding)
        contracts = strat.compute_position_size(5.0, 1.0)
        assert contracts == 4  # risk-based (margin not binding with $50 NT margin)

    def test_sizing_with_scale(self):
        """Position scale of 0.5 should halve contracts."""
        strat = MESMeanReversionStrategy(capital=10_000.0)
        # raw = 4.0, scaled = floor(4.0 * 0.5) = 2
        contracts = strat.compute_position_size(5.0, 0.5)
        assert contracts == 2

    def test_insufficient_capital_returns_zero(self):
        """When risk budget is too small for 1 contract, return 0 (no trade)."""
        strat = MESMeanReversionStrategy(capital=100.0)
        # dollar_risk = 100 * 0.01 = 1.0
        # raw = 1.0 / (50.0 * 5.0) = 0.004
        # scaled = floor(0.004 * 1.0) = 0 → returns 0 (skip trade)
        contracts = strat.compute_position_size(50.0, 1.0)
        assert contracts == 0

    def test_margin_cap(self):
        """Contracts should not exceed 50% of margin capacity."""
        strat = MESMeanReversionStrategy(capital=10_000.0)
        # margin cap = 10000 * 0.50 / 50 = 100 (NT $50 margin)
        # With very small stop: raw = 100 / (0.01 * 5.0) = 2000
        # Should be capped at 100
        contracts = strat.compute_position_size(0.01, 1.0)
        assert contracts == 100  # margin cap

    def test_zero_stop_returns_zero(self):
        strat = MESMeanReversionStrategy(capital=10_000.0)
        assert strat.compute_position_size(0.0, 1.0) == 0

    def test_negative_stop_returns_zero(self):
        strat = MESMeanReversionStrategy(capital=10_000.0)
        assert strat.compute_position_size(-5.0, 1.0) == 0


# ══════════════════════════════════════════════════════════════════════════
# Time Filter Tests
# ══════════════════════════════════════════════════════════════════════════

class TestTimeFilter:
    """Verify time-based filtering."""

    def test_within_trading_window(self):
        strat = MESMeanReversionStrategy()
        ts = ET.localize(datetime(2024, 3, 1, 11, 0))
        assert strat._in_trading_window(ts) is True

    def test_before_trading_window(self):
        strat = MESMeanReversionStrategy()
        ts = ET.localize(datetime(2024, 3, 1, 9, 30))
        assert strat._in_trading_window(ts) is False

    def test_after_trading_window(self):
        strat = MESMeanReversionStrategy()
        ts = ET.localize(datetime(2024, 3, 1, 14, 5))
        assert strat._in_trading_window(ts) is False

    def test_at_window_start(self):
        strat = MESMeanReversionStrategy()
        ts = ET.localize(datetime(2024, 3, 1, 10, 0))
        assert strat._in_trading_window(ts) is True

    def test_at_window_end(self):
        strat = MESMeanReversionStrategy()
        ts = ET.localize(datetime(2024, 3, 1, 14, 0))
        assert strat._in_trading_window(ts) is False  # end is exclusive

    def test_flatten_time(self):
        strat = MESMeanReversionStrategy()
        ts = ET.localize(datetime(2024, 3, 1, 15, 55))
        assert strat._is_flatten_time(ts) is True

    def test_before_flatten_time(self):
        strat = MESMeanReversionStrategy()
        ts = ET.localize(datetime(2024, 3, 1, 15, 50))
        assert strat._is_flatten_time(ts) is False

    def test_no_entries_outside_window(self):
        """Bars outside 10:00-14:00 ET should never produce entries."""
        start = ET.localize(datetime(2024, 3, 1, 14, 30))  # After window
        n = 30
        timestamps = [start + timedelta(minutes=5 * i) for i in range(n)]
        rng = np.random.RandomState(99)
        closes = 4500.0 + np.cumsum(rng.randn(n) * 10.0)

        df = pd.DataFrame({
            "open": closes - 0.5, "high": closes + 2.0,
            "low": closes - 2.0, "close": closes,
            "volume": [1000] * n,
        }, index=pd.DatetimeIndex(timestamps, name="timestamp"))

        vix = _make_vix_series(df, value=18.0)
        strat = MESMeanReversionStrategy(capital=10_000.0)
        entries, _ = strat.generate_signals(df, vix_series=vix)
        assert entries.sum() == 0


# ══════════════════════════════════════════════════════════════════════════
# Exit Logic Tests
# ══════════════════════════════════════════════════════════════════════════

class TestExitLogic:
    """Test stop loss, take profit, and time stop exits."""

    def test_long_stop_loss(self):
        """A LONG position should close when price hits stop."""
        strat = MESMeanReversionStrategy(capital=10_000.0)
        ts = ET.localize(datetime(2024, 3, 1, 11, 0))

        # Manually set position
        strat.position = Position(
            side=Side.LONG,
            entry_price=4500.0,
            entry_time=ts,
            contracts=1,
            stop_price=4490.0,
            atr_at_entry=5.0,
        )

        vwap_data = {"vwap": 4500.0, "std": 2.0}
        result = strat._manage_position(4489.0, ts + timedelta(minutes=5), vwap_data, 5.0)
        assert result is not None
        assert result.exit_reason == "stop_loss"
        assert result.pnl < 0

    def test_long_take_profit(self):
        """A LONG position should close when price reaches target."""
        strat = MESMeanReversionStrategy(capital=10_000.0)
        ts = ET.localize(datetime(2024, 3, 1, 11, 0))

        strat.position = Position(
            side=Side.LONG,
            entry_price=4490.0,
            entry_time=ts,
            contracts=1,
            stop_price=4480.0,
            atr_at_entry=5.0,
            target_price=4500.0,
        )

        vwap_data = {"vwap": 4500.0, "std": 2.0}
        result = strat._manage_position(4500.0, ts + timedelta(minutes=10), vwap_data, 5.0)
        assert result is not None
        assert result.exit_reason == "take_profit"
        assert result.pnl > 0

    def test_time_stop_flattens_position(self):
        """Position should be closed at 3:55 PM ET."""
        df = _make_long_entry_scenario()
        vix = _make_vix_series(df, value=18.0)
        strat = MESMeanReversionStrategy(capital=10_000.0)

        # Feed bars to get an entry
        strat.generate_signals(df, vix_series=vix)

        # If we have trades, check if any have time_stop
        # (The scenario may not produce time_stop since it's only ~2 hours of bars)
        # Instead, test the flatten detection directly
        ts_before = ET.localize(datetime(2024, 3, 1, 15, 54))
        ts_at = ET.localize(datetime(2024, 3, 1, 15, 55))
        ts_after = ET.localize(datetime(2024, 3, 1, 15, 56))

        assert strat._is_flatten_time(ts_before) is False
        assert strat._is_flatten_time(ts_at) is True
        assert strat._is_flatten_time(ts_after) is True

    def test_trailing_stop_activated(self):
        """Trailing stop should activate and ratchet up after favorable excursion."""
        strat = MESMeanReversionStrategy(
            params={"trailing_stop_factor": 0.6},
            capital=10_000.0,
        )
        ts = ET.localize(datetime(2024, 3, 1, 11, 0))

        strat.position = Position(
            side=Side.LONG,
            entry_price=4500.0,
            entry_time=ts,
            contracts=1,
            stop_price=4490.0,
            atr_at_entry=5.0,
        )

        # Price moved up by 1 ATR (5.0 points) — exceeds breakeven_atr_mult(0.7) * 5.0 = 3.5
        vwap_data = {"vwap": 4510.0, "std": 2.0}
        result = strat._manage_position(4505.0, ts + timedelta(minutes=5), vwap_data, 5.0)
        assert result is None  # Still open
        assert strat.position.trailing_active is True
        # trail stop = entry + 0.6 * max_favorable = 4500 + 0.6 * 5.0 = 4503.0
        assert strat.position.stop_price == 4503.0


# ══════════════════════════════════════════════════════════════════════════
# Parameter Persistence Tests
# ══════════════════════════════════════════════════════════════════════════

class TestParameterPersistence:
    """Test save/load of strategy parameters."""

    def test_save_and_load(self, tmp_path):
        """Params should round-trip through JSON."""
        params = {
            "bb_period": 25,
            "bb_sigma": 1.5,
            "atr_stop_multiplier": 3.0,
            "vwap_deviation_entry": 0.75,
        }
        strat = MESMeanReversionStrategy(params=params)
        path = str(tmp_path / "params.json")
        strat.save_params(path)

        loaded = MESMeanReversionStrategy.from_json(path)
        assert loaded.params["bb_period"] == 25
        assert loaded.params["bb_sigma"] == 1.5
        assert loaded.params["atr_stop_multiplier"] == 3.0
        assert loaded.params["vwap_deviation_entry"] == 0.75

    def test_default_params_preserved(self):
        """Unspecified params should retain defaults."""
        strat = MESMeanReversionStrategy(params={"bb_period": 30})
        assert strat.params["bb_period"] == 30
        assert strat.params["bb_sigma"] == 1.5  # Default


# ══════════════════════════════════════════════════════════════════════════
# Vectorized vs Event-Driven Consistency
# ══════════════════════════════════════════════════════════════════════════

class TestSignalConsistency:
    """Compare vectorized and event-driven signal generation."""

    def test_vectorized_entry_direction_matches(self):
        """Vectorized long/short entries should fire on the same bars as
        the event-driven version's entries (for entry signals only)."""
        df = _make_bar_df(500, seed=77)
        vix = _make_vix_series(df, value=20.0)

        strat_event = MESMeanReversionStrategy(capital=10_000.0)
        ev_entries, _ = strat_event.generate_signals(df, vix_series=vix)

        strat_vec = MESMeanReversionStrategy(capital=10_000.0)
        long_entries, short_entries = strat_vec.generate_signals_vectorized(df, vix_series=vix)
        vec_entries = long_entries | short_entries

        # The vectorized version fires on every qualifying bar (no position gating),
        # while event-driven only fires when flat. So vectorized entries should be
        # a superset of event-driven entries.
        ev_set = set(df.index[ev_entries])
        vec_set = set(df.index[vec_entries])
        # Every event-driven entry should also be a vectorized entry
        missed = ev_set - vec_set
        assert len(missed) == 0, (
            f"{len(missed)} event-driven entries not found in vectorized signals"
        )


# ══════════════════════════════════════════════════════════════════════════
# Trade PnL Math Tests
# ══════════════════════════════════════════════════════════════════════════

class TestTradePnL:
    """Verify trade P&L calculations."""

    def test_long_pnl_calculation(self):
        strat = MESMeanReversionStrategy(capital=10_000.0)
        ts = ET.localize(datetime(2024, 3, 1, 11, 0))

        strat.position = Position(
            side=Side.LONG,
            entry_price=4500.0,
            entry_time=ts,
            contracts=2,
            stop_price=4490.0,
            atr_at_entry=5.0,
        )

        record = strat._close_position(4505.0, ts + timedelta(minutes=30), "take_profit")
        # PnL = (4505 - 4500) * 5.0 * 2 = 50.0
        expected = (4505.0 - 4500.0) * 5.0 * 2
        assert record.pnl == pytest.approx(expected, abs=1e-6)

    def test_capital_updates_on_close(self):
        strat = MESMeanReversionStrategy(capital=10_000.0)
        ts = ET.localize(datetime(2024, 3, 1, 11, 0))

        strat.position = Position(
            side=Side.LONG,
            entry_price=4500.0,
            entry_time=ts,
            contracts=1,
            stop_price=4490.0,
            atr_at_entry=5.0,
        )

        strat._close_position(4510.0, ts + timedelta(minutes=30), "take_profit")
        # PnL = (4510-4500) * 5.0 * 1 = 50
        assert strat.capital == pytest.approx(10050.0, abs=1e-6)
