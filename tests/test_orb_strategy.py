"""
Tests for VIX-Adaptive ORB Strategy (Strategy B).
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz
import pytest

from strategies.orb_strategy import (
    VIXAdaptiveORBStrategy,
    ORBPosition,
    ORBTradeRecord,
    Side,
)

ET = pytz.timezone("US/Eastern")


# ── Helpers ──────────────────────────────────────────────────────────

def _make_or_bars(
    or_high: float = 5425.0,
    or_low: float = 5418.0,
    or_duration_minutes: int = 15,
    start: datetime = None,
) -> list:
    """Create bars that build an Opening Range with known high/low.

    Generates enough 5-min bars to span the OR duration, plus one bar
    at the OR end time to trigger completion.

    Returns list of (timestamp, open, high, low, close, volume) tuples.
    """
    if start is None:
        start = ET.localize(datetime(2024, 6, 3, 9, 30))
    # Number of bars: need bars up to and including or_end_time
    n_or_bars = or_duration_minutes // 5 + 1  # e.g. 15min → bars at 9:30, 9:35, 9:40, 9:45
    bars = []
    mid = (or_high + or_low) / 2.0
    for i in range(n_or_bars):
        ts = start + timedelta(minutes=5 * i)
        if i == 0:
            h, l = or_high, mid
        elif i == 1:
            h, l = mid, or_low
        else:
            h, l = mid + 0.5, mid - 0.5
        c = (h + l) / 2.0
        bars.append((ts, c, h, l, c, 2000))
    return bars


def _warmup_atr(strat: VIXAdaptiveORBStrategy, n: int = 20) -> None:
    """Feed enough bars to initialize ATR before the OR day."""
    base = ET.localize(datetime(2024, 5, 31, 10, 0))
    for i in range(n):
        ts = base + timedelta(minutes=5 * i)
        strat.on_bar(ts, 5420.0, 5422.0, 5418.0, 5420.0, 1000)


def _make_orb_df(
    n_days: int = 5,
    base_price: float = 5420.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate multi-day RTH OHLCV for ORB testing."""
    rng = np.random.RandomState(seed)
    bars_per_day = 78  # 5-min bars in RTH
    timestamps = []
    opens, highs, lows, closes, volumes = [], [], [], [], []

    day = ET.localize(datetime(2024, 6, 3, 9, 30))
    for d in range(n_days):
        price = base_price
        for b in range(bars_per_day):
            ts = day + timedelta(minutes=5 * b)
            timestamps.append(ts)
            price += rng.randn() * 2.0
            o = price + rng.randn() * 0.5
            h = max(o, price) + abs(rng.randn()) * 1.5
            l = min(o, price) - abs(rng.randn()) * 1.5
            opens.append(o)
            highs.append(h)
            lows.append(l)
            closes.append(price)
            volumes.append(rng.randint(1000, 5000))
        day += timedelta(days=1)
        while day.weekday() >= 5:
            day += timedelta(days=1)
        day = day.replace(hour=9, minute=30)

    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": volumes,
    }, index=pd.DatetimeIndex(timestamps, name="timestamp"))


def _make_vix_series(df: pd.DataFrame, value: float = 20.0) -> pd.Series:
    dates = sorted(set(df.index.date))
    return pd.Series(value, index=dates, name="vix")


# ── Tests ────────────────────────────────────────────────────────────

class TestOpeningRangeBuild:
    """Test Opening Range construction."""

    def test_or_builds_correctly(self):
        """OR should track highest high and lowest low."""
        strat = VIXAdaptiveORBStrategy(params={"vix_or_durations": {"NORMAL": 15}})
        _warmup_atr(strat)

        or_bars = _make_or_bars(or_high=5425.0, or_low=5418.0)
        for ts, o, h, l, c, v in or_bars:
            strat.on_bar(ts, o, h, l, c, v)

        assert strat._or_complete is True
        assert strat._or_high == pytest.approx(5425.0, abs=0.01)
        assert strat._or_low == pytest.approx(5418.0, abs=0.01)
        assert strat._or_width == pytest.approx(7.0, abs=0.01)

    def test_or_not_complete_before_duration(self):
        """OR should not be complete before duration elapses."""
        strat = VIXAdaptiveORBStrategy(params={"vix_or_durations": {"NORMAL": 15}})
        _warmup_atr(strat)

        # Only feed 2 of 4 bars (need bar at 9:45 to complete)
        or_bars = _make_or_bars()
        for ts, o, h, l, c, v in or_bars[:2]:
            strat.on_bar(ts, o, h, l, c, v)

        assert strat._or_complete is False


class TestVIXAdaptiveDuration:
    """Test OR duration adapts to VIX regime."""

    def test_low_vix_5min(self):
        strat = VIXAdaptiveORBStrategy()
        strat.set_vix(12.0)
        assert strat._get_or_duration() == 5

    def test_normal_vix_15min(self):
        strat = VIXAdaptiveORBStrategy()
        strat.set_vix(20.0)
        assert strat._get_or_duration() == 15

    def test_high_vix_30min(self):
        strat = VIXAdaptiveORBStrategy()
        strat.set_vix(30.0)
        assert strat._get_or_duration() == 30

    def test_extreme_vix_30min(self):
        strat = VIXAdaptiveORBStrategy()
        strat.set_vix(40.0)
        assert strat._get_or_duration() == 30


class TestBreakoutSignals:
    """Test breakout detection and signal generation."""

    def test_long_breakout(self):
        """Close above OR_high with positive VWAP slope → LONG."""
        strat = VIXAdaptiveORBStrategy(
            params={"vix_or_durations": {"NORMAL": 15}},
            capital=20_000.0,
        )
        _warmup_atr(strat)

        # Build OR
        or_bars = _make_or_bars(or_high=5425.0, or_low=5418.0)
        for ts, o, h, l, c, v in or_bars:
            strat.on_bar(ts, o, h, l, c, v)

        assert strat._or_complete is True

        # Feed a few bars within OR range to build VWAP slope
        for i in range(5):
            ts = or_bars[-1][0] + timedelta(minutes=5 * (i + 1))
            strat.on_bar(ts, 5422.0 + i * 0.5, 5423.0 + i * 0.5,
                         5421.0 + i * 0.5, 5422.0 + i * 0.5, 3000)

        # Breakout bar: close above OR_high
        breakout_ts = or_bars[-1][0] + timedelta(minutes=35)
        strat.on_bar(breakout_ts, 5424.0, 5427.0, 5424.0, 5426.0, 5000)

        assert strat.position is not None
        assert strat.position.side == Side.LONG

    def test_short_breakout(self):
        """Close below OR_low with negative VWAP slope → SHORT."""
        strat = VIXAdaptiveORBStrategy(
            params={"vix_or_durations": {"NORMAL": 15}},
            capital=20_000.0,
        )
        _warmup_atr(strat)

        or_bars = _make_or_bars(or_high=5425.0, or_low=5418.0)
        for ts, o, h, l, c, v in or_bars:
            strat.on_bar(ts, o, h, l, c, v)

        # Declining VWAP slope
        for i in range(5):
            ts = or_bars[-1][0] + timedelta(minutes=5 * (i + 1))
            strat.on_bar(ts, 5421.0 - i * 0.5, 5422.0 - i * 0.5,
                         5420.0 - i * 0.5, 5421.0 - i * 0.5, 3000)

        # Breakout bar: close below OR_low
        breakout_ts = or_bars[-1][0] + timedelta(minutes=35)
        strat.on_bar(breakout_ts, 5419.0, 5419.0, 5416.0, 5417.0, 5000)

        assert strat.position is not None
        assert strat.position.side == Side.SHORT


class TestFilterRejection:
    """Test that filters correctly reject bad signals."""

    def test_time_filter_rejects_late_entry(self):
        """No entries after 2:00 PM."""
        strat = VIXAdaptiveORBStrategy(capital=20_000.0)
        _warmup_atr(strat)

        # Build OR
        or_bars = _make_or_bars()
        for ts, o, h, l, c, v in or_bars:
            strat.on_bar(ts, o, h, l, c, v)

        # Feed slope bars
        for i in range(5):
            ts = or_bars[-1][0] + timedelta(minutes=5 * (i + 1))
            strat.on_bar(ts, 5422.0, 5423.0, 5421.0, 5422.0 + i * 0.5, 3000)

        # Breakout bar at 2:30 PM
        late_ts = ET.localize(datetime(2024, 6, 3, 14, 30))
        strat.on_bar(late_ts, 5424.0, 5427.0, 5424.0, 5426.0, 5000)
        assert strat.position is None


class TestExitManagement:
    """Test exit logic."""

    def test_stop_loss_long(self):
        strat = VIXAdaptiveORBStrategy(capital=20_000.0)
        ts = ET.localize(datetime(2024, 6, 3, 10, 30))
        strat._current_date = ts.date()

        strat.position = ORBPosition(
            side=Side.LONG,
            entry_price=5426.0,
            entry_time=ts,
            contracts=1,
            stop_price=5422.5,
            target_price=5433.0,
            or_width=7.0,
        )
        strat._or_duration_minutes = 15

        result = strat._manage_exits(5422.0, ts + timedelta(minutes=5))
        assert result is not None
        assert result.exit_reason == "stop_loss"

    def test_target_hit_long(self):
        strat = VIXAdaptiveORBStrategy(capital=20_000.0)
        ts = ET.localize(datetime(2024, 6, 3, 10, 30))
        strat._current_date = ts.date()

        strat.position = ORBPosition(
            side=Side.LONG,
            entry_price=5426.0,
            entry_time=ts,
            contracts=1,
            stop_price=5422.5,
            target_price=5433.0,
            or_width=7.0,
        )
        strat._or_duration_minutes = 15

        result = strat._manage_exits(5433.5, ts + timedelta(minutes=30))
        assert result is not None
        assert result.exit_reason == "target_hit"
        assert result.pnl > 0

    def test_trailing_stop_to_breakeven(self):
        """After 0.75 * OR_width profit, stop moves to entry."""
        strat = VIXAdaptiveORBStrategy(capital=20_000.0)
        ts = ET.localize(datetime(2024, 6, 3, 10, 30))
        strat._current_date = ts.date()

        strat.position = ORBPosition(
            side=Side.LONG,
            entry_price=5426.0,
            entry_time=ts,
            contracts=1,
            stop_price=5422.5,
            target_price=5433.0,
            or_width=7.0,
        )
        strat._or_duration_minutes = 15

        # Move price up by 0.75 * 7.0 = 5.25 → close at 5431.25
        result = strat._manage_exits(5431.5, ts + timedelta(minutes=10))
        assert result is None  # Still open
        assert strat.position.breakeven_moved is True
        assert strat.position.stop_price == 5426.0  # Moved to entry

    def test_pnl_calculation(self):
        strat = VIXAdaptiveORBStrategy(capital=20_000.0)
        ts = ET.localize(datetime(2024, 6, 3, 10, 30))
        strat._current_date = ts.date()
        strat._or_duration_minutes = 15

        strat.position = ORBPosition(
            side=Side.LONG,
            entry_price=5426.0,
            entry_time=ts,
            contracts=2,
            stop_price=5422.5,
            target_price=5433.0,
            or_width=7.0,
        )

        record = strat._close_position(5433.0, ts + timedelta(minutes=30), "target_hit")
        # PnL = (5433 - 5426) * 5.0 * 2 = 70.0
        assert record.pnl == pytest.approx(70.0, abs=1e-6)
        assert record.or_width == 7.0
        assert record.or_duration_minutes == 15


class TestDailyReset:
    """OR state resets on new trading day."""

    def test_or_resets_on_new_day(self):
        strat = VIXAdaptiveORBStrategy()
        _warmup_atr(strat)

        # Day 1: build OR
        or_bars = _make_or_bars()
        for ts, o, h, l, c, v in or_bars:
            strat.on_bar(ts, o, h, l, c, v)
        assert strat._or_complete is True

        # Day 2: new day should reset
        day2 = ET.localize(datetime(2024, 6, 4, 9, 30))
        strat.on_bar(day2, 5420.0, 5422.0, 5418.0, 5420.0, 1000)
        assert strat._or_complete is False  # Reset for new day

    def test_only_one_trade_per_day(self):
        strat = VIXAdaptiveORBStrategy(capital=20_000.0)
        strat._traded_today = True
        strat._or_complete = True
        strat._or_high = 5425.0
        strat._or_low = 5418.0
        strat._or_width = 7.0

        ts = ET.localize(datetime(2024, 6, 3, 11, 0))
        strat._current_date = ts.date()
        # Even with valid breakout, should not enter
        strat.on_bar(ts, 5424.0, 5427.0, 5424.0, 5426.0, 5000)
        assert strat.position is None


class TestGenerateSignals:
    """Test vectorized signal generation."""

    def test_returns_correct_shape(self):
        df = _make_orb_df(n_days=3)
        vix = _make_vix_series(df, 20.0)
        strat = VIXAdaptiveORBStrategy(capital=20_000.0)
        result = strat.generate_signals(df, vix_series=vix)
        assert len(result) == len(df)
        assert "signal" in result.columns
        assert "or_width" in result.columns

    def test_signals_are_valid_values(self):
        df = _make_orb_df(n_days=5)
        vix = _make_vix_series(df, 20.0)
        strat = VIXAdaptiveORBStrategy(capital=20_000.0)
        result = strat.generate_signals(df, vix_series=vix)
        assert set(result["signal"].unique()).issubset({-1, 0, 1})
