"""
Tests for the bar aggregator.

Verifies that the BarAggregator correctly produces 1-min and 5-min
OHLCV bars from synthetic 5-second bar data.
"""

from datetime import datetime, timedelta

import pytz
import pytest

from data.aggregator import BarAggregator, OHLCVBar


class TestBarAggregator:
    """Verify aggregator produces correct OHLCV from synthetic data."""

    def _make_5sec_bars(self, n: int, start: datetime, base_price: float = 100.0):
        """Generate n synthetic 5-second bars."""
        bars = []
        for i in range(n):
            ts = start + timedelta(seconds=5 * i)
            o = base_price + i * 0.1
            h = o + 0.5
            l = o - 0.3
            c = o + 0.2
            v = 100 + i
            bars.append((ts, o, h, l, c, v))
        return bars

    def test_1min_bar_emission(self):
        """12 five-second bars should produce exactly 1 one-minute bar."""
        agg = BarAggregator(symbol="MES")
        completed_bars = []
        agg.on_bar_complete = lambda bar: completed_bars.append(bar)

        start = datetime(2024, 1, 2, 10, 0, 0, tzinfo=pytz.UTC)
        bars = self._make_5sec_bars(12, start)

        for ts, o, h, l, c, v in bars:
            agg.process_bar(ts, o, h, l, c, v)

        # Should have emitted 1 one-minute bar and 0 five-minute bars
        one_min_bars = [b for b in completed_bars if b.bar_size == "1min"]
        assert len(one_min_bars) == 1

    def test_5min_bar_emission(self):
        """60 five-second bars should produce exactly 1 five-minute bar."""
        agg = BarAggregator(symbol="MES")
        completed_bars = []
        agg.on_bar_complete = lambda bar: completed_bars.append(bar)

        start = datetime(2024, 1, 2, 10, 0, 0, tzinfo=pytz.UTC)
        bars = self._make_5sec_bars(60, start)

        for ts, o, h, l, c, v in bars:
            agg.process_bar(ts, o, h, l, c, v)

        five_min_bars = [b for b in completed_bars if b.bar_size == "5min"]
        assert len(five_min_bars) == 1

    def test_ohlcv_correctness(self):
        """Verify that the aggregated 1-min bar has correct OHLCV values."""
        agg = BarAggregator(symbol="MES")
        completed_bars = []
        agg.on_bar_complete = lambda bar: completed_bars.append(bar)

        start = datetime(2024, 1, 2, 10, 0, 0, tzinfo=pytz.UTC)
        bars = self._make_5sec_bars(12, start)

        # Track expected values manually
        expected_open = bars[0][1]       # open of first bar
        expected_high = max(b[2] for b in bars)   # max of all highs
        expected_low = min(b[3] for b in bars)    # min of all lows
        expected_close = bars[-1][4]     # close of last bar
        expected_volume = sum(b[5] for b in bars)

        for ts, o, h, l, c, v in bars:
            agg.process_bar(ts, o, h, l, c, v)

        one_min_bars = [b for b in completed_bars if b.bar_size == "1min"]
        assert len(one_min_bars) == 1

        bar = one_min_bars[0]
        assert bar.open == pytest.approx(expected_open, abs=1e-6)
        assert bar.high == pytest.approx(expected_high, abs=1e-6)
        assert bar.low == pytest.approx(expected_low, abs=1e-6)
        assert bar.close == pytest.approx(expected_close, abs=1e-6)
        assert bar.volume == expected_volume

    def test_multiple_1min_bars(self):
        """24 five-second bars should produce exactly 2 one-minute bars."""
        agg = BarAggregator(symbol="MES")
        completed_bars = []
        agg.on_bar_complete = lambda bar: completed_bars.append(bar)

        start = datetime(2024, 1, 2, 10, 0, 0, tzinfo=pytz.UTC)
        bars = self._make_5sec_bars(24, start)

        for ts, o, h, l, c, v in bars:
            agg.process_bar(ts, o, h, l, c, v)

        one_min_bars = [b for b in completed_bars if b.bar_size == "1min"]
        assert len(one_min_bars) == 2

    def test_partial_bar_not_emitted(self):
        """Fewer than 12 five-second bars should NOT produce a 1-min bar."""
        agg = BarAggregator(symbol="MES")
        completed_bars = []
        agg.on_bar_complete = lambda bar: completed_bars.append(bar)

        start = datetime(2024, 1, 2, 10, 0, 0, tzinfo=pytz.UTC)
        bars = self._make_5sec_bars(11, start)

        for ts, o, h, l, c, v in bars:
            agg.process_bar(ts, o, h, l, c, v)

        one_min_bars = [b for b in completed_bars if b.bar_size == "1min"]
        assert len(one_min_bars) == 0

    def test_session_boundary_flushes(self):
        """Bars across session boundary (18:00 ET) should flush partial bars."""
        et = pytz.timezone("US/Eastern")
        agg = BarAggregator(symbol="MES")
        completed_bars = []
        agg.on_bar_complete = lambda bar: completed_bars.append(bar)

        # Feed 6 bars before 18:00 ET
        pre_boundary = et.localize(datetime(2024, 1, 2, 17, 59, 30))
        for i in range(6):
            ts = pre_boundary + timedelta(seconds=5 * i)
            agg.process_bar(ts, 100.0, 101.0, 99.0, 100.5, 100)

        # Feed 1 bar after 18:00 ET to trigger boundary detection
        post_boundary = et.localize(datetime(2024, 1, 2, 18, 0, 5))
        agg.process_bar(post_boundary, 100.0, 101.0, 99.0, 100.5, 100)

        # The partial bar should have been flushed
        assert len(completed_bars) > 0

    def test_symbol_propagation(self):
        """Completed bars should carry the correct symbol."""
        agg = BarAggregator(symbol="MES")
        completed_bars = []
        agg.on_bar_complete = lambda bar: completed_bars.append(bar)

        start = datetime(2024, 1, 2, 10, 0, 0, tzinfo=pytz.UTC)
        bars = self._make_5sec_bars(12, start)

        for ts, o, h, l, c, v in bars:
            agg.process_bar(ts, o, h, l, c, v)

        assert all(b.symbol == "MES" for b in completed_bars)

    def test_multiple_callbacks(self):
        """Multiple callbacks should all be invoked."""
        agg = BarAggregator(symbol="MES")
        results_a = []
        results_b = []
        agg.add_callback(lambda bar: results_a.append(bar))
        agg.add_callback(lambda bar: results_b.append(bar))

        start = datetime(2024, 1, 2, 10, 0, 0, tzinfo=pytz.UTC)
        bars = self._make_5sec_bars(12, start)

        for ts, o, h, l, c, v in bars:
            agg.process_bar(ts, o, h, l, c, v)

        assert len(results_a) == len(results_b)
        assert len(results_a) > 0
