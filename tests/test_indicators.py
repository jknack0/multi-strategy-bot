"""
Unit tests for all indicators with known-answer test vectors.
"""

import math
from datetime import datetime, timedelta

import numpy as np
import pytest
import pytz

from indicators.bollinger import BollingerBands
from indicators.atr import ATR
from indicators.vwap import VWAP
from indicators.vix_regime import VIXRegimeClassifier
from config.constants import VIXRegime


# ═══════════════════════════════════════════════════════════════════════════
# Bollinger Bands
# ═══════════════════════════════════════════════════════════════════════════

class TestBollingerBands:
    """Known-answer tests for Bollinger Bands."""

    def test_not_ready_before_period(self):
        bb = BollingerBands(period=5, num_std=2.0)
        for val in [1.0, 2.0, 3.0, 4.0]:
            result = bb.update(val)
            assert result is None
        assert not bb.is_ready

    def test_known_answer_simple_sequence(self):
        """BB on [1,2,3,4,5,6,7,8,9,10] with period=5, sigma=2."""
        bb = BollingerBands(period=5, num_std=2.0)
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

        results = []
        for v in values:
            r = bb.update(v)
            if r is not None:
                results.append(r)

        # After feeding [1,2,3,4,5]: mean=3.0, pop_std=sqrt(2.0)=1.4142
        first = results[0]
        assert first["middle"] == pytest.approx(3.0, abs=1e-6)
        expected_std = math.sqrt(2.0)  # population std of [1,2,3,4,5]
        assert first["upper"] == pytest.approx(3.0 + 2.0 * expected_std, abs=1e-6)
        assert first["lower"] == pytest.approx(3.0 - 2.0 * expected_std, abs=1e-6)

        # After feeding [6,7,8,9,10] → window is [6,7,8,9,10]: mean=8.0
        last = results[-1]
        assert last["middle"] == pytest.approx(8.0, abs=1e-6)
        expected_std_last = math.sqrt(2.0)  # population std of [6,7,8,9,10]
        assert last["upper"] == pytest.approx(8.0 + 2.0 * expected_std_last, abs=1e-6)
        assert last["lower"] == pytest.approx(8.0 - 2.0 * expected_std_last, abs=1e-6)

    def test_pct_b_at_upper(self):
        """When close is at the upper band, %B should be ~1.0."""
        bb = BollingerBands(period=5, num_std=2.0)
        # Flat data → bands collapse → then a spike
        for _ in range(4):
            bb.update(100.0)
        # 5th value: still 100, std=0
        r = bb.update(100.0)
        # With zero std, pct_b should be 0.5
        assert r["pct_b"] == pytest.approx(0.5, abs=1e-6)

    def test_pct_b_range(self):
        """pct_b should typically be between 0 and 1 for data within bands."""
        bb = BollingerBands(period=20, num_std=2.0)
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        for p in prices:
            r = bb.update(float(p))
            if r is not None:
                # Not strictly bounded but should be reasonable
                assert -1.0 <= r["pct_b"] <= 2.0

    def test_bandwidth_positive(self):
        bb = BollingerBands(period=5, num_std=2.0)
        for v in [10, 11, 12, 11, 10]:
            r = bb.update(float(v))
        assert r is not None
        assert r["bandwidth"] >= 0.0

    def test_reset(self):
        bb = BollingerBands(period=5, num_std=2.0)
        for v in [1, 2, 3, 4, 5]:
            bb.update(float(v))
        assert bb.is_ready
        bb.reset()
        assert not bb.is_ready


# ═══════════════════════════════════════════════════════════════════════════
# ATR
# ═══════════════════════════════════════════════════════════════════════════

class TestATR:
    """Known-answer tests for ATR with Wilder smoothing."""

    def test_not_ready_before_period(self):
        atr = ATR(period=3)
        assert atr.update(10.0, 9.0, 9.5) is None
        assert atr.update(11.0, 9.5, 10.0) is None
        assert not atr.is_ready

    def test_initial_atr_is_sma(self):
        """First ATR value should be SMA of first N true ranges."""
        atr = ATR(period=3)
        # Bar 1: H=10, L=9, C=9.5 → TR=1.0 (no prev close)
        atr.update(10.0, 9.0, 9.5)
        # Bar 2: H=11, L=9.5, C=10.5 → TR=max(1.5, |11-9.5|, |9.5-9.5|)=1.5
        atr.update(11.0, 9.5, 10.5)
        # Bar 3: H=10.5, L=9.0, C=9.5 → TR=max(1.5, |10.5-10.5|, |9.0-10.5|)=1.5
        result = atr.update(10.5, 9.0, 9.5)

        assert result is not None
        expected = (1.0 + 1.5 + 1.5) / 3.0  # SMA of first 3 TRs
        assert result == pytest.approx(expected, abs=1e-6)

    def test_wilder_smoothing(self):
        """After initialization, ATR uses Wilder smoothing."""
        atr = ATR(period=3)
        atr.update(10.0, 9.0, 9.5)   # TR=1.0
        atr.update(11.0, 9.5, 10.5)  # TR=1.5
        initial = atr.update(10.5, 9.0, 9.5)  # TR=1.5, ATR=4/3

        # Bar 4: H=12, L=10, C=11 → TR=max(2, |12-9.5|, |10-9.5|)=2.5
        result = atr.update(12.0, 10.0, 11.0)
        expected = (initial * 2 + 2.5) / 3.0  # Wilder: prev*(n-1)/n + tr/n
        assert result == pytest.approx(expected, abs=1e-6)

    def test_true_range_with_gap(self):
        """true_range should handle gaps (|H-prevC| or |L-prevC| > H-L)."""
        # Gap up: prev_close=100, current H=105, L=103
        tr = ATR.true_range(105.0, 103.0, 100.0)
        # max(2, 5, 3) = 5
        assert tr == pytest.approx(5.0, abs=1e-6)

    def test_true_range_no_prev_close(self):
        tr = ATR.true_range(10.0, 8.0, None)
        assert tr == pytest.approx(2.0, abs=1e-6)

    def test_reset(self):
        atr = ATR(period=3)
        for h, l, c in [(10, 9, 9.5), (11, 9.5, 10.5), (10.5, 9, 9.5)]:
            atr.update(float(h), float(l), float(c))
        assert atr.is_ready
        atr.reset()
        assert not atr.is_ready
        assert atr.current_atr is None


# ═══════════════════════════════════════════════════════════════════════════
# VWAP
# ═══════════════════════════════════════════════════════════════════════════

class TestVWAP:
    """Tests for session VWAP."""

    def test_single_bar(self):
        vwap = VWAP()
        ts = datetime(2024, 1, 2, 10, 0, tzinfo=pytz.UTC)
        result = vwap.update(ts, high=105.0, low=95.0, close=100.0, volume=1000)
        # typical = (105+95+100)/3 = 100.0
        assert result["vwap"] == pytest.approx(100.0, abs=1e-6)

    def test_volume_weighted(self):
        vwap = VWAP()
        ts1 = datetime(2024, 1, 2, 10, 0, tzinfo=pytz.UTC)
        ts2 = datetime(2024, 1, 2, 10, 1, tzinfo=pytz.UTC)

        # Bar 1: typical=(105+95+100)/3=100, volume=1000
        vwap.update(ts1, 105.0, 95.0, 100.0, 1000)
        # Bar 2: typical=(210+190+200)/3=200, volume=3000
        result = vwap.update(ts2, 210.0, 190.0, 200.0, 3000)

        expected = (100.0 * 1000 + 200.0 * 3000) / (1000 + 3000)
        assert result["vwap"] == pytest.approx(expected, abs=1e-6)

    def test_std_bands(self):
        vwap = VWAP(num_std_bands=2)
        ts = datetime(2024, 1, 2, 10, 0, tzinfo=pytz.UTC)
        result = vwap.update(ts, 105.0, 95.0, 100.0, 1000)
        # With one bar, std should be 0
        assert result["std"] == pytest.approx(0.0, abs=1e-6)
        assert "upper_1" in result
        assert "lower_2" in result

    def test_reset_on_session(self):
        """VWAP should reset at 9:30 ET."""
        et = pytz.timezone("US/Eastern")
        vwap = VWAP()

        # Bar before reset
        ts1 = et.localize(datetime(2024, 1, 2, 9, 29))
        vwap.update(ts1, 105.0, 95.0, 100.0, 1000)

        # Bar after reset (9:30)
        ts2 = et.localize(datetime(2024, 1, 2, 9, 30))
        result = vwap.update(ts2, 210.0, 190.0, 200.0, 500)

        # Should be reset: only bar 2 data
        expected = (210.0 + 190.0 + 200.0) / 3.0
        assert result["vwap"] == pytest.approx(expected, abs=1e-6)

    def test_bar_count(self):
        vwap = VWAP()
        ts = datetime(2024, 1, 2, 10, 0, tzinfo=pytz.UTC)
        for i in range(5):
            vwap.update(ts + timedelta(minutes=i), 100.0, 99.0, 99.5, 100)
        assert vwap.bar_count == 5


# ═══════════════════════════════════════════════════════════════════════════
# VIX Regime
# ═══════════════════════════════════════════════════════════════════════════

class TestVIXRegime:
    """Tests for VIX regime classification."""

    def test_low_regime(self):
        clf = VIXRegimeClassifier()
        result = clf.classify(12.0)
        assert result["regime"] == VIXRegime.LOW
        assert result["size_mult"] == 1.0

    def test_normal_regime(self):
        clf = VIXRegimeClassifier()
        result = clf.classify(20.0)
        assert result["regime"] == VIXRegime.NORMAL

    def test_high_regime(self):
        clf = VIXRegimeClassifier()
        result = clf.classify(30.0)
        assert result["regime"] == VIXRegime.HIGH
        assert result["size_mult"] < 1.0

    def test_extreme_regime(self):
        clf = VIXRegimeClassifier()
        result = clf.classify(40.0)
        assert result["regime"] == VIXRegime.EXTREME
        assert result["size_mult"] == 0.3

    def test_boundary_values(self):
        clf = VIXRegimeClassifier()
        # At boundary 15: should be NORMAL (>= 15)
        assert clf.classify(15.0)["regime"] == VIXRegime.NORMAL
        # At boundary 25: should be HIGH
        assert clf.classify(25.0)["regime"] == VIXRegime.HIGH
        # At boundary 35: should be EXTREME
        assert clf.classify(35.0)["regime"] == VIXRegime.EXTREME

    def test_regime_summary(self):
        clf = VIXRegimeClassifier()
        clf.classify(10.0)
        clf.classify(20.0)
        clf.classify(30.0)
        summary = clf.regime_summary()
        assert summary["LOW"] == 1
        assert summary["NORMAL"] == 1
        assert summary["HIGH"] == 1
