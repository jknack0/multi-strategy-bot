"""Tests for VWAPSlope indicator."""

import pytest
from indicators.vwap_slope import VWAPSlope


class TestVWAPSlope:
    """Known-answer tests for VWAP slope."""

    def test_positive_slope(self):
        """Increasing VWAP values should produce positive slope."""
        vs = VWAPSlope(lookback=5)
        values = [100.0, 101.0, 102.0, 103.0, 104.0]
        for v in values[:-1]:
            result = vs.update(v)
        result = vs.update(values[-1])
        assert result is not None
        assert result > 0

    def test_negative_slope(self):
        """Decreasing VWAP values should produce negative slope."""
        vs = VWAPSlope(lookback=5)
        values = [104.0, 103.0, 102.0, 101.0, 100.0]
        result = None
        for v in values:
            result = vs.update(v)
        assert result is not None
        assert result < 0

    def test_flat_slope(self):
        """Constant VWAP values should produce zero slope."""
        vs = VWAPSlope(lookback=5)
        for _ in range(5):
            result = vs.update(100.0)
        assert result is not None
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_returns_none_before_full(self):
        """Should return None until buffer is full."""
        vs = VWAPSlope(lookback=5)
        for _ in range(4):
            result = vs.update(100.0)
            assert result is None
        assert not vs.is_ready

    def test_is_ready_when_full(self):
        vs = VWAPSlope(lookback=3)
        vs.update(1.0)
        vs.update(2.0)
        assert not vs.is_ready
        vs.update(3.0)
        assert vs.is_ready

    def test_known_slope_value(self):
        """OLS slope of [0,1,2,3,4] with x=[0,1,2,3,4] should be 1.0."""
        vs = VWAPSlope(lookback=5)
        for v in [0.0, 1.0, 2.0, 3.0, 4.0]:
            result = vs.update(v)
        assert result == pytest.approx(1.0, abs=1e-10)

    def test_normalized_slope(self):
        """Normalized slope should divide by ATR."""
        vs = VWAPSlope(lookback=5)
        for v in [0.0, 1.0, 2.0, 3.0, 4.0]:
            vs.update(v)
        norm = vs.get_normalized_slope(atr=2.0)
        assert norm == pytest.approx(0.5, abs=1e-10)

    def test_normalized_slope_none_when_not_ready(self):
        vs = VWAPSlope(lookback=5)
        vs.update(1.0)
        assert vs.get_normalized_slope(atr=1.0) is None

    def test_normalized_slope_none_when_zero_atr(self):
        vs = VWAPSlope(lookback=3)
        for v in [1.0, 2.0, 3.0]:
            vs.update(v)
        assert vs.get_normalized_slope(atr=0.0) is None

    def test_reset(self):
        vs = VWAPSlope(lookback=3)
        for v in [1.0, 2.0, 3.0]:
            vs.update(v)
        assert vs.is_ready
        vs.reset()
        assert not vs.is_ready
        assert vs.current_slope is None

    def test_invalid_lookback(self):
        with pytest.raises(ValueError):
            VWAPSlope(lookback=1)
