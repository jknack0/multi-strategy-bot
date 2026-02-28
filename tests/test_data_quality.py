"""Tests for data quality checks."""

import numpy as np
import pandas as pd
import pytest

from data.data_quality import (
    DataQualityChecker,
    validate_bars,
    generate_quality_report,
)


@pytest.fixture
def checker():
    return DataQualityChecker(sigma_threshold=5.0)


def _make_df(n=50, freq="1min"):
    """Create a clean OHLCV DataFrame for testing."""
    rng = np.random.RandomState(42)
    dates = pd.date_range("2024-01-02 09:30", periods=n, freq=freq)
    close = 5000.0 + np.cumsum(rng.randn(n) * 0.5)
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    open_ = close + rng.randn(n) * 0.3
    # Ensure OHLC validity
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    volume = rng.randint(100, 5000, n)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


class TestOHLCValidity:
    """Verify OHLC validation catches invalid bars."""

    def test_high_less_than_low(self, checker):
        """A bar where high < low should be flagged."""
        df = _make_df(10)
        # Corrupt one bar: swap high and low
        df.iloc[5, df.columns.get_loc("high")] = df.iloc[5]["low"] - 1.0
        issues = checker.check_ohlc_validity(df)
        assert len(issues) >= 1
        assert any(i.issue_type == "ohlc_invalid" for i in issues)

    def test_high_less_than_open(self, checker):
        """High < open should be flagged."""
        df = _make_df(10)
        df.iloc[3, df.columns.get_loc("high")] = df.iloc[3]["open"] - 1.0
        issues = checker.check_ohlc_validity(df)
        assert len(issues) >= 1

    def test_low_greater_than_close(self, checker):
        """Low > close should be flagged."""
        df = _make_df(10)
        df.iloc[3, df.columns.get_loc("low")] = df.iloc[3]["close"] + 1.0
        issues = checker.check_ohlc_validity(df)
        assert len(issues) >= 1

    def test_clean_data_passes(self, checker):
        """Properly formed OHLC should have no issues."""
        df = _make_df(50)
        issues = checker.check_ohlc_validity(df)
        assert len(issues) == 0


class TestZeroVolume:
    """Verify zero-volume bar detection."""

    def test_zero_volume_detected(self, checker):
        """Bars with volume == 0 should be flagged."""
        df = _make_df(20)
        df.iloc[5, df.columns.get_loc("volume")] = 0
        df.iloc[12, df.columns.get_loc("volume")] = 0
        issues = checker.check_zero_volume(df)
        assert len(issues) == 2
        assert all(i.issue_type == "zero_volume" for i in issues)

    def test_no_zero_volume(self, checker):
        """All positive volumes should produce no issues."""
        df = _make_df(20)
        issues = checker.check_zero_volume(df)
        assert len(issues) == 0


class TestOutlierDetection:
    """Verify outlier detection catches extreme moves."""

    def test_spike_detected(self, checker):
        """A 10-sigma spike should be flagged as an outlier."""
        df = _make_df(100)
        # Inject a massive spike at bar 50
        normal_std = df["close"].pct_change().std()
        df.iloc[50, df.columns.get_loc("close")] = df.iloc[49]["close"] * (1 + 10 * normal_std)
        issues = checker.check_outliers(df, column="close")
        assert len(issues) >= 1
        assert any(i.issue_type == "outlier" for i in issues)

    def test_normal_data_no_outliers(self, checker):
        """Normal price movements should not be flagged."""
        df = _make_df(100)
        issues = checker.check_outliers(df)
        assert len(issues) == 0


class TestDuplicateTimestamps:
    """Verify duplicate timestamp detection."""

    def test_duplicates_detected(self):
        """Duplicate timestamps should be caught by validate_bars."""
        df = _make_df(20)
        # Create a duplicate by repeating index entry
        df2 = pd.concat([df, df.iloc[[5]]])
        result = validate_bars(df2)
        assert len(result["duplicate_timestamps"]) >= 1

    def test_no_duplicates(self):
        """Unique timestamps should produce empty list."""
        df = _make_df(20)
        result = validate_bars(df)
        assert len(result["duplicate_timestamps"]) == 0


class TestGapDetection:
    """Verify gap detection in bar data."""

    def test_gap_detected(self, checker):
        """A 5-minute gap in 1-minute data should be flagged."""
        df = _make_df(20, freq="1min")
        # Drop bars 10-13 to create a gap
        df = df.drop(df.index[10:14])
        issues = checker.check_gaps(df, bar_size="1min")
        assert len(issues) >= 1
        assert any(i.issue_type == "gap" for i in issues)

    def test_no_gaps(self, checker):
        """Continuous 1-minute data should have no gaps."""
        df = _make_df(20, freq="1min")
        issues = checker.check_gaps(df, bar_size="1min")
        assert len(issues) == 0


class TestValidateBars:
    """Test the convenience validate_bars() function."""

    def test_returns_all_keys(self):
        df = _make_df(50)
        result = validate_bars(df)
        assert "ohlc_violations" in result
        assert "zero_volume" in result
        assert "outliers" in result
        assert "duplicate_timestamps" in result
        assert "gaps" in result

    def test_clean_data(self):
        df = _make_df(50)
        result = validate_bars(df)
        assert len(result["ohlc_violations"]) == 0
        assert len(result["zero_volume"]) == 0
        assert len(result["duplicate_timestamps"]) == 0


class TestGenerateQualityReport:
    """Test human-readable report generation."""

    def test_returns_string(self):
        df = _make_df(50)
        report = generate_quality_report(df, "MES", "1min")
        assert isinstance(report, str)
        assert "MES" in report
        assert "Total bars: 50" in report

    def test_reports_issues(self):
        df = _make_df(20)
        df.iloc[5, df.columns.get_loc("volume")] = 0
        report = generate_quality_report(df, "MES", "1min")
        assert "Zero-volume: 1" in report
