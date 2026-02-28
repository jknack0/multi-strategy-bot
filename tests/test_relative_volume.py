"""Tests for RelativeVolume indicator."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz
import pytest

from indicators.relative_volume import RelativeVolume

ET = pytz.timezone("US/Eastern")


def _make_historical_data(n_days: int = 25, bars_per_day: int = 78) -> pd.DataFrame:
    """Build synthetic RTH data with known volume pattern."""
    timestamps = []
    volumes = []
    base_price = 4500.0

    start_date = ET.localize(datetime(2024, 1, 2, 9, 30))
    day = start_date

    for d in range(n_days):
        for b in range(bars_per_day):
            ts = day + timedelta(minutes=5 * b)
            timestamps.append(ts)
            # Known pattern: volume peaks at open (9:30), declines through day
            # 9:30 = 3000, 10:00 = 2500, ..., 15:55 = 500
            vol = max(500, 3000 - b * 30)
            volumes.append(vol)
        day += timedelta(days=1)
        while day.weekday() >= 5:
            day += timedelta(days=1)
        day = day.replace(hour=9, minute=30, second=0, microsecond=0)

    closes = np.full(len(timestamps), base_price)
    return pd.DataFrame({
        "open": closes, "high": closes + 1, "low": closes - 1,
        "close": closes, "volume": volumes,
    }, index=pd.DatetimeIndex(timestamps, name="timestamp"))


class TestRelativeVolume:
    """Tests for relative volume indicator."""

    def test_build_profile(self):
        """Profile should contain time slots."""
        df = _make_historical_data()
        rvol = RelativeVolume(df)
        assert len(rvol.profile) > 0
        assert "09:30" in rvol.profile

    def test_relative_volume_at_average(self):
        """When current equals average, rvol should be ~1.0."""
        df = _make_historical_data()
        rvol = RelativeVolume(df)
        avg_930 = rvol.profile.get("09:30", 0)
        ts = ET.localize(datetime(2024, 6, 3, 9, 30))
        result = rvol.get_relative_volume(ts, avg_930)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_relative_volume_double(self):
        """When current is 2x average, rvol should be 2.0."""
        df = _make_historical_data()
        rvol = RelativeVolume(df)
        avg_930 = rvol.profile.get("09:30", 1000)
        ts = ET.localize(datetime(2024, 6, 3, 9, 30))
        result = rvol.get_relative_volume(ts, avg_930 * 2)
        assert result == pytest.approx(2.0, abs=0.01)

    def test_missing_slot_returns_neutral(self):
        """Unknown time slot returns 1.0."""
        df = _make_historical_data()
        rvol = RelativeVolume(df)
        # 17:00 is outside RTH, no profile entry
        ts = ET.localize(datetime(2024, 6, 3, 17, 0))
        result = rvol.get_relative_volume(ts, 1000)
        assert result == 1.0

    def test_zero_volume_returns_neutral(self):
        """Zero current volume returns 1.0."""
        df = _make_historical_data()
        rvol = RelativeVolume(df)
        ts = ET.localize(datetime(2024, 6, 3, 10, 0))
        assert rvol.get_relative_volume(ts, 0) == 1.0

    def test_no_historical_data(self):
        """With no data, always returns 1.0."""
        rvol = RelativeVolume()
        ts = ET.localize(datetime(2024, 6, 3, 10, 0))
        assert rvol.get_relative_volume(ts, 5000) == 1.0

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df.index.name = "timestamp"
        rvol = RelativeVolume(df)
        assert len(rvol.profile) == 0
