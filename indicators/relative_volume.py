"""
Relative Volume indicator — compares current bar volume to historical
time-of-day average.

Normalizes for intraday volume patterns (e.g., 10 AM naturally higher than 2 PM).
"""

from typing import Dict, Optional

import pandas as pd


class RelativeVolume:
    """Compare current bar volume to historical average at the same time of day.

    Usage:
        rvol = RelativeVolume(historical_data, min_days=20)
        ratio = rvol.get_relative_volume(timestamp, current_volume)
    """

    def __init__(
        self,
        historical_data: Optional[pd.DataFrame] = None,
        min_days: int = 20,
    ) -> None:
        self.min_days = min_days
        self._volume_profile: Dict[str, float] = {}
        if historical_data is not None and not historical_data.empty:
            self._volume_profile = self._build_profile(historical_data)

    def _build_profile(self, data: pd.DataFrame) -> Dict[str, float]:
        """Build time-of-day volume profile.

        Groups bars by HH:MM, computes mean volume per slot.
        Only uses RTH bars (9:30-16:00).
        """
        if data.empty or "volume" not in data.columns:
            return {}

        idx = data.index
        # Convert to time-of-day strings
        if hasattr(idx, "tz") and idx.tz is not None:
            import pytz
            et = pytz.timezone("US/Eastern")
            et_idx = idx.tz_convert(et)
        elif hasattr(idx, "tz_localize"):
            import pytz
            et = pytz.timezone("US/Eastern")
            try:
                et_idx = idx.tz_localize(et)
            except TypeError:
                et_idx = idx
        else:
            et_idx = idx

        hours = et_idx.hour
        minutes = et_idx.minute

        # Filter RTH (9:30-16:00)
        bar_minutes = hours * 60 + minutes
        rth_mask = (bar_minutes >= 570) & (bar_minutes < 960)  # 9:30=570, 16:00=960

        rth_data = data[rth_mask].copy()
        if rth_data.empty:
            return {}

        rth_et_idx = et_idx[rth_mask]
        time_slots = [f"{h:02d}:{m:02d}" for h, m in zip(rth_et_idx.hour, rth_et_idx.minute)]

        rth_data = rth_data.copy()
        rth_data["_time_slot"] = time_slots
        profile = rth_data.groupby("_time_slot")["volume"].mean().to_dict()
        return profile

    def get_relative_volume(self, timestamp, current_volume: float) -> float:
        """Return current_volume / avg_volume_at_this_time.

        If no historical data for this slot, returns 1.0 (neutral).
        """
        if not self._volume_profile or current_volume <= 0:
            return 1.0

        # Extract HH:MM from timestamp
        if hasattr(timestamp, "astimezone"):
            import pytz
            et = pytz.timezone("US/Eastern")
            try:
                ts_et = timestamp.astimezone(et)
            except (ValueError, TypeError):
                ts_et = timestamp
        elif hasattr(timestamp, "hour"):
            ts_et = timestamp
        else:
            return 1.0

        slot = f"{ts_et.hour:02d}:{ts_et.minute:02d}"
        avg_vol = self._volume_profile.get(slot, 0.0)
        if avg_vol <= 0:
            return 1.0
        return current_volume / avg_vol

    @property
    def profile(self) -> Dict[str, float]:
        """Return the volume profile dict."""
        return self._volume_profile
