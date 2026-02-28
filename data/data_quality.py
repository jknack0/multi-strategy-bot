"""
Data quality checks for OHLCV bar data.

Detects gaps, outliers (>5 sigma), zero-volume bars, and generates
daily quality reports.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class QualityIssue:
    """A single data quality issue."""
    timestamp: datetime
    issue_type: str  # 'gap', 'outlier', 'zero_volume', 'ohlc_invalid'
    description: str
    severity: str = "warning"  # 'info', 'warning', 'critical'


@dataclass
class DailyQualityReport:
    """Summary quality report for a day of data."""
    date: datetime
    symbol: str
    bar_size: str
    total_bars: int
    expected_bars: int
    gap_count: int
    outlier_count: int
    zero_volume_count: int
    ohlc_invalid_count: int
    issues: List[QualityIssue] = field(default_factory=list)

    @property
    def completeness_pct(self) -> float:
        if self.expected_bars == 0:
            return 0.0
        return (self.total_bars / self.expected_bars) * 100.0

    @property
    def is_clean(self) -> bool:
        return (
            self.gap_count == 0
            and self.outlier_count == 0
            and self.ohlc_invalid_count == 0
        )


class DataQualityChecker:
    """Performs data quality checks on OHLCV data."""

    def __init__(
        self,
        sigma_threshold: float = 5.0,
        expected_bars_per_day: Optional[Dict[str, int]] = None,
    ) -> None:
        self.sigma_threshold = sigma_threshold
        self.expected_bars_per_day = expected_bars_per_day or {
            "1min": 1380,   # ~23 hours of futures trading
            "5min": 276,    # ~23 hours / 5
        }

    def check_gaps(
        self,
        df: pd.DataFrame,
        bar_size: str = "1min",
    ) -> List[QualityIssue]:
        """Detect time gaps in bar data.

        A gap is defined as a missing bar where the time delta
        exceeds the expected bar interval by more than 2x.
        """
        issues: List[QualityIssue] = []
        if df.empty or len(df) < 2:
            return issues

        interval_map = {"1min": timedelta(minutes=1), "5min": timedelta(minutes=5)}
        expected_delta = interval_map.get(bar_size, timedelta(minutes=1))
        max_delta = expected_delta * 2

        timestamps = pd.to_datetime(df.index if isinstance(df.index, pd.DatetimeIndex) else df["timestamp"])
        deltas = timestamps.diff().dropna()

        for i, delta in enumerate(deltas):
            if delta > max_delta:
                ts = timestamps.iloc[i]
                issues.append(
                    QualityIssue(
                        timestamp=ts.to_pydatetime() if hasattr(ts, 'to_pydatetime') else ts,
                        issue_type="gap",
                        description=f"Gap of {delta} detected (expected ~{expected_delta})",
                        severity="warning",
                    )
                )

        return issues

    def check_outliers(
        self,
        df: pd.DataFrame,
        column: str = "close",
    ) -> List[QualityIssue]:
        """Detect price outliers using z-score > sigma_threshold.

        Computes returns and flags any bar where the absolute return
        exceeds sigma_threshold standard deviations.
        """
        issues: List[QualityIssue] = []
        if df.empty or len(df) < 20:
            return issues

        prices = df[column].astype(float)
        returns = prices.pct_change().dropna()

        if returns.std() == 0:
            return issues

        z_scores = (returns - returns.mean()) / returns.std()
        outlier_mask = z_scores.abs() > self.sigma_threshold

        for idx in z_scores[outlier_mask].index:
            ts = idx if isinstance(idx, datetime) else df.index[0]
            issues.append(
                QualityIssue(
                    timestamp=ts.to_pydatetime() if hasattr(ts, 'to_pydatetime') else ts,
                    issue_type="outlier",
                    description=f"Price move of {z_scores[idx]:.1f} sigma in {column}",
                    severity="critical",
                )
            )

        return issues

    def check_zero_volume(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Detect bars with zero volume."""
        issues: List[QualityIssue] = []
        if df.empty or "volume" not in df.columns:
            return issues

        zero_mask = df["volume"] == 0
        timestamps = df.index if isinstance(df.index, pd.DatetimeIndex) else df.get("timestamp", pd.Series())

        for idx in df[zero_mask].index:
            ts = idx if isinstance(idx, datetime) else datetime.now()
            issues.append(
                QualityIssue(
                    timestamp=ts.to_pydatetime() if hasattr(ts, 'to_pydatetime') else ts,
                    issue_type="zero_volume",
                    description="Zero volume bar",
                    severity="info",
                )
            )

        return issues

    def check_ohlc_validity(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Check OHLC logical consistency: high >= max(open,close), low <= min(open,close)."""
        issues: List[QualityIssue] = []
        if df.empty:
            return issues

        invalid = (
            (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["low"] > df["open"])
            | (df["low"] > df["close"])
            | (df["high"] < df["low"])
        )

        for idx in df[invalid].index:
            ts = idx if isinstance(idx, datetime) else datetime.now()
            row = df.loc[idx]
            issues.append(
                QualityIssue(
                    timestamp=ts.to_pydatetime() if hasattr(ts, 'to_pydatetime') else ts,
                    issue_type="ohlc_invalid",
                    description=(
                        f"Invalid OHLC: O={row['open']}, H={row['high']}, "
                        f"L={row['low']}, C={row['close']}"
                    ),
                    severity="critical",
                )
            )

        return issues

    def generate_daily_report(
        self,
        df: pd.DataFrame,
        symbol: str,
        bar_size: str = "1min",
        date: Optional[datetime] = None,
    ) -> DailyQualityReport:
        """Generate a comprehensive quality report for one day of data."""
        if date is None:
            date = datetime.now()

        gaps = self.check_gaps(df, bar_size)
        outliers = self.check_outliers(df)
        zero_vol = self.check_zero_volume(df)
        ohlc_issues = self.check_ohlc_validity(df)

        all_issues = gaps + outliers + zero_vol + ohlc_issues
        expected = self.expected_bars_per_day.get(bar_size, 0)

        report = DailyQualityReport(
            date=date,
            symbol=symbol,
            bar_size=bar_size,
            total_bars=len(df),
            expected_bars=expected,
            gap_count=len(gaps),
            outlier_count=len(outliers),
            zero_volume_count=len(zero_vol),
            ohlc_invalid_count=len(ohlc_issues),
            issues=all_issues,
        )

        logger.info(
            "Quality report for %s %s on %s: %d bars (%.1f%% complete), "
            "%d gaps, %d outliers, %d zero-vol, %d invalid OHLC",
            symbol,
            bar_size,
            date.date(),
            report.total_bars,
            report.completeness_pct,
            report.gap_count,
            report.outlier_count,
            report.zero_volume_count,
            report.ohlc_invalid_count,
        )

        return report
