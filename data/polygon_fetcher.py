"""
Polygon.io REST client for fetching historical OHLCV bar data.

Supports paginated fetching of 2+ years of 1-min and 5-min bars
for futures symbols (MES / ES) with rate limiting.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from polygon import RESTClient
from polygon.rest.models import Agg

from config.settings import (
    POLYGON_API_KEY,
    POLYGON_RATE_LIMIT_CALLS,
    POLYGON_RATE_LIMIT_PERIOD,
)
from data.questdb_client import QuestDBClient

logger = logging.getLogger(__name__)


class PolygonFetcher:
    """Fetches historical OHLCV data from Polygon.io."""

    def __init__(
        self,
        api_key: str = POLYGON_API_KEY,
        rate_limit_calls: int = POLYGON_RATE_LIMIT_CALLS,
        rate_limit_period: float = POLYGON_RATE_LIMIT_PERIOD,
    ) -> None:
        if not api_key:
            raise ValueError("POLYGON_API_KEY is required")
        self.client = RESTClient(api_key=api_key)
        self.rate_limit_calls = rate_limit_calls
        self.rate_limit_period = rate_limit_period
        self._call_timestamps: list = []

    def _rate_limit(self) -> None:
        """Simple sliding-window rate limiter."""
        now = time.monotonic()
        # Remove timestamps outside the window
        self._call_timestamps = [
            t for t in self._call_timestamps
            if now - t < self.rate_limit_period
        ]
        if len(self._call_timestamps) >= self.rate_limit_calls:
            sleep_time = self.rate_limit_period - (now - self._call_timestamps[0])
            if sleep_time > 0:
                logger.info("Rate limit reached, sleeping %.1fs", sleep_time)
                time.sleep(sleep_time)
        self._call_timestamps.append(time.monotonic())

    def fetch_bars(
        self,
        symbol: str = "MES",
        multiplier: int = 1,
        timespan: str = "minute",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars from Polygon.io with automatic pagination.

        Args:
            symbol: Ticker symbol. For MES futures use the appropriate
                     Polygon ticker (e.g., 'I:MES1!' or the continuous contract).
            multiplier: Bar size multiplier (1 for 1-min, 5 for 5-min)
            timespan: 'minute', 'hour', 'day'
            start: Start datetime (default: 2 years ago)
            end: End datetime (default: now)

        Returns:
            DataFrame with columns [timestamp, open, high, low, close, volume, vwap]
        """
        if end is None:
            end = datetime.utcnow()
        if start is None:
            start = end - timedelta(days=730)

        all_bars: list = []
        current_start = start

        while current_start < end:
            # Polygon max results per request
            chunk_end = min(current_start + timedelta(days=30), end)
            self._rate_limit()

            try:
                aggs = self.client.get_aggs(
                    ticker=symbol,
                    multiplier=multiplier,
                    timespan=timespan,
                    from_=current_start.strftime("%Y-%m-%d"),
                    to=chunk_end.strftime("%Y-%m-%d"),
                    limit=50000,
                )

                if aggs:
                    for agg in aggs:
                        all_bars.append({
                            "timestamp": datetime.utcfromtimestamp(agg.timestamp / 1000),
                            "open": agg.open,
                            "high": agg.high,
                            "low": agg.low,
                            "close": agg.close,
                            "volume": agg.volume or 0,
                            "vwap": agg.vwap or 0.0,
                        })
                    logger.info(
                        "Fetched %d bars for %s (%s to %s)",
                        len(aggs),
                        symbol,
                        current_start.date(),
                        chunk_end.date(),
                    )
                else:
                    logger.debug("No bars returned for %s to %s", current_start.date(), chunk_end.date())

            except Exception as exc:
                logger.error(
                    "Error fetching %s bars: %s. Continuing...", symbol, exc
                )

            current_start = chunk_end + timedelta(seconds=1)

        if not all_bars:
            logger.warning("No bars fetched for %s", symbol)
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume", "vwap"]
            )

        df = pd.DataFrame(all_bars)
        df.drop_duplicates(subset=["timestamp"], inplace=True)
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        logger.info("Total bars fetched for %s: %d", symbol, len(df))
        return df

    def fetch_and_store(
        self,
        symbol: str = "MES",
        bar_sizes: Optional[list] = None,
        questdb_client: Optional[QuestDBClient] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> None:
        """Fetch bars and store directly to QuestDB.

        Args:
            symbol: Instrument symbol
            bar_sizes: List of (multiplier, timespan, label) tuples.
                       Default: [(1, 'minute', '1min'), (5, 'minute', '5min')]
            questdb_client: QuestDBClient instance (creates one if None)
            start: Start datetime
            end: End datetime
        """
        if bar_sizes is None:
            bar_sizes = [
                (1, "minute", "1min"),
                (5, "minute", "5min"),
            ]
        if questdb_client is None:
            questdb_client = QuestDBClient()

        for multiplier, timespan, label in bar_sizes:
            logger.info("Fetching %s %s bars...", symbol, label)
            df = self.fetch_bars(
                symbol=symbol,
                multiplier=multiplier,
                timespan=timespan,
                start=start,
                end=end,
            )
            if df.empty:
                logger.warning("No %s bars to store for %s", label, symbol)
                continue

            questdb_client.ingest_dataframe(df, symbol=symbol, bar_size=label)
            logger.info("Stored %d %s bars for %s to QuestDB", len(df), label, symbol)

        questdb_client.close_ilp()
