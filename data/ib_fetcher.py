"""
IB historical data fetcher for MES/ES futures.

Paginated fetching of up to 1 year of 1-min and 5-min bars
from Interactive Brokers with IB pacing compliance.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from data.ib_connector import IBConnector
from data.questdb_client import QuestDBClient

logger = logging.getLogger(__name__)

# IB max duration per request by bar size
_CHUNK_CONFIG = {
    "1 min": {"duration": "1 D", "step": timedelta(days=1)},
    "5 mins": {"duration": "1 W", "step": timedelta(weeks=1)},
}


class IBFetcher:
    """Fetches historical OHLCV data from Interactive Brokers."""

    def __init__(
        self,
        connector: Optional[IBConnector] = None,
        pacing_delay: float = 10.0,
    ) -> None:
        """
        Args:
            connector: IBConnector instance (creates and connects one if None)
            pacing_delay: Seconds to sleep between requests (IB pacing rule)
        """
        self._owns_connector = connector is None
        self.connector = connector or IBConnector()
        self.pacing_delay = pacing_delay

    def _ensure_connected(self) -> None:
        if not self.connector.is_connected:
            self.connector.connect()

    def fetch_bars(
        self,
        symbol: str = "MES",
        bar_size: str = "1 min",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        what_to_show: str = "TRADES",
        use_rth: bool = False,
    ) -> pd.DataFrame:
        """Fetch historical bars from IB with automatic pagination.

        Args:
            symbol: Futures symbol (e.g., 'MES', 'ES')
            bar_size: '1 min' or '5 mins'
            start: Start datetime (default: 365 days ago)
            end: End datetime (default: now)
            what_to_show: IB data type
            use_rth: Regular trading hours only

        Returns:
            DataFrame with columns [timestamp, open, high, low, close, volume, vwap]
        """
        if end is None:
            end = datetime.utcnow()
        if start is None:
            start = end - timedelta(days=365)

        config = _CHUNK_CONFIG.get(bar_size)
        if config is None:
            raise ValueError(
                f"Unsupported bar_size '{bar_size}'. Use one of: {list(_CHUNK_CONFIG)}"
            )

        self._ensure_connected()

        # Resolve front-month via ContFuture, then build a specific Future
        # contract from it (IB doesn't allow endDateTime on ContFuture)
        from ib_insync import Future
        cont = self.connector.make_continuous_futures_contract(symbol)
        cont = self.connector.qualify_contract(cont)
        contract = Future(
            conId=cont.conId,
            symbol=cont.symbol,
            lastTradeDateOrContractMonth=cont.lastTradeDateOrContractMonth,
            multiplier=cont.multiplier,
            exchange=cont.exchange,
            currency=cont.currency,
            localSymbol=cont.localSymbol,
            tradingClass=cont.tradingClass,
        )
        contract = self.connector.qualify_contract(contract)
        logger.info("Resolved contract: %s", contract.localSymbol)

        all_bars: list = []
        current_end = end
        chunk_num = 0

        while current_end > start:
            chunk_num += 1
            end_str = current_end.strftime("%Y%m%d-%H:%M:%S")

            try:
                bars = self.connector.get_historical_bars(
                    symbol=symbol,
                    duration=config["duration"],
                    bar_size=bar_size,
                    what_to_show=what_to_show,
                    use_rth=use_rth,
                    end_datetime=end_str,
                    contract=contract,
                )

                if bars:
                    for bar in bars:
                        ts = bar.date
                        if isinstance(ts, str):
                            ts = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                        if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
                            ts = ts.replace(tzinfo=None)
                        if ts < start:
                            continue
                        all_bars.append({
                            "timestamp": ts,
                            "open": bar.open,
                            "high": bar.high,
                            "low": bar.low,
                            "close": bar.close,
                            "volume": int(bar.volume),
                            "vwap": getattr(bar, "average", 0.0) or 0.0,
                        })
                    logger.info(
                        "Chunk %d: fetched %d bars for %s %s (ending %s)",
                        chunk_num, len(bars), symbol, bar_size, end_str,
                    )
                else:
                    logger.debug(
                        "Chunk %d: no bars returned for %s ending %s",
                        chunk_num, symbol, end_str,
                    )

            except Exception as exc:
                logger.error(
                    "Chunk %d: error fetching %s bars: %s. Continuing...",
                    chunk_num, symbol, exc,
                )

            current_end -= config["step"]

            # IB pacing: sleep between requests
            if current_end > start:
                logger.debug("Pacing delay: %.1fs", self.pacing_delay)
                time.sleep(self.pacing_delay)

        if not all_bars:
            logger.warning("No bars fetched for %s %s", symbol, bar_size)
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume", "vwap"]
            )

        df = pd.DataFrame(all_bars)
        df.drop_duplicates(subset=["timestamp"], inplace=True)
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        logger.info("Total bars fetched for %s %s: %d", symbol, bar_size, len(df))
        return df

    def fetch_and_store(
        self,
        symbol: str = "MES",
        bar_sizes: Optional[list] = None,
        questdb_client: Optional[QuestDBClient] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> None:
        """Fetch bars from IB and store to QuestDB.

        Args:
            symbol: Instrument symbol
            bar_sizes: List of (ib_bar_size, label) tuples.
                       Default: [('1 min', '1min'), ('5 mins', '5min')]
            questdb_client: QuestDBClient instance (creates one if None)
            start: Start datetime (default: 365 days ago)
            end: End datetime (default: now)
        """
        if bar_sizes is None:
            bar_sizes = [
                ("1 min", "1min"),
                ("5 mins", "5min"),
            ]
        if questdb_client is None:
            questdb_client = QuestDBClient()

        questdb_client.create_table()

        for ib_bar_size, label in bar_sizes:
            # Resume from last stored timestamp if no explicit start
            fetch_start = start
            if fetch_start is None:
                latest = questdb_client.get_latest_timestamp(symbol, label)
                if latest is not None:
                    fetch_start = latest + timedelta(seconds=1)
                    logger.info(
                        "Resuming %s %s from %s", symbol, label, fetch_start,
                    )

            logger.info("Fetching %s %s bars from IB...", symbol, label)
            df = self.fetch_bars(
                symbol=symbol,
                bar_size=ib_bar_size,
                start=fetch_start,
                end=end,
            )
            if df.empty:
                logger.warning("No %s bars to store for %s", label, symbol)
                continue

            questdb_client.ingest_dataframe(df, symbol=symbol, bar_size=label)
            logger.info(
                "Stored %d %s bars for %s to QuestDB", len(df), label, symbol,
            )

        questdb_client.close_ilp()

        if self._owns_connector:
            self.connector.disconnect()
