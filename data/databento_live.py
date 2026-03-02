"""
Databento live market data connector.

Subscribes to CME MES data and dispatches to registered callbacks:
  - 1-min OHLCV bars (ohlcv-1m): (timestamp, open, high, low, close, volume)
  - L1 top-of-book quotes (mbp-1): (timestamp, bid, ask, bid_size, ask_size)
"""

import logging
from datetime import datetime, timezone
from typing import Callable, List, Optional

import databento as db

logger = logging.getLogger(__name__)

# Databento stores prices as int64 fixed-point with 1e-9 scale.
# The Python SDK may expose floats directly — we check at runtime.
FIXED_PRICE_SCALE = 1_000_000_000


def _convert_price(raw) -> float:
    """Convert a Databento price field (int fixed-point or float) to float."""
    if isinstance(raw, int):
        return raw / FIXED_PRICE_SCALE
    return float(raw)


class DatabentoLiveConnector:
    """Connects to Databento live API and dispatches OHLCV bars and L1 quotes.

    Usage:
        connector = DatabentoLiveConnector(api_key="db-...")
        connector.add_callback(on_bar)      # fn(timestamp, o, h, l, c, volume)
        connector.add_l1_callback(on_quote)  # fn(timestamp, bid, ask, bid_sz, ask_sz)
        connector.start()
        # ... runs on background thread ...
        connector.stop()
    """

    def __init__(
        self,
        api_key: str,
        dataset: str = "GLBX.MDP3",
        symbols: Optional[List[str]] = None,
        stype_in: str = "parent",
        enable_l1: bool = False,
    ) -> None:
        if not api_key:
            raise ValueError("DATABENTO_API_KEY is required")
        self._api_key = api_key
        self._dataset = dataset
        self._symbols = symbols or ["MES.FUT"]
        self._stype_in = stype_in
        self._enable_l1 = enable_l1
        self._callbacks: List[Callable] = []
        self._l1_callbacks: List[Callable] = []
        self._client: Optional[db.Live] = None

    def add_callback(self, callback: Callable) -> None:
        """Register a bar callback: fn(timestamp, open, high, low, close, volume)."""
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def add_l1_callback(self, callback: Callable) -> None:
        """Register an L1 callback: fn(timestamp, bid, ask, bid_size, ask_size)."""
        if callback not in self._l1_callbacks:
            self._l1_callbacks.append(callback)

    def start(self) -> None:
        """Subscribe and begin streaming. Non-blocking (runs on background thread)."""
        self._client = db.Live(key=self._api_key)

        # OHLCV 1-min bars
        logger.info(
            "Subscribing: dataset=%s schema=ohlcv-1m symbols=%s",
            self._dataset, self._symbols,
        )
        self._client.subscribe(
            dataset=self._dataset,
            schema="ohlcv-1m",
            stype_in=self._stype_in,
            symbols=self._symbols,
        )

        # L1 top-of-book (optional)
        if self._enable_l1:
            logger.info(
                "Subscribing: dataset=%s schema=mbp-1 symbols=%s",
                self._dataset, self._symbols,
            )
            self._client.subscribe(
                dataset=self._dataset,
                schema="mbp-1",
                stype_in=self._stype_in,
                symbols=self._symbols,
            )

        self._client.add_callback(self._on_record)
        self._client.start()
        logger.info("Databento live stream started")

    def stop(self) -> None:
        """Gracefully close the live connection."""
        if self._client is not None:
            try:
                self._client.stop()
            except Exception as e:
                logger.warning("Error stopping Databento client: %s", e)
            self._client = None
            logger.info("Databento live stream stopped")

    def _on_record(self, record: db.DBNRecord) -> None:
        """Dispatch incoming Databento records to the appropriate handler."""
        if isinstance(record, db.OHLCVMsg):
            self._handle_ohlcv(record)
        elif isinstance(record, db.MBP1Msg):
            self._handle_l1(record)
        elif isinstance(record, db.ErrorMsg):
            logger.error("Databento error: %s", record.err)
        elif isinstance(record, db.SymbolMappingMsg):
            logger.info(
                "Symbol mapping: %s -> instrument_id=%d",
                record.stype_in_symbol, record.instrument_id,
            )

    def _handle_ohlcv(self, ohlcv: db.OHLCVMsg) -> None:
        """Convert OHLCVMsg to standard bar format and dispatch to callbacks."""
        ts = datetime.fromtimestamp(ohlcv.ts_event / 1e9, tz=timezone.utc)

        o = _convert_price(ohlcv.open)
        h = _convert_price(ohlcv.high)
        l = _convert_price(ohlcv.low)
        c = _convert_price(ohlcv.close)
        v = int(ohlcv.volume)

        for cb in self._callbacks:
            try:
                cb(ts, o, h, l, c, v)
            except Exception as e:
                logger.error("OHLCV callback error: %s", e, exc_info=True)

    def _handle_l1(self, msg: db.MBP1Msg) -> None:
        """Convert MBP1Msg to L1 quote and dispatch to callbacks."""
        if not self._l1_callbacks:
            return

        ts = datetime.fromtimestamp(msg.ts_event / 1e9, tz=timezone.utc)

        # MBP1Msg has a single level in msg.levels[0]
        level = msg.levels[0]
        bid = _convert_price(level.bid_px)
        ask = _convert_price(level.ask_px)
        bid_sz = int(level.bid_sz)
        ask_sz = int(level.ask_sz)

        for cb in self._l1_callbacks:
            try:
                cb(ts, bid, ask, bid_sz, ask_sz)
            except Exception as e:
                logger.error("L1 callback error: %s", e, exc_info=True)
