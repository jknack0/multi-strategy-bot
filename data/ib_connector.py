"""
Interactive Brokers Gateway connector using ib_insync.

Handles connection management, bar subscriptions, order placement,
and position/account queries with automatic reconnection.
"""

import logging
import time
from typing import Callable, Dict, List, Optional

from ib_insync import IB, Contract, ContFuture, Future, MarketOrder, LimitOrder, util
from ib_insync.objects import BarData, Position, AccountValue

from config.settings import (
    IB_HOST,
    IB_PORT,
    IB_CLIENT_ID,
    IB_TIMEOUT,
    IB_READONLY,
)
from config.constants import INSTRUMENTS, InstrumentSpec

logger = logging.getLogger(__name__)


class IBConnector:
    """Manages connection to IB Gateway and provides trading methods."""

    MAX_RETRIES: int = 5
    BASE_BACKOFF: float = 2.0  # seconds

    def __init__(
        self,
        host: str = IB_HOST,
        port: int = IB_PORT,
        client_id: int = IB_CLIENT_ID,
        timeout: int = IB_TIMEOUT,
        readonly: bool = IB_READONLY,
    ) -> None:
        self.host = host
        self.port = port
        self.client_id = client_id
        self.timeout = timeout
        self.readonly = readonly
        self.ib = IB()
        self._bar_callbacks: Dict[str, List[Callable]] = {}

    # ── Connection Management ────────────────────────────────────────────

    def connect(self) -> None:
        """Connect to IB Gateway with exponential backoff retry."""
        for attempt in range(self.MAX_RETRIES):
            try:
                self.ib.connect(
                    host=self.host,
                    port=self.port,
                    clientId=self.client_id,
                    timeout=self.timeout,
                    readonly=self.readonly,
                )
                logger.info(
                    "Connected to IB Gateway at %s:%d (client %d)",
                    self.host,
                    self.port,
                    self.client_id,
                )
                self.ib.disconnectedEvent += self._on_disconnect
                return
            except Exception as exc:
                wait = self.BASE_BACKOFF ** (attempt + 1)
                logger.warning(
                    "Connection attempt %d/%d failed: %s. Retrying in %.1fs",
                    attempt + 1,
                    self.MAX_RETRIES,
                    exc,
                    wait,
                )
                time.sleep(wait)
        raise ConnectionError(
            f"Failed to connect to IB Gateway after {self.MAX_RETRIES} attempts"
        )

    def _on_disconnect(self) -> None:
        """Handle unexpected disconnections by reconnecting."""
        logger.warning("Disconnected from IB Gateway, attempting reconnect...")
        try:
            self.connect()
        except ConnectionError:
            logger.error("Reconnection failed")

    def disconnect(self) -> None:
        """Gracefully disconnect from IB Gateway."""
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from IB Gateway")

    @property
    def is_connected(self) -> bool:
        return self.ib.isConnected()

    # ── Contract Helpers ─────────────────────────────────────────────────

    @staticmethod
    def make_futures_contract(
        symbol: str = "MES",
        exchange: str = "CME",
        currency: str = "USD",
    ) -> Future:
        """Create an IB Future contract for the given symbol."""
        spec = INSTRUMENTS.get(symbol)
        if spec:
            exchange = spec.exchange
            currency = spec.currency
        return Future(
            symbol=symbol,
            exchange=exchange,
            currency=currency,
        )

    @staticmethod
    def make_continuous_futures_contract(
        symbol: str = "MES",
        exchange: str = "CME",
        currency: str = "USD",
    ) -> ContFuture:
        """Create a continuous futures contract for historical data."""
        spec = INSTRUMENTS.get(symbol)
        if spec:
            exchange = spec.exchange
            currency = spec.currency
        return ContFuture(
            symbol=symbol,
            exchange=exchange,
            currency=currency,
        )

    def qualify_contract(self, contract: Contract) -> Contract:
        """Qualify a contract so IB fills in the conId and other details."""
        qualified = self.ib.qualifyContracts(contract)
        if qualified:
            return qualified[0]
        raise ValueError(f"Could not qualify contract: {contract}")

    # ── Market Data ──────────────────────────────────────────────────────

    def subscribe_bars(
        self,
        symbol: str,
        bar_size: str = "5 secs",
        what_to_show: str = "TRADES",
        use_rth: bool = False,
        callback: Optional[Callable[[BarData], None]] = None,
    ) -> None:
        """Subscribe to real-time bars for the given symbol."""
        contract = self.make_futures_contract(symbol)
        contract = self.qualify_contract(contract)

        bars = self.ib.reqRealTimeBars(
            contract,
            barSize=5,
            whatToShow=what_to_show,
            useRTH=use_rth,
        )

        if callback:
            key = f"{symbol}_{bar_size}"
            self._bar_callbacks.setdefault(key, []).append(callback)
            bars.updateEvent += callback

        logger.info("Subscribed to %s %s bars", symbol, bar_size)

    def get_historical_bars(
        self,
        symbol: str,
        duration: str = "1 D",
        bar_size: str = "1 min",
        what_to_show: str = "TRADES",
        use_rth: bool = False,
        end_datetime: str = "",
        contract: Optional[Contract] = None,
    ) -> List[BarData]:
        """Fetch historical bars from IB.

        Args:
            symbol: Instrument symbol (e.g., 'MES')
            duration: Duration string (e.g., '1 D', '1 W')
            bar_size: Bar size setting (e.g., '1 min', '5 mins')
            what_to_show: Data type ('TRADES', 'MIDPOINT', etc.)
            use_rth: Regular trading hours only
            end_datetime: End date/time as 'YYYYMMDD HH:MM:SS' or '' for now
            contract: Pre-qualified Contract to use (skips contract creation)
        """
        if contract is None:
            contract = self.make_futures_contract(symbol)
            contract = self.qualify_contract(contract)
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime=end_datetime,
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=use_rth,
            formatDate=1,
        )
        logger.info("Fetched %d historical bars for %s", len(bars), symbol)
        return bars

    # ── Order Management ─────────────────────────────────────────────────

    def place_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        order_type: str = "MKT",
        limit_price: Optional[float] = None,
    ):
        """Place an order on IB.

        Args:
            symbol: Instrument symbol (e.g., 'MES')
            action: 'BUY' or 'SELL'
            quantity: Number of contracts
            order_type: 'MKT' or 'LMT'
            limit_price: Required if order_type is 'LMT'

        Returns:
            ib_insync Trade object
        """
        contract = self.make_futures_contract(symbol)
        contract = self.qualify_contract(contract)

        if order_type == "LMT":
            if limit_price is None:
                raise ValueError("limit_price required for LMT orders")
            order = LimitOrder(action, quantity, limit_price)
        else:
            order = MarketOrder(action, quantity)

        trade = self.ib.placeOrder(contract, order)
        logger.info(
            "Placed %s %s %d %s @ %s",
            order_type,
            action,
            quantity,
            symbol,
            limit_price or "MKT",
        )
        return trade

    # ── Account & Positions ──────────────────────────────────────────────

    def get_positions(self) -> List[Position]:
        """Return current positions."""
        return self.ib.positions()

    def get_account_summary(self) -> List[AccountValue]:
        """Return account summary values."""
        return self.ib.accountSummary()

    def get_net_liquidation(self) -> float:
        """Return net liquidation value of the account."""
        summary = self.get_account_summary()
        for item in summary:
            if item.tag == "NetLiquidation" and item.currency == "USD":
                return float(item.value)
        return 0.0

    # ── Event Loop ───────────────────────────────────────────────────────

    def sleep(self, seconds: float = 0.1) -> None:
        """Run the IB event loop for the specified duration."""
        self.ib.sleep(seconds)

    def run(self) -> None:
        """Run the IB event loop indefinitely."""
        self.ib.run()
