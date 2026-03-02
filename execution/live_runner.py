"""
Live paper-trading runner for multi-strategy MES bot.

Streams 1-min OHLCV bars from Databento, feeds them to Strategy A (BB/VWAP
mean reversion) and Strategy B (VIX-adaptive ORB), tracks paper PnL internally,
and persists bars/trades to QuestDB and Supabase.

Usage:
    uv run python execution/live_runner.py --capital 20000
    uv run python execution/live_runner.py --dry-run --replay-days 1
"""

import argparse
import json
import logging
import signal
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pytz

logger = logging.getLogger(__name__)
ET = pytz.timezone("US/Eastern")
ROOT = Path(__file__).resolve().parent.parent

# Supabase strategy IDs (must match strategies table)
STRATEGY_A_ID = "bb_vwap_mr_v1"
STRATEGY_B_ID = "orb_vix_v1"

# L1 buffer settings
L1_BUFFER_SIZE = 500       # flush after N ticks
L1_FLUSH_INTERVAL = 30.0   # flush after N seconds

# ANSI color codes for console output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


class LiveRunner:
    """Live paper-trading runner for multi-strategy MES bot.

    Wires Databento live data -> strategies -> risk management -> persistence.
    Strategies manage positions internally (no broker orders).
    """

    def __init__(
        self,
        capital: float = 20_000.0,
        strategy_a_params_path: str = "config/strategy_a_params.json",
        strategy_b_params_path: str = "config/strategy_b_params.json",
        alloc_a: float = 0.50,
        alloc_b: float = 0.50,
        dry_run: bool = False,
        replay_days: int = 1,
        enable_l1: bool = False,
    ) -> None:
        self._capital = capital
        self._dry_run = dry_run
        self._replay_days = replay_days
        self._enable_l1 = enable_l1
        self._stop_event = threading.Event()

        # Capital allocation
        cap_a = capital * alloc_a
        cap_b = capital * alloc_b
        logger.info(
            "Capital: $%.0f | A=%.0f%% ($%.0f) B=%.0f%% ($%.0f)",
            capital, alloc_a * 100, cap_a, alloc_b * 100, cap_b,
        )

        # Load strategy params
        a_path = ROOT / strategy_a_params_path
        b_path = ROOT / strategy_b_params_path
        with open(a_path) as f:
            params_a = json.load(f)
        with open(b_path) as f:
            params_b = json.load(f)
        logger.info("Strategy A params: %s", json.dumps(params_a))
        logger.info("Strategy B params: %s", json.dumps(params_b))

        # Initialize strategies
        from strategies.mean_reversion import MESMeanReversionStrategy
        from strategies.orb_strategy import VIXAdaptiveORBStrategy

        self.strategy_a = MESMeanReversionStrategy(params=params_a, capital=cap_a)
        self.strategy_b = VIXAdaptiveORBStrategy(params=params_b, capital=cap_b)

        # Risk management
        from risk.risk_manager import RiskManager

        allocations = {"strategy_a": alloc_a, "strategy_b": alloc_b}
        self.risk_manager = RiskManager(
            total_capital=capital,
            strategy_allocations=allocations,
        )

        # QuestDB (optional — graceful degradation)
        self._questdb = None
        try:
            from data.questdb_client import QuestDBClient
            self._questdb = QuestDBClient()
            logger.info("QuestDB connected")
        except Exception as e:
            logger.warning("QuestDB unavailable: %s (bars won't be persisted)", e)

        # Supabase (optional — graceful degradation)
        self._supabase = None
        try:
            from data.supabase_client import SupabaseClient
            self._supabase = SupabaseClient()
            logger.info("Supabase connected")
            # Register strategies (idempotent upsert)
            self._supabase.upsert_strategy(
                strategy_id=STRATEGY_A_ID, name="BB/VWAP Mean Reversion",
                version="1.0", config=params_a, status="paper",
            )
            self._supabase.upsert_strategy(
                strategy_id=STRATEGY_B_ID, name="VIX-Adaptive ORB",
                version="1.0", config=params_b, status="paper",
            )
        except Exception as e:
            logger.warning("Supabase unavailable: %s (trades won't be persisted)", e)

        # Databento connector (only needed for live mode)
        self._connector = None
        if not dry_run:
            from config.settings import DATABENTO_API_KEY
            from data.databento_live import DatabentoLiveConnector
            self._connector = DatabentoLiveConnector(
                api_key=DATABENTO_API_KEY, enable_l1=enable_l1,
            )

        # Daily state tracking
        self._current_date: Optional[datetime] = None
        self._bar_count: int = 0
        self._daily_pnl_a: float = 0.0
        self._daily_pnl_b: float = 0.0
        self._daily_trades_a: int = 0
        self._daily_trades_b: int = 0

        # L1 tick buffer (flushed periodically to Supabase)
        self._l1_buffer: list = []
        self._l1_lock = threading.Lock()
        self._l1_flush_timer: Optional[threading.Timer] = None
        self._l1_total: int = 0

    # -- Main entry point ----------------------------------------------

    def run(self) -> None:
        """Main entry point. Blocks until shutdown."""
        print(f"\n{BOLD}{'='*60}{RESET}")
        print(f"{BOLD}  Multi-Strategy MES Live Runner{RESET}")
        print(f"{BOLD}{'='*60}{RESET}")
        print(f"  Capital: ${self._capital:,.0f}")
        print(f"  Mode:    {'DRY RUN (replay)' if self._dry_run else 'LIVE (Databento)'}")
        print(f"{'='*60}\n")

        # Fetch initial VIX
        vix = self._fetch_vix()
        self.strategy_a.set_vix(vix)
        self.strategy_b.set_vix(vix)
        print(f"  VIX: {vix:.2f}")

        if self._dry_run:
            self._replay_historical()
            return

        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        try:
            signal.signal(signal.SIGBREAK, self._signal_handler)
        except AttributeError:
            pass  # SIGBREAK not available on non-Windows

        # Start live stream
        self._connector.add_callback(self._on_bar)
        if self._enable_l1:
            self._connector.add_l1_callback(self._on_l1_tick)
            self._start_l1_flush_timer()
        self._connector.start()
        print(f"\n  {GREEN}Listening for MES 1-min bars from Databento...{RESET}")
        print(f"  Press Ctrl+C to stop\n")

        # Block main thread until shutdown signal
        self._stop_event.wait()
        self._shutdown()

    def _signal_handler(self, signum, frame) -> None:
        """Handle SIGINT/SIGBREAK for graceful shutdown."""
        logger.info("Shutdown signal received")
        self._stop_event.set()

    # -- Core bar handler ----------------------------------------------

    def _on_bar(
        self,
        timestamp: datetime,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: int,
    ) -> None:
        """Called on each 1-min bar from Databento (or replay)."""
        # Convert to ET for display and day detection
        if timestamp.tzinfo is None:
            et_ts = ET.localize(timestamp)
        else:
            et_ts = timestamp.astimezone(ET)

        self._bar_count += 1

        # Day boundary check
        self._check_new_day(et_ts)

        # Persist bar to QuestDB
        self._persist_bar(timestamp, open_, high, low, close, volume)

        # Circuit breaker: flatten all if HALTED
        if self.risk_manager.circuit_breaker.should_flatten_all():
            self._force_flatten_all(et_ts, close)
            return

        # -- Strategy A --
        had_pos_a = self.strategy_a.position is not None
        result_a = self.strategy_a.on_bar(timestamp, open_, high, low, close, volume)

        # Detect new entry
        if not had_pos_a and self.strategy_a.position is not None:
            pos = self.strategy_a.position
            side_str = pos.side.value if hasattr(pos.side, "value") else str(pos.side)
            self._log_entry("A", side_str, pos.entry_price, pos.contracts,
                            pos.stop_price, pos.target_price, et_ts)
            self.risk_manager.register_position(
                "strategy_a", pos.contracts, pos.entry_price,
                self.strategy_a.spec.point_value,
            )

        # Detect exit
        if result_a is not None:
            self._log_exit("A", result_a, et_ts)
            self.risk_manager.update_pnl("strategy_a", result_a.pnl)
            self.risk_manager.close_position("strategy_a")
            self._daily_pnl_a += result_a.pnl
            self._daily_trades_a += 1
            self._persist_trade(STRATEGY_A_ID, result_a)

        # -- Strategy B --
        had_pos_b = self.strategy_b.position is not None
        result_b = self.strategy_b.on_bar(timestamp, open_, high, low, close, volume)

        # Detect new entry
        if not had_pos_b and self.strategy_b.position is not None:
            pos = self.strategy_b.position
            side_str = pos.side if isinstance(pos.side, str) else str(pos.side)
            self._log_entry("B", side_str, pos.entry_price, pos.contracts,
                            pos.stop_price, pos.target_price, et_ts)
            self.risk_manager.register_position(
                "strategy_b", pos.contracts, pos.entry_price,
                self.strategy_b.spec.point_value,
            )

        # Detect exit
        if result_b is not None:
            self._log_exit("B", result_b, et_ts)
            self.risk_manager.update_pnl("strategy_b", result_b.pnl)
            self.risk_manager.close_position("strategy_b")
            self._daily_pnl_b += result_b.pnl
            self._daily_trades_b += 1
            self._persist_trade(STRATEGY_B_ID, result_b)

    # -- Day boundary handling -----------------------------------------

    def _check_new_day(self, et_ts: datetime) -> None:
        """Detect day boundary, reset daily state, refresh VIX."""
        current_date = et_ts.date()
        if self._current_date is not None and current_date != self._current_date:
            # Print daily summary for previous day
            self._daily_summary()

            # Reset daily state
            self._daily_pnl_a = 0.0
            self._daily_pnl_b = 0.0
            self._daily_trades_a = 0
            self._daily_trades_b = 0
            self._bar_count = 0

            # Risk manager day/week boundary
            if current_date.weekday() == 0:  # Monday
                self.risk_manager.on_new_week()
            else:
                self.risk_manager.on_new_day()

            # Refresh VIX
            vix = self._fetch_vix()
            self.strategy_a.set_vix(vix)
            self.strategy_b.set_vix(vix)
            logger.info("New day %s | VIX: %.2f", current_date, vix)

        self._current_date = current_date

    # -- VIX fetching --------------------------------------------------

    def _fetch_vix(self) -> float:
        """Try QuestDB first (latest VIX close), fall back to yfinance."""
        # QuestDB
        if self._questdb is not None:
            try:
                df = self._questdb.query(
                    "SELECT close FROM ohlcv "
                    "WHERE symbol = 'VIX' AND bar_size = '1day' "
                    "ORDER BY timestamp DESC LIMIT 1"
                )
                if not df.empty:
                    vix_val = float(df.iloc[0]["close"])
                    logger.info("VIX from QuestDB: %.2f", vix_val)
                    return vix_val
            except Exception as e:
                logger.warning("QuestDB VIX query failed: %s", e)

        # yfinance fallback
        try:
            import yfinance as yf
            ticker = yf.Ticker("^VIX")
            hist = ticker.history(period="5d")
            if not hist.empty:
                vix_val = float(hist["Close"].iloc[-1])
                logger.info("VIX from Yahoo Finance: %.2f", vix_val)
                return vix_val
        except Exception as e:
            logger.warning("Yahoo Finance VIX fetch failed: %s", e)

        logger.warning("Using default VIX = 20.0")
        return 20.0

    # -- Persistence ---------------------------------------------------

    def _persist_bar(self, timestamp, o, h, l, c, v) -> None:
        """Store bar in QuestDB and Supabase."""
        if self._questdb is not None:
            try:
                self._questdb.ingest_bar(
                    symbol="MES", bar_size="1min",
                    timestamp=timestamp,
                    open_=o, high=h, low=l, close=c, volume=v,
                )
            except Exception as e:
                logger.error("QuestDB bar ingest failed: %s", e)

        if self._supabase is not None:
            try:
                self._supabase.upsert_bar(
                    symbol="MES", timeframe="1m", open_time=timestamp,
                    open_=o, high=h, low=l, close=c, volume=v,
                )
            except Exception as e:
                logger.error("Supabase bar upsert failed: %s", e)

    def _persist_trade(self, strategy_name: str, trade_record) -> None:
        """Store trade in Supabase."""
        if self._supabase is None:
            return
        try:
            self._supabase.insert_trades_from_records(
                trade_records=[trade_record],
                strategy_id=strategy_name,
                symbol="MES",
            )
        except Exception as e:
            logger.error("Supabase trade insert failed: %s", e)

    # -- L1 tick handling ------------------------------------------------

    def _on_l1_tick(
        self, timestamp: datetime, bid: float, ask: float,
        bid_size: int, ask_size: int,
    ) -> None:
        """Buffer an L1 tick for batch persistence."""
        from data.supabase_client import _ts_to_iso

        record = {
            "symbol": "MES",
            "ts": _ts_to_iso(timestamp),
            "bid": bid,
            "ask": ask,
            "bid_size": bid_size,
            "ask_size": ask_size,
            "source": "databento",
        }
        with self._l1_lock:
            self._l1_buffer.append(record)
            if len(self._l1_buffer) >= L1_BUFFER_SIZE:
                self._flush_l1_buffer()

    def _flush_l1_buffer(self) -> None:
        """Flush buffered L1 ticks to Supabase. Caller must hold _l1_lock."""
        if not self._l1_buffer or self._supabase is None:
            return
        batch = self._l1_buffer.copy()
        self._l1_buffer.clear()

        # Release lock before network call
        try:
            count = self._supabase.insert_l1_batch(batch)
            self._l1_total += count
            logger.debug("Flushed %d L1 ticks (total: %d)", count, self._l1_total)
        except Exception as e:
            logger.error("Supabase L1 batch insert failed: %s", e)

    def _flush_l1_timer_callback(self) -> None:
        """Timer callback to flush L1 buffer periodically."""
        with self._l1_lock:
            self._flush_l1_buffer()
        self._start_l1_flush_timer()

    def _start_l1_flush_timer(self) -> None:
        """Start (or restart) the periodic L1 flush timer."""
        if self._l1_flush_timer is not None:
            self._l1_flush_timer.cancel()
        self._l1_flush_timer = threading.Timer(
            L1_FLUSH_INTERVAL, self._flush_l1_timer_callback,
        )
        self._l1_flush_timer.daemon = True
        self._l1_flush_timer.start()

    # -- Circuit breaker flatten ---------------------------------------

    def _force_flatten_all(self, et_ts: datetime, close: float) -> None:
        """Force-close all positions when circuit breaker hits HALTED."""
        print(f"\n  {RED}{BOLD}[CIRCUIT BREAKER] HALTED — flattening all positions{RESET}")

        if self.strategy_a.position is not None:
            result = self.strategy_a._close_position(close, et_ts, "circuit_breaker")
            self._log_exit("A", result, et_ts)
            self.risk_manager.update_pnl("strategy_a", result.pnl)
            self.risk_manager.close_position("strategy_a")
            self._daily_pnl_a += result.pnl
            self._persist_trade(STRATEGY_A_ID, result)

        if self.strategy_b.position is not None:
            result = self.strategy_b._close_position(close, et_ts, "circuit_breaker")
            self._log_exit("B", result, et_ts)
            self.risk_manager.update_pnl("strategy_b", result.pnl)
            self.risk_manager.close_position("strategy_b")
            self._daily_pnl_b += result.pnl
            self._persist_trade(STRATEGY_B_ID, result)

    # -- Console logging -----------------------------------------------

    def _log_entry(self, strat: str, side: str, price: float, contracts: int,
                   stop: float, target: float, et_ts: datetime) -> None:
        ts_str = et_ts.strftime("%H:%M:%S")
        color = GREEN if side == "LONG" else RED
        print(
            f"  {ts_str} {CYAN}[ENTRY]{RESET} Strat {strat} | "
            f"{color}{side}{RESET} {contracts} MES @ {price:.2f} | "
            f"Stop: {stop:.2f} | Target: {target:.2f}"
        )

    def _log_exit(self, strat: str, trade, et_ts: datetime) -> None:
        ts_str = et_ts.strftime("%H:%M:%S")
        pnl = trade.pnl
        color = GREEN if pnl >= 0 else RED
        side_str = trade.side if isinstance(trade.side, str) else trade.side.value
        print(
            f"  {ts_str} {YELLOW}[EXIT]{RESET}  Strat {strat} | "
            f"{side_str} {trade.contracts} MES @ {trade.exit_price:.2f} | "
            f"PnL: {color}${pnl:+,.2f}{RESET} | {trade.exit_reason}"
        )

    def _daily_summary(self) -> None:
        """Print end-of-day PnL summary."""
        total = self._daily_pnl_a + self._daily_pnl_b
        total_trades = self._daily_trades_a + self._daily_trades_b
        color = GREEN if total >= 0 else RED
        cb_state = self.risk_manager.circuit_breaker.state.value

        print(f"\n  {BOLD}{'-'*55}{RESET}")
        print(
            f"  {BOLD}[DAILY]{RESET} {self._current_date} | "
            f"PnL: {color}${total:+,.2f}{RESET} "
            f"(A: ${self._daily_pnl_a:+,.2f}, B: ${self._daily_pnl_b:+,.2f}) | "
            f"Trades: {total_trades} | CB: {cb_state} | Bars: {self._bar_count}"
        )
        print(
            f"  Equity: A=${self.strategy_a.capital:,.2f} "
            f"B=${self.strategy_b.capital:,.2f} "
            f"Total=${self.strategy_a.capital + self.strategy_b.capital:,.2f}"
        )
        print(f"  {BOLD}{'-'*55}{RESET}\n")

    # -- Dry-run replay ------------------------------------------------

    def _replay_historical(self) -> None:
        """Replay recent bars from QuestDB for pipeline testing."""
        import pandas as pd

        df = None

        # Try QuestDB first
        if self._questdb is not None:
            end = datetime.now(ET)
            start = end - timedelta(days=self._replay_days + 2)  # buffer for weekends
            df = self._questdb.get_bars("MES", "1min", start=start, end=end)
            if df.empty:
                print(f"  {YELLOW}No bars in QuestDB, falling back to Parquet...{RESET}")
                df = None

        # Fall back to Parquet
        if df is None:
            parquet_path = ROOT / "data" / "MES_1min.parquet"
            if not parquet_path.exists():
                print(f"  {RED}No data source available for replay{RESET}")
                return

            print(f"  Loading from Parquet...")
            df = pd.read_parquet(parquet_path)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            # RTH filter
            if df.index.tz is None:
                et_idx = df.index.tz_localize("UTC").tz_convert(ET)
            else:
                et_idx = df.index.tz_convert(ET)
            mask = (et_idx.hour >= 9) & (et_idx.hour < 16)
            df = df.loc[mask].copy()
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC").tz_convert(ET)

            # Trim to replay_days
            unique_dates = sorted(set(df.index.date))
            if len(unique_dates) > self._replay_days:
                cutoff = unique_dates[-self._replay_days]
                df = df[df.index.date >= cutoff]

        print(f"  Replaying {len(df):,} bars ({self._replay_days} day{'s' if self._replay_days > 1 else ''})...")
        print()

        for ts, row in df.iterrows():
            if not isinstance(ts, datetime):
                ts = pd.Timestamp(ts).to_pydatetime()
            self._on_bar(
                ts,
                float(row["open"]), float(row["high"]),
                float(row["low"]), float(row["close"]),
                int(row["volume"]),
            )

        # Final daily summary
        self._daily_summary()

        total_a = len(self.strategy_a.trades)
        total_b = len(self.strategy_b.trades)
        pnl_a = sum(t.pnl for t in self.strategy_a.trades)
        pnl_b = sum(t.pnl for t in self.strategy_b.trades)

        print(f"\n  {BOLD}Replay complete:{RESET}")
        print(f"    Strategy A: {total_a} trades, PnL ${pnl_a:+,.2f}")
        print(f"    Strategy B: {total_b} trades, PnL ${pnl_b:+,.2f}")
        print(f"    Combined:   {total_a + total_b} trades, PnL ${pnl_a + pnl_b:+,.2f}")

    # -- Shutdown ------------------------------------------------------

    def _shutdown(self) -> None:
        """Graceful cleanup."""
        print(f"\n  {YELLOW}Shutting down...{RESET}")

        if self._connector is not None:
            self._connector.stop()

        # Flush remaining L1 ticks
        if self._l1_flush_timer is not None:
            self._l1_flush_timer.cancel()
        with self._l1_lock:
            self._flush_l1_buffer()

        self._daily_summary()

        if self._questdb is not None:
            try:
                self._questdb.close_ilp()
            except Exception:
                pass

        total_a = len(self.strategy_a.trades)
        total_b = len(self.strategy_b.trades)
        print(f"  Session total: A={total_a} trades, B={total_b} trades")
        if self._l1_total > 0:
            print(f"  L1 ticks persisted: {self._l1_total:,}")
        print(f"  {GREEN}Shutdown complete{RESET}")


# -- CLI ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Live paper trading runner for MES")
    parser.add_argument("--capital", type=float, default=20_000.0,
                        help="Total capital (default: $20,000)")
    parser.add_argument("--alloc-a", type=float, default=0.50,
                        help="Capital allocation to Strategy A (default: 0.50)")
    parser.add_argument("--alloc-b", type=float, default=0.50,
                        help="Capital allocation to Strategy B (default: 0.50)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Replay recent data instead of live stream")
    parser.add_argument("--replay-days", type=int, default=1,
                        help="Days of history to replay in dry-run mode (default: 1)")
    parser.add_argument("--strategy-a-params", default="config/strategy_a_params.json")
    parser.add_argument("--strategy-b-params", default="config/strategy_b_params.json")
    parser.add_argument("--enable-l1", action="store_true",
                        help="Stream and persist L1 top-of-book quotes to Supabase")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    runner = LiveRunner(
        capital=args.capital,
        strategy_a_params_path=args.strategy_a_params,
        strategy_b_params_path=args.strategy_b_params,
        alloc_a=args.alloc_a,
        alloc_b=args.alloc_b,
        dry_run=args.dry_run,
        replay_days=args.replay_days,
        enable_l1=args.enable_l1,
    )
    runner.run()


if __name__ == "__main__":
    main()
