"""
Backfill MES 1-min historical bars from Interactive Brokers into QuestDB.

Stores each chunk immediately to QuestDB (no data loss on crash/interrupt).
IB provides up to 1 year of 1-min data. QuestDB dedup handles re-runs safely.

Prerequisites:
    - IB Gateway or TWS running on localhost:4002 (paper) or 4001 (live)
    - Market data subscription for MES (CME)
    - QuestDB running on localhost:9000

Usage:
    uv run python scripts/backfill_mes.py
    uv run python scripts/backfill_mes.py --days 180
    uv run python scripts/backfill_mes.py --start 2025-06-01 --end 2026-02-28
    uv run python scripts/backfill_mes.py --dry-run
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.ib_connector import IBConnector
from data.questdb_client import QuestDBClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# IB pacing: 1-min bars max 1 day per request
_CHUNK_CONFIG = {
    "1 min": {"duration": "1 D", "step": timedelta(days=1)},
    "5 mins": {"duration": "1 W", "step": timedelta(weeks=1)},
}


def backfill_incremental(
    connector: IBConnector,
    qdb: QuestDBClient,
    symbol: str,
    ib_bar_size: str,
    label: str,
    start: datetime,
    end: datetime,
    pacing_delay: float,
):
    """Fetch bars from IB chunk-by-chunk, storing each chunk immediately."""
    from ib_insync import Future

    config = _CHUNK_CONFIG[ib_bar_size]

    # Resolve front-month contract
    cont = connector.make_continuous_futures_contract(symbol)
    cont = connector.qualify_contract(cont)
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
    contract = connector.qualify_contract(contract)
    logger.info("Resolved contract: %s", contract.localSymbol)

    qdb.create_table()

    total_bars = 0
    total_chunks = 0
    current_end = end
    t0 = time.time()

    while current_end > start:
        total_chunks += 1
        end_str = current_end.strftime("%Y%m%d-%H:%M:%S")

        try:
            bars = connector.get_historical_bars(
                symbol=symbol,
                duration=config["duration"],
                bar_size=ib_bar_size,
                what_to_show="TRADES",
                use_rth=False,
                end_datetime=end_str,
                contract=contract,
            )

            if bars:
                # Build ILP lines and store immediately
                lines = []
                chunk_count = 0
                for bar in bars:
                    ts = bar.date
                    if isinstance(ts, str):
                        ts = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                    if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
                        ts = ts.replace(tzinfo=None)
                    if ts < start:
                        continue
                    lines.append(
                        qdb.build_ilp_line(
                            symbol=symbol,
                            bar_size=label,
                            timestamp=ts,
                            open_=float(bar.open),
                            high=float(bar.high),
                            low=float(bar.low),
                            close=float(bar.close),
                            volume=int(bar.volume),
                            vwap=float(getattr(bar, "average", 0.0) or 0.0),
                        )
                    )
                    chunk_count += 1

                # Flush to QuestDB immediately
                if lines:
                    qdb.ingest_bars_batch(lines)
                    total_bars += chunk_count

                elapsed = time.time() - t0
                logger.info(
                    "Chunk %d: +%d bars (total: %d, %.1f min elapsed) ending %s",
                    total_chunks, chunk_count, total_bars, elapsed / 60, end_str[:8],
                )
            else:
                logger.debug("Chunk %d: no bars ending %s", total_chunks, end_str[:8])

        except Exception as exc:
            logger.error("Chunk %d: error: %s. Continuing...", total_chunks, exc)

        current_end -= config["step"]

        if current_end > start:
            time.sleep(pacing_delay)

    qdb.close_ilp()
    elapsed = time.time() - t0
    logger.info(
        "Backfill complete: %d bars in %d chunks (%.1f min)",
        total_bars, total_chunks, elapsed / 60,
    )
    return total_bars


def main():
    parser = argparse.ArgumentParser(description="Backfill MES 1-min bars from IBKR")
    parser.add_argument(
        "--days", type=int, default=365,
        help="Number of days to look back (default: 365, max IB allows)",
    )
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD (overrides --days)")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD (default: now)")
    parser.add_argument(
        "--bar-sizes", nargs="+", default=["1min"],
        choices=["1min", "5min", "both"],
        help="Bar sizes to fetch (default: 1min)",
    )
    parser.add_argument(
        "--pacing", type=float, default=10.0,
        help="Seconds between IB requests (default: 10, IB pacing rule)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be fetched without actually fetching",
    )
    args = parser.parse_args()

    # Parse dates
    end = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.now(timezone.utc).replace(tzinfo=None)
    if args.start:
        start = datetime.strptime(args.start, "%Y-%m-%d")
    else:
        start = end - timedelta(days=args.days)

    # Determine bar sizes
    if "both" in args.bar_sizes:
        bar_sizes = [("1 min", "1min"), ("5 mins", "5min")]
    else:
        mapping = {"1min": ("1 min", "1min"), "5min": ("5 mins", "5min")}
        bar_sizes = [mapping[b] for b in args.bar_sizes]

    # Check existing data
    qdb = QuestDBClient()
    for _, label in bar_sizes:
        count = qdb.count_bars("MES", label)
        latest = qdb.get_latest_timestamp("MES", label)
        logger.info("Existing MES %s data: %d bars, latest: %s", label, count, latest)

    n_days = (end - start).days
    logger.info("Backfill range: %s to %s (%d days)", start.date(), end.date(), n_days)
    logger.info("Bar sizes: %s", [label for _, label in bar_sizes])
    logger.info("Pacing delay: %.1fs between requests", args.pacing)

    est_minutes = n_days * args.pacing / 60
    logger.info("Estimated time: ~%.0f minutes for 1-min bars", est_minutes)

    if args.dry_run:
        logger.info("DRY RUN: would fetch %d days of data. Exiting.", n_days)
        return

    connector = IBConnector()
    try:
        connector.connect()
        logger.info("Connected to IB Gateway")

        for ib_bar_size, label in bar_sizes:
            logger.info("Starting backfill for MES %s...", label)
            backfill_incremental(
                connector=connector,
                qdb=qdb,
                symbol="MES",
                ib_bar_size=ib_bar_size,
                label=label,
                start=start,
                end=end,
                pacing_delay=args.pacing,
            )

        # Final report
        for _, label in bar_sizes:
            count = qdb.count_bars("MES", label)
            latest = qdb.get_latest_timestamp("MES", label)
            logger.info("Final - MES %s: %d bars, latest: %s", label, count, latest)

    except ConnectionError as exc:
        logger.error("Could not connect to IB Gateway: %s", exc)
        logger.error("Make sure IB Gateway/TWS is running on port 4002 (paper) or 4001 (live)")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted. Progress is saved - re-run to resume.")
    finally:
        connector.disconnect()


if __name__ == "__main__":
    main()
