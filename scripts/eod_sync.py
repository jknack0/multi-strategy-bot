"""
End-of-day sync: push today's bars from QuestDB to Supabase.

Run after market close (post 16:00 ET). Idempotent — safe to re-run.
QuestDB remains the live hot-path store; Supabase is the analytics warehouse.

Usage:
    uv run python scripts/eod_sync.py
    uv run python scripts/eod_sync.py --date 2026-02-28
    uv run python scripts/eod_sync.py --days 5
    uv run python scripts/eod_sync.py --dry-run
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.questdb_client import QuestDBClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ET = pytz.timezone("US/Eastern")


def sync_date(qdb: QuestDBClient, sb, date: datetime, symbol: str,
              bar_sizes: list, dry_run: bool) -> int:
    """Sync one calendar date from QuestDB → Supabase. Returns rows synced."""
    start = date.replace(hour=0, minute=0, second=0, microsecond=0)
    end = date.replace(hour=23, minute=59, second=59, microsecond=0)
    total = 0

    for qdb_label, sb_timeframe in bar_sizes:
        df = qdb.get_bars(symbol, qdb_label, start=start, end=end, limit=5_000_000)

        if df.empty:
            logger.info("  %s %s %s: no bars", date.strftime("%Y-%m-%d"), symbol, qdb_label)
            continue

        # Drop non-OHLCV columns so upsert_market_data gets a clean df
        drop_cols = [c for c in ("symbol", "bar_size") if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        logger.info(
            "  %s %s %s: %d bars (%s → %s)",
            date.strftime("%Y-%m-%d"), symbol, qdb_label,
            len(df), df.index.min(), df.index.max(),
        )

        if dry_run:
            total += len(df)
            continue

        count = sb.upsert_market_data(
            df, symbol=symbol, timeframe=sb_timeframe, source="questdb",
        )
        total += count

    return total


def main():
    parser = argparse.ArgumentParser(description="EOD sync: QuestDB → Supabase")
    parser.add_argument(
        "--date", type=str,
        help="Date to sync (YYYY-MM-DD). Default: today ET",
    )
    parser.add_argument(
        "--days", type=int, default=1,
        help="Number of days to sync back from --date (default: 1)",
    )
    parser.add_argument("--symbol", default="MES", help="Symbol (default: MES)")
    parser.add_argument(
        "--bar-sizes", nargs="+", default=["1min"],
        choices=["1min", "5min", "both"],
        help="Bar sizes to sync (default: 1min)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Query QuestDB and print stats without writing to Supabase",
    )
    args = parser.parse_args()

    # Resolve date range
    if args.date:
        end_date = datetime.strptime(args.date, "%Y-%m-%d")
    else:
        end_date = datetime.now(ET).replace(tzinfo=None)

    dates = [end_date - timedelta(days=i) for i in range(args.days)]
    dates.reverse()

    # Map bar size labels
    if "both" in args.bar_sizes:
        bar_sizes = [("1min", "1m"), ("5min", "5m")]
    else:
        mapping = {"1min": ("1min", "1m"), "5min": ("5min", "5m")}
        bar_sizes = [mapping[b] for b in args.bar_sizes]

    qdb = QuestDBClient()

    # Defer Supabase import so --dry-run works without credentials
    sb = None
    if not args.dry_run:
        from data.supabase_client import SupabaseClient
        sb = SupabaseClient()
        if not sb.health_check():
            logger.error("Supabase health check failed")
            sys.exit(1)
        logger.info("Supabase connection OK")

    logger.info(
        "Syncing %d day(s): %s → %s | %s %s",
        len(dates),
        dates[0].strftime("%Y-%m-%d"),
        dates[-1].strftime("%Y-%m-%d"),
        args.symbol,
        [label for _, label in bar_sizes],
    )

    t0 = time.time()
    grand_total = 0
    for date in dates:
        count = sync_date(qdb, sb, date, args.symbol, bar_sizes, args.dry_run)
        grand_total += count

    elapsed = time.time() - t0

    if args.dry_run:
        logger.info("DRY RUN: would sync %d bars. Exiting.", grand_total)
    else:
        logger.info(
            "Synced %d bars in %.1f seconds (%.0f rows/sec)",
            grand_total, elapsed, grand_total / max(elapsed, 0.001),
        )


if __name__ == "__main__":
    main()
