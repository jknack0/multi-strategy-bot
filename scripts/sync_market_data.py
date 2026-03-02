"""
Sync local parquet OHLCV data to Supabase market_data table.

Reads a parquet file and upserts rows into Supabase. Idempotent — safe to re-run.
Supabase upserts on (symbol, timeframe, open_time).

Usage:
    uv run python scripts/sync_market_data.py data/MES_1min.parquet --symbol MES --timeframe 1m
    uv run python scripts/sync_market_data.py data/MES_1min.parquet --dry-run
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Sync parquet OHLCV data to Supabase market_data"
    )
    parser.add_argument(
        "parquet_path",
        nargs="?",
        default="data/MES_1min.parquet",
        help="Path to parquet file (default: data/MES_1min.parquet)",
    )
    parser.add_argument("--symbol", default="MES", help="Symbol name (default: MES)")
    parser.add_argument(
        "--timeframe", default="1m", help="Timeframe label (default: 1m)"
    )
    parser.add_argument(
        "--source", default="databento", help="Data source tag (default: databento)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Rows per upsert batch (default: 1000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read parquet and print stats without uploading",
    )
    args = parser.parse_args()

    # Read parquet
    parquet_path = Path(args.parquet_path)
    if not parquet_path.exists():
        logger.error("Parquet file not found: %s", parquet_path)
        sys.exit(1)

    logger.info("Reading %s ...", parquet_path)
    df = pd.read_parquet(parquet_path)

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df.set_index("timestamp", inplace=True)
        elif "open_time" in df.columns:
            df.set_index("open_time", inplace=True)
        else:
            logger.error(
                "Cannot determine timestamp column. Expected DatetimeIndex or 'timestamp'/'open_time' column."
            )
            sys.exit(1)

    logger.info("Rows:       %d", len(df))
    logger.info("Date range: %s → %s", df.index.min(), df.index.max())
    logger.info("Columns:    %s", list(df.columns))

    if args.dry_run:
        logger.info("DRY RUN: would upsert %d rows to market_data. Exiting.", len(df))
        return

    # Import here so --dry-run works without Supabase credentials
    from data.supabase_client import SupabaseClient

    client = SupabaseClient(batch_size=args.batch_size)

    # Health check
    if not client.health_check():
        logger.error("Supabase health check failed. Check SUPABASE_URL and SUPABASE_KEY.")
        sys.exit(1)
    logger.info("Supabase connection OK")

    # Check existing count
    existing = client.count_market_data(args.symbol, args.timeframe)
    logger.info("Existing %s %s rows in Supabase: %d", args.symbol, args.timeframe, existing)

    # Upsert
    t0 = time.time()
    count = client.upsert_market_data(
        df, symbol=args.symbol, timeframe=args.timeframe, source=args.source
    )
    elapsed = time.time() - t0

    # Final count
    final = client.count_market_data(args.symbol, args.timeframe)
    logger.info("Upserted %d rows in %.1f seconds (%.0f rows/sec)", count, elapsed, count / max(elapsed, 0.001))
    logger.info("Final %s %s row count: %d", args.symbol, args.timeframe, final)


if __name__ == "__main__":
    main()
