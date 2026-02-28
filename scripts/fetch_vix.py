"""Fetch daily VIX data from Yahoo Finance and store in QuestDB.

Usage:
    uv run python scripts/fetch_vix.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yfinance as yf
import pandas as pd

from data.questdb_client import QuestDBClient


def fetch_vix(start: str = "2025-02-01", end: str = "2026-03-01") -> pd.DataFrame:
    """Download VIX daily OHLCV from Yahoo Finance."""
    raw = yf.download("^VIX", start=start, end=end, progress=False)
    if raw.empty:
        raise RuntimeError("No VIX data returned from Yahoo Finance")

    # yfinance returns multi-level columns with ticker; flatten
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = pd.DataFrame({
        "timestamp": raw.index,
        "open": raw["Open"].values,
        "high": raw["High"].values,
        "low": raw["Low"].values,
        "close": raw["Close"].values,
        "volume": raw["Volume"].values.astype(int),
        "vwap": 0.0,
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def main():
    print("Fetching VIX daily data from Yahoo Finance...")
    df = fetch_vix()
    print(f"  Downloaded {len(df)} trading days")
    print(f"  Date range: {df['timestamp'].iloc[0].date()} to {df['timestamp'].iloc[-1].date()}")
    print(f"  VIX range: {df['close'].min():.1f} - {df['close'].max():.1f}")

    qdb = QuestDBClient()
    qdb.ingest_dataframe(df, symbol="VIX", bar_size="1day")
    qdb.close_ilp()

    # Verify
    count = qdb.count_bars("VIX", "1day")
    print(f"  Stored in QuestDB: {count} rows")
    print("Done.")


if __name__ == "__main__":
    main()
