"""
Load Databento MES 1-second data -> aggregate to 1-min bars -> save to Parquet + QuestDB.

Supports both CSV and Parquet input (parquet is much faster for large files).

Outputs:
  - data/MES_1min.parquet  (primary, for backtesting)
  - QuestDB ohlcv table    (optional, for live trading / SQL queries)

Usage:
    uv run python data/load_databento.py /path/to/MES_databento.csv
    uv run python data/load_databento.py /path/to/MES_databento.parquet
    uv run python data/load_databento.py /path/to/data.parquet --no-questdb
    uv run python data/load_databento.py /path/to/data.csv --dry-run
"""

import argparse
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import QUESTDB_HOST, QUESTDB_ILP_PORT

TABLE = "ohlcv"
SYMBOL = "MES"
BAR_SIZE = "1min"
CHUNK_SIZE = 5_000_000  # rows per chunk
ILP_BATCH = 10_000  # ILP lines per TCP send
PARQUET_PATH = Path(__file__).resolve().parent / "MES_1min.parquet"


def aggregate_1s_to_1min(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1-second bars to 1-minute OHLCV bars."""
    agg = df.resample("1min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["open"])
    # Recompute VWAP from typical price * volume (approximation from 1s bars)
    if "vwap" in df.columns:
        vwap_agg = df.resample("1min").apply(
            lambda x: np.average(x["vwap"], weights=x["volume"])
            if x["volume"].sum() > 0 else np.nan
        )
        if isinstance(vwap_agg, pd.DataFrame):
            agg["vwap"] = (agg["high"] + agg["low"] + agg["close"]) / 3.0
        else:
            agg["vwap"] = vwap_agg
    else:
        agg["vwap"] = (agg["high"] + agg["low"] + agg["close"]) / 3.0
    agg["vwap"] = agg["vwap"].fillna((agg["high"] + agg["low"] + agg["close"]) / 3.0)
    return agg


def build_ilp_lines(df_1min: pd.DataFrame) -> list:
    """Build ILP lines from a 1-min DataFrame with DatetimeIndex."""
    lines = []
    # Must convert to ns for QuestDB ILP
    ts_ns = df_1min.index.as_unit("ns").asi8
    opens = df_1min["open"].values
    highs = df_1min["high"].values
    lows = df_1min["low"].values
    closes = df_1min["close"].values
    volumes = df_1min["volume"].values.astype(np.int64)
    vwaps = df_1min["vwap"].values

    for i in range(len(df_1min)):
        lines.append(
            f"{TABLE},symbol={SYMBOL},bar_size={BAR_SIZE} "
            f"open={opens[i]},high={highs[i]},low={lows[i]},close={closes[i]},"
            f"volume={volumes[i]}i,vwap={vwaps[i]} "
            f"{ts_ns[i]}\n"
        )
    return lines


def send_ilp_batch(sock: socket.socket, lines: list) -> None:
    """Send a batch of ILP lines over TCP."""
    payload = "".join(lines).encode()
    sock.sendall(payload)


def iter_chunks_csv(csv_path: Path):
    """Yield DataFrames from CSV in chunks with DatetimeIndex (UTC)."""
    reader = pd.read_csv(
        csv_path,
        chunksize=CHUNK_SIZE,
        dtype={"timestamp": np.float64, "open": np.float64, "high": np.float64,
               "low": np.float64, "close": np.float64, "volume": np.int64,
               "vwap": np.float64},
    )
    for chunk in reader:
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], unit="s", utc=True)
        chunk.set_index("timestamp", inplace=True)
        yield chunk


def iter_chunks_parquet(parquet_path: Path):
    """Yield DataFrames from Parquet row groups with DatetimeIndex (UTC)."""
    pf = pq.ParquetFile(str(parquet_path))
    for i in range(pf.metadata.num_row_groups):
        chunk = pf.read_row_group(i).to_pandas()
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], unit="s", utc=True)
        chunk.set_index("timestamp", inplace=True)
        yield chunk


def main():
    parser = argparse.ArgumentParser(description="Load Databento MES data -> Parquet + QuestDB")
    parser.add_argument("input_path", help="Path to MES .csv or .parquet file")
    parser.add_argument("--dry-run", action="store_true", help="Aggregate only, don't save")
    parser.add_argument("--no-questdb", action="store_true", help="Skip QuestDB ingestion")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found")
        sys.exit(1)

    is_parquet = input_path.suffix.lower() in (".parquet", ".pq")

    print("=" * 70)
    print("  Databento MES 1s -> 1min Loader")
    print("=" * 70)
    print(f"  Source: {input_path} ({input_path.stat().st_size / 1e9:.1f} GB)")
    print(f"  Format: {'Parquet' if is_parquet else 'CSV'}")

    # Connect to QuestDB (optional)
    sock = None
    use_questdb = not args.dry_run and not args.no_questdb
    if use_questdb:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((QUESTDB_HOST, QUESTDB_ILP_PORT))
            print(f"  QuestDB ILP connected at {QUESTDB_HOST}:{QUESTDB_ILP_PORT}")
        except ConnectionRefusedError:
            print("  [WARN] QuestDB not available, skipping ILP ingestion")
            sock = None

    t_start = time.time()
    total_1s_bars = 0
    total_1min_bars = 0
    total_sent = 0
    chunk_num = 0
    all_1min_chunks = []

    # Buffer for rows that span chunk boundaries
    leftover = None

    chunks = iter_chunks_parquet(input_path) if is_parquet else iter_chunks_csv(input_path)

    for chunk in chunks:
        chunk_num += 1
        total_1s_bars += len(chunk)

        # Prepend leftover from previous chunk
        if leftover is not None:
            chunk = pd.concat([leftover, chunk])

        # Hold back the last minute (might be split across chunks)
        last_minute = chunk.index[-1].floor("min")
        leftover = chunk.loc[chunk.index >= last_minute].copy()
        chunk = chunk.loc[chunk.index < last_minute]

        if chunk.empty:
            continue

        # Aggregate to 1-min
        df_1min = aggregate_1s_to_1min(chunk)
        total_1min_bars += len(df_1min)
        all_1min_chunks.append(df_1min)

        # QuestDB ingestion
        if sock is not None:
            lines = build_ilp_lines(df_1min)
            for batch_start in range(0, len(lines), ILP_BATCH):
                batch = lines[batch_start:batch_start + ILP_BATCH]
                send_ilp_batch(sock, batch)
                total_sent += len(batch)

        elapsed = time.time() - t_start
        rate = total_1s_bars / elapsed if elapsed > 0 else 0
        print(
            f"  Chunk {chunk_num}: {total_1s_bars:,} 1s bars -> {total_1min_bars:,} 1min bars "
            f"({rate:,.0f} rows/s, {elapsed:.0f}s)",
            flush=True,
        )

    # Process final leftover
    if leftover is not None and not leftover.empty:
        df_1min = aggregate_1s_to_1min(leftover)
        total_1min_bars += len(df_1min)
        all_1min_chunks.append(df_1min)
        if sock is not None:
            lines = build_ilp_lines(df_1min)
            send_ilp_batch(sock, lines)
            total_sent += len(lines)

    if sock is not None:
        sock.close()

    # Write Parquet
    if not args.dry_run and all_1min_chunks:
        df_all = pd.concat(all_1min_chunks)
        df_all.sort_index(inplace=True)
        df_all.to_parquet(PARQUET_PATH, engine="pyarrow", compression="snappy")
        parquet_mb = PARQUET_PATH.stat().st_size / 1e6
        print(f"\n  Parquet: {PARQUET_PATH} ({parquet_mb:.1f} MB)")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Input:  {total_1s_bars:>12,} 1s bars")
    print(f"  Output: {total_1min_bars:>12,} 1min bars")
    if total_sent > 0:
        print(f"  QuestDB:{total_sent:>12,} ILP lines sent")

    # Date range
    if all_1min_chunks:
        first_ts = all_1min_chunks[0].index[0]
        last_ts = all_1min_chunks[-1].index[-1]
        print(f"  Range:  {first_ts} to {last_ts}")
    print("=" * 70)


if __name__ == "__main__":
    main()
