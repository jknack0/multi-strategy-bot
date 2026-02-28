"""
06 — ORB Analysis: Understand the Opening Range before building the strategy.

1. Compute OR for every trading day across multiple durations
2. OR Width Distribution (histogram stats)
3. Breakout Success Rate (measured-move target vs stop)
4. VIX Regime Analysis (which duration works best per regime)

Usage:
    uv run python notebooks/06_orb_analysis.py --synthetic
    uv run python notebooks/06_orb_analysis.py --bar-size 5min
"""

import argparse
import sys
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from indicators.atr import ATR

ET = pytz.timezone("US/Eastern")


# ── Data ────────────────────────────────────────────────────────────

def generate_synthetic_ohlcv(n: int = 50_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    mu, theta, sigma = 5420.0, 0.03, 3.0
    closes = np.empty(n)
    closes[0] = mu
    for i in range(1, n):
        closes[i] = closes[i - 1] + theta * (mu - closes[i - 1]) + sigma * rng.randn()
    noise = rng.uniform(0.5, 2.5, n)
    opens = closes + rng.randn(n) * 0.5
    highs = np.maximum(opens, closes) + noise
    lows = np.minimum(opens, closes) - noise
    volumes = rng.randint(500, 5000, n)

    bars_per_day = 78
    n_days = n // bars_per_day + 1
    dates = pd.bdate_range("2023-01-03", periods=n_days, freq="B")
    timestamps = []
    for d in dates:
        day_start = ET.localize(d.replace(hour=9, minute=30))
        for j in range(bars_per_day):
            timestamps.append(day_start + timedelta(minutes=5 * j))
            if len(timestamps) >= n:
                break
        if len(timestamps) >= n:
            break

    return pd.DataFrame(
        {"open": opens[:n], "high": highs[:n], "low": lows[:n],
         "close": closes[:n], "volume": volumes[:n]},
        index=pd.DatetimeIndex(timestamps[:n], name="timestamp"),
    )


def generate_synthetic_vix(df: pd.DataFrame, seed: int = 42) -> pd.Series:
    rng = np.random.RandomState(seed)
    unique_dates = sorted(set(df.index.date))
    vix_vals = 18.0 + np.cumsum(rng.randn(len(unique_dates)) * 0.5)
    vix_vals = np.clip(vix_vals, 10, 50)
    return pd.Series(vix_vals, index=unique_dates, name="vix")


def load_data(use_synthetic: bool, bar_size: str = "5min") -> tuple:
    if use_synthetic:
        df = generate_synthetic_ohlcv()
        return df, generate_synthetic_vix(df)
    try:
        from data.questdb_client import QuestDBClient
        qdb = QuestDBClient()
        df = qdb.get_bars("MES", bar_size, limit=500_000)
        if df.empty:
            raise RuntimeError("Empty")
        try:
            from scripts.fetch_vix import fetch_vix
            # Try loading from QuestDB first
            vix_df = qdb.get_bars("VIX", "1day", limit=500_000)
            if not vix_df.empty:
                vix = pd.Series(vix_df["close"].values, index=vix_df.index.date, name="vix")
            else:
                vix = generate_synthetic_vix(df)
        except Exception:
            vix = generate_synthetic_vix(df)
        return df, vix
    except Exception:
        df = generate_synthetic_ohlcv()
        return df, generate_synthetic_vix(df)


# ── OR Computation ──────────────────────────────────────────────────

def compute_daily_or(df: pd.DataFrame, or_duration_minutes: int) -> pd.DataFrame:
    """Compute Opening Range for each day.

    Returns DataFrame: date, or_high, or_low, or_width, daily_atr
    """
    if df.index.tz is None:
        et_idx = df.index.tz_localize(ET)
    else:
        et_idx = df.index.tz_convert(ET)

    df_et = df.copy()
    df_et["_date"] = et_idx.date
    df_et["_time"] = et_idx.time

    rth_open = dt_time(9, 30)
    or_end = (datetime(2000, 1, 1, 9, 30) + timedelta(minutes=or_duration_minutes)).time()

    # Compute ATR per bar for daily ATR reference
    atr_ind = ATR(14)
    atr_vals = []
    for _, row in df.iterrows():
        a = atr_ind.update(row["high"], row["low"], row["close"])
        atr_vals.append(a if a is not None else np.nan)
    df_et["_atr"] = atr_vals

    results = []
    for date, day_df in df_et.groupby("_date"):
        or_mask = (day_df["_time"] >= rth_open) & (day_df["_time"] < or_end)
        or_bars = day_df[or_mask]
        if len(or_bars) < 1:
            continue
        or_high = or_bars["high"].max()
        or_low = or_bars["low"].min()
        or_width = or_high - or_low

        # Daily ATR: use last available
        daily_atr = day_df["_atr"].dropna().iloc[-1] if day_df["_atr"].notna().any() else np.nan

        results.append({
            "date": date,
            "or_high": or_high,
            "or_low": or_low,
            "or_width": or_width,
            "daily_atr": daily_atr,
            "or_width_atr_pct": or_width / daily_atr if daily_atr > 0 else np.nan,
        })

    return pd.DataFrame(results)


def compute_breakout_results(
    df: pd.DataFrame,
    or_df: pd.DataFrame,
    or_duration_minutes: int,
) -> pd.DataFrame:
    """For each day, check if a breakout hit target or stop."""
    if df.index.tz is None:
        et_idx = df.index.tz_localize(ET)
    else:
        et_idx = df.index.tz_convert(ET)

    df_et = df.copy()
    df_et["_date"] = et_idx.date
    df_et["_time"] = et_idx.time

    or_end = (datetime(2000, 1, 1, 9, 30) + timedelta(minutes=or_duration_minutes)).time()
    max_entry = dt_time(14, 0)

    results = []
    or_lookup = or_df.set_index("date")

    for date, day_df in df_et.groupby("_date"):
        if date not in or_lookup.index:
            continue
        row = or_lookup.loc[date]
        or_high, or_low, or_width = row["or_high"], row["or_low"], row["or_width"]
        if or_width <= 0:
            continue

        # Only check bars after OR
        post_or = day_df[(day_df["_time"] >= or_end) & (day_df["_time"] < max_entry)]

        for side in ["LONG", "SHORT"]:
            if side == "LONG":
                entry_level = or_high
                target = entry_level + 1.0 * or_width
                stop = entry_level - 0.5 * or_width
            else:
                entry_level = or_low
                target = entry_level - 1.0 * or_width
                stop = entry_level + 0.5 * or_width

            # Find breakout bar
            if side == "LONG":
                breakout_mask = post_or["close"] > entry_level
            else:
                breakout_mask = post_or["close"] < entry_level

            if not breakout_mask.any():
                continue

            breakout_idx = breakout_mask.idxmax()
            entry_price = post_or.loc[breakout_idx, "close"]

            # Check subsequent bars for target/stop
            after_entry = day_df.loc[breakout_idx:]
            hit_target = False
            hit_stop = False
            for _, bar in after_entry.iterrows():
                if side == "LONG":
                    if bar["low"] <= stop:
                        hit_stop = True
                        break
                    if bar["high"] >= target:
                        hit_target = True
                        break
                else:
                    if bar["high"] >= stop:
                        hit_stop = True
                        break
                    if bar["low"] <= target:
                        hit_target = True
                        break

            results.append({
                "date": date,
                "side": side,
                "or_width": or_width,
                "entry_price": entry_price,
                "hit_target": hit_target,
                "hit_stop": hit_stop,
                "no_resolution": not hit_target and not hit_stop,
            })

    return pd.DataFrame(results)


# ── Main ────────────────────────────────────────────────────────────

def main(use_synthetic: bool = False, bar_size: str = "5min"):
    print("=" * 70)
    print("  Opening Range Breakout Analysis")
    print("=" * 70)

    df, vix_series = load_data(use_synthetic, bar_size)
    n_days = len(set(df.index.date))
    print(f"Data: {len(df)} bars, {n_days} trading days")

    durations = [5, 10, 15, 20, 30]

    # ── OR Width Distribution ──
    print(f"\n{'='*70}")
    print("  OR Width Distribution by Duration")
    header = f"  {'Duration':>8s} {'Mean':>7s} {'Median':>7s} {'Std':>7s} {'Min':>7s} {'Max':>7s} {'%ATR':>7s}"
    print(header)
    print("  " + "-" * 55)

    or_data = {}
    for dur in durations:
        or_df = compute_daily_or(df, dur)
        or_data[dur] = or_df
        w = or_df["or_width"]
        pct = or_df["or_width_atr_pct"]
        print(
            f"  {dur:6d}m {w.mean():7.2f} {w.median():7.2f} {w.std():7.2f} "
            f"{w.min():7.2f} {w.max():7.2f} {pct.mean()*100:6.1f}%"
        )

    # ── Breakout Success Rate ──
    print(f"\n{'='*70}")
    print("  Breakout Success Rate (1x OR target, 0.5x OR stop)")
    header = f"  {'Duration':>8s} {'Trades':>7s} {'WinRate':>8s} {'Avg R:R':>8s} {'Long WR':>8s} {'Short WR':>8s}"
    print(header)
    print("  " + "-" * 50)

    for dur in durations:
        bo_df = compute_breakout_results(df, or_data[dur], dur)
        if bo_df.empty:
            print(f"  {dur:6d}m {'N/A':>7s}")
            continue
        total = len(bo_df)
        wins = bo_df["hit_target"].sum()
        losses = bo_df["hit_stop"].sum()
        wr = wins / total if total > 0 else 0
        avg_rr = 2.0 * wr  # target is 2x stop, so E[RR] ~ 2*WR

        long_bo = bo_df[bo_df["side"] == "LONG"]
        short_bo = bo_df[bo_df["side"] == "SHORT"]
        lwr = long_bo["hit_target"].sum() / len(long_bo) if len(long_bo) > 0 else 0
        swr = short_bo["hit_target"].sum() / len(short_bo) if len(short_bo) > 0 else 0

        print(
            f"  {dur:6d}m {total:7d} {wr*100:7.1f}% {avg_rr:8.2f} "
            f"{lwr*100:7.1f}% {swr*100:7.1f}%"
        )

    # ── VIX Regime Analysis ──
    print(f"\n{'='*70}")
    print("  Best OR Duration by VIX Regime")
    vix_dict = vix_series.to_dict() if vix_series is not None else {}

    for regime_name, vix_range in [("LOW (<15)", (0, 15)), ("NORMAL (15-25)", (15, 25)), ("HIGH (>=25)", (25, 100))]:
        print(f"\n  {regime_name}:")
        best_wr = 0
        best_dur = 0

        for dur in durations:
            bo_df = compute_breakout_results(df, or_data[dur], dur)
            if bo_df.empty:
                continue
            # Filter by VIX regime
            regime_dates = [d for d, v in vix_dict.items() if vix_range[0] <= v < vix_range[1]]
            regime_bo = bo_df[bo_df["date"].isin(regime_dates)]
            if len(regime_bo) < 5:
                continue
            wr = regime_bo["hit_target"].sum() / len(regime_bo)
            if wr > best_wr:
                best_wr = wr
                best_dur = dur
            print(f"    {dur:4d}m: {len(regime_bo):4d} trades, WR={wr*100:.1f}%")

        if best_dur > 0:
            print(f"    >> Best: {best_dur}m (WR={best_wr*100:.1f}%)")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--bar-size", default="5min", choices=["1min", "5min"])
    args = parser.parse_args()
    main(use_synthetic=args.synthetic, bar_size=args.bar_size)
