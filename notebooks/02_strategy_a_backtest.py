"""
02 — Strategy A Full Backtest Pipeline.

Loads data, computes indicators, runs MESMeanReversionStrategy through
VectorBT, and reports baseline performance metrics.

Usage:
    uv run python notebooks/02_strategy_a_backtest.py
    uv run python notebooks/02_strategy_a_backtest.py --synthetic
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtesting.metrics import compute_all_metrics
from config.constants import INSTRUMENTS
from strategies.mean_reversion import MESMeanReversionStrategy

ET = pytz.timezone("US/Eastern")


# ── Data Loading ─────────────────────────────────────────────────────────

def load_from_questdb() -> pd.DataFrame:
    from data.questdb_client import QuestDBClient
    qdb = QuestDBClient()
    df = qdb.get_bars("MES", "5min", limit=500_000)
    if df.empty:
        raise RuntimeError("No data in QuestDB")
    return df


def generate_synthetic_ohlcv(
    n: int = 50_000, seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic OHLCV data with mean-reverting characteristics."""
    rng = np.random.RandomState(seed)
    mu = 4500.0
    theta = 0.03
    sigma = 3.0

    closes = np.empty(n)
    closes[0] = mu
    for i in range(1, n):
        closes[i] = (
            closes[i - 1]
            + theta * (mu - closes[i - 1])
            + sigma * rng.randn()
        )

    # Build OHLCV from closes
    noise = rng.uniform(0.5, 2.5, n)
    opens = closes + rng.randn(n) * 0.5
    highs = np.maximum(opens, closes) + noise
    lows = np.minimum(opens, closes) - noise
    volumes = rng.randint(500, 5000, n)

    # Create trading-hours index (9:30-16:00 ET, 5-min bars → 78 bars/day)
    bars_per_day = 78
    n_days = n // bars_per_day + 1
    dates = pd.bdate_range("2022-01-03", periods=n_days, freq="B")

    timestamps = []
    for d in dates:
        day_start = ET.localize(d.replace(hour=9, minute=30))
        for j in range(bars_per_day):
            timestamps.append(day_start + pd.Timedelta(minutes=5 * j))
            if len(timestamps) >= n:
                break
        if len(timestamps) >= n:
            break

    timestamps = timestamps[:n]

    df = pd.DataFrame(
        {
            "open": opens[:n],
            "high": highs[:n],
            "low": lows[:n],
            "close": closes[:n],
            "volume": volumes[:n],
        },
        index=pd.DatetimeIndex(timestamps, name="timestamp"),
    )
    return df


def generate_synthetic_vix(df: pd.DataFrame, seed: int = 42) -> pd.Series:
    """Generate synthetic daily VIX series."""
    rng = np.random.RandomState(seed)
    dates = df.index.date
    unique_dates = sorted(set(dates))
    vix_vals = 18.0 + np.cumsum(rng.randn(len(unique_dates)) * 0.5)
    vix_vals = np.clip(vix_vals, 10, 50)
    return pd.Series(vix_vals, index=unique_dates, name="vix")


# ── Main ─────────────────────────────────────────────────────────────────

def main(use_synthetic: bool = False) -> dict:
    print("=" * 70)
    print("  Strategy A: MES Mean Reversion — Full Backtest")
    print("=" * 70)

    # 1. Load data
    if use_synthetic:
        print("\n[INFO] Using synthetic data")
        df = generate_synthetic_ohlcv()
        vix_series = generate_synthetic_vix(df)
    else:
        try:
            df = load_from_questdb()
            vix_series = generate_synthetic_vix(df)
            print(f"\n[INFO] Loaded {len(df)} bars from QuestDB")
        except Exception as exc:
            print(f"\n[WARN] QuestDB unavailable ({exc}), using synthetic data")
            df = generate_synthetic_ohlcv()
            vix_series = generate_synthetic_vix(df)

    print(f"Data range: {df.index[0]} → {df.index[-1]}")
    print(f"Total bars: {len(df)}")

    # 2. Generate signals via event-driven strategy
    print("\n[INFO] Generating signals...")
    strategy = MESMeanReversionStrategy(capital=10_000.0)
    entries, exits = strategy.generate_signals(df, vix_series=vix_series)

    n_entries = entries.sum()
    n_exits = exits.sum()
    print(f"Entry signals: {n_entries}")
    print(f"Exit signals:  {n_exits}")
    print(f"Total event-driven trades: {len(strategy.trades)}")

    if len(strategy.trades) == 0:
        print("\n[WARN] No trades generated. Check data alignment or parameters.")
        return {}

    # 3. Compute metrics from event-driven trades
    trade_pnls = np.array([t.pnl for t in strategy.trades])
    equity_curve = np.cumsum(np.concatenate([[10000.0], trade_pnls]))
    daily_returns = np.diff(equity_curve) / equity_curve[:-1]

    metrics = compute_all_metrics(
        daily_returns=daily_returns,
        trades=trade_pnls,
        equity_curve=equity_curve,
    )

    # Compute additional stats
    n_days = len(set(df.index.date))
    avg_trades_per_day = len(strategy.trades) / max(n_days, 1)

    hold_times = [
        (t.exit_time - t.entry_time).total_seconds() / 60
        for t in strategy.trades
    ]
    avg_hold_minutes = np.mean(hold_times) if hold_times else 0

    exit_reasons = {}
    for t in strategy.trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    # 4. Print results
    print("\n" + "=" * 70)
    print("  BASELINE METRICS")
    print("=" * 70)
    print(f"  Sharpe Ratio     : {metrics['sharpe']:.3f}")
    print(f"  Sortino Ratio    : {metrics['sortino']:.3f}")
    print(f"  Max Drawdown     : {metrics['max_drawdown'] * 100:.1f}%")
    print(f"  Win Rate         : {metrics['win_rate'] * 100:.1f}%")
    print(f"  Profit Factor    : {metrics['profit_factor']:.2f}")
    print(f"  Calmar Ratio     : {metrics['calmar']:.3f}")
    print(f"  Expectancy       : ${metrics['expectancy']:.2f}")
    print(f"  Total Trades     : {metrics['total_trades']}")
    print(f"  Total Return     : {metrics['total_return'] * 100:.1f}%")
    print(f"  Avg Trades/Day   : {avg_trades_per_day:.2f}")
    print(f"  Avg Hold Time    : {avg_hold_minutes:.0f} min")

    print("\n  Exit Reasons:")
    for reason, count in sorted(exit_reasons.items()):
        pct = count / len(strategy.trades) * 100
        print(f"    {reason:15s}: {count:4d} ({pct:.1f}%)")

    print("=" * 70)

    metrics["avg_trades_per_day"] = avg_trades_per_day
    metrics["avg_hold_minutes"] = avg_hold_minutes
    metrics["exit_reasons"] = exit_reasons
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    main(use_synthetic=args.synthetic)
