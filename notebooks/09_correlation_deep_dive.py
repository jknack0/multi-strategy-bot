"""
09 — Correlation Deep Dive between Strategy A and Strategy B.

1. Daily return correlation (Pearson, Spearman, Kendall)
2. Rolling 20-day correlation
3. Regime-conditional correlation
4. Drawdown overlap analysis
5. Portfolio benefit analysis

Usage:
    uv run python notebooks/09_correlation_deep_dive.py --synthetic
    uv run python notebooks/09_correlation_deep_dive.py --bar-size 5min
"""

import argparse
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtesting.metrics import annualized_sharpe, max_drawdown
from strategies.orb_strategy import VIXAdaptiveORBStrategy
from strategies.mean_reversion import MESMeanReversionStrategy

ET = pytz.timezone("US/Eastern")


# ── Data (reuse from 07) ────────────────────────────────────────────

def generate_synthetic_ohlcv(n: int = 50_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    mu, theta, sigma = 5420.0, 0.03, 4.0
    closes = np.empty(n)
    closes[0] = mu
    for i in range(1, n):
        closes[i] = closes[i - 1] + theta * (mu - closes[i - 1]) + sigma * rng.randn()
    noise = rng.uniform(0.5, 3.0, n)
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
        vix_df = qdb.get_bars("VIX", "1day", limit=500_000)
        if not vix_df.empty:
            vix = pd.Series(vix_df["close"].values, index=vix_df.index.date, name="vix")
        else:
            vix = generate_synthetic_vix(df)
        return df, vix
    except Exception:
        df = generate_synthetic_ohlcv()
        return df, generate_synthetic_vix(df)


def get_daily_returns(trades, dates, capital=10_000.0):
    """Convert trade list to daily returns Series."""
    daily = pd.Series(0.0, index=dates)
    for t in trades:
        d = t.entry_time
        if hasattr(d, "tzinfo") and d.tzinfo is not None:
            d = d.replace(tzinfo=None)
        d = pd.Timestamp(d.date() if hasattr(d, "date") else d)
        if d in daily.index:
            daily.loc[d] += t.pnl / capital
    return daily


# ── Main ────────────────────────────────────────────────────────────

def main(use_synthetic: bool = False, bar_size: str = "5min"):
    print("=" * 70)
    print("  Correlation Deep Dive: Strategy A vs Strategy B")
    print("=" * 70)

    df, vix_series = load_data(use_synthetic, bar_size)

    # Run both strategies
    strat_a = MESMeanReversionStrategy(
        params={"long_only": True, "max_daily_losses": 3, "max_daily_loss_dollars": 500.0},
        capital=10_000.0,
    )
    strat_a.generate_signals(df, vix_series=vix_series)

    strat_b = VIXAdaptiveORBStrategy(capital=20_000.0)
    strat_b.generate_signals(df, vix_series=vix_series)

    print(f"  Strategy A: {len(strat_a.trades)} trades")
    print(f"  Strategy B: {len(strat_b.trades)} trades")

    if len(strat_a.trades) < 5 or len(strat_b.trades) < 5:
        print("  [WARN] Not enough trades for correlation analysis")
        return

    dates = pd.bdate_range(min(df.index.date), max(df.index.date))
    daily_a = get_daily_returns(strat_a.trades, dates, 10_000.0)
    daily_b = get_daily_returns(strat_b.trades, dates, 20_000.0)

    # ── 1. Overall Correlation ──
    print(f"\n{'-'*70}")
    print("  1. Daily Return Correlations")
    print("-" * 70)

    active = (daily_a != 0) | (daily_b != 0)
    da = daily_a[active]
    db = daily_b[active]

    if len(da) > 5:
        pearson = da.corr(db)
        spearman = da.rank().corr(db.rank())
        kendall = da.corr(db, method="kendall") if len(da) > 10 else 0.0
    else:
        pearson = spearman = kendall = 0.0

    print(f"  Pearson:  {pearson:+.3f}  {'PASS' if abs(pearson) < 0.15 else 'CHECK' if abs(pearson) < 0.30 else 'WARN'}")
    print(f"  Spearman: {spearman:+.3f}  {'PASS' if abs(spearman) < 0.20 else 'CHECK' if abs(spearman) < 0.30 else 'WARN'}")
    print(f"  Kendall:  {kendall:+.3f}")

    # ── 2. Rolling Correlation ──
    print(f"\n{'-'*70}")
    print("  2. Rolling 20-Day Correlation")
    print("-" * 70)

    if len(daily_a) >= 40:
        rolling_corr = daily_a.rolling(20).corr(daily_b)
        rolling_corr = rolling_corr.dropna()
        if len(rolling_corr) > 0:
            pct_above_030 = (rolling_corr.abs() > 0.30).sum() / len(rolling_corr) * 100
            print(f"  Mean rolling correlation:   {rolling_corr.mean():+.3f}")
            print(f"  Max rolling correlation:    {rolling_corr.max():+.3f}")
            print(f"  Min rolling correlation:    {rolling_corr.min():+.3f}")
            print(f"  % days |corr| > 0.30:       {pct_above_030:.1f}%  {'PASS' if pct_above_030 < 15 else 'WARN'}")
        else:
            print("  Not enough data for rolling correlation")
    else:
        print("  Not enough data for rolling correlation")

    # ── 3. Regime-Conditional Correlation ──
    print(f"\n{'-'*70}")
    print("  3. Regime-Conditional Correlation")
    print("-" * 70)

    vix_dict = vix_series.to_dict() if vix_series is not None else {}
    regimes = [("LOW (<15)", 0, 15), ("NORMAL (15-25)", 15, 25), ("HIGH (>=25)", 25, 100)]

    for name, vlo, vhi in regimes:
        regime_dates = [pd.Timestamp(d) for d, v in vix_dict.items() if vlo <= v < vhi]
        common_dates = daily_a.index.intersection(regime_dates)
        ra = daily_a.loc[common_dates]
        rb = daily_b.loc[common_dates]
        active_r = (ra != 0) | (rb != 0)
        if active_r.sum() > 5:
            corr = ra[active_r].corr(rb[active_r])
            print(f"  {name:20s}: corr={corr:+.3f} ({active_r.sum()} active days)")
        else:
            print(f"  {name:20s}: insufficient data ({active_r.sum()} active days)")

    # ── 4. Drawdown Overlap ──
    print(f"\n{'-'*70}")
    print("  4. Drawdown Overlap Analysis")
    print("-" * 70)

    # Worst 10 days for each strategy
    worst_a_days = daily_a.nsmallest(10).index
    worst_b_days = daily_b.nsmallest(10).index

    b_on_a_worst = daily_b.loc[worst_a_days]
    a_on_b_worst = daily_a.loc[worst_b_days]

    print(f"  When A has worst 10 days:")
    print(f"    A avg return:  {daily_a.loc[worst_a_days].mean()*100:+.3f}%")
    print(f"    B avg return:  {b_on_a_worst.mean()*100:+.3f}%  {'GOOD' if b_on_a_worst.mean() >= 0 else 'OK' if b_on_a_worst.mean() > -0.005 else 'CONCERN'}")

    print(f"  When B has worst 10 days:")
    print(f"    B avg return:  {daily_b.loc[worst_b_days].mean()*100:+.3f}%")
    print(f"    A avg return:  {a_on_b_worst.mean()*100:+.3f}%  {'GOOD' if a_on_b_worst.mean() >= 0 else 'OK' if a_on_b_worst.mean() > -0.005 else 'CONCERN'}")

    overlap = set(worst_a_days) & set(worst_b_days)
    print(f"  Overlap in worst 10 days: {len(overlap)}/10  {'GOOD' if len(overlap) <= 2 else 'CONCERN'}")

    # ── 5. Portfolio Benefit ──
    print(f"\n{'-'*70}")
    print("  5. Portfolio Benefit")
    print("-" * 70)

    combined = 0.40 * daily_a + 0.30 * daily_b
    equity_a = np.cumprod(1.0 + daily_a.values) * 10_000
    equity_b = np.cumprod(1.0 + daily_b.values) * 20_000
    equity_c = np.cumprod(1.0 + combined.values) * 20_000

    sr_a = annualized_sharpe(daily_a.values)
    sr_b = annualized_sharpe(daily_b.values)
    sr_combined = annualized_sharpe(combined.values)
    dd_a = max_drawdown(equity_a)
    dd_b = max_drawdown(equity_b)
    dd_combined = max_drawdown(equity_c)

    print(f"  {'':20s} {'Sharpe':>8s} {'MaxDD':>8s}")
    print(f"  {'Strategy A':20s} {sr_a:+8.3f} {dd_a*100:7.1f}%")
    print(f"  {'Strategy B':20s} {sr_b:+8.3f} {dd_b*100:7.1f}%")
    print(f"  {'Combined (40/30)':20s} {sr_combined:+8.3f} {dd_combined*100:7.1f}%")

    # Theoretical portfolio Sharpe
    rho = pearson
    n_strats = 2
    sr_avg = (sr_a + sr_b) / 2
    if 1 + (n_strats - 1) * rho > 0:
        sr_theoretical = sr_avg * np.sqrt(n_strats / (1 + (n_strats - 1) * rho))
    else:
        sr_theoretical = 0.0
    print(f"\n  Theoretical portfolio Sharpe (given rho={rho:.3f}): {sr_theoretical:+.3f}")
    print(f"  Diversification benefit: {'YES' if sr_combined > max(sr_a, sr_b) else 'NO'}")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--bar-size", default="5min", choices=["1min", "5min"])
    args = parser.parse_args()
    main(use_synthetic=args.synthetic, bar_size=args.bar_size)
