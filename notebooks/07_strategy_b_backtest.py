"""
07 — Strategy B Backtest: Full backtest and validation of VIX-Adaptive ORB.

1. Run with DEFAULT_PARAMS on available data
2. Print baseline metrics
3. Compute daily return correlation with Strategy A
4. Parameter sweep
5. CPCV on top parameter sets
6. Combined A+B portfolio analysis
7. Monte Carlo

Usage:
    uv run python notebooks/07_strategy_b_backtest.py --synthetic
    uv run python notebooks/07_strategy_b_backtest.py --bar-size 5min
"""

import argparse
import itertools
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtesting.metrics import compute_all_metrics, annualized_sharpe
from backtesting.monte_carlo import MonteCarloSimulator
from strategies.orb_strategy import VIXAdaptiveORBStrategy
from strategies.mean_reversion import MESMeanReversionStrategy

ET = pytz.timezone("US/Eastern")

PARAMS_FILE = Path(__file__).resolve().parent.parent / "config" / "strategy_b_params.json"


# ── Data ────────────────────────────────────────────────────────────

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


# ── Run Strategy ────────────────────────────────────────────────────

def run_strategy_b(df, vix_series, params=None, capital=20_000.0):
    """Run ORB strategy and return metrics dict."""
    strat = VIXAdaptiveORBStrategy(params=params, capital=capital)
    strat.generate_signals(df, vix_series=vix_series)

    if len(strat.trades) < 2:
        return {"sharpe": 0.0, "n_trades": len(strat.trades), "trades": strat.trades}

    trade_pnls = np.array([t.pnl for t in strat.trades])
    equity = np.cumsum(np.concatenate([[capital], trade_pnls]))
    returns = np.diff(equity) / equity[:-1]
    metrics = compute_all_metrics(returns, trade_pnls, equity)
    metrics["trades"] = strat.trades
    return metrics


def run_strategy_a(df, vix_series, capital=10_000.0):
    """Run Strategy A for correlation comparison."""
    params = {"long_only": True, "max_daily_losses": 3, "max_daily_loss_dollars": 500.0}
    strat = MESMeanReversionStrategy(params=params, capital=capital)
    strat.generate_signals(df, vix_series=vix_series)

    if len(strat.trades) < 2:
        return {"sharpe": 0.0, "n_trades": len(strat.trades), "trades": strat.trades}

    trade_pnls = np.array([t.pnl for t in strat.trades])
    equity = np.cumsum(np.concatenate([[capital], trade_pnls]))
    returns = np.diff(equity) / equity[:-1]
    metrics = compute_all_metrics(returns, trade_pnls, equity)
    metrics["trades"] = strat.trades
    return metrics


def compute_daily_returns(trades, start_date, end_date, capital=10_000.0):
    """Convert trade list to daily return series."""
    dates = pd.bdate_range(start_date, end_date)
    daily = pd.Series(0.0, index=dates)
    for t in trades:
        d = t.entry_time.date() if hasattr(t.entry_time, "date") else t.entry_time
        if hasattr(t.entry_time, "tzinfo") and t.entry_time.tzinfo is not None:
            d = t.entry_time.replace(tzinfo=None).date()
        d = pd.Timestamp(d)
        if d in daily.index:
            daily.loc[d] += t.pnl / capital
    return daily


# ── Main ────────────────────────────────────────────────────────────

def main(use_synthetic: bool = False, bar_size: str = "5min"):
    print("=" * 70)
    print("  Strategy B: VIX-Adaptive ORB Backtest")
    print("=" * 70)

    df, vix_series = load_data(use_synthetic, bar_size)
    n_days = len(set(df.index.date))
    print(f"Data: {len(df)} bars, {n_days} trading days")

    # ── 1. Baseline ──
    print(f"\n{'-'*70}")
    print("  1. Baseline (DEFAULT_PARAMS)")
    print("-" * 70)

    metrics_b = run_strategy_b(df, vix_series)
    trades_b = metrics_b.get("trades", [])
    n_trades = len(trades_b)
    total_pnl = sum(t.pnl for t in trades_b)

    print(f"  Trades:       {n_trades}")
    print(f"  Sharpe:       {metrics_b.get('sharpe', 0):+.3f}")
    print(f"  Sortino:      {metrics_b.get('sortino', 0):+.3f}")
    print(f"  Max DD:       {metrics_b.get('max_drawdown', 0)*100:.1f}%")
    print(f"  Win Rate:     {metrics_b.get('win_rate', 0)*100:.1f}%")
    print(f"  Profit Factor:{metrics_b.get('profit_factor', 0):.2f}")
    print(f"  Total PnL:    ${total_pnl:+,.0f}")

    # ── 2. Correlation with Strategy A ──
    print(f"\n{'-'*70}")
    print("  2. Correlation with Strategy A")
    print("-" * 70)

    metrics_a = run_strategy_a(df, vix_series)
    trades_a = metrics_a.get("trades", [])

    if len(trades_a) >= 2 and len(trades_b) >= 2:
        dates = sorted(set(df.index.date))
        start_date = min(dates)
        end_date = max(dates)

        daily_a = compute_daily_returns(trades_a, start_date, end_date, 10_000.0)
        daily_b = compute_daily_returns(trades_b, start_date, end_date, 20_000.0)

        # Align
        common = daily_a.index.intersection(daily_b.index)
        da = daily_a.loc[common]
        db = daily_b.loc[common]

        # Only compute correlation on days where at least one strategy traded
        active = (da != 0) | (db != 0)
        if active.sum() > 5:
            pearson = da[active].corr(db[active])
            spearman = da[active].rank().corr(db[active].rank())
        else:
            pearson = spearman = 0.0

        print(f"  Strategy A: {len(trades_a)} trades, Sharpe={metrics_a.get('sharpe', 0):+.3f}")
        print(f"  Strategy B: {len(trades_b)} trades, Sharpe={metrics_b.get('sharpe', 0):+.3f}")
        print(f"  Pearson correlation:  {pearson:+.3f}  {'PASS' if abs(pearson) < 0.30 else 'WARN'}")
        print(f"  Spearman correlation: {spearman:+.3f}  {'PASS' if abs(spearman) < 0.30 else 'WARN'}")
    else:
        print("  [WARN] Not enough trades for correlation analysis")
        pearson = spearman = 0.0

    # ── 3. Parameter sweep ──
    print(f"\n{'-'*70}")
    print("  3. Parameter Sweep")
    print("-" * 70)

    param_grid = {
        "or_width_min_atr_fraction": [0.2, 0.3, 0.4],
        "volume_threshold": [1.0, 1.5, 2.0],
        "stop_or_multiplier": [0.3, 0.5, 0.75],
        "target_or_multiplier": [0.75, 1.0, 1.5],
    }

    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    print(f"  {len(combos)} parameter combinations")

    # Use reduced dataset for sweep (faster), full for baseline/MC
    sweep_bars = min(len(df), 15_000)
    df_sweep = df.iloc[:sweep_bars]
    vix_sweep = vix_series

    results = []
    for combo in combos:
        params = dict(zip(keys, combo))
        m = run_strategy_b(df_sweep, vix_sweep, params=params)
        trades = m.get("trades", [])
        results.append({
            **params,
            "sharpe": m.get("sharpe", 0),
            "n_trades": len(trades),
            "win_rate": m.get("win_rate", 0),
            "profit_factor": m.get("profit_factor", 0),
            "max_drawdown": m.get("max_drawdown", 0),
            "total_pnl": sum(t.pnl for t in trades),
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("sharpe", ascending=False)

    print(f"\n  Top 10 parameter sets:")
    header = f"  {'Sharpe':>7s} {'Trades':>7s} {'WR':>6s} {'PF':>6s} {'DD':>6s} {'PnL':>9s} | Params"
    print(header)
    print("  " + "-" * 65)
    for _, row in results_df.head(10).iterrows():
        params_str = f"atr_frac={row['or_width_min_atr_fraction']}, vol={row['volume_threshold']}, stop={row['stop_or_multiplier']}, tgt={row['target_or_multiplier']}"
        print(
            f"  {row['sharpe']:+7.3f} {row['n_trades']:7.0f} "
            f"{row['win_rate']*100:5.1f}% {row['profit_factor']:5.2f} "
            f"{row['max_drawdown']*100:5.1f}% ${row['total_pnl']:+8,.0f} | {params_str}"
        )

    # Save best params
    if len(results_df) > 0:
        best = results_df.iloc[0]
        best_params = {k: float(best[k]) for k in keys}
        PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(PARAMS_FILE, "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"\n  Best params saved to {PARAMS_FILE}")

    # ── 4. Combined portfolio ──
    print(f"\n{'-'*70}")
    print("  4. Combined A+B Portfolio")
    print("-" * 70)

    if len(trades_a) >= 2 and len(trades_b) >= 2:
        # A at 40%, B at 30%
        combined_daily = 0.40 * da + 0.30 * db
        combined_sharpe = annualized_sharpe(combined_daily.values)
        a_sharpe = annualized_sharpe(da.values)
        b_sharpe = annualized_sharpe(db.values)

        print(f"  Strategy A Sharpe: {a_sharpe:+.3f}")
        print(f"  Strategy B Sharpe: {b_sharpe:+.3f}")
        print(f"  Combined Sharpe:   {combined_sharpe:+.3f}")
        print(f"  Diversification benefit: {'YES' if combined_sharpe > max(a_sharpe, b_sharpe) else 'NO'}")
    else:
        print("  [WARN] Not enough trades for portfolio analysis")

    # ── 5. Monte Carlo ──
    print(f"\n{'-'*70}")
    print("  5. Monte Carlo (Strategy B, 1000 paths)")
    print("-" * 70)

    if len(trades_b) >= 2:
        trade_pnls = np.array([t.pnl for t in trades_b])
        mc = MonteCarloSimulator(n_paths=1000, seed=42)
        mc_result = mc.simulate(trade_pnls, initial_capital=20_000.0)

        print(f"\n  {'Percentile':>12s} {'Final Equity':>14s} {'Max DD':>8s} {'Sharpe':>8s}")
        print("  " + "-" * 45)
        for p in [5, 25, 50, 75, 95]:
            print(
                f"  {p:11d}th ${mc_result['final_equity_pcts'][p]:13,.2f} "
                f"{mc_result['max_drawdown_pcts'][p]*100:7.1f}% "
                f"{mc_result['sharpe_pcts'][p]:8.3f}"
            )
        print(f"\n  Prob. positive: {mc_result['prob_positive']*100:.1f}%")
    else:
        print("  [WARN] Not enough trades for Monte Carlo")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--bar-size", default="5min", choices=["1min", "5min"])
    args = parser.parse_args()
    main(use_synthetic=args.synthetic, bar_size=args.bar_size)
