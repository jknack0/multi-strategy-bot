"""
04 — Regime Analysis: Segment backtest by VIX regime.

Runs the strategy on each VIX regime slice independently, compares
filtered vs unfiltered performance, and runs Monte Carlo simulation.

Uses optimized params from config/strategy_a_params.json if available,
and loads real VIX data from QuestDB.

Usage:
    uv run python notebooks/04_regime_analysis.py --bar-size 1min
    uv run python notebooks/04_regime_analysis.py --synthetic
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtesting.metrics import compute_all_metrics, annualized_sharpe
from backtesting.monte_carlo import MonteCarloSimulator
from config.constants import VIXRegime
from strategies.mean_reversion import MESMeanReversionStrategy

# Best params from optimization — long-only is the key improvement
OPTIMIZED_PARAMS = {
    "long_only": True,
    "max_daily_losses": 3,
    "max_daily_loss_dollars": 500.0,
}

PARAMS_FILE = Path(__file__).resolve().parent.parent / "config" / "strategy_a_params.json"

ET = pytz.timezone("US/Eastern")


# ── Data (reuse synthetic generators) ────────────────────────────────────

def generate_synthetic_ohlcv(n: int = 50_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    mu = 4500.0
    theta = 0.03
    sigma = 3.0

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


def load_vix_from_questdb() -> pd.Series:
    """Load real VIX daily closes from QuestDB, return as date-indexed Series."""
    from data.questdb_client import QuestDBClient
    qdb = QuestDBClient()
    vix_df = qdb.get_bars("VIX", "1day", limit=500_000)
    if vix_df.empty:
        return pd.Series(dtype=float, name="vix")
    vix_series = pd.Series(
        vix_df["close"].values,
        index=vix_df.index.date,
        name="vix",
    )
    return vix_series


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
        # Try real VIX first, fall back to synthetic
        vix = load_vix_from_questdb()
        if vix.empty:
            print("  [WARN] No real VIX data in QuestDB, using synthetic VIX")
            vix = generate_synthetic_vix(df)
        else:
            print(f"  Loaded real VIX data: {len(vix)} days, "
                  f"range {vix.min():.1f}-{vix.max():.1f}")
        return df, vix
    except Exception:
        df = generate_synthetic_ohlcv()
        return df, generate_synthetic_vix(df)


# ── Run Strategy on Subset ───────────────────────────────────────────────

def load_best_params() -> dict:
    """Load base params from file, then overlay optimized params on top."""
    base = {}
    if PARAMS_FILE.exists():
        with open(PARAMS_FILE) as f:
            base = json.load(f)
    # Always apply our optimized overrides
    base.update(OPTIMIZED_PARAMS)
    return base


def run_on_subset(
    df: pd.DataFrame,
    vix_series: pd.Series,
    capital: float = 10_000.0,
    params: dict = None,
) -> dict:
    """Run strategy on a data subset and return metrics + trades."""
    if len(df) < 100:
        return {"sharpe": 0.0, "n_trades": 0, "trades": []}

    strat = MESMeanReversionStrategy(capital=capital, params=params)
    strat.generate_signals(df, vix_series=vix_series)

    if len(strat.trades) < 2:
        return {"sharpe": 0.0, "n_trades": len(strat.trades), "trades": strat.trades}

    trade_pnls = np.array([t.pnl for t in strat.trades])
    equity = np.cumsum(np.concatenate([[capital], trade_pnls]))
    returns = np.diff(equity) / equity[:-1]
    metrics = compute_all_metrics(returns, trade_pnls, equity)
    metrics["trades"] = strat.trades
    return metrics


# ── Main ─────────────────────────────────────────────────────────────────

def _print_strat_line(label: str, strat, capital: float = 10_000.0):
    """Print a single strategy result line."""
    if len(strat.trades) < 2:
        print(f"  {label:20s}: No trades")
        return None
    pnls = np.array([t.pnl for t in strat.trades])
    equity = np.cumsum(np.concatenate([[capital], pnls]))
    rets = np.diff(equity) / equity[:-1]
    m = compute_all_metrics(rets, pnls, equity)
    total_pnl = float(np.sum(pnls))
    print(
        f"  {label:20s}: Sharpe={m['sharpe']:+.3f}, "
        f"MaxDD={m['max_drawdown']*100:.1f}%, "
        f"Trades={m['total_trades']}, "
        f"WinRate={m['win_rate']*100:.1f}%, "
        f"PF={m['profit_factor']:.2f}, "
        f"PnL=${total_pnl:+,.0f}"
    )
    return m


def main(use_synthetic: bool = False, bar_size: str = "5min") -> dict:
    print("=" * 70)
    print("  Strategy A: VIX Regime Analysis")
    print("=" * 70)

    best_params = load_best_params()
    df, vix_series = load_data(use_synthetic, bar_size)
    print(f"Data: {len(df)} bars, {len(set(df.index.date))} trading days")
    print(f"Optimized params: { {k: v for k, v in best_params.items() if k in OPTIMIZED_PARAMS} }")

    # Map each bar to its VIX regime
    bar_dates = pd.Series(df.index.date, index=df.index)
    bar_vix = bar_dates.map(lambda d: vix_series.get(d, 20.0))

    # Show VIX distribution
    n_low = int((bar_vix < 15).sum())
    n_mid = int(((bar_vix >= 15) & (bar_vix <= 25)).sum())
    n_high = int(((bar_vix > 25) & (bar_vix <= 35)).sum())
    n_extreme = int((bar_vix > 35).sum())
    print(f"VIX distribution: LOW(<15)={n_low}, NORMAL(15-25)={n_mid}, "
          f"HIGH(25-35)={n_high}, EXTREME(>35)={n_extreme} bars")

    regime_slices = {
        "VIX < 15": df[bar_vix < 15],
        "VIX 15-25": df[(bar_vix >= 15) & (bar_vix <= 25)],
        "VIX 25-35": df[(bar_vix > 25) & (bar_vix <= 35)],
        "VIX > 35": df[bar_vix > 35],
    }

    # ── Per-Regime Metrics (with optimized params) ──
    print("\n" + "=" * 70)
    print("  Per-Regime Performance (optimized params)")
    header = (
        f"  {'Regime':12s} {'Bars':>7} {'Trades':>7} {'Sharpe':>8} "
        f"{'Sortino':>8} {'MaxDD':>7} {'WinRate':>8} {'PF':>6} {'PnL':>9}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    regime_results = {}
    for label, subset in regime_slices.items():
        result = run_on_subset(subset, vix_series, params=best_params)
        regime_results[label] = result
        nt = result.get("total_trades", result.get("n_trades", 0))
        trades_list = result.get("trades", [])
        total_pnl = sum(t.pnl for t in trades_list) if trades_list else 0.0
        print(
            f"  {label:12s} {len(subset):7d} {nt:7d} "
            f"{result.get('sharpe', 0):8.3f} "
            f"{result.get('sortino', 0):8.3f} "
            f"{result.get('max_drawdown', 0) * 100:6.1f}% "
            f"{result.get('win_rate', 0) * 100:7.1f}% "
            f"{result.get('profit_factor', 0):6.2f} "
            f"${total_pnl:+8,.0f}"
        )

    # ── Default vs Optimized vs VIX-Filtered ──
    print("\n" + "-" * 70)
    print("  Comparison: Default vs Optimized vs VIX-Filtered")
    print("-" * 70)

    # Default params, no VIX filter
    strat_default = MESMeanReversionStrategy(capital=10_000.0)
    strat_default.generate_signals(df, vix_series=None)
    _print_strat_line("Default (no VIX)", strat_default)

    # Optimized params, no VIX filter
    strat_opt_novix = MESMeanReversionStrategy(capital=10_000.0, params=best_params)
    strat_opt_novix.generate_signals(df, vix_series=None)
    _print_strat_line("Optimized (no VIX)", strat_opt_novix)

    # Optimized params + VIX filter
    strat_filtered = MESMeanReversionStrategy(capital=10_000.0, params=best_params)
    strat_filtered.generate_signals(df, vix_series=vix_series)
    _print_strat_line("Optimized + VIX", strat_filtered)

    # ── Monte Carlo on best variant ──
    best_strat = strat_filtered if len(strat_filtered.trades) >= len(strat_opt_novix.trades) // 2 else strat_opt_novix
    best_label = "VIX-Filtered" if best_strat is strat_filtered else "Optimized"

    print("\n" + "-" * 70)
    print(f"  Monte Carlo Simulation ({best_label}, 1000 paths)")
    print("-" * 70)

    if len(best_strat.trades) >= 2:
        trade_pnls = np.array([t.pnl for t in best_strat.trades])
        mc = MonteCarloSimulator(n_paths=1000, seed=42)
        mc_result = mc.simulate(trade_pnls, initial_capital=10_000.0)

        print(f"\n  {'Percentile':>12s} {'Final Equity':>14s} {'Max DD':>8s} {'Sharpe':>8s}")
        print("  " + "-" * 45)
        for p in [5, 25, 50, 75, 95]:
            print(
                f"  {p:11d}th ${mc_result['final_equity_pcts'][p]:13,.2f} "
                f"{mc_result['max_drawdown_pcts'][p]*100:7.1f}% "
                f"{mc_result['sharpe_pcts'][p]:8.3f}"
            )

        # Probability of positive return at 6 months
        # Approximate: use half the trades as "6 month equivalent"
        n_6mo = len(trade_pnls) // 4  # rough
        if n_6mo > 0:
            rng = np.random.RandomState(42)
            positive_count = 0
            for _ in range(1000):
                sample = rng.choice(trade_pnls, size=n_6mo, replace=True)
                if np.sum(sample) > 0:
                    positive_count += 1
            prob_positive = positive_count / 1000
            print(f"\n  Prob. positive return at ~6 months: {prob_positive*100:.1f}%")
        else:
            prob_positive = 0.0

        print(f"\n  5th percentile Sharpe   : {mc_result['sharpe_pcts'][5]:.3f}")
        print(f"  95th percentile Max DD  : {mc_result['max_drawdown_pcts'][95]*100:.1f}%")
    else:
        print("  [WARN] Not enough trades for Monte Carlo simulation")
        mc_result = {}
        prob_positive = 0.0

    print("\n" + "=" * 70)

    return {
        "regime_results": {k: {kk: vv for kk, vv in v.items() if kk != "trades"}
                           for k, v in regime_results.items()},
        "mc_result_summary": mc_result.get("summary", None),
        "prob_positive_6mo": prob_positive,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument(
        "--bar-size",
        default="5min",
        choices=["1min", "5min", "15min", "1h"],
        help="Bar size to use (default: 5min)",
    )
    args = parser.parse_args()
    main(use_synthetic=args.synthetic, bar_size=args.bar_size)
