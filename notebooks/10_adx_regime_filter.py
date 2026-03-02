"""
10 — ADX Regime Filter Validation for Strategy A.

Sweeps ADX threshold values on ~2 years of data to see if filtering out
strong-trend periods improves mean reversion performance.

ADX > threshold => skip entries (market is trending, bad for MR)
ADX = 0 (disabled) => baseline (no filter)

Usage:
    uv run python notebooks/10_adx_regime_filter.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtesting.metrics import annualized_sharpe, compute_all_metrics
from strategies.mean_reversion import (
    MESMeanReversionStrategy,
    precompute_indicators,
)

ET = pytz.timezone("US/Eastern")
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "adx_regime_filter"
PARQUET_PATH = Path(__file__).resolve().parent.parent / "data" / "MES_1min.parquet"
PARAMS_PATH = Path(__file__).resolve().parent.parent / "config" / "strategy_a_params.json"


def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    if idx.tz is None:
        et_idx = idx.tz_localize("UTC").tz_convert(ET)
    else:
        et_idx = idx.tz_convert(ET)
    mask = (et_idx.hour >= 9) & (et_idx.hour < 16)
    filtered = df.loc[mask].copy()
    if filtered.index.tz is None:
        filtered.index = filtered.index.tz_localize("UTC").tz_convert(ET)
    elif str(filtered.index.tz) != "US/Eastern":
        filtered.index = filtered.index.tz_convert(ET)
    return filtered


def load_data_2yr():
    """Load ~2 years of 1-min data from Parquet."""
    df_raw = pd.read_parquet(PARQUET_PATH)
    if not isinstance(df_raw.index, pd.DatetimeIndex):
        df_raw.index = pd.to_datetime(df_raw.index)
    df = filter_rth(df_raw)

    # Take last ~5 years
    cutoff = df.index[-1] - pd.Timedelta(days=1825)
    df = df.loc[df.index >= cutoff]

    # Load VIX from QuestDB
    try:
        from data.questdb_client import QuestDBClient
        qdb = QuestDBClient()
        vix_df = qdb.get_bars("VIX", "1day", limit=10_000)
        if not vix_df.empty:
            vix = pd.Series(vix_df["close"].values, index=vix_df.index.date, name="vix")
        else:
            vix = None
    except Exception:
        vix = None

    return df, vix


def run_with_adx_threshold(df, vix, base_params, adx_threshold, precomputed, adx_sizing=False):
    """Run strategy with a specific ADX threshold, return metrics."""
    params = dict(base_params)
    params["adx_threshold"] = adx_threshold
    params["adx_sizing"] = adx_sizing

    strat = MESMeanReversionStrategy(params=params, capital=10_000.0)
    strat.generate_signals_fast(df, precomputed=precomputed)

    trades = strat.trades
    if len(trades) < 5:
        return {
            "adx_threshold": adx_threshold,
            "n_trades": len(trades),
            "sharpe": 0.0,
            "total_pnl": sum(t.pnl for t in trades),
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "avg_pnl": 0.0,
            "trades": trades,
        }

    trade_pnls = np.array([t.pnl for t in trades])
    equity = np.cumsum(np.concatenate([[10_000.0], trade_pnls]))
    returns = np.diff(equity) / equity[:-1]
    metrics = compute_all_metrics(returns, trade_pnls, equity)

    return {
        "adx_threshold": adx_threshold,
        "n_trades": len(trades),
        "sharpe": metrics.get("sharpe", 0.0),
        "total_pnl": trade_pnls.sum(),
        "win_rate": metrics.get("win_rate", 0.0),
        "profit_factor": metrics.get("profit_factor", 0.0),
        "max_drawdown": metrics.get("max_drawdown", 0.0),
        "avg_pnl": trade_pnls.mean(),
        "trades": trades,
    }


def plot_results(results, filename):
    """Plot sweep results: Sharpe, PnL, trade count, win rate vs ADX threshold."""
    thresholds = [r["adx_threshold"] for r in results]
    labels = ["OFF" if t == 0 else str(int(t)) for t in thresholds]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Sharpe
    sharpes = [r["sharpe"] for r in results]
    colors = ['green' if s > results[0]["sharpe"] else 'steelblue' if s == results[0]["sharpe"] else 'salmon' for s in sharpes]
    colors[0] = 'gray'  # baseline
    axes[0, 0].bar(labels, sharpes, color=colors, edgecolor='black', alpha=0.8)
    axes[0, 0].set_title("Sharpe Ratio")
    axes[0, 0].set_xlabel("ADX Threshold")
    axes[0, 0].axhline(y=results[0]["sharpe"], color='gray', linestyle='--', linewidth=1, label=f'Baseline: {results[0]["sharpe"]:.3f}')
    axes[0, 0].legend()
    for i, s in enumerate(sharpes):
        axes[0, 0].text(i, s + 0.01, f"{s:.3f}", ha='center', fontsize=8)

    # Total PnL
    pnls = [r["total_pnl"] for r in results]
    colors_pnl = ['green' if p > results[0]["total_pnl"] else 'gray' if p == results[0]["total_pnl"] else 'salmon' for p in pnls]
    colors_pnl[0] = 'gray'
    axes[0, 1].bar(labels, pnls, color=colors_pnl, edgecolor='black', alpha=0.8)
    axes[0, 1].set_title("Total PnL ($)")
    axes[0, 1].set_xlabel("ADX Threshold")
    axes[0, 1].axhline(y=results[0]["total_pnl"], color='gray', linestyle='--', linewidth=1)
    for i, p in enumerate(pnls):
        axes[0, 1].text(i, p, f"${p:,.0f}", ha='center', fontsize=8, va='bottom' if p >= 0 else 'top')

    # Trade count
    counts = [r["n_trades"] for r in results]
    axes[0, 2].bar(labels, counts, color='steelblue', edgecolor='black', alpha=0.8)
    axes[0, 2].set_title("Number of Trades")
    axes[0, 2].set_xlabel("ADX Threshold")
    for i, c in enumerate(counts):
        axes[0, 2].text(i, c + 1, str(c), ha='center', fontsize=8)

    # Win Rate
    wrs = [r["win_rate"] * 100 for r in results]
    axes[1, 0].bar(labels, wrs, color='steelblue', edgecolor='black', alpha=0.8)
    axes[1, 0].set_title("Win Rate (%)")
    axes[1, 0].set_xlabel("ADX Threshold")
    axes[1, 0].axhline(y=50, color='gray', linestyle='--', linewidth=0.5)
    for i, wr in enumerate(wrs):
        axes[1, 0].text(i, wr + 0.3, f"{wr:.1f}%", ha='center', fontsize=8)

    # Profit Factor
    pfs = [r["profit_factor"] for r in results]
    axes[1, 1].bar(labels, pfs, color='steelblue', edgecolor='black', alpha=0.8)
    axes[1, 1].set_title("Profit Factor")
    axes[1, 1].set_xlabel("ADX Threshold")
    axes[1, 1].axhline(y=1.0, color='gray', linestyle='--', linewidth=0.5)
    for i, pf in enumerate(pfs):
        axes[1, 1].text(i, pf + 0.01, f"{pf:.2f}", ha='center', fontsize=8)

    # Max Drawdown
    dds = [r["max_drawdown"] * 100 for r in results]
    axes[1, 2].bar(labels, dds, color='salmon', edgecolor='black', alpha=0.8)
    axes[1, 2].set_title("Max Drawdown (%)")
    axes[1, 2].set_xlabel("ADX Threshold")
    for i, dd in enumerate(dds):
        axes[1, 2].text(i, dd + 0.2, f"{dd:.1f}%", ha='center', fontsize=8)

    plt.suptitle("Strategy A — ADX Regime Filter Sweep (5yr data)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_equity_comparison(results, filename):
    """Overlay equity curves for baseline vs best ADX filter."""
    baseline = results[0]
    best = max(results[1:], key=lambda r: r["sharpe"])

    fig, ax = plt.subplots(figsize=(16, 6))

    for r, label, color, lw in [
        (baseline, f"Baseline (no ADX filter)", 'gray', 1.5),
        (best, f"ADX <= {int(best['adx_threshold'])}", 'blue', 2.0),
    ]:
        trades = r["trades"]
        if not trades:
            continue
        pnls = [t.pnl for t in trades]
        equity = np.cumsum([10_000.0] + pnls)
        dates = [trades[0].entry_time] + [t.exit_time for t in trades]
        ax.plot(dates, equity, color=color, linewidth=lw, label=f"{label} (Sharpe={r['sharpe']:.3f}, {r['n_trades']} trades)")

    ax.axhline(y=10_000, color='black', linewidth=0.5, linestyle='--')
    ax.set_title("Equity Curve: Baseline vs Best ADX Filter")
    ax.set_ylabel("Equity ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_three_equity(baseline, best_gate, best_sizing, filename):
    """Overlay equity curves for baseline vs gate vs sizing."""
    fig, ax = plt.subplots(figsize=(16, 6))

    for r, label, color, lw in [
        (baseline, "Baseline (no ADX)", 'gray', 1.5),
        (best_gate, f"Gate ADX<={int(best_gate['adx_threshold'])}", 'red', 1.8),
        (best_sizing, f"Sizing @{int(best_sizing['adx_threshold'])}", 'blue', 2.0),
    ]:
        trades = r["trades"]
        if not trades:
            continue
        pnls = [t.pnl for t in trades]
        equity = np.cumsum([10_000.0] + pnls)
        dates = [trades[0].entry_time] + [t.exit_time for t in trades]
        ax.plot(dates, equity, color=color, linewidth=lw,
                label=f"{label} (Sharpe={r['sharpe']:.3f}, {r['n_trades']} trades, ${r['total_pnl']:+,.0f})")

    ax.axhline(y=10_000, color='black', linewidth=0.5, linestyle='--')
    ax.set_title("Equity Curve: Baseline vs Gate vs Sizing")
    ax.set_ylabel("Equity ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def main():
    print("=" * 70)
    print("  Strategy A — ADX Regime Filter Validation")
    print("=" * 70)

    # Load data
    t0 = time.time()
    df, vix = load_data_2yr()
    n_days = len(set(df.index.date))
    print(f"\nData: {len(df):,} RTH 1-min bars, {n_days} trading days")
    print(f"  Range: {df.index[0].date()} to {df.index[-1].date()}")
    if vix is not None:
        print(f"  VIX: {len(vix)} days")
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Load base params
    with open(PARAMS_PATH) as f:
        base_params = json.load(f)
    print(f"\nBase params: bb={base_params['bb_period']}/{base_params['bb_sigma']}, "
          f"atr={base_params['atr_stop_multiplier']}, "
          f"vwap={base_params['vwap_deviation_entry']}, "
          f"tp={base_params['take_profit_pct']}")

    # Precompute indicators once
    print("\nPrecomputing indicators...")
    t_pre = time.time()
    ema_periods = [base_params["trend_ema_period"]] if base_params.get("trend_ema_period", 0) > 0 else []
    precomputed = precompute_indicators(
        df, vix_series=vix,
        bb_periods=[base_params["bb_period"]],
        ema_periods=ema_periods,
        atr_period=14,
        trade_start=(base_params["trade_start_hour"], base_params["trade_start_minute"]),
        trade_end=(base_params["trade_end_hour"], base_params["trade_end_minute"]),
    )
    print(f"  Done in {time.time()-t_pre:.1f}s")

    # ADX distribution analysis
    adx_vals = precomputed["adx_vals"]
    valid_adx = adx_vals[~np.isnan(adx_vals)]
    print(f"\nADX distribution ({len(valid_adx):,} bars):")
    for pct in [10, 25, 50, 75, 90]:
        print(f"  P{pct}: {np.percentile(valid_adx, pct):.1f}")
    print(f"  Mean: {valid_adx.mean():.1f}, Std: {valid_adx.std():.1f}")

    # -- 1. Gate mode sweep ---------------------------------------------
    thresholds = [0, 20, 25, 30, 35, 40]

    print(f"\n{'-'*70}")
    print(f"  1. GATE MODE (binary on/off)")
    print(f"{'-'*70}")
    print(f"  {'Threshold':>10} {'Trades':>7} {'Sharpe':>8} {'WR':>7} {'PF':>7} {'PnL':>10} {'MaxDD':>7}")
    print("  " + "-" * 60)

    gate_results = []
    for thresh in thresholds:
        t_start = time.time()
        r = run_with_adx_threshold(df, vix, base_params, thresh, precomputed, adx_sizing=False)
        elapsed = time.time() - t_start
        gate_results.append(r)

        label = "OFF" if thresh == 0 else f"ADX<={thresh}"
        print(f"  {label:>10} {r['n_trades']:7d} {r['sharpe']:+8.3f} "
              f"{r['win_rate']*100:6.1f}% {r['profit_factor']:6.2f} "
              f"${r['total_pnl']:+9,.0f} {r['max_drawdown']*100:6.1f}%  "
              f"({elapsed:.1f}s)")

    # -- 2. Sizing mode sweep ------------------------------------------
    # Higher thresholds make more sense for sizing — ADX linearly scales
    # position from 100% at ADX=0 down to 0% at threshold
    sizing_thresholds = [25, 30, 35, 40, 50]

    print(f"\n{'-'*70}")
    print(f"  2. SIZING MODE (linear scale: size = (threshold - ADX) / threshold)")
    print(f"{'-'*70}")
    print(f"  {'Threshold':>10} {'Trades':>7} {'Sharpe':>8} {'WR':>7} {'PF':>7} {'PnL':>10} {'MaxDD':>7}")
    print("  " + "-" * 60)

    sizing_results = []
    for thresh in sizing_thresholds:
        t_start = time.time()
        r = run_with_adx_threshold(df, vix, base_params, thresh, precomputed, adx_sizing=True)
        r["label"] = f"size@{thresh}"
        elapsed = time.time() - t_start
        sizing_results.append(r)

        print(f"  {"size@"+str(thresh):>10} {r['n_trades']:7d} {r['sharpe']:+8.3f} "
              f"{r['win_rate']*100:6.1f}% {r['profit_factor']:6.2f} "
              f"${r['total_pnl']:+9,.0f} {r['max_drawdown']*100:6.1f}%  "
              f"({elapsed:.1f}s)")

    # -- Summary -------------------------------------------------------
    baseline = gate_results[0]
    best_gate = max(gate_results[1:], key=lambda r: r["sharpe"])
    best_sizing = max(sizing_results, key=lambda r: r["sharpe"])

    print(f"\n{'='*70}")
    print(f"  RESULTS COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Mode':<20} {'Sharpe':>8} {'Trades':>7} {'WR':>7} {'PF':>7} {'PnL':>10} {'MaxDD':>7}")
    print("  " + "-" * 70)

    for label, r in [
        ("Baseline (no ADX)", baseline),
        (f"Gate ADX<={int(best_gate['adx_threshold'])}", best_gate),
        (f"Sizing @{int(best_sizing['adx_threshold'])}", best_sizing),
    ]:
        print(f"  {label:<20} {r['sharpe']:+8.3f} {r['n_trades']:7d} "
              f"{r['win_rate']*100:6.1f}% {r['profit_factor']:6.2f} "
              f"${r['total_pnl']:+9,.0f} {r['max_drawdown']*100:6.1f}%")

    # Generate plots
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\nGenerating plots...")
    plot_results(gate_results, OUT_DIR / "01_adx_gate_sweep.png")

    # Equity comparison: baseline vs best gate vs best sizing
    plot_three_equity(baseline, best_gate, best_sizing, OUT_DIR / "02_equity_comparison.png")

    print(f"\nAll outputs saved to: {OUT_DIR}/")
    print(f"Total runtime: {time.time()-t0:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
