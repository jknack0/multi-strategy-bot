"""
08 — Trading Window Comparison for Strategy A.

Tests whether avoiding the first 90 min after open and last 90 min before close
improves mean reversion performance.

Windows tested:
  - Current:  10:00 - 14:00 ET (4.0 hrs)
  - Proposed: 11:00 - 14:30 ET (3.5 hrs)
  - Wide:     10:00 - 14:30 ET (4.5 hrs)
  - Late:     10:30 - 14:30 ET (4.0 hrs)
  - Narrow:   11:00 - 14:00 ET (3.0 hrs)
  - Midday:   11:00 - 13:30 ET (2.5 hrs)

Usage:
    uv run python notebooks/08_trading_window_test.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtesting.metrics import annualized_sharpe
from strategies.mean_reversion import (
    MESMeanReversionStrategy,
    Side,
    precompute_indicators,
)

ET = pytz.timezone("US/Eastern")

TRADING_WINDOWS = {
    "10:00-14:00 (current)":  ((10, 0),  (14, 0)),
    "11:00-14:30 (proposed)": ((11, 0),  (14, 30)),
    "10:00-14:30 (wide)":     ((10, 0),  (14, 30)),
    "10:30-14:30 (late)":     ((10, 30), (14, 30)),
    "11:00-14:00 (narrow)":   ((11, 0),  (14, 0)),
    "11:00-13:30 (midday)":   ((11, 0),  (13, 30)),
}

# Parameter sets to test each window with
PARAM_SETS = {
    "default": {
        "bb_period": 20, "bb_sigma": 2.0, "atr_stop_multiplier": 3.5,
        "vwap_deviation_entry": 1.0, "trend_ema_period": 0,
        "take_profit_pct": 1.0, "trailing_stop_factor": 0.6,
        "min_rr_ratio": 0.0, "long_only": False,
    },
    "long_only": {
        "bb_period": 20, "bb_sigma": 2.0, "atr_stop_multiplier": 3.5,
        "vwap_deviation_entry": 1.0, "trend_ema_period": 0,
        "take_profit_pct": 1.0, "trailing_stop_factor": 0.6,
        "min_rr_ratio": 0.0, "long_only": True,
    },
    "tight_bb": {
        "bb_period": 15, "bb_sigma": 1.5, "atr_stop_multiplier": 2.5,
        "vwap_deviation_entry": 0.75, "trend_ema_period": 0,
        "take_profit_pct": 0.75, "trailing_stop_factor": 0.4,
        "min_rr_ratio": 0.0, "long_only": False,
    },
    "tight_bb_long": {
        "bb_period": 15, "bb_sigma": 1.5, "atr_stop_multiplier": 2.5,
        "vwap_deviation_entry": 0.75, "trend_ema_period": 0,
        "take_profit_pct": 0.75, "trailing_stop_factor": 0.4,
        "min_rr_ratio": 0.0, "long_only": True,
    },
    "wide_bb": {
        "bb_period": 30, "bb_sigma": 2.5, "atr_stop_multiplier": 3.5,
        "vwap_deviation_entry": 1.25, "trend_ema_period": 0,
        "take_profit_pct": 1.0, "trailing_stop_factor": 0.6,
        "min_rr_ratio": 0.0, "long_only": False,
    },
    "wide_bb_long": {
        "bb_period": 30, "bb_sigma": 2.5, "atr_stop_multiplier": 3.5,
        "vwap_deviation_entry": 1.25, "trend_ema_period": 0,
        "take_profit_pct": 1.0, "trailing_stop_factor": 0.6,
        "min_rr_ratio": 0.0, "long_only": True,
    },
}


def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only RTH bars (9:00-16:00 ET) with 30min pre-market warmup."""
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


def run_backtest(df, vix_series, params, trade_start, trade_end):
    """Run a single backtest with specific trading window."""
    full_params = MESMeanReversionStrategy.DEFAULT_PARAMS.copy()
    full_params.update(params)
    full_params["trade_start_hour"] = trade_start[0]
    full_params["trade_start_minute"] = trade_start[1]
    full_params["trade_end_hour"] = trade_end[0]
    full_params["trade_end_minute"] = trade_end[1]

    ema_periods = [full_params["trend_ema_period"]] if full_params.get("trend_ema_period", 0) > 0 else []
    precomputed = precompute_indicators(
        df, vix_series=vix_series,
        bb_periods=[full_params["bb_period"]],
        ema_periods=ema_periods,
        atr_period=14,
        trade_start=trade_start,
        trade_end=trade_end,
    )

    capital = 10_000.0
    strat = MESMeanReversionStrategy(params=full_params, capital=capital)
    strat.generate_signals_fast(df, precomputed=precomputed)

    n_trades = len(strat.trades)
    if n_trades < 2:
        return {
            "n_trades": n_trades, "sharpe": float("nan"),
            "win_rate": 0.0, "total_pnl": 0.0,
            "avg_pnl": 0.0, "max_dd": 0.0,
            "profit_factor": 0.0, "n_longs": 0, "n_shorts": 0,
        }

    trade_pnls = np.array([t.pnl for t in strat.trades])
    n_longs = sum(1 for t in strat.trades if t.side == Side.LONG)
    n_shorts = sum(1 for t in strat.trades if t.side == Side.SHORT)

    equity = np.cumsum(np.concatenate([[capital], trade_pnls]))
    returns = np.diff(equity) / equity[:-1]
    sharpe = annualized_sharpe(returns)

    wins = trade_pnls[trade_pnls > 0]
    losses = trade_pnls[trade_pnls <= 0]
    win_rate = len(wins) / n_trades if n_trades > 0 else 0.0
    total_pnl = float(trade_pnls.sum())
    avg_pnl = float(trade_pnls.mean())

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(dd.min())

    # Profit factor
    gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 0.001
    pf = gross_profit / gross_loss

    return {
        "n_trades": n_trades, "sharpe": sharpe,
        "win_rate": win_rate, "total_pnl": total_pnl,
        "avg_pnl": avg_pnl, "max_dd": max_dd,
        "profit_factor": pf, "n_longs": n_longs, "n_shorts": n_shorts,
    }


def main():
    t0 = time.time()
    print("=" * 90)
    print("  Trading Window Comparison — Strategy A Mean Reversion")
    print("=" * 90)

    # Load data
    from data.questdb_client import QuestDBClient
    qdb = QuestDBClient()
    df_raw = qdb.get_bars("MES", "1min", limit=500_000)
    if df_raw.empty:
        raise RuntimeError("No 1-min data in QuestDB")
    df = filter_rth(df_raw)
    print(f"\nData: {len(df_raw)} raw 1-min bars -> {len(df)} RTH bars")
    print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Trading days: {len(set(df.index.date))}")

    # Load VIX
    vix_df = qdb.get_bars("VIX", "1day", limit=500_000)
    if not vix_df.empty:
        vix_series = pd.Series(vix_df["close"].values, index=vix_df.index.date, name="vix")
        print(f"  VIX: {len(vix_series)} days (range: {vix_series.min():.1f}-{vix_series.max():.1f})")
    else:
        print("  [WARN] No VIX data, using default 20.0")
        vix_series = None

    # Run all combinations
    results = []
    total_combos = len(TRADING_WINDOWS) * len(PARAM_SETS)
    combo_num = 0

    for window_name, (start, end) in TRADING_WINDOWS.items():
        for param_name, params in PARAM_SETS.items():
            combo_num += 1
            t_start = time.time()
            result = run_backtest(df, vix_series, params, start, end)
            elapsed = time.time() - t_start
            result["window"] = window_name
            result["params"] = param_name
            results.append(result)
            print(f"  [{combo_num:2d}/{total_combos}] {window_name:25s} | {param_name:15s} | "
                  f"Sharpe={result['sharpe']:+7.3f}  Trades={result['n_trades']:4d}  "
                  f"WR={result['win_rate']:.0%}  PnL=${result['total_pnl']:+8.1f}  ({elapsed:.1f}s)")

    # Summary table grouped by window
    print(f"\n{'='*90}")
    print("  RESULTS BY TRADING WINDOW (averaged across param sets)")
    print("=" * 90)
    print(f"  {'Window':25s} {'Avg Sharpe':>10} {'Avg WR':>8} {'Avg PnL':>10} {'Avg Trades':>10} {'Best Sharpe':>11}")
    print("  " + "-" * 80)

    for window_name in TRADING_WINDOWS:
        window_results = [r for r in results if r["window"] == window_name]
        avg_sharpe = np.nanmean([r["sharpe"] for r in window_results])
        avg_wr = np.mean([r["win_rate"] for r in window_results])
        avg_pnl = np.mean([r["total_pnl"] for r in window_results])
        avg_trades = np.mean([r["n_trades"] for r in window_results])
        best_sharpe = np.nanmax([r["sharpe"] for r in window_results])
        print(f"  {window_name:25s} {avg_sharpe:+10.3f} {avg_wr:7.0%} ${avg_pnl:+9.1f} {avg_trades:10.0f} {best_sharpe:+11.3f}")

    # Best overall combos
    valid = [r for r in results if not np.isnan(r["sharpe"])]
    valid.sort(key=lambda x: x["sharpe"], reverse=True)

    print(f"\n{'='*90}")
    print("  TOP 10 OVERALL COMBINATIONS")
    print("=" * 90)
    print(f"  {'Rank':>4} {'Window':25s} {'Params':15s} {'Sharpe':>8} {'WR':>6} {'Trades':>7} {'PnL':>10} {'PF':>6} {'MaxDD':>8} {'L/S':>7}")
    print("  " + "-" * 105)
    for i, r in enumerate(valid[:10]):
        print(f"  {i+1:4d} {r['window']:25s} {r['params']:15s} "
              f"{r['sharpe']:+8.3f} {r['win_rate']:5.0%} {r['n_trades']:7d} "
              f"${r['total_pnl']:+9.1f} {r['profit_factor']:6.2f} "
              f"{r['max_dd']:+7.1%} {r['n_longs']:3d}/{r['n_shorts']:<3d}")

    # Worst combos for contrast
    print(f"\n  BOTTOM 5:")
    print("  " + "-" * 105)
    for i, r in enumerate(valid[-5:]):
        print(f"  {len(valid)-4+i:4d} {r['window']:25s} {r['params']:15s} "
              f"{r['sharpe']:+8.3f} {r['win_rate']:5.0%} {r['n_trades']:7d} "
              f"${r['total_pnl']:+9.1f} {r['profit_factor']:6.2f} "
              f"{r['max_dd']:+7.1%} {r['n_longs']:3d}/{r['n_shorts']:<3d}")

    total = time.time() - t0
    print(f"\n  Total runtime: {total:.1f}s")
    print("=" * 90)


if __name__ == "__main__":
    main()
