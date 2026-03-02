"""
11 — Combined Portfolio Backtest: Run both optimized strategies together.

Runs Strategy A (BB/VWAP Mean Reversion) and Strategy B (VIX-Adaptive ORB)
with their saved optimized params, then combines into a portfolio with
capital allocation. Shows individual + combined metrics.

Usage:
    uv run python notebooks/11_combined_portfolio_backtest.py
    uv run python notebooks/11_combined_portfolio_backtest.py --days 252
    uv run python notebooks/11_combined_portfolio_backtest.py --days 504
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtesting.metrics import (
    compute_all_metrics,
    annualized_sharpe,
    max_drawdown,
    max_drawdown_detail,
    max_drawdown_duration,
    profit_factor,
    avg_trade_pnl,
)
from backtesting.monte_carlo import MonteCarloSimulator
from strategies.mean_reversion import MESMeanReversionStrategy
from strategies.orb_strategy import VIXAdaptiveORBStrategy
from indicators.relative_volume import RelativeVolume

ET = pytz.timezone("US/Eastern")
ROOT = Path(__file__).resolve().parent.parent
PARQUET_PATH = ROOT / "data" / "MES_1min.parquet"


# ── Data ────────────────────────────────────────────────────────────

def load_data(max_days: int = 0):
    """Load 1-min data, filter RTH, resample to 5-min, load VIX."""
    df = pd.read_parquet(PARQUET_PATH)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    print(f"  Loaded {len(df):,} bars from Parquet")

    # RTH filter (9:00-16:00 ET)
    idx = df.index
    if idx.tz is None:
        et_idx = idx.tz_localize("UTC").tz_convert(ET)
    else:
        et_idx = idx.tz_convert(ET)
    mask = (et_idx.hour >= 9) & (et_idx.hour < 16)
    df = df.loc[mask].copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(ET)
    elif str(df.index.tz) != "US/Eastern":
        df.index = df.index.tz_convert(ET)

    # Trim to recent N days
    if max_days > 0:
        unique_dates = sorted(set(df.index.date))
        if len(unique_dates) > max_days:
            cutoff = unique_dates[-max_days]
            df = df[df.index.date >= cutoff]
            print(f"  Trimmed to last {max_days} trading days ({len(df):,} bars)")

    # 5-min resample for Strategy A
    df_5min = df.resample("5min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()

    # VIX
    from data.questdb_client import QuestDBClient
    qdb = QuestDBClient()
    vix_df = qdb.get_bars("VIX", "1day", limit=10_000)
    if not vix_df.empty:
        vix = pd.Series(vix_df["close"].values, index=vix_df.index.date, name="vix")
    else:
        vix = pd.Series(dtype=float)

    return df, df_5min, vix


def compute_daily_pnl(trades, capital):
    """Convert trade list to daily PnL series."""
    daily = {}
    for t in trades:
        entry = t.entry_time
        if hasattr(entry, "date"):
            d = entry.date()
        else:
            d = entry
        d = pd.Timestamp(d)
        daily[d] = daily.get(d, 0.0) + t.pnl
    return pd.Series(daily, name="pnl").sort_index()


def print_metrics_table(name, trades, capital):
    """Print a detailed metrics block for a strategy."""
    if len(trades) < 2:
        print(f"  {name}: Not enough trades ({len(trades)})")
        return {}

    trade_pnls = np.array([t.pnl for t in trades])
    equity = np.cumsum(np.concatenate([[capital], trade_pnls]))
    returns = np.diff(equity) / equity[:-1]
    metrics = compute_all_metrics(returns, trade_pnls, equity)
    pnl_detail = avg_trade_pnl(trade_pnls)
    dd_detail = max_drawdown_detail(equity)

    total_pnl = np.sum(trade_pnls)
    winners = trade_pnls[trade_pnls > 0]
    losers = trade_pnls[trade_pnls < 0]

    print(f"  {name}")
    print(f"    Trades:       {len(trades):>6d}")
    print(f"    Total PnL:    ${total_pnl:>+10,.2f}  ({total_pnl/capital*100:+.1f}%)")
    print(f"    Sharpe:       {metrics['sharpe']:>+10.3f}")
    print(f"    Sortino:      {metrics['sortino']:>+10.3f}")
    print(f"    Profit Factor:{metrics['profit_factor']:>10.2f}")
    print(f"    Win Rate:     {metrics['win_rate']*100:>9.1f}%")
    print(f"    Avg Win:      ${pnl_detail['avg_win']:>+10,.2f}")
    print(f"    Avg Loss:     ${pnl_detail['avg_loss']:>+10,.2f}")
    print(f"    R:R Ratio:    {pnl_detail['reward_risk_ratio']:>10.2f}")
    print(f"    Max DD:       {dd_detail['max_dd_pct']*100:>9.1f}%  (${dd_detail['max_dd_dollars']:>,.0f})")
    print(f"    Expectancy:   ${metrics['expectancy']:>+10,.2f}/trade")

    # Long/short breakdown if trades have side attribute
    if hasattr(trades[0], "side"):
        longs = [t for t in trades if t.side == "LONG"]
        shorts = [t for t in trades if t.side == "SHORT"]
        if longs:
            l_wins = sum(1 for t in longs if t.pnl > 0)
            print(f"    Longs:        {len(longs):>6d}  WR {l_wins/len(longs)*100:.1f}%  PnL ${sum(t.pnl for t in longs):+,.0f}")
        if shorts:
            s_wins = sum(1 for t in shorts if t.pnl > 0)
            print(f"    Shorts:       {len(shorts):>6d}  WR {s_wins/len(shorts)*100:.1f}%  PnL ${sum(t.pnl for t in shorts):+,.0f}")

    return metrics


# ── Main ────────────────────────────────────────────────────────────

def main(max_days: int = 0):
    print("=" * 70)
    print("  Combined Portfolio Backtest: Strategy A + Strategy B")
    print("=" * 70)

    # ── Load data ──
    t0 = time.time()
    df_1min, df_5min, vix = load_data(max_days)
    n_days = len(set(df_1min.index.date))
    dates = sorted(set(df_1min.index.date))
    print(f"  {len(df_1min):,} 1-min bars, {len(df_5min):,} 5-min bars, {n_days} trading days")
    print(f"  Date range: {min(dates)} to {max(dates)}")
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # ── Load params ──
    with open(ROOT / "config" / "strategy_a_params.json") as f:
        params_a = json.load(f)
    with open(ROOT / "config" / "strategy_b_params.json") as f:
        params_b = json.load(f)

    print(f"\n  Strategy A params: {json.dumps(params_a)}")
    print(f"  Strategy B params: {json.dumps(params_b)}")

    # ── Capital allocation ──
    total_capital = 20_000.0
    alloc_a = 0.50  # 50% to Strategy A
    alloc_b = 0.50  # 50% to Strategy B
    cap_a = total_capital * alloc_a
    cap_b = total_capital * alloc_b
    print(f"\n  Total capital: ${total_capital:,.0f}")
    print(f"  Allocation: A={alloc_a:.0%} (${cap_a:,.0f}), B={alloc_b:.0%} (${cap_b:,.0f})")

    # ── Run Strategy A ──
    # Use bar size from params (default 1-min for more trades)
    a_bar_size = params_a.get("bar_size", "1min")
    a_data = df_1min if a_bar_size == "1min" else df_5min
    print(f"\n{'-'*70}")
    print(f"  1. Strategy A: BB/VWAP Mean Reversion ({a_bar_size})")
    print("-" * 70)
    t1 = time.time()
    strat_a = MESMeanReversionStrategy(params=params_a, capital=cap_a)
    strat_a.generate_signals(a_data, vix_series=vix)
    trades_a = strat_a.trades
    print(f"  Ran in {time.time()-t1:.1f}s")
    metrics_a = print_metrics_table("Strategy A", trades_a, cap_a)

    # ── Run Strategy B ──
    print(f"\n{'-'*70}")
    print("  2. Strategy B: VIX-Adaptive ORB (1-min)")
    print("-" * 70)
    t2 = time.time()
    strat_b = VIXAdaptiveORBStrategy(params=params_b, capital=cap_b)
    strat_b.generate_signals(df_1min, vix_series=vix)
    trades_b = strat_b.trades
    print(f"  Ran in {time.time()-t2:.1f}s")
    metrics_b = print_metrics_table("Strategy B", trades_b, cap_b)

    # ── Combined Portfolio ──
    print(f"\n{'-'*70}")
    print("  3. Combined Portfolio")
    print("-" * 70)

    # Build daily PnL series for each strategy
    daily_a = compute_daily_pnl(trades_a, cap_a)
    daily_b = compute_daily_pnl(trades_b, cap_b)

    # Align to common date range
    all_dates = pd.bdate_range(min(dates), max(dates))
    daily_a = daily_a.reindex(all_dates, fill_value=0.0)
    daily_b = daily_b.reindex(all_dates, fill_value=0.0)
    daily_combined = daily_a + daily_b

    # Combined equity curve
    equity_a = cap_a + np.cumsum(daily_a.values)
    equity_b = cap_b + np.cumsum(daily_b.values)
    equity_combined = total_capital + np.cumsum(daily_combined.values)

    # Combined metrics
    returns_combined = daily_combined.values / np.concatenate([[total_capital], equity_combined[:-1]])
    combined_pnls = daily_combined.values[daily_combined.values != 0]

    sharpe_c = annualized_sharpe(returns_combined)
    dd_c = max_drawdown_detail(equity_combined)
    pf_c = profit_factor(combined_pnls)

    total_pnl_a = np.sum([t.pnl for t in trades_a])
    total_pnl_b = np.sum([t.pnl for t in trades_b])
    total_pnl_combined = total_pnl_a + total_pnl_b

    print(f"  Combined Sharpe:     {sharpe_c:+.3f}")
    print(f"  Combined PnL:        ${total_pnl_combined:+,.2f}  ({total_pnl_combined/total_capital*100:+.1f}%)")
    print(f"    from A:            ${total_pnl_a:+,.2f}")
    print(f"    from B:            ${total_pnl_b:+,.2f}")
    print(f"  Combined Max DD:     {dd_c['max_dd_pct']*100:.1f}%  (${dd_c['max_dd_dollars']:,.0f})")
    print(f"  Combined PF:         {pf_c:.2f}")
    print(f"  Final equity:        ${equity_combined[-1]:,.2f}")

    # ── Correlation ──
    print(f"\n{'-'*70}")
    print("  4. Strategy Correlation")
    print("-" * 70)

    active = (daily_a != 0) | (daily_b != 0)
    if active.sum() > 5:
        pearson = daily_a[active].corr(daily_b[active])
        spearman = daily_a[active].rank().corr(daily_b[active].rank())
    else:
        pearson = spearman = 0.0

    # Individual daily-return Sharpes
    ret_a = daily_a.values / np.concatenate([[cap_a], equity_a[:-1]])
    ret_b = daily_b.values / np.concatenate([[cap_b], equity_b[:-1]])
    sharpe_a = annualized_sharpe(ret_a)
    sharpe_b = annualized_sharpe(ret_b)

    print(f"  Pearson correlation:  {pearson:+.3f}  {'GOOD' if abs(pearson) < 0.30 else 'WARN'}")
    print(f"  Spearman correlation: {spearman:+.3f}  {'GOOD' if abs(spearman) < 0.30 else 'WARN'}")
    print(f"  Strategy A Sharpe:    {sharpe_a:+.3f}")
    print(f"  Strategy B Sharpe:    {sharpe_b:+.3f}")
    print(f"  Combined Sharpe:      {sharpe_c:+.3f}")
    benefit = sharpe_c > max(sharpe_a, sharpe_b)
    print(f"  Diversification:      {'YES — combined > best individual' if benefit else 'NO — combined <= best individual'}")

    # ── Monthly breakdown ──
    print(f"\n{'-'*70}")
    print("  5. Monthly PnL Breakdown")
    print("-" * 70)

    monthly_a = daily_a.resample("ME").sum()
    monthly_b = daily_b.resample("ME").sum()
    monthly_c = daily_combined.resample("ME").sum()

    print(f"  {'Month':>10s} {'Strat A':>10s} {'Strat B':>10s} {'Combined':>10s} {'Both +':>7s}")
    print("  " + "-" * 50)
    for dt in monthly_c.index:
        label = dt.strftime("%Y-%m")
        a_val = monthly_a.get(dt, 0)
        b_val = monthly_b.get(dt, 0)
        c_val = monthly_c.get(dt, 0)
        both_pos = "  Y" if a_val > 0 and b_val > 0 else "  -" if a_val >= 0 or b_val >= 0 else "  N"
        print(f"  {label:>10s} ${a_val:>+9,.0f} ${b_val:>+9,.0f} ${c_val:>+9,.0f} {both_pos}")

    pos_months = sum(1 for v in monthly_c.values if v > 0)
    total_months = len(monthly_c)
    print(f"\n  Positive months: {pos_months}/{total_months} ({pos_months/total_months*100:.0f}%)")

    # ── Worst days ──
    print(f"\n{'-'*70}")
    print("  6. Worst 10 Days")
    print("-" * 70)
    worst = daily_combined.sort_values().head(10)
    print(f"  {'Date':>12s} {'Combined':>10s} {'Strat A':>10s} {'Strat B':>10s}")
    print("  " + "-" * 45)
    for dt, val in worst.items():
        a_val = daily_a.get(dt, 0)
        b_val = daily_b.get(dt, 0)
        print(f"  {dt.strftime('%Y-%m-%d'):>12s} ${val:>+9,.2f} ${a_val:>+9,.2f} ${b_val:>+9,.2f}")

    # ── Monte Carlo on combined ──
    print(f"\n{'-'*70}")
    print("  7. Monte Carlo (combined daily PnL, 1000 paths)")
    print("-" * 70)

    nonzero_daily = daily_combined.values[daily_combined.values != 0]
    if len(nonzero_daily) >= 10:
        mc = MonteCarloSimulator(n_paths=1000, seed=42)
        mc_result = mc.simulate(nonzero_daily, initial_capital=total_capital)

        print(f"\n  {'Percentile':>12s} {'Final Equity':>14s} {'Max DD':>8s} {'Sharpe':>8s}")
        print("  " + "-" * 45)
        for p in [5, 25, 50, 75, 95]:
            dd_val = mc_result["max_drawdown_pcts"][p]
            dd_pct = dd_val * 100 if dd_val <= 1.0 else dd_val
            print(
                f"  {p:11d}th ${mc_result['final_equity_pcts'][p]:13,.2f} "
                f"{dd_pct:7.1f}% "
                f"{mc_result['sharpe_pcts'][p]:8.3f}"
            )
        prob = mc_result["prob_positive"]
        prob_pct = prob * 100 if prob <= 1.0 else prob
        print(f"\n  Prob. positive: {prob_pct:.1f}%")

    # ── Summary ──
    print(f"\n{'='*70}")
    print("  PORTFOLIO SUMMARY")
    print("=" * 70)
    print(f"  Capital:     ${total_capital:>10,.0f}")
    print(f"  Period:      {min(dates)} to {max(dates)} ({n_days} days)")
    print(f"  Total PnL:   ${total_pnl_combined:>+10,.2f} ({total_pnl_combined/total_capital*100:+.1f}%)")
    print(f"  Sharpe:      {sharpe_c:>+10.3f}")
    print(f"  Max DD:      {dd_c['max_dd_pct']*100:>9.1f}%")
    print(f"  Correlation: {pearson:>+10.3f}")
    print(f"  A trades:    {len(trades_a):>10d}  (${total_pnl_a:+,.0f})")
    print(f"  B trades:    {len(trades_b):>10d}  (${total_pnl_b:+,.0f})")
    print(f"  Total trades:{len(trades_a)+len(trades_b):>10d}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=0, help="Limit to N most recent trading days")
    args = parser.parse_args()
    main(max_days=args.days)
