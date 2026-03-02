"""
09 — Strategy A Entry/Exit Diagnostics.

Visualizes where mean reversion enters and exits, broken down by:
  1. Time-of-day PnL heatmap
  2. Exit reason breakdown (stop_loss vs take_profit vs time_stop vs trailing)
  3. Win/loss by entry hour
  4. Trade duration distribution
  5. Sample days with price + BB + VWAP + entry/exit markers
  6. MFE/MAE analysis (how far trades go right vs wrong)

Saves charts to outputs/strategy_a_diagnostics/

Usage:
    uv run python notebooks/09_strategy_a_diagnostics.py
"""

import sys
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import pytz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategies.mean_reversion import (
    MESMeanReversionStrategy,
    Side,
    precompute_indicators,
)

ET = pytz.timezone("US/Eastern")
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "strategy_a_diagnostics"


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


def run_strategy(df, vix_series, params=None):
    """Run Strategy A and return strategy object with all trades."""
    full_params = MESMeanReversionStrategy.DEFAULT_PARAMS.copy()
    if params:
        full_params.update(params)

    ema_periods = [full_params["trend_ema_period"]] if full_params.get("trend_ema_period", 0) > 0 else []
    precomputed = precompute_indicators(
        df, vix_series=vix_series,
        bb_periods=[full_params["bb_period"]],
        ema_periods=ema_periods,
        atr_period=14,
        trade_start=(full_params["trade_start_hour"], full_params["trade_start_minute"]),
        trade_end=(full_params["trade_end_hour"], full_params["trade_end_minute"]),
    )

    strat = MESMeanReversionStrategy(params=full_params, capital=10_000.0)
    strat.generate_signals_fast(df, precomputed=precomputed)
    return strat, precomputed


def plot_exit_reason_breakdown(trades, filename):
    """Pie chart + bar chart of exit reasons and their PnL."""
    reasons = defaultdict(lambda: {"count": 0, "pnl": 0.0, "wins": 0, "losses": 0})
    for t in trades:
        r = t.exit_reason
        reasons[r]["count"] += 1
        reasons[r]["pnl"] += t.pnl
        if t.pnl > 0:
            reasons[r]["wins"] += 1
        else:
            reasons[r]["losses"] += 1

    labels = list(reasons.keys())
    counts = [reasons[l]["count"] for l in labels]
    pnls = [reasons[l]["pnl"] for l in labels]
    win_rates = [reasons[l]["wins"] / reasons[l]["count"] * 100 for l in labels]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Pie chart of counts
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    axes[0].pie(counts, labels=[f"{l}\n({c})" for l, c in zip(labels, counts)],
                colors=colors, autopct='%1.0f%%', startangle=90)
    axes[0].set_title("Exit Reason Distribution")

    # PnL by reason
    bar_colors = ['green' if p > 0 else 'red' for p in pnls]
    axes[1].bar(labels, pnls, color=bar_colors, edgecolor='black', alpha=0.7)
    axes[1].set_title("Total PnL by Exit Reason")
    axes[1].set_ylabel("PnL ($)")
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    for i, (l, p) in enumerate(zip(labels, pnls)):
        axes[1].text(i, p, f"${p:.0f}", ha='center',
                     va='bottom' if p >= 0 else 'top', fontsize=9)

    # Win rate by reason
    axes[2].bar(labels, win_rates, color='steelblue', edgecolor='black', alpha=0.7)
    axes[2].set_title("Win Rate by Exit Reason")
    axes[2].set_ylabel("Win Rate (%)")
    axes[2].set_ylim(0, 100)
    axes[2].axhline(y=50, color='gray', linewidth=0.5, linestyle='--')
    for i, wr in enumerate(win_rates):
        axes[2].text(i, wr + 1, f"{wr:.0f}%", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_entry_hour_analysis(trades, filename):
    """PnL and trade count by entry hour."""
    hours = defaultdict(lambda: {"count": 0, "pnl": 0.0, "wins": 0, "longs": 0, "shorts": 0})
    for t in trades:
        et = t.entry_time
        if hasattr(et, 'astimezone'):
            try:
                et = et.astimezone(ET)
            except Exception:
                pass
        h = et.hour
        hours[h]["count"] += 1
        hours[h]["pnl"] += t.pnl
        if t.pnl > 0:
            hours[h]["wins"] += 1
        if t.side == Side.LONG:
            hours[h]["longs"] += 1
        else:
            hours[h]["shorts"] += 1

    sorted_hours = sorted(hours.keys())
    counts = [hours[h]["count"] for h in sorted_hours]
    pnls = [hours[h]["pnl"] for h in sorted_hours]
    win_rates = [hours[h]["wins"] / hours[h]["count"] * 100 if hours[h]["count"] > 0 else 0 for h in sorted_hours]
    avg_pnls = [hours[h]["pnl"] / hours[h]["count"] if hours[h]["count"] > 0 else 0 for h in sorted_hours]
    longs = [hours[h]["longs"] for h in sorted_hours]
    shorts = [hours[h]["shorts"] for h in sorted_hours]
    labels = [f"{h}:00" for h in sorted_hours]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Trade count by hour
    axes[0, 0].bar(labels, longs, label='Long', color='green', alpha=0.6)
    axes[0, 0].bar(labels, shorts, bottom=longs, label='Short', color='red', alpha=0.6)
    axes[0, 0].set_title("Trade Count by Entry Hour")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].legend()

    # Total PnL by hour
    bar_colors = ['green' if p > 0 else 'red' for p in pnls]
    axes[0, 1].bar(labels, pnls, color=bar_colors, edgecolor='black', alpha=0.7)
    axes[0, 1].set_title("Total PnL by Entry Hour")
    axes[0, 1].set_ylabel("PnL ($)")
    axes[0, 1].axhline(y=0, color='black', linewidth=0.5)

    # Average PnL per trade by hour
    bar_colors2 = ['green' if p > 0 else 'red' for p in avg_pnls]
    axes[1, 0].bar(labels, avg_pnls, color=bar_colors2, edgecolor='black', alpha=0.7)
    axes[1, 0].set_title("Average PnL per Trade by Entry Hour")
    axes[1, 0].set_ylabel("Avg PnL ($)")
    axes[1, 0].axhline(y=0, color='black', linewidth=0.5)

    # Win rate by hour
    axes[1, 1].bar(labels, win_rates, color='steelblue', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title("Win Rate by Entry Hour")
    axes[1, 1].set_ylabel("Win Rate (%)")
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].axhline(y=50, color='gray', linewidth=0.5, linestyle='--')
    for i, wr in enumerate(win_rates):
        axes[1, 1].text(i, wr + 1, f"{wr:.0f}%", ha='center', fontsize=9)

    plt.suptitle("Strategy A — Entry Hour Analysis", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_trade_duration(trades, filename):
    """Distribution of trade hold times."""
    durations_min = []
    for t in trades:
        dt = (t.exit_time - t.entry_time).total_seconds() / 60
        durations_min.append(dt)

    wins = [d for d, t in zip(durations_min, trades) if t.pnl > 0]
    losses = [d for d, t in zip(durations_min, trades) if t.pnl <= 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    bins = np.arange(0, max(durations_min) + 10, 10)
    axes[0].hist(wins, bins=bins, alpha=0.6, color='green', label=f'Winners ({len(wins)})')
    axes[0].hist(losses, bins=bins, alpha=0.6, color='red', label=f'Losers ({len(losses)})')
    axes[0].set_title("Trade Duration Distribution")
    axes[0].set_xlabel("Duration (minutes)")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    # Duration vs PnL scatter
    pnls = [t.pnl for t in trades]
    colors = ['green' if p > 0 else 'red' for p in pnls]
    axes[1].scatter(durations_min, pnls, c=colors, alpha=0.5, s=20)
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].set_title("Duration vs PnL")
    axes[1].set_xlabel("Duration (minutes)")
    axes[1].set_ylabel("PnL ($)")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_mfe_mae(trades, filename):
    """Max Favorable Excursion vs Max Adverse Excursion analysis."""
    # We need to approximate MFE/MAE from entry/exit since Position tracks max_favorable
    # but TradeRecord doesn't. Use PnL and exit reason as proxy.
    pnls = [t.pnl for t in trades]
    entry_prices = [t.entry_price for t in trades]
    exit_prices = [t.exit_price for t in trades]

    # Points moved
    moves = []
    for t in trades:
        if t.side == Side.LONG:
            move = t.exit_price - t.entry_price
        else:
            move = t.entry_price - t.exit_price
        moves.append(move)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # PnL distribution
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    all_pnls = sorted(pnls)
    bins = 30
    axes[0].hist(wins, bins=bins, alpha=0.6, color='green', label=f'Winners ({len(wins)})')
    axes[0].hist(losses, bins=bins, alpha=0.6, color='red', label=f'Losers ({len(losses)})')
    axes[0].axvline(x=0, color='black', linewidth=1)
    axes[0].set_title("PnL Distribution")
    axes[0].set_xlabel("PnL ($)")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    axes[0].axvline(x=avg_win, color='green', linewidth=1, linestyle='--', label=f'Avg Win: ${avg_win:.1f}')
    axes[0].axvline(x=avg_loss, color='red', linewidth=1, linestyle='--', label=f'Avg Loss: ${avg_loss:.1f}')
    axes[0].legend()

    # Points moved by exit reason
    reasons = set(t.exit_reason for t in trades)
    reason_moves = {r: [] for r in reasons}
    for t, m in zip(trades, moves):
        reason_moves[t.exit_reason].append(m)

    positions = []
    labels = []
    colors = []
    color_map = {'stop_loss': 'red', 'take_profit': 'green', 'time_stop': 'orange', 'trailing_stop': 'blue'}
    for r in sorted(reasons):
        positions.append(reason_moves[r])
        labels.append(f"{r}\n(n={len(reason_moves[r])})")
        colors.append(color_map.get(r, 'gray'))

    bp = axes[1].boxplot(positions, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].set_title("Points Moved by Exit Reason")
    axes[1].set_ylabel("Points (+ = favorable)")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_equity_curve(trades, capital, filename):
    """Cumulative PnL / equity curve."""
    if not trades:
        return
    pnls = [t.pnl for t in trades]
    equity = np.cumsum([capital] + pnls)
    dates = [trades[0].entry_time] + [t.exit_time for t in trades]

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={'height_ratios': [3, 1]})

    # Equity curve
    axes[0].plot(dates, equity, 'b-', linewidth=1)
    axes[0].fill_between(dates, capital, equity, where=equity >= capital,
                         color='green', alpha=0.1)
    axes[0].fill_between(dates, capital, equity, where=equity < capital,
                         color='red', alpha=0.1)
    axes[0].axhline(y=capital, color='gray', linewidth=0.5, linestyle='--')
    axes[0].set_title("Strategy A Equity Curve")
    axes[0].set_ylabel("Equity ($)")

    # Drawdown
    peak = np.maximum.accumulate(equity)
    dd_pct = (equity - peak) / peak * 100
    axes[1].fill_between(dates, 0, dd_pct, color='red', alpha=0.3)
    axes[1].plot(dates, dd_pct, 'r-', linewidth=0.5)
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_ylim(min(dd_pct) * 1.1, 1)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_sample_days(df, trades, precomputed, n_days=6, filename=None):
    """Plot price action + BB + VWAP + entries/exits for sample days."""
    trade_dates = set()
    trades_by_date = defaultdict(list)
    for t in trades:
        et = t.entry_time
        if hasattr(et, 'date'):
            d = et.date() if not callable(et.date) else et.date()
        else:
            d = pd.Timestamp(et).date()
        trade_dates.add(d)
        trades_by_date[d].append(t)

    # Pick days with most trades, spread across the date range
    sorted_dates = sorted(trade_dates)
    if len(sorted_dates) <= n_days:
        sample_dates = sorted_dates
    else:
        # Pick evenly spaced dates that have trades
        indices = np.linspace(0, len(sorted_dates) - 1, n_days, dtype=int)
        sample_dates = [sorted_dates[i] for i in indices]

    if not sample_dates:
        print("  No trades to plot sample days")
        return

    n_cols = 2
    n_rows = (len(sample_dates) + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    bb_period = 20  # default
    bb_means = precomputed["bb_means"][bb_period]
    bb_stds = precomputed["bb_stds"][bb_period]
    bb_sigma = 2.0
    vwap_vals = precomputed["vwap_vals"]
    closes = precomputed["closes"]

    for idx, d in enumerate(sample_dates):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        # Get bars for this day
        day_mask = df.index.date == d
        day_df = df.loc[day_mask]
        if day_df.empty:
            continue

        day_indices = np.where(day_mask)[0]
        times = day_df.index

        # Price
        ax.plot(times, day_df["close"].values, 'k-', linewidth=0.8, label='Close')

        # BB bands
        day_bb_mean = bb_means[day_indices]
        day_bb_upper = day_bb_mean + bb_sigma * bb_stds[day_indices]
        day_bb_lower = day_bb_mean - bb_sigma * bb_stds[day_indices]
        ax.fill_between(times, day_bb_lower, day_bb_upper, alpha=0.1, color='blue')
        ax.plot(times, day_bb_upper, 'b--', linewidth=0.5, alpha=0.5)
        ax.plot(times, day_bb_lower, 'b--', linewidth=0.5, alpha=0.5)

        # VWAP
        day_vwap = vwap_vals[day_indices]
        ax.plot(times, day_vwap, 'purple', linewidth=1, alpha=0.7, label='VWAP')

        # Entries and exits
        day_trades = trades_by_date.get(d, [])
        for t in day_trades:
            entry_color = 'green' if t.side == Side.LONG else 'red'
            marker = '^' if t.side == Side.LONG else 'v'
            ax.plot(t.entry_time, t.entry_price, marker, color=entry_color,
                    markersize=12, zorder=5)

            exit_color = 'green' if t.pnl > 0 else 'red'
            ax.plot(t.exit_time, t.exit_price, 'x', color=exit_color,
                    markersize=10, markeredgewidth=2, zorder=5)

            # Draw line from entry to exit
            line_color = 'green' if t.pnl > 0 else 'red'
            ax.plot([t.entry_time, t.exit_time], [t.entry_price, t.exit_price],
                    '-', color=line_color, alpha=0.3, linewidth=1)

            # Annotate PnL
            mid_time = t.entry_time + (t.exit_time - t.entry_time) / 2
            ax.annotate(f"${t.pnl:.0f}\n{t.exit_reason}",
                        xy=(mid_time, (t.entry_price + t.exit_price) / 2),
                        fontsize=7, ha='center', color=line_color)

        n_day_trades = len(day_trades)
        day_pnl = sum(t.pnl for t in day_trades)
        ax.set_title(f"{d} — {n_day_trades} trades, PnL: ${day_pnl:.0f}", fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(len(sample_dates), n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    plt.suptitle("Strategy A — Sample Trading Days (^ = entry, x = exit)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")


def plot_long_vs_short(trades, filename):
    """Compare long vs short performance."""
    longs = [t for t in trades if t.side == Side.LONG]
    shorts = [t for t in trades if t.side == Side.SHORT]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Cumulative PnL
    if longs:
        long_cum = np.cumsum([t.pnl for t in sorted(longs, key=lambda t: t.entry_time)])
        axes[0].plot(range(len(long_cum)), long_cum, 'g-', label=f'Longs ({len(longs)})')
    if shorts:
        short_cum = np.cumsum([t.pnl for t in sorted(shorts, key=lambda t: t.entry_time)])
        axes[0].plot(range(len(short_cum)), short_cum, 'r-', label=f'Shorts ({len(shorts)})')
    axes[0].axhline(y=0, color='black', linewidth=0.5)
    axes[0].set_title("Cumulative PnL: Long vs Short")
    axes[0].set_xlabel("Trade #")
    axes[0].set_ylabel("Cumulative PnL ($)")
    axes[0].legend()

    # Win rate comparison
    long_wr = sum(1 for t in longs if t.pnl > 0) / len(longs) * 100 if longs else 0
    short_wr = sum(1 for t in shorts if t.pnl > 0) / len(shorts) * 100 if shorts else 0
    axes[1].bar(['Long', 'Short'], [long_wr, short_wr],
                color=['green', 'red'], alpha=0.7, edgecolor='black')
    axes[1].set_title("Win Rate: Long vs Short")
    axes[1].set_ylabel("Win Rate (%)")
    axes[1].set_ylim(0, 100)
    axes[1].axhline(y=50, color='gray', linewidth=0.5, linestyle='--')
    for i, wr in enumerate([long_wr, short_wr]):
        axes[1].text(i, wr + 1, f"{wr:.1f}%", ha='center')

    # Avg PnL comparison
    long_avg = np.mean([t.pnl for t in longs]) if longs else 0
    short_avg = np.mean([t.pnl for t in shorts]) if shorts else 0
    colors = ['green' if long_avg > 0 else 'red', 'green' if short_avg > 0 else 'red']
    axes[2].bar(['Long', 'Short'], [long_avg, short_avg],
                color=colors, alpha=0.7, edgecolor='black')
    axes[2].set_title("Avg PnL per Trade: Long vs Short")
    axes[2].set_ylabel("Avg PnL ($)")
    axes[2].axhline(y=0, color='black', linewidth=0.5)
    for i, avg in enumerate([long_avg, short_avg]):
        axes[2].text(i, avg, f"${avg:.1f}", ha='center',
                     va='bottom' if avg >= 0 else 'top')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def print_summary(trades):
    """Print text summary of trade statistics."""
    if not trades:
        print("  No trades!")
        return

    pnls = np.array([t.pnl for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    durations = [(t.exit_time - t.entry_time).total_seconds() / 60 for t in trades]

    print(f"\n  Total trades: {len(trades)}")
    print(f"  Longs: {sum(1 for t in trades if t.side == Side.LONG)}, "
          f"Shorts: {sum(1 for t in trades if t.side == Side.SHORT)}")
    print(f"  Win rate: {len(wins)/len(trades)*100:.1f}%")
    print(f"  Total PnL: ${pnls.sum():.2f}")
    print(f"  Avg PnL/trade: ${pnls.mean():.2f}")
    print(f"  Avg winner: ${wins.mean():.2f}" if len(wins) else "  No winners")
    print(f"  Avg loser: ${losses.mean():.2f}" if len(losses) else "  No losers")
    gross_profit = wins.sum() if len(wins) else 0
    gross_loss = abs(losses.sum()) if len(losses) else 0.001
    print(f"  Profit factor: {gross_profit/gross_loss:.2f}")
    print(f"  Avg duration: {np.mean(durations):.0f} min")
    print(f"  Median duration: {np.median(durations):.0f} min")

    # Exit reason counts
    reasons = defaultdict(int)
    for t in trades:
        reasons[t.exit_reason] += 1
    print(f"\n  Exit reasons:")
    for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
        wr = sum(1 for t in trades if t.exit_reason == r and t.pnl > 0) / c * 100
        avg = np.mean([t.pnl for t in trades if t.exit_reason == r])
        print(f"    {r:15s}: {c:4d} ({c/len(trades)*100:5.1f}%) | WR: {wr:5.1f}% | Avg: ${avg:+.1f}")

    # Per-hour breakdown
    print(f"\n  Entry hour breakdown:")
    hours = defaultdict(lambda: {"n": 0, "pnl": 0.0, "wins": 0})
    for t in trades:
        h = t.entry_time.hour
        hours[h]["n"] += 1
        hours[h]["pnl"] += t.pnl
        if t.pnl > 0:
            hours[h]["wins"] += 1
    for h in sorted(hours):
        d = hours[h]
        wr = d["wins"] / d["n"] * 100
        avg = d["pnl"] / d["n"]
        print(f"    {h:02d}:00 | {d['n']:4d} trades | WR: {wr:5.1f}% | "
              f"Total: ${d['pnl']:+8.1f} | Avg: ${avg:+6.1f}")


def main():
    print("=" * 70)
    print("  Strategy A — Entry/Exit Diagnostics")
    print("=" * 70)

    # Load data
    from data.questdb_client import QuestDBClient
    qdb = QuestDBClient()
    df_raw = qdb.get_bars("MES", "1min", limit=500_000)
    if df_raw.empty:
        raise RuntimeError("No 1-min data in QuestDB")
    df = filter_rth(df_raw)
    print(f"\nData: {len(df_raw)} raw -> {len(df)} RTH 1-min bars")
    print(f"  Range: {df.index[0].date()} to {df.index[-1].date()}")

    # Load VIX
    vix_df = qdb.get_bars("VIX", "1day", limit=500_000)
    if not vix_df.empty:
        vix_series = pd.Series(vix_df["close"].values, index=vix_df.index.date, name="vix")
        print(f"  VIX: {len(vix_series)} days ({vix_series.min():.1f}-{vix_series.max():.1f})")
    else:
        vix_series = None
        print("  [WARN] No VIX data")

    # Create output directory
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load params from config JSON
    import json
    params_path = Path(__file__).resolve().parent.parent / "config" / "strategy_a_params.json"
    with open(params_path) as f:
        config_params = json.load(f)
    print(f"\nLoaded params from {params_path.name}: long_only={config_params.get('long_only')}")

    print("Running Strategy A...")
    strat, precomputed = run_strategy(df, vix_series, params=config_params)
    trades = strat.trades
    print_summary(trades)

    if not trades:
        print("No trades generated - nothing to diagnose!")
        return

    # Generate all diagnostic plots
    print("\nGenerating diagnostic plots...")
    plot_exit_reason_breakdown(trades, OUT_DIR / "01_exit_reasons.png")
    plot_entry_hour_analysis(trades, OUT_DIR / "02_entry_hours.png")
    plot_trade_duration(trades, OUT_DIR / "03_trade_duration.png")
    plot_mfe_mae(trades, OUT_DIR / "04_pnl_distribution.png")
    plot_equity_curve(trades, 10_000.0, OUT_DIR / "05_equity_curve.png")
    plot_long_vs_short(trades, OUT_DIR / "06_long_vs_short.png")
    plot_sample_days(df, trades, precomputed, n_days=8, filename=OUT_DIR / "07_sample_days.png")

    print(f"\nAll diagnostics saved to: {OUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
