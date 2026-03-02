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
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtesting.cpcv_validator import CPCVValidator
from backtesting.metrics import compute_all_metrics, annualized_sharpe
from backtesting.monte_carlo import MonteCarloSimulator
from indicators.relative_volume import RelativeVolume
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


def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only RTH bars (9:00-16:00 ET).

    Includes 9:00-9:29 for ATR/VWAP warmup before the 9:30 OR open.
    Works with both tz-naive UTC and tz-aware ET indexes.
    """
    idx = df.index
    if idx.tz is None:
        # Assume UTC, convert to ET
        et_idx = idx.tz_localize("UTC").tz_convert(ET)
    else:
        et_idx = idx.tz_convert(ET)
    hour = et_idx.hour
    mask = (hour >= 9) & (hour < 16)
    filtered = df.loc[mask].copy()
    # Ensure index is ET-aware for the strategy
    if filtered.index.tz is None:
        filtered.index = filtered.index.tz_localize("UTC").tz_convert(ET)
    elif str(filtered.index.tz) != "US/Eastern":
        filtered.index = filtered.index.tz_convert(ET)
    return filtered


def resample_to_5min(df_1min: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-min RTH bars to 5-min bars for Strategy A."""
    df5 = df_1min.resample("5min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    return df5


PARQUET_PATH = Path(__file__).resolve().parent.parent / "data" / "MES_1min.parquet"


def load_data(use_synthetic: bool, bar_size: str = "5min", max_days: int = 0) -> tuple:
    """Load data, returning (df_1min, df_5min, vix_series).

    Reads from Parquet first (fast), falls back to QuestDB.
    If max_days > 0, keeps only the most recent N trading days.
    """
    if use_synthetic:
        df = generate_synthetic_ohlcv()
        return df, df, generate_synthetic_vix(df)
    try:
        # Try Parquet first (much faster than QuestDB for large datasets)
        if PARQUET_PATH.exists() and bar_size == "1min":
            df = pd.read_parquet(PARQUET_PATH)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            print(f"  Loaded {len(df):,} bars from Parquet")
        else:
            from data.questdb_client import QuestDBClient
            qdb = QuestDBClient()
            df = qdb.get_bars("MES", bar_size, limit=10_000_000)
            if df.empty:
                raise RuntimeError("Empty")
        # Filter to RTH only
        df = filter_rth(df)
        # Limit to most recent N trading days
        if max_days > 0:
            unique_dates = sorted(set(df.index.date))
            if len(unique_dates) > max_days:
                cutoff = unique_dates[-max_days]
                df = df[df.index.date >= cutoff]
                print(f"  Trimmed to last {max_days} trading days ({len(df):,} bars)")
        # Resample to 5-min
        if bar_size == "1min":
            df_5min = resample_to_5min(df)
        else:
            df_5min = df
        # VIX from QuestDB (small dataset, fast)
        from data.questdb_client import QuestDBClient
        qdb = QuestDBClient()
        vix_df = qdb.get_bars("VIX", "1day", limit=10_000)
        if not vix_df.empty:
            vix = pd.Series(vix_df["close"].values, index=vix_df.index.date, name="vix")
        else:
            vix = generate_synthetic_vix(df)
        return df, df_5min, vix
    except Exception as e:
        print(f"  [WARN] Data load failed: {e}, using synthetic")
        df = generate_synthetic_ohlcv()
        return df, df, generate_synthetic_vix(df)


# ── Run Strategy ────────────────────────────────────────────────────

def precompute_bar_data(df, vix_series):
    """Pre-extract numpy arrays and timestamps once for fast sweeps."""
    from datetime import datetime as dt_cls
    n = len(df)
    opens = df["open"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    volumes = df["volume"].values.astype(np.int64)

    timestamps = []
    for ts in df.index:
        if not isinstance(ts, dt_cls):
            ts = pd.Timestamp(ts).to_pydatetime()
        if ts.tzinfo is None:
            ts = ET.localize(ts)
        else:
            ts = ts.astimezone(ET)
        timestamps.append(ts)

    vix_dict = vix_series.to_dict() if vix_series is not None else {}

    return {
        "opens": opens, "highs": highs, "lows": lows,
        "closes": closes, "volumes": volumes,
        "timestamps": timestamps, "vix_dict": vix_dict, "n": n,
    }


def run_strategy_b_fast(precomputed, params=None, capital=20_000.0, rvol=None):
    """Run ORB strategy using precomputed bar data (avoids repeated timestamp conversion)."""
    strat = VIXAdaptiveORBStrategy(params=params, capital=capital)
    strat.reset()
    if rvol is not None:
        strat.set_relative_volume(rvol)

    ts_list = precomputed["timestamps"]
    opens = precomputed["opens"]
    highs = precomputed["highs"]
    lows = precomputed["lows"]
    closes = precomputed["closes"]
    volumes = precomputed["volumes"]
    vix_dict = precomputed["vix_dict"]
    n = precomputed["n"]

    prev_date = None
    for i in range(n):
        ts = ts_list[i]
        d = ts.date()
        if d != prev_date:
            prev_date = d
            if d in vix_dict:
                strat.set_vix(float(vix_dict[d]))

        strat.on_bar(
            timestamp=ts,
            open_=opens[i],
            high=highs[i],
            low=lows[i],
            close=closes[i],
            volume=int(volumes[i]),
        )

    if len(strat.trades) < 2:
        return {"sharpe": 0.0, "n_trades": len(strat.trades), "trades": strat.trades}

    trade_pnls = np.array([t.pnl for t in strat.trades])
    equity = np.cumsum(np.concatenate([[capital], trade_pnls]))
    returns = np.diff(equity) / equity[:-1]
    metrics = compute_all_metrics(returns, trade_pnls, equity)
    metrics["trades"] = strat.trades
    return metrics


def run_strategy_b(df, vix_series, params=None, capital=20_000.0, rvol=None):
    """Run ORB strategy and return metrics dict."""
    strat = VIXAdaptiveORBStrategy(params=params, capital=capital)
    if rvol is not None:
        strat.set_relative_volume(rvol)
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
    params_file = Path(__file__).resolve().parent.parent / "config" / "strategy_a_params.json"
    if params_file.exists():
        with open(params_file) as f:
            params = json.load(f)
    else:
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

def main(use_synthetic: bool = False, bar_size: str = "1min", max_days: int = 0):
    print("=" * 70)
    print("  Strategy B: VIX-Adaptive ORB Optimizer")
    print("=" * 70)

    t_load = time.time()
    df, df_5min, vix_series = load_data(use_synthetic, bar_size, max_days=max_days)
    n_days = len(set(df.index.date))
    print(f"Data: {len(df):,} bars ({bar_size}), {len(df_5min):,} bars (5min), {n_days} trading days (loaded in {time.time()-t_load:.1f}s)")

    # Build RelativeVolume profile from historical data
    print("  Building relative volume profile...", flush=True)
    rvol = RelativeVolume(df, min_days=20)
    print(f"  Volume profile: {len(rvol.profile)} time slots")

    # ── 1. Parameter sweep ──
    print(f"\n{'-'*70}")
    print("  1. Parameter Sweep")
    print("-" * 70)

    param_grid = {
        "or_width_min_atr_fraction": [0.15, 0.25, 0.4],
        "volume_threshold": [0.0, 1.0, 1.5, 2.0],
        "stop_or_multiplier": [0.3, 0.5, 0.75, 1.0],
        "target_or_multiplier": [0.75, 1.0, 1.5, 2.0],
        "trail_trigger_or": [0.0, 0.75],
        "max_entry_hour": [12, 14],
    }

    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    n_combos = len(combos)
    print(f"  {n_combos} parameter combinations on {len(df):,} bars, {n_days} days")

    # Pre-compute shared data once (avoids repeated timestamp conversion)
    print("  Pre-computing bar data...", flush=True)
    t_pre = time.time()
    pre = precompute_bar_data(df, vix_series)
    print(f"  Pre-compute done in {time.time()-t_pre:.1f}s")

    t_sweep = time.time()
    results = []
    for idx, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        m = run_strategy_b_fast(pre, params=params, rvol=rvol)
        trades = m.get("trades", [])
        longs = [t for t in trades if t.side == "LONG"]
        shorts = [t for t in trades if t.side == "SHORT"]
        long_wins = sum(1 for t in longs if t.pnl > 0)
        short_wins = sum(1 for t in shorts if t.pnl > 0)
        results.append({
            **params,
            "sharpe": m.get("sharpe", 0),
            "n_trades": len(trades),
            "win_rate": m.get("win_rate", 0),
            "profit_factor": m.get("profit_factor", 0),
            "max_drawdown": m.get("max_drawdown", 0),
            "total_pnl": sum(t.pnl for t in trades),
            "n_longs": len(longs),
            "n_shorts": len(shorts),
            "long_wr": long_wins / len(longs) if longs else 0,
            "short_wr": short_wins / len(shorts) if shorts else 0,
            "long_pnl": sum(t.pnl for t in longs),
            "short_pnl": sum(t.pnl for t in shorts),
        })
        if (idx + 1) % 25 == 0 or idx == n_combos - 1:
            elapsed = time.time() - t_sweep
            rate = (idx + 1) / elapsed
            eta = (n_combos - idx - 1) / rate if rate > 0 else 0
            print(f"    [{idx+1}/{n_combos}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining", flush=True)

    results_df = pd.DataFrame(results)
    # Only rank combos with meaningful trade count
    min_trades = max(10, n_days // 10)  # at least ~10% of days
    viable = results_df[results_df["n_trades"] >= min_trades].copy()
    if viable.empty:
        viable = results_df[results_df["n_trades"] >= 5].copy()
    if viable.empty:
        viable = results_df.copy()
    viable = viable.sort_values("sharpe", ascending=False)

    print(f"\n  Sweep done in {time.time()-t_sweep:.1f}s")
    print(f"  Viable combos (>={min_trades} trades): {len(viable)}/{n_combos}")

    print(f"\n  Top 10 parameter sets:")
    header = f"  {'Sharpe':>7s} {'Trades':>7s} {'WR':>6s} {'PF':>6s} {'DD':>6s} {'PnL':>9s} | Params"
    print(header)
    print("  " + "-" * 80)
    for _, row in viable.head(10).iterrows():
        params_str = (
            f"atr_frac={row['or_width_min_atr_fraction']}, "
            f"vol={row['volume_threshold']}, "
            f"stop={row['stop_or_multiplier']}, "
            f"tgt={row['target_or_multiplier']}, "
            f"trail={row['trail_trigger_or']}, "
            f"maxhr={int(row['max_entry_hour'])}"
        )
        print(
            f"  {row['sharpe']:+7.3f} {row['n_trades']:7.0f} "
            f"{row['win_rate']*100:5.1f}% {row['profit_factor']:5.2f} "
            f"{row['max_drawdown']*100:5.1f}% ${row['total_pnl']:+8,.0f} | {params_str}"
        )

    # ── Long vs Short breakdown for top 10 ──
    print(f"\n  Long vs Short breakdown (top 10):")
    header2 = (
        f"  {'Sharpe':>7s} | {'Longs':>6s} {'L-WR':>6s} {'L-PnL':>9s} | "
        f"{'Shorts':>6s} {'S-WR':>6s} {'S-PnL':>9s} | Params"
    )
    print(header2)
    print("  " + "-" * 95)
    for _, row in viable.head(10).iterrows():
        params_str = (
            f"stop={row['stop_or_multiplier']}, "
            f"tgt={row['target_or_multiplier']}, "
            f"trail={row['trail_trigger_or']}, "
            f"maxhr={int(row['max_entry_hour'])}"
        )
        print(
            f"  {row['sharpe']:+7.3f} | "
            f"{row['n_longs']:5.0f}  {row['long_wr']*100:5.1f}% ${row['long_pnl']:+8,.0f} | "
            f"{row['n_shorts']:5.0f}  {row['short_wr']*100:5.1f}% ${row['short_pnl']:+8,.0f} | {params_str}"
        )

    # ── 2. CPCV on top 20 ──
    print(f"\n{'-'*70}")
    print("  2. CPCV Validation (top 20 configs)")
    print("-" * 70)

    top_n = min(20, len(viable))
    cpcv_results = []

    for rank, (_, row) in enumerate(viable.head(top_n).iterrows()):
        params = {k: float(row[k]) if k != "max_entry_hour" else int(row[k]) for k in keys}

        def make_strategy_fn(p, rv):
            def strategy_fn(data_slice):
                strat = VIXAdaptiveORBStrategy(params=p, capital=20_000.0)
                if rv is not None:
                    strat.set_relative_volume(rv)
                strat.generate_signals(data_slice, vix_series=vix_series)
                if len(strat.trades) < 2:
                    return np.zeros(1)
                trade_pnls = np.array([t.pnl for t in strat.trades])
                equity = np.cumsum(np.concatenate([[20_000.0], trade_pnls]))
                returns = np.diff(equity) / equity[:-1]
                return returns
            return strategy_fn

        validator = CPCVValidator(n_groups=6, k_test=2, purge_bars=60)
        try:
            result = validator.validate(df, make_strategy_fn(params, rvol))
            pbo = result["pbo"]
            oos_sharpe = result["mean_oos_sharpe"]
        except Exception as e:
            pbo = 1.0
            oos_sharpe = 0.0

        passed = pbo < 0.40
        tag = "PASS" if passed else "FAIL"
        params_str = (
            f"atr_frac={params['or_width_min_atr_fraction']}, "
            f"stop={params['stop_or_multiplier']}, "
            f"tgt={params['target_or_multiplier']}, "
            f"trail={params['trail_trigger_or']}, "
            f"vol={params['volume_threshold']}, "
            f"maxhr={params['max_entry_hour']}"
        )
        print(
            f"  [{tag}] #{rank+1:2d} Sharpe={row['sharpe']:+.3f} "
            f"PBO={pbo:.1%} OOS_Sharpe={oos_sharpe:+.1f} "
            f"trades={int(row['n_trades'])} | {params_str}"
        )
        cpcv_results.append({
            **params,
            "sharpe": row["sharpe"],
            "n_trades": int(row["n_trades"]),
            "pbo": pbo,
            "oos_sharpe": oos_sharpe,
            "passed": passed,
        })

    cpcv_df = pd.DataFrame(cpcv_results)
    passed_df = cpcv_df[cpcv_df["passed"]].sort_values("sharpe", ascending=False)
    print(f"\n  CPCV validated: {len(passed_df)}/{top_n} (PBO < 40%)")

    if len(passed_df) > 0:
        print(f"\n  CPCV-validated configs:")
        for _, row in passed_df.iterrows():
            params_str = (
                f"atr_frac={row['or_width_min_atr_fraction']}, "
                f"stop={row['stop_or_multiplier']}, "
                f"tgt={row['target_or_multiplier']}, "
                f"trail={row['trail_trigger_or']}, "
                f"vol={row['volume_threshold']}, "
                f"maxhr={int(row['max_entry_hour'])}"
            )
            print(
                f"    Sharpe={row['sharpe']:+.3f} PBO={row['pbo']:.1%} "
                f"OOS={row['oos_sharpe']:+.1f} trades={int(row['n_trades'])} | {params_str}"
            )

    # ── 3. Save best params ──
    print(f"\n{'-'*70}")
    print("  3. Save Best Params")
    print("-" * 70)

    # Prefer CPCV-validated, fall back to best raw Sharpe
    if len(passed_df) > 0:
        save_row = passed_df.iloc[0]
        source = "CPCV-validated"
    elif len(viable) > 0:
        save_row = viable.iloc[0]
        source = "best raw Sharpe (CPCV failed)"
    else:
        save_row = None
        source = None

    if save_row is not None:
        save_params = {
            "or_width_min_atr_fraction": float(save_row["or_width_min_atr_fraction"]),
            "volume_threshold": float(save_row["volume_threshold"]),
            "stop_or_multiplier": float(save_row["stop_or_multiplier"]),
            "target_or_multiplier": float(save_row["target_or_multiplier"]),
            "trail_trigger_or": float(save_row["trail_trigger_or"]),
            "max_entry_hour": int(save_row["max_entry_hour"]),
            "max_entry_minute": 0,
        }
        PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(PARAMS_FILE, "w") as f:
            json.dump(save_params, f, indent=2)
        print(f"  Saved {source} to {PARAMS_FILE}")
        print(f"  {json.dumps(save_params, indent=2)}")
    else:
        print("  [WARN] No viable configs to save")
        save_params = None

    # ── 4. Correlation with Strategy A ──
    print(f"\n{'-'*70}")
    print("  4. Correlation with Strategy A")
    print("-" * 70)

    metrics_a = run_strategy_a(df_5min, vix_series)
    trades_a = metrics_a.get("trades", [])

    if save_params is not None:
        metrics_b_best = run_strategy_b(df, vix_series, params=save_params, rvol=rvol)
    else:
        metrics_b_best = run_strategy_b(df, vix_series, rvol=rvol)
    trades_b_best = metrics_b_best.get("trades", [])

    if len(trades_a) >= 2 and len(trades_b_best) >= 2:
        dates = sorted(set(df.index.date))
        start_date = min(dates)
        end_date = max(dates)

        daily_a = compute_daily_returns(trades_a, start_date, end_date, 10_000.0)
        daily_b = compute_daily_returns(trades_b_best, start_date, end_date, 20_000.0)

        common = daily_a.index.intersection(daily_b.index)
        da = daily_a.loc[common]
        db = daily_b.loc[common]

        active = (da != 0) | (db != 0)
        if active.sum() > 5:
            pearson = da[active].corr(db[active])
            spearman = da[active].rank().corr(db[active].rank())
        else:
            pearson = spearman = 0.0

        combined_daily = 0.40 * da + 0.30 * db
        combined_sharpe = annualized_sharpe(combined_daily.values)
        a_sharpe = annualized_sharpe(da.values)
        b_sharpe = annualized_sharpe(db.values)

        print(f"  Strategy A: {len(trades_a)} trades, Sharpe={metrics_a.get('sharpe', 0):+.3f}")
        print(f"  Strategy B: {len(trades_b_best)} trades, Sharpe={metrics_b_best.get('sharpe', 0):+.3f}")
        print(f"  Pearson correlation:  {pearson:+.3f}  {'PASS' if abs(pearson) < 0.30 else 'WARN'}")
        print(f"  Spearman correlation: {spearman:+.3f}  {'PASS' if abs(spearman) < 0.30 else 'WARN'}")
        print(f"  Combined A+B Sharpe:  {combined_sharpe:+.3f}")
        print(f"  Diversification:      {'YES' if combined_sharpe > max(a_sharpe, b_sharpe) else 'NO'}")
    else:
        print("  [WARN] Not enough trades for correlation analysis")

    # ── 5. Monte Carlo ──
    print(f"\n{'-'*70}")
    print("  5. Monte Carlo (best params, 1000 paths)")
    print("-" * 70)

    if len(trades_b_best) >= 2:
        trade_pnls = np.array([t.pnl for t in trades_b_best])
        mc = MonteCarloSimulator(n_paths=1000, seed=42)
        mc_result = mc.simulate(trade_pnls, initial_capital=20_000.0)

        print(f"\n  {'Percentile':>12s} {'Final Equity':>14s} {'Max DD':>8s} {'Sharpe':>8s}")
        print("  " + "-" * 45)
        for p in [5, 25, 50, 75, 95]:
            dd_val = mc_result['max_drawdown_pcts'][p]
            dd_pct = dd_val * 100 if dd_val <= 1.0 else dd_val
            print(
                f"  {p:11d}th ${mc_result['final_equity_pcts'][p]:13,.2f} "
                f"{dd_pct:7.1f}% "
                f"{mc_result['sharpe_pcts'][p]:8.3f}"
            )
        prob = mc_result['prob_positive']
        prob_pct = prob * 100 if prob <= 1.0 else prob
        print(f"\n  Prob. positive: {prob_pct:.1f}%")
    else:
        print("  [WARN] Not enough trades for Monte Carlo")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--bar-size", default="1min", choices=["1min", "5min"])
    parser.add_argument("--days", type=int, default=0, help="Limit to N most recent trading days")
    args = parser.parse_args()
    main(use_synthetic=args.synthetic, bar_size=args.bar_size, max_days=args.days)
