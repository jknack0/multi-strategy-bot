"""
03 — Parameter Optimization with CPCV Validation.

Two-phase grid search: vectorized screening of all combos, then full
backtest on top candidates. CPCV validates the best; PBO > 40% rejected.
The best CPCV-validated set is saved to config/strategy_a_params.json.

Usage:
    uv run python notebooks/03_parameter_optimization.py
    uv run python notebooks/03_parameter_optimization.py --synthetic
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtesting.cpcv_validator import CPCVValidator
from backtesting.metrics import annualized_sharpe
from strategies.mean_reversion import (
    MESMeanReversionStrategy,
    precompute_indicators,
)

ET = pytz.timezone("US/Eastern")


# ── Data Generation (reuse from 02) ─────────────────────────────────────

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


def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only RTH bars (9:00-16:00 ET)."""
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


PARQUET_PATH = Path(__file__).resolve().parent.parent / "data" / "MES_1min.parquet"


def load_data(use_synthetic: bool, bar_size: str = "1min") -> tuple:
    if use_synthetic:
        df = generate_synthetic_ohlcv()
        vix = generate_synthetic_vix(df)
        return df, vix
    try:
        # Try Parquet first (much faster for large datasets)
        if PARQUET_PATH.exists() and bar_size == "1min":
            df_raw = pd.read_parquet(PARQUET_PATH)
            if not isinstance(df_raw.index, pd.DatetimeIndex):
                df_raw.index = pd.to_datetime(df_raw.index)
            print(f"  Loaded {len(df_raw):,} bars from Parquet")
        else:
            from data.questdb_client import QuestDBClient
            qdb = QuestDBClient()
            df_raw = qdb.get_bars("MES", "1min", limit=10_000_000)
            if df_raw.empty:
                raise RuntimeError("Empty")
            print(f"  Loaded {len(df_raw):,} bars from QuestDB")
        df = filter_rth(df_raw)
        if bar_size == "5min":
            df = df.resample("5min").agg({
                "open": "first", "high": "max",
                "low": "min", "close": "last", "volume": "sum",
            }).dropna()
        print(f"  Filtered to {len(df):,} RTH {bar_size} bars, {len(set(df.index.date)):,} trading days")
        # Load real VIX from QuestDB
        from data.questdb_client import QuestDBClient
        qdb = QuestDBClient()
        vix_df = qdb.get_bars("VIX", "1day", limit=10_000)
        if not vix_df.empty:
            vix = pd.Series(vix_df["close"].values, index=vix_df.index.date, name="vix")
            print(f"  Loaded {len(vix)} days of real VIX data (range: {vix.min():.1f}-{vix.max():.1f})")
        else:
            vix = generate_synthetic_vix(df)
            print("  [WARN] No VIX data, using synthetic")
        return df, vix
    except Exception as exc:
        print(f"  [WARN] Data load failed ({exc}), using synthetic data")
        return generate_synthetic_ohlcv(), generate_synthetic_vix(generate_synthetic_ohlcv())


# ── Grid Search ──────────────────────────────────────────────────────────

PARAM_GRID = {
    "bb_period": [15, 20, 25, 30],
    "bb_sigma": [1.0, 1.5, 2.0],          # added 1.0 (tighter = more entries)
    "atr_stop_multiplier": [1.5, 2.0, 2.5, 3.0, 3.5],
    "vwap_deviation_entry": [0.25, 0.5, 0.75, 1.0],  # lowered (looser = more entries)
    "trend_ema_period": [0],               # removed EMA filter (was killing trades)
    "take_profit_pct": [0.5, 0.75, 1.0],
    "trailing_stop_factor": [0.0, 0.4, 0.6],
    "min_rr_ratio": [0.0],
    "long_only": [True],
    # ADX regime filter (0 = disabled, >0 = gate threshold)
    "adx_threshold": [0, 20, 25, 30],      # re-test without EMA stacking
    "adx_sizing": [False],
    # (start_h, start_m, end_h, end_m)
    "trade_window": [
        (9, 30, 15, 0),    # full RTH core
        (10, 0, 15, 0),    # slightly later start
        (10, 0, 14, 0),    # current default
        (10, 0, 13, 0),    # morning focus
        (11, 0, 14, 0),    # skip first hour
    ],
}


def build_window_masks(df: pd.DataFrame) -> dict:
    """Pre-compute in_window boolean masks for each trade_window option."""
    idx = df.index
    if idx.tz is None:
        et_index = idx.tz_localize(ET)
    else:
        et_index = idx.tz_convert(ET)
    hours = et_index.hour
    minutes = et_index.minute
    bar_minutes = hours * 60 + minutes

    masks = {}
    for window in PARAM_GRID["trade_window"]:
        sh, sm, eh, em = window
        start_min = sh * 60 + sm
        end_min = eh * 60 + em
        masks[window] = np.asarray((bar_minutes >= start_min) & (bar_minutes < end_min))
    return masks


def vectorized_screen(precomputed: dict, params: dict, window_masks: dict) -> dict:
    """Score a parameter set using vectorized entry conditions (no position mgmt).

    Returns approximate signal count and a quick score based on whether
    price moved toward VWAP within 10 bars after each entry signal.
    """
    closes = precomputed["closes"]
    bb_period = params["bb_period"]
    bb_sigma = params["bb_sigma"]
    n = len(closes)

    bb_mean = precomputed["bb_means"][bb_period]
    bb_std = precomputed["bb_stds"][bb_period]

    vwap_vals = precomputed["vwap_vals"]
    vwap_stds = precomputed["vwap_stds"]
    vwap_dev = params["vwap_deviation_entry"]
    atr_vals = precomputed["atr_vals"]

    # Use per-combo window mask
    trade_window = params.get("trade_window", (10, 0, 14, 0))
    in_window = window_masks[trade_window]

    # VIX filter
    vix_per_bar = precomputed["vix_per_bar"]
    vix_mask = vix_per_bar <= 35
    eff_sigma = np.where(vix_per_bar > 25, 2.5, bb_sigma)
    adj_upper = bb_mean + eff_sigma * bb_std
    adj_lower = bb_mean - eff_sigma * bb_std

    ready = ~np.isnan(bb_mean) & ~np.isnan(atr_vals)

    # Trend filter
    trend_period = params.get("trend_ema_period", 0)
    ema_vals_dict = precomputed.get("ema_vals", {})
    if trend_period > 0 and trend_period in ema_vals_dict:
        ema_arr = ema_vals_dict[trend_period]
        ema_ready = ~np.isnan(ema_arr)
        trend_long_ok = ema_ready & (closes >= ema_arr)
        trend_short_ok = ema_ready & (closes <= ema_arr)
    else:
        trend_long_ok = np.ones(n, dtype=bool)
        trend_short_ok = np.ones(n, dtype=bool)

    # ADX regime filter
    adx_threshold = params.get("adx_threshold", 0)
    adx_arr = precomputed.get("adx_vals")
    if adx_threshold > 0 and adx_arr is not None:
        adx_mask = np.isnan(adx_arr) | (adx_arr <= adx_threshold)
    else:
        adx_mask = np.ones(n, dtype=bool)

    long_entries = (
        ready & in_window & vix_mask & trend_long_ok & adx_mask
        & (closes <= adj_lower)
        & (closes < vwap_vals - vwap_dev * vwap_stds)
    )

    long_only = params.get("long_only", False)
    if long_only:
        short_entries = np.zeros(n, dtype=bool)
    else:
        short_entries = (
            ready & in_window & vix_mask & trend_short_ok & adx_mask
            & (closes >= adj_upper)
            & (closes > vwap_vals + vwap_dev * vwap_stds)
        )

    # R:R filter
    tp_pct = params.get("take_profit_pct", 1.0)
    min_rr = params.get("min_rr_ratio", 0.0)
    if min_rr > 0:
        stop_dist = params["atr_stop_multiplier"] * atr_vals
        long_target_dist = tp_pct * (vwap_vals - closes)
        long_rr = np.where(stop_dist > 0, long_target_dist / stop_dist, 0.0)
        long_entries = long_entries & (long_rr >= min_rr)
        short_target_dist = tp_pct * (closes - vwap_vals)
        short_rr = np.where(stop_dist > 0, short_target_dist / stop_dist, 0.0)
        short_entries = short_entries & (short_rr >= min_rr)

    n_signals = int(long_entries.sum() + short_entries.sum())

    # Quick hit-rate: for each signal, did price move toward VWAP within 10 bars?
    lookahead = 10
    hits = 0
    total = 0
    long_idx = np.where(long_entries)[0]
    for idx in long_idx:
        if idx + lookahead < n:
            future_max = np.max(closes[idx + 1: idx + 1 + lookahead])
            if future_max > closes[idx]:
                hits += 1
            total += 1

    short_idx = np.where(short_entries)[0]
    for idx in short_idx:
        if idx + lookahead < n:
            future_min = np.min(closes[idx + 1: idx + 1 + lookahead])
            if future_min < closes[idx]:
                hits += 1
            total += 1

    hit_rate = hits / total if total > 0 else 0.0
    # Combined score: penalize too few signals, reward high hit rate
    approx_score = hit_rate * min(n_signals, 200) / 200.0

    return {
        "n_signals": n_signals,
        "hit_rate": hit_rate,
        "approx_score": approx_score,
        "params": params,
    }


def run_single_config_fast(
    df: pd.DataFrame,
    params: dict,
    precomputed: dict,
    window_masks: dict,
    capital: float = 10_000.0,
) -> dict:
    """Run a single parameter configuration using precomputed indicators."""
    # Apply trade_window to strategy params and precomputed mask
    trade_window = params.get("trade_window", (10, 0, 14, 0))
    full_params = dict(params)
    full_params["trade_start_hour"] = trade_window[0]
    full_params["trade_start_minute"] = trade_window[1]
    full_params["trade_end_hour"] = trade_window[2]
    full_params["trade_end_minute"] = trade_window[3]

    pc = dict(precomputed)
    pc["in_window"] = window_masks[trade_window]

    strat = MESMeanReversionStrategy(params=full_params, capital=capital)
    strat.generate_signals_fast(df, precomputed=pc)

    if len(strat.trades) < 5:
        return {"sharpe": -999.0, "n_trades": len(strat.trades), "params": params}

    trade_pnls = np.array([t.pnl for t in strat.trades])
    equity = np.cumsum(np.concatenate([[capital], trade_pnls]))
    if np.any(equity[:-1] <= 0):
        return {"sharpe": -999.0, "n_trades": len(strat.trades), "params": params}
    returns = np.diff(equity) / equity[:-1]
    sharpe = annualized_sharpe(returns)

    return {"sharpe": sharpe, "n_trades": len(strat.trades), "params": params}


def grid_search(
    df: pd.DataFrame,
    vix_series: pd.Series,
    precomputed: dict,
    window_masks: dict,
    top_n_full: int = 30,
) -> list:
    """Two-phase grid search: vectorized screening, then full backtest on top N."""
    keys = list(PARAM_GRID.keys())
    combos = list(itertools.product(*[PARAM_GRID[k] for k in keys]))

    # Phase 1: Vectorized screening
    t1 = time.time()
    print(f"\n  Phase 1: Vectorized screening of {len(combos)} combinations...")
    screening_results = []
    for combo in combos:
        params = dict(zip(keys, combo))
        score = vectorized_screen(precomputed, params, window_masks)
        screening_results.append(score)

    screening_results.sort(key=lambda x: x["approx_score"], reverse=True)
    elapsed1 = time.time() - t1
    best_screen = screening_results[0]
    print(f"  Screening complete in {elapsed1:.1f}s")
    print(f"  Best screening score: {best_screen['approx_score']:.3f} "
          f"({best_screen['n_signals']} signals, "
          f"{best_screen['hit_rate']*100:.0f}% hit rate)")

    # Phase 2: Full backtest on top candidates
    candidates = screening_results[:top_n_full]
    t2 = time.time()
    print(f"\n  Phase 2: Full backtest on top {len(candidates)} candidates...")
    results = []
    for i, candidate in enumerate(candidates):
        t_start = time.time()
        result = run_single_config_fast(df, candidate["params"], precomputed, window_masks)
        elapsed = time.time() - t_start
        results.append(result)
        remaining = elapsed * (len(candidates) - i - 1)
        print(f"    [{i+1}/{len(candidates)}] Sharpe={result['sharpe']:7.3f}  "
              f"Trades={result['n_trades']:4d}  "
              f"({elapsed:.1f}s, ~{remaining:.0f}s remaining)")

    elapsed2 = time.time() - t2
    print(f"  Full backtest complete in {elapsed2:.1f}s")

    results.sort(key=lambda x: float("-inf") if np.isnan(x["sharpe"]) else x["sharpe"], reverse=True)
    return results


# ── CPCV Validation ──────────────────────────────────────────────────────

def cpcv_validate_top_n(
    df: pd.DataFrame,
    vix_series: pd.Series,
    top_results: list,
    n: int = 10,
) -> list:
    """Run CPCV on top N parameter sets using fast signal generation."""
    validator = CPCVValidator(n_groups=6, k_test=2, purge_bars=60)
    validated = []

    t0 = time.time()
    print(f"\nRunning CPCV on top {n} parameter sets...")
    for i, result in enumerate(top_results[:n]):
        t_start = time.time()
        params = result["params"]
        ema_str = f", ema={params.get('trend_ema_period', 0)}" if params.get("trend_ema_period", 0) > 0 else ""
        tp_str = f", tp={params.get('take_profit_pct', 1.0)}" if params.get("take_profit_pct", 1.0) != 1.0 else ""
        trail_str = f", trail={params.get('trailing_stop_factor', 0.0)}" if params.get("trailing_stop_factor", 0.0) > 0 else ""
        adx_str = f", adx<={params.get('adx_threshold', 0)}" if params.get("adx_threshold", 0) > 0 else ""
        tw = params.get("trade_window", (10, 0, 14, 0))
        win_str = f", win={tw[0]}:{tw[1]:02d}-{tw[2]}:{tw[3]:02d}"
        print(f"  [{i+1}/{n}] bb={params['bb_period']}/{params['bb_sigma']}, "
              f"atr={params['atr_stop_multiplier']}, "
              f"vwap={params['vwap_deviation_entry']}"
              f"{ema_str}{tp_str}{trail_str}{adx_str}{win_str} "
              f"(raw Sharpe={result['sharpe']:.3f})")

        # Use a closure with precomputed indicators per data subset
        def make_strategy_fn(p):
            def strategy_fn(data: pd.DataFrame) -> np.ndarray:
                ema_p = [p["trend_ema_period"]] if p.get("trend_ema_period", 0) > 0 else []
                trade_window = p.get("trade_window", (10, 0, 14, 0))
                full_p = dict(p)
                full_p["trade_start_hour"] = trade_window[0]
                full_p["trade_start_minute"] = trade_window[1]
                full_p["trade_end_hour"] = trade_window[2]
                full_p["trade_end_minute"] = trade_window[3]
                pc = precompute_indicators(
                    data, vix_series=vix_series,
                    bb_periods=[full_p["bb_period"]], ema_periods=ema_p,
                    atr_period=14,
                    trade_start=(trade_window[0], trade_window[1]),
                    trade_end=(trade_window[2], trade_window[3]),
                )
                strat = MESMeanReversionStrategy(params=full_p, capital=10_000.0)
                strat.generate_signals_fast(data, precomputed=pc)
                if len(strat.trades) < 2:
                    return np.zeros(len(data))
                trade_pnls = np.array([t.pnl for t in strat.trades])
                equity = np.cumsum(np.concatenate([[10000.0], trade_pnls]))
                if np.any(equity[:-1] <= 0):
                    return np.zeros(len(data))
                returns = np.diff(equity) / equity[:-1]
                padded = np.zeros(len(data))
                padded[: len(returns)] = returns
                return padded
            return strategy_fn

        cpcv_result = validator.validate(df, make_strategy_fn(params))
        pbo = cpcv_result["pbo"]

        elapsed = time.time() - t_start
        remaining = elapsed * (n - i - 1)
        oos_sharpe = np.mean(cpcv_result["oos_scores"]) if cpcv_result["oos_scores"] else 0.0
        print(f"    PBO={pbo:.3f}, OOS Sharpe={oos_sharpe:.3f} "
              f"({elapsed:.1f}s, ~{remaining:.0f}s remaining)", end="")

        if pbo > 0.40:
            print(" -> REJECTED (PBO > 40%)")
        else:
            print(" -> PASSED")
            validated.append({
                "params": params,
                "raw_sharpe": result["sharpe"],
                "pbo": pbo,
                "oos_sharpe": oos_sharpe,
                "n_trades": result["n_trades"],
                "cpcv_is_scores": cpcv_result["is_scores"],
                "cpcv_oos_scores": cpcv_result["oos_scores"],
            })

    total = time.time() - t0
    print(f"  CPCV validation complete in {total:.1f}s")
    return validated


# ── Main ─────────────────────────────────────────────────────────────────

def run_optimization(df, vix_series, bar_label: str) -> dict:
    """Run full grid search + CPCV for a given dataset. Returns best result."""
    print(f"\n{'='*70}")
    print(f"  Optimizing on {bar_label}: {len(df)} bars, {len(set(df.index.date))} days")
    print(f"{'='*70}")

    # Pre-compute shared indicators once (use widest window for base)
    t_pre = time.time()
    print("\nPre-computing shared indicators...")
    ema_periods = [p for p in PARAM_GRID.get("trend_ema_period", []) if p > 0]
    precomputed = precompute_indicators(
        df,
        vix_series=vix_series,
        bb_periods=PARAM_GRID["bb_period"],
        ema_periods=ema_periods,
        atr_period=14,
    )
    # Build per-window masks (cheap array ops, no indicator recomputation)
    window_masks = build_window_masks(df)
    print(f"  Done in {time.time() - t_pre:.1f}s")
    print(f"  Trading windows: {len(PARAM_GRID['trade_window'])}")

    # Grid search (two-phase) — use top 200 to avoid screening bias
    results = grid_search(df, vix_series, precomputed, window_masks, top_n_full=200)

    # Show top 10
    print(f"\nTop 10 by raw Sharpe ({bar_label}):")
    print(f"  {'Rank':>4} {'Sharpe':>8} {'Trades':>7}  Parameters")
    print("  " + "-" * 65)
    for i, r in enumerate(results[:10]):
        p = r["params"]
        extras = []
        if p.get("trend_ema_period", 0) > 0:
            extras.append(f"ema={p['trend_ema_period']}")
        if p.get("take_profit_pct", 1.0) != 1.0:
            extras.append(f"tp={p['take_profit_pct']}")
        if p.get("trailing_stop_factor", 0.0) > 0:
            extras.append(f"trail={p['trailing_stop_factor']}")
        if p.get("min_rr_ratio", 0.0) > 0:
            extras.append(f"rr={p['min_rr_ratio']}")
        if p.get("adx_threshold", 0) > 0:
            extras.append(f"adx<={p['adx_threshold']}")
        tw = p.get("trade_window", (10, 0, 14, 0))
        extras.append(f"win={tw[0]}:{tw[1]:02d}-{tw[2]}:{tw[3]:02d}")
        extra_str = ", " + ", ".join(extras) if extras else ""
        print(f"  {i+1:4d} {r['sharpe']:8.3f} {r['n_trades']:7d}  "
              f"bb={p['bb_period']}/{p['bb_sigma']}, "
              f"atr={p['atr_stop_multiplier']}, vwap={p['vwap_deviation_entry']}"
              f"{extra_str}")

    # CPCV validation on top 20
    validated = cpcv_validate_top_n(df, vix_series, results, n=20)

    if not validated:
        print(f"\n[WARN] No parameter sets passed CPCV validation for {bar_label}.")
        # Use best raw Sharpe as fallback (even if not CPCV-validated)
        fallback_params = results[0]["params"].copy() if results else MESMeanReversionStrategy.DEFAULT_PARAMS.copy()
        fallback_params["long_only"] = True
        return {"params": fallback_params,
                "pbo": 1.0, "oos_sharpe": 0.0, "raw_sharpe": results[0]["sharpe"] if results else 0.0,
                "bar_label": bar_label, "validated": []}
    else:
        validated.sort(key=lambda x: x["oos_sharpe"], reverse=True)
        best = validated[0]

        print(f"\n{'='*70}")
        print(f"  CPCV-VALIDATED RESULTS ({bar_label})")
        print("=" * 70)
        print(f"  {'Rank':>4} {'OOS Sharpe':>11} {'PBO':>6} {'Raw Sharpe':>11}  Parameters")
        print("  " + "-" * 65)
        for i, v in enumerate(validated):
            p = v["params"]
            extras = []
            if p.get("trend_ema_period", 0) > 0:
                extras.append(f"ema={p['trend_ema_period']}")
            if p.get("take_profit_pct", 1.0) != 1.0:
                extras.append(f"tp={p['take_profit_pct']}")
            if p.get("trailing_stop_factor", 0.0) > 0:
                extras.append(f"trail={p['trailing_stop_factor']}")
            if p.get("adx_threshold", 0) > 0:
                extras.append(f"adx<={p['adx_threshold']}")
            tw = p.get("trade_window", (10, 0, 14, 0))
            extras.append(f"win={tw[0]}:{tw[1]:02d}-{tw[2]}:{tw[3]:02d}")
            extra_str = ", " + ", ".join(extras) if extras else ""
            print(f"  {i+1:4d} {v['oos_sharpe']:11.3f} {v['pbo']:6.3f} "
                  f"{v['raw_sharpe']:11.3f}  "
                  f"bb={p['bb_period']}/{p['bb_sigma']}, "
                  f"atr={p['atr_stop_multiplier']}, vwap={p['vwap_deviation_entry']}"
                  f"{extra_str}")

        best["bar_label"] = bar_label
        best["validated"] = validated
        return best


def main(use_synthetic: bool = False, max_days: int = 0) -> dict:
    t_total = time.time()
    n_combos = 1
    for v in PARAM_GRID.values():
        n_combos *= len(v)
    print("=" * 70)
    print("  Strategy A: Parameter Optimization + CPCV Validation")
    print(f"  Long-only on 1-min bars | {n_combos} combinations")
    print("=" * 70)

    df, vix_series = load_data(use_synthetic, bar_size="1min")

    # Optionally limit to most recent N trading days
    if max_days > 0:
        unique_dates = sorted(set(df.index.date))
        if len(unique_dates) > max_days:
            cutoff_date = unique_dates[-max_days]
            df = df[df.index.date >= cutoff_date]
            print(f"  Trimmed to last {max_days} trading days: {len(df):,} bars, "
                  f"{len(set(df.index.date)):,} days")

    best = run_optimization(df, vix_series, "1-min bars (long-only)")

    # Save optimal params
    best_params = best["params"].copy()
    # Convert trade_window tuple to individual params for JSON
    tw = best_params.pop("trade_window", (10, 0, 14, 0))
    best_params["trade_start_hour"] = tw[0]
    best_params["trade_start_minute"] = tw[1]
    best_params["trade_end_hour"] = tw[2]
    best_params["trade_end_minute"] = tw[3]
    best_params["bar_size"] = "1min"
    best_params["adx_period"] = 14  # Fixed; only threshold is optimized

    output_path = Path(__file__).resolve().parent.parent / "config" / "strategy_a_params.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(best_params, f, indent=2)

    total_time = time.time() - t_total
    print(f"\n  Best params saved to {output_path}")
    print(f"    Window     : {tw[0]}:{tw[1]:02d}-{tw[2]}:{tw[3]:02d}")
    print(f"    OOS Sharpe : {best.get('oos_sharpe', 0):.3f}")
    print(f"    PBO        : {best.get('pbo', 0):.3f}")
    print(f"    Params     : {best_params}")
    print(f"\n  Total runtime: {total_time:.1f}s")
    print("=" * 70)

    return {
        "best_params": best_params,
        "oos_sharpe": best.get("oos_sharpe", 0),
        "pbo": best.get("pbo", 0),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--days", type=int, default=0,
                        help="Limit to most recent N trading days (0=all)")
    args = parser.parse_args()
    main(use_synthetic=args.synthetic, max_days=args.days)
