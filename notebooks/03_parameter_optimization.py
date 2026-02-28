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


def load_data(use_synthetic: bool, bar_size: str = "5min") -> tuple:
    if use_synthetic:
        df = generate_synthetic_ohlcv()
        vix = generate_synthetic_vix(df)
        return df, vix
    try:
        from data.questdb_client import QuestDBClient
        qdb = QuestDBClient()
        df = qdb.get_bars("MES", bar_size, limit=500_000)
        if df.empty:
            raise RuntimeError("Empty")
        vix = generate_synthetic_vix(df)
        return df, vix
    except Exception:
        return generate_synthetic_ohlcv(), generate_synthetic_vix(generate_synthetic_ohlcv())


# ── Grid Search ──────────────────────────────────────────────────────────

PARAM_GRID = {
    "bb_period": [15, 20, 25, 30],
    "bb_sigma": [1.5, 2.0, 2.5],
    "atr_stop_multiplier": [1.5, 2.0, 2.5, 3.0],
    "vwap_deviation_entry": [0.5, 0.75, 1.0, 1.25],
    "trend_ema_period": [0, 200],
    "take_profit_pct": [0.5, 0.75, 1.0],
    "trailing_stop_factor": [0.0, 0.4],
    "min_rr_ratio": [0.0],
}


def vectorized_screen(precomputed: dict, params: dict) -> dict:
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

    in_window = precomputed["in_window"]

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

    long_entries = (
        ready & in_window & vix_mask & trend_long_ok
        & (closes <= adj_lower)
        & (closes < vwap_vals - vwap_dev * vwap_stds)
    )

    long_only = params.get("long_only", False)
    if long_only:
        short_entries = np.zeros(n, dtype=bool)
    else:
        short_entries = (
            ready & in_window & vix_mask & trend_short_ok
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
    capital: float = 10_000.0,
) -> dict:
    """Run a single parameter configuration using precomputed indicators."""
    strat = MESMeanReversionStrategy(params=params, capital=capital)
    strat.generate_signals_fast(df, precomputed=precomputed)

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
        score = vectorized_screen(precomputed, params)
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
        result = run_single_config_fast(df, candidate["params"], precomputed)
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
        print(f"  [{i+1}/{n}] bb={params['bb_period']}/{params['bb_sigma']}, "
              f"atr={params['atr_stop_multiplier']}, "
              f"vwap={params['vwap_deviation_entry']}"
              f"{ema_str}{tp_str}{trail_str} "
              f"(raw Sharpe={result['sharpe']:.3f})")

        # Use a closure with precomputed indicators per data subset
        def make_strategy_fn(p):
            def strategy_fn(data: pd.DataFrame) -> np.ndarray:
                ema_p = [p["trend_ema_period"]] if p.get("trend_ema_period", 0) > 0 else []
                pc = precompute_indicators(
                    data, vix_series=vix_series,
                    bb_periods=[p["bb_period"]], ema_periods=ema_p,
                    atr_period=14,
                )
                strat = MESMeanReversionStrategy(params=p, capital=10_000.0)
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

def main(use_synthetic: bool = False, bar_size: str = "5min") -> dict:
    t_total = time.time()
    print("=" * 70)
    print("  Strategy A: Parameter Optimization + CPCV Validation")
    print("=" * 70)

    df, vix_series = load_data(use_synthetic, bar_size)
    print(f"Data: {len(df)} bars ({bar_size})")

    # Pre-compute shared indicators once
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
    print(f"  Done in {time.time() - t_pre:.1f}s")

    # Grid search (two-phase)
    results = grid_search(df, vix_series, precomputed)

    # Show top 10
    print("\nTop 10 by raw Sharpe:")
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
        extra_str = ", " + ", ".join(extras) if extras else ""
        print(f"  {i+1:4d} {r['sharpe']:8.3f} {r['n_trades']:7d}  "
              f"bb={p['bb_period']}/{p['bb_sigma']}, "
              f"atr={p['atr_stop_multiplier']}, vwap={p['vwap_deviation_entry']}"
              f"{extra_str}")

    # CPCV validation on top 10
    validated = cpcv_validate_top_n(df, vix_series, results, n=10)

    if not validated:
        print("\n[WARN] No parameter sets passed CPCV validation (PBO <= 40%).")
        print("Using default parameters as fallback.")
        best = {"params": MESMeanReversionStrategy.DEFAULT_PARAMS.copy(),
                "pbo": 1.0, "oos_sharpe": 0.0, "raw_sharpe": 0.0}
    else:
        # Select best by CPCV-validated OOS Sharpe
        validated.sort(key=lambda x: x["oos_sharpe"], reverse=True)
        best = validated[0]

        print("\n" + "=" * 70)
        print("  CPCV-VALIDATED RESULTS")
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
            extra_str = ", " + ", ".join(extras) if extras else ""
            print(f"  {i+1:4d} {v['oos_sharpe']:11.3f} {v['pbo']:6.3f} "
                  f"{v['raw_sharpe']:11.3f}  "
                  f"bb={p['bb_period']}/{p['bb_sigma']}, "
                  f"atr={p['atr_stop_multiplier']}, vwap={p['vwap_deviation_entry']}"
                  f"{extra_str}")

    # Save optimal params
    output_path = Path(__file__).resolve().parent.parent / "config" / "strategy_a_params.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(best["params"], f, indent=2)

    total_time = time.time() - t_total
    print(f"\n  Best CPCV-validated parameters saved to {output_path}")
    print(f"    OOS Sharpe : {best.get('oos_sharpe', 0):.3f}")
    print(f"    PBO        : {best.get('pbo', 0):.3f}")
    print(f"    Params     : {best['params']}")
    print(f"\n  Total runtime: {total_time:.1f}s")
    print("=" * 70)

    return {
        "best_params": best["params"],
        "oos_sharpe": best.get("oos_sharpe", 0),
        "pbo": best.get("pbo", 0),
        "n_validated": len(validated),
        "n_grid_total": len(results),
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
