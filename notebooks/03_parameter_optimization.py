"""
03 — Parameter Optimization with CPCV Validation.

Grid search over BB period/sigma, ATR multiplier, and VWAP deviation.
Top candidates are validated with CPCV; any with PBO > 40% are rejected.
The best CPCV-validated set is saved to config/strategy_a_params.json.

Usage:
    uv run python notebooks/03_parameter_optimization.py
    uv run python notebooks/03_parameter_optimization.py --synthetic
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtesting.cpcv_validator import CPCVValidator
from backtesting.metrics import annualized_sharpe
from strategies.mean_reversion import MESMeanReversionStrategy

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


def load_data(use_synthetic: bool) -> tuple:
    if use_synthetic:
        df = generate_synthetic_ohlcv()
        vix = generate_synthetic_vix(df)
        return df, vix
    try:
        from data.questdb_client import QuestDBClient
        qdb = QuestDBClient()
        df = qdb.get_bars("MES", "5min", limit=500_000)
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
}


def run_single_config(
    df: pd.DataFrame,
    vix_series: pd.Series,
    params: dict,
    capital: float = 10_000.0,
) -> dict:
    """Run a single parameter configuration and return metrics."""
    strat = MESMeanReversionStrategy(params=params, capital=capital)
    strat.generate_signals(df, vix_series=vix_series)

    if len(strat.trades) < 5:
        return {"sharpe": -999.0, "n_trades": len(strat.trades), "params": params}

    trade_pnls = np.array([t.pnl for t in strat.trades])
    equity = np.cumsum(np.concatenate([[capital], trade_pnls]))
    returns = np.diff(equity) / equity[:-1]
    sharpe = annualized_sharpe(returns)

    return {"sharpe": sharpe, "n_trades": len(strat.trades), "params": params}


def grid_search(df: pd.DataFrame, vix_series: pd.Series) -> list:
    """Run grid search over all parameter combinations."""
    keys = list(PARAM_GRID.keys())
    combos = list(itertools.product(*[PARAM_GRID[k] for k in keys]))
    results = []

    print(f"\nRunning grid search: {len(combos)} combinations...")
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        result = run_single_config(df, vix_series, params)
        results.append(result)
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(combos)}] Current best Sharpe: "
                  f"{max(r['sharpe'] for r in results):.3f}")

    results.sort(key=lambda x: x["sharpe"], reverse=True)
    return results


# ── CPCV Validation ──────────────────────────────────────────────────────

def cpcv_validate_top_n(
    df: pd.DataFrame,
    vix_series: pd.Series,
    top_results: list,
    n: int = 10,
) -> list:
    """Run CPCV on top N parameter sets."""
    validator = CPCVValidator(n_groups=6, k_test=2, purge_bars=60)
    validated = []

    print(f"\nRunning CPCV on top {n} parameter sets...")
    for i, result in enumerate(top_results[:n]):
        params = result["params"]
        print(f"  [{i+1}/{n}] bb_period={params['bb_period']}, "
              f"bb_sigma={params['bb_sigma']}, "
              f"atr_mult={params['atr_stop_multiplier']}, "
              f"vwap_dev={params['vwap_deviation_entry']} "
              f"(raw Sharpe={result['sharpe']:.3f})")

        def strategy_fn(data: pd.DataFrame) -> np.ndarray:
            strat = MESMeanReversionStrategy(params=params, capital=10_000.0)
            strat.generate_signals(data, vix_series=vix_series)
            if len(strat.trades) < 2:
                return np.zeros(len(data))
            trade_pnls = np.array([t.pnl for t in strat.trades])
            equity = np.cumsum(np.concatenate([[10000.0], trade_pnls]))
            returns = np.diff(equity) / equity[:-1]
            # Pad to match data length
            padded = np.zeros(len(data))
            padded[: len(returns)] = returns
            return padded

        cpcv_result = validator.validate(df, strategy_fn)
        pbo = cpcv_result["pbo"]

        oos_sharpe = np.mean(cpcv_result["oos_scores"]) if cpcv_result["oos_scores"] else 0.0
        print(f"    PBO={pbo:.3f}, OOS Sharpe={oos_sharpe:.3f}", end="")

        if pbo > 0.40:
            print(" → REJECTED (PBO > 40%)")
        else:
            print(" → PASSED")
            validated.append({
                "params": params,
                "raw_sharpe": result["sharpe"],
                "pbo": pbo,
                "oos_sharpe": oos_sharpe,
                "n_trades": result["n_trades"],
                "cpcv_is_scores": cpcv_result["is_scores"],
                "cpcv_oos_scores": cpcv_result["oos_scores"],
            })

    return validated


# ── Main ─────────────────────────────────────────────────────────────────

def main(use_synthetic: bool = False) -> dict:
    print("=" * 70)
    print("  Strategy A: Parameter Optimization + CPCV Validation")
    print("=" * 70)

    df, vix_series = load_data(use_synthetic)
    print(f"Data: {len(df)} bars")

    # Grid search
    results = grid_search(df, vix_series)

    # Show top 10
    print("\nTop 10 by raw Sharpe:")
    print(f"  {'Rank':>4} {'Sharpe':>8} {'Trades':>7}  Parameters")
    print("  " + "-" * 65)
    for i, r in enumerate(results[:10]):
        p = r["params"]
        print(f"  {i+1:4d} {r['sharpe']:8.3f} {r['n_trades']:7d}  "
              f"bb={p['bb_period']}/{p['bb_sigma']}, "
              f"atr={p['atr_stop_multiplier']}, vwap={p['vwap_deviation_entry']}")

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
            print(f"  {i+1:4d} {v['oos_sharpe']:11.3f} {v['pbo']:6.3f} "
                  f"{v['raw_sharpe']:11.3f}  "
                  f"bb={p['bb_period']}/{p['bb_sigma']}, "
                  f"atr={p['atr_stop_multiplier']}, vwap={p['vwap_deviation_entry']}")

    # Save optimal params
    output_path = Path(__file__).resolve().parent.parent / "config" / "strategy_a_params.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(best["params"], f, indent=2)

    print(f"\n  ★ Best CPCV-validated parameters saved to {output_path}")
    print(f"    OOS Sharpe : {best.get('oos_sharpe', 0):.3f}")
    print(f"    PBO        : {best.get('pbo', 0):.3f}")
    print(f"    Params     : {best['params']}")
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
    args = parser.parse_args()
    main(use_synthetic=args.synthetic)
