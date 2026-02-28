"""
01 — Stationarity Analysis for MES Returns.

Runs ADF test, computes Hurst exponent, and estimates half-life of
mean reversion to validate the premise of Strategy A.

Usage:
    uv run python notebooks/01_stationarity_analysis.py
    uv run python notebooks/01_stationarity_analysis.py --bar-size 1min
    uv run python notebooks/01_stationarity_analysis.py --synthetic
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Data Loading ─────────────────────────────────────────────────────────

def load_data_questdb(bar_size: str = "5min") -> pd.Series:
    """Load MES closes from QuestDB."""
    from data.questdb_client import QuestDBClient
    qdb = QuestDBClient()
    df = qdb.get_bars("MES", bar_size, limit=500_000)
    if df.empty:
        raise RuntimeError("No data in QuestDB. Run ib_fetcher first.")
    return df["close"]


def generate_synthetic_data(n: int = 100_000, seed: int = 42) -> pd.Series:
    """Generate synthetic mean-reverting price series for testing."""
    rng = np.random.RandomState(seed)
    # Ornstein-Uhlenbeck process (mean-reverting)
    mu = 4500.0
    theta = 0.05   # Speed of mean reversion
    sigma = 2.0
    dt = 1.0

    prices = np.empty(n)
    prices[0] = mu
    for i in range(1, n):
        prices[i] = (
            prices[i - 1]
            + theta * (mu - prices[i - 1]) * dt
            + sigma * np.sqrt(dt) * rng.randn()
        )

    idx = pd.date_range("2022-01-01", periods=n, freq="5min")  # synthetic always 5min
    return pd.Series(prices, index=idx, name="close")


# Bar size to minutes mapping
_BAR_MINUTES = {"1min": 1, "5min": 5, "15min": 15, "1h": 60}


# ── ADF Test ─────────────────────────────────────────────────────────────

def run_adf_test(log_returns: np.ndarray) -> dict:
    """Run Augmented Dickey-Fuller test on log returns."""
    result = adfuller(log_returns, maxlag=None, regression="c", autolag="AIC")
    return {
        "adf_statistic": result[0],
        "p_value": result[1],
        "lags_used": result[2],
        "n_obs": result[3],
        "critical_values": result[4],
        "is_stationary": result[1] < 0.05,
    }


# ── Hurst Exponent ───────────────────────────────────────────────────────

def hurst_exponent(series: np.ndarray, max_lag: int = 100) -> float:
    """Compute Hurst exponent via rescaled range (R/S) method.

    H < 0.5: mean-reverting
    H = 0.5: random walk
    H > 0.5: trending
    """
    lags = range(2, max_lag)
    rs_values = []

    for lag in lags:
        # Split into non-overlapping subseries
        n_chunks = len(series) // lag
        if n_chunks == 0:
            break

        rs_for_lag = []
        for j in range(n_chunks):
            chunk = series[j * lag : (j + 1) * lag]
            mean_chunk = np.mean(chunk)
            deviations = chunk - mean_chunk
            cumulative = np.cumsum(deviations)
            r = np.max(cumulative) - np.min(cumulative)
            s = np.std(chunk, ddof=1)
            if s > 0:
                rs_for_lag.append(r / s)

        if rs_for_lag:
            rs_values.append((lag, np.mean(rs_for_lag)))

    if len(rs_values) < 2:
        return 0.5

    log_lags = np.log([v[0] for v in rs_values])
    log_rs = np.log([v[1] for v in rs_values])

    slope, _, _, _, _ = stats.linregress(log_lags, log_rs)
    return float(slope)


# ── Half-Life of Mean Reversion ──────────────────────────────────────────

def half_life_ols(prices: np.ndarray) -> float:
    """Estimate half-life of mean reversion via OLS on ΔP = α + β * P_{t-1} + ε.

    Half-life = -ln(2) / β
    """
    lagged = prices[:-1]
    delta = np.diff(prices)

    # OLS: delta = alpha + beta * lagged
    X = np.column_stack([np.ones(len(lagged)), lagged])
    beta_hat = np.linalg.lstsq(X, delta, rcond=None)[0]
    beta = beta_hat[1]

    if beta >= 0:
        return float("inf")  # Not mean-reverting
    return float(-np.log(2) / beta)


# ── Main ─────────────────────────────────────────────────────────────────

def main(use_synthetic: bool = False, bar_size: str = "5min") -> dict:
    """Run complete stationarity analysis and return results dict."""
    bar_min = _BAR_MINUTES.get(bar_size, 5)
    print("=" * 70)
    print(f"  MES {bar_size} Stationarity Analysis")
    print("=" * 70)

    if use_synthetic:
        print("\n[INFO] Using synthetic mean-reverting data (Ornstein-Uhlenbeck)")
        prices = generate_synthetic_data()
    else:
        try:
            prices = load_data_questdb(bar_size)
        except Exception as exc:
            print(f"\n[WARN] QuestDB unavailable ({exc}), falling back to synthetic data")
            prices = generate_synthetic_data()

    prices_arr = prices.values.astype(float)
    log_returns = np.diff(np.log(prices_arr))
    log_returns = log_returns[np.isfinite(log_returns)]

    print(f"\nData: {len(prices_arr)} bars, {len(log_returns)} log returns")
    print(f"Price range: {prices_arr.min():.2f} - {prices_arr.max():.2f}")

    # ── ADF Test ──
    print("\n" + "-" * 50)
    print("Augmented Dickey-Fuller Test (log returns)")
    print("-" * 50)

    adf = run_adf_test(log_returns)
    print(f"  ADF Statistic : {adf['adf_statistic']:.4f}")
    print(f"  p-value       : {adf['p_value']:.6f}")
    print(f"  Lags Used     : {adf['lags_used']}")
    print(f"  N Observations: {adf['n_obs']}")
    print("  Critical Values:")
    for level, value in adf["critical_values"].items():
        print(f"    {level}: {value:.4f}")

    if adf["is_stationary"]:
        print("\n  ✓ Log returns ARE stationary (p < 0.05)")
    else:
        print("\n  ✗ WARNING: Log returns are NOT stationary (p >= 0.05)")

    # ── Hurst Exponent ──
    print("\n" + "-" * 50)
    print("Hurst Exponent (Rescaled Range)")
    print("-" * 50)

    H = hurst_exponent(log_returns, max_lag=min(200, len(log_returns) // 10))
    print(f"  H = {H:.4f}")
    if H < 0.5:
        print(f"  ✓ Mean-reverting (H < 0.5)")
    elif H == 0.5:
        print(f"  ~ Random walk (H ≈ 0.5)")
    else:
        print(f"  ✗ WARNING: Trending (H > 0.5) — mean reversion may not be valid")

    # ── Half-Life ──
    print("\n" + "-" * 50)
    print("Half-Life of Mean Reversion (OLS)")
    print("-" * 50)

    hl = half_life_ols(prices_arr)
    print(f"  Half-life = {hl:.1f} bars")
    if hl < float("inf"):
        print(f"  At {bar_size} bars, that's ~{hl * bar_min:.0f} minutes or ~{hl * bar_min / 60:.1f} hours")
    else:
        print("  ✗ WARNING: No mean reversion detected (β >= 0)")

    # ── Summary ──
    print("\n" + "=" * 70)
    stationarity = "stationary" if adf["is_stationary"] else "non-stationary"
    print(
        f"MES {bar_size} returns are [{stationarity}], "
        f"H=[{H:.4f}], half-life=[{hl:.1f}] bars"
    )

    warnings = []
    if H >= 0.5:
        warnings.append("Hurst exponent >= 0.5 (trending, not mean-reverting)")
    if not adf["is_stationary"]:
        warnings.append("ADF p-value >= 0.05 (non-stationary)")

    if warnings:
        print("\n⚠ WARNINGS:")
        for w in warnings:
            print(f"  - {w}")
        print("  Mean reversion strategy may not be valid for this data.")
    else:
        print("\n✓ Data supports mean-reversion hypothesis.")
    print("=" * 70)

    return {
        "adf": adf,
        "hurst": H,
        "half_life_bars": hl,
        "is_mean_reverting": H < 0.5 and adf["is_stationary"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument(
        "--bar-size",
        default="5min",
        choices=list(_BAR_MINUTES.keys()),
        help="Bar size to analyse (default: 5min)",
    )
    args = parser.parse_args()
    main(use_synthetic=args.synthetic, bar_size=args.bar_size)
