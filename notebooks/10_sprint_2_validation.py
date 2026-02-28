"""
10 — Sprint 2 Validation Report.

Comprehensive validation of Strategy B + Risk Engine.

Usage:
    uv run python notebooks/10_sprint_2_validation.py --synthetic
    uv run python notebooks/10_sprint_2_validation.py --bar-size 5min
"""

import argparse
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtesting.metrics import annualized_sharpe, max_drawdown, compute_all_metrics
from backtesting.monte_carlo import MonteCarloSimulator
from risk.circuit_breaker import CircuitBreaker, CircuitBreakerState
from risk.risk_manager import RiskManager
from risk.dynamic_stops import DynamicStopCalculator
from strategies.orb_strategy import VIXAdaptiveORBStrategy
from strategies.mean_reversion import MESMeanReversionStrategy

ET = pytz.timezone("US/Eastern")


def _check(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


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
        vix_df = qdb.get_bars("VIX", "1day", limit=500_000)
        if not vix_df.empty:
            vix = pd.Series(vix_df["close"].values, index=vix_df.index.date, name="vix")
        else:
            vix = generate_synthetic_vix(df)
        return df, vix
    except Exception:
        df = generate_synthetic_ohlcv()
        return df, generate_synthetic_vix(df)


def get_daily_returns(trades, dates, capital):
    daily = pd.Series(0.0, index=dates)
    for t in trades:
        d = t.entry_time
        if hasattr(d, "tzinfo") and d.tzinfo is not None:
            d = d.replace(tzinfo=None)
        d = pd.Timestamp(d.date() if hasattr(d, "date") else d)
        if d in daily.index:
            daily.loc[d] += t.pnl / capital
    return daily


# ── Risk Engine Tests ───────────────────────────────────────────────

def run_risk_tests():
    """Run all risk engine validation tests."""
    results = {}

    # CB L1
    cb = CircuitBreaker(capital=20_000.0)
    cb.on_trade_update(-200.0)
    results["Circuit breaker L1"] = cb.state == CircuitBreakerState.REDUCED

    # CB L2
    cb2 = CircuitBreaker(capital=20_000.0)
    cb2.on_trade_update(-400.0)
    results["Circuit breaker L2"] = cb2.state == CircuitBreakerState.NO_NEW_ENTRIES

    # CB L3
    cb3 = CircuitBreaker(capital=20_000.0)
    cb3.on_trade_update(-600.0)
    results["Circuit breaker L3"] = cb3.state == CircuitBreakerState.HALTED and cb3.should_flatten_all()

    # Weekly limit
    cb4 = CircuitBreaker(capital=20_000.0, weekly_limit_pct=0.05)
    for _ in range(5):
        cb4.on_trade_update(-220.0)
        cb4.on_new_day()
    results["Weekly limit"] = cb4._weekly_size_override == 0.5

    # Position sizing
    rm = RiskManager(total_capital=20_000.0)
    sz = rm.position_size(20_000.0, 0.01, 5420.0, 5412.0, 5.0, 2455.0)
    results["Position sizing"] = sz > 0 and sz <= 4

    # Margin cap
    rm2 = RiskManager(total_capital=10_000.0)
    big = rm2.position_size(10_000.0, 0.01, 5420.0, 5419.99, 5.0, 2455.0)
    margin_cap = int(10_000.0 * 0.5 / 2455.0)
    results["Margin cap enforcement"] = big <= margin_cap

    return results


# ── Main ────────────────────────────────────────────────────────────

def main(use_synthetic: bool = False, bar_size: str = "5min"):
    df, vix_series = load_data(use_synthetic, bar_size)
    n_days = len(set(df.index.date))

    # Run strategies
    strat_a = MESMeanReversionStrategy(
        params={"long_only": True, "max_daily_losses": 3, "max_daily_loss_dollars": 500.0},
        capital=10_000.0,
    )
    strat_a.generate_signals(df, vix_series=vix_series)

    strat_b = VIXAdaptiveORBStrategy(capital=20_000.0)
    strat_b.generate_signals(df, vix_series=vix_series)

    # Strategy B metrics
    b_trades = strat_b.trades
    b_pnls = np.array([t.pnl for t in b_trades]) if b_trades else np.array([])
    b_has_trades = len(b_trades) >= 2
    if b_has_trades:
        b_equity = np.cumsum(np.concatenate([[20_000.0], b_pnls]))
        b_returns = np.diff(b_equity) / b_equity[:-1]
        b_metrics = compute_all_metrics(b_returns, b_pnls, b_equity)
    else:
        b_metrics = {"sharpe": 0, "win_rate": 0, "profit_factor": 0, "max_drawdown": 0}

    b_sharpe = b_metrics.get("sharpe", 0)
    b_win_rate = b_metrics.get("win_rate", 0)
    b_pf = b_metrics.get("profit_factor", 0)
    b_dd = b_metrics.get("max_drawdown", 0)
    b_trades_per_day = len(b_trades) / n_days if n_days > 0 else 0
    b_avg_pnl = float(np.mean(b_pnls)) if len(b_pnls) > 0 else 0

    # Correlation
    dates = pd.bdate_range(min(df.index.date), max(df.index.date))
    daily_a = get_daily_returns(strat_a.trades, dates, 10_000.0)
    daily_b = get_daily_returns(b_trades, dates, 20_000.0)

    active = (daily_a != 0) | (daily_b != 0)
    da = daily_a[active]
    db = daily_b[active]

    if len(da) > 5:
        pearson = da.corr(db)
        spearman = da.rank().corr(db.rank())
    else:
        pearson = spearman = 0.0

    # Combined portfolio
    combined = 0.40 * daily_a + 0.30 * daily_b
    sr_combined = annualized_sharpe(combined.values)
    sr_a = annualized_sharpe(daily_a.values)
    sr_b = annualized_sharpe(daily_b.values)
    equity_c = np.cumprod(1.0 + combined.values) * 20_000
    dd_combined = max_drawdown(equity_c)

    # Monte Carlo on combined
    combined_arr = combined.values[combined.values != 0]
    mc_5_sharpe = 0.0
    mc_95_dd = 0.0
    if len(combined_arr) > 5:
        mc = MonteCarloSimulator(n_paths=1000, seed=42)
        # Convert returns to dollar PnL for MC
        mc_pnls = combined_arr * 20_000
        mc_result = mc.simulate(mc_pnls, initial_capital=20_000.0)
        mc_5_sharpe = mc_result["sharpe_pcts"][5]
        mc_95_dd = mc_result["max_drawdown_pcts"][95]

    # Risk engine tests
    risk_results = run_risk_tests()

    # ── Print Report ──
    print()
    print("=" * 65)
    print(" SPRINT 2 VALIDATION REPORT")
    print("=" * 65)

    print(f"\n STRATEGY B: VIX-ADAPTIVE ORB")
    print(f"   Sharpe:               {b_sharpe:+7.3f}     (informational)")
    print(f"   Avg trades per day:   {b_trades_per_day:7.1f}     {_check(0.1 <= b_trades_per_day <= 3.0)}")
    print(f"   Avg profit per trade: ${b_avg_pnl:7.2f}   (informational)")
    print(f"   Win Rate:             {b_win_rate*100:6.1f}%")
    print(f"   Profit Factor:        {b_pf:7.2f}")
    print(f"   Max Drawdown:         {b_dd*100:6.1f}%")
    print(f"   Total Trades:         {len(b_trades):7d}")

    print(f"\n CORRELATION WITH STRATEGY A")
    print(f"   Pearson correlation:  {pearson:+7.3f}     {_check(abs(pearson) < 0.30)}")
    print(f"   Spearman correlation: {spearman:+7.3f}     {_check(abs(spearman) < 0.30)}")

    print(f"\n COMBINED PORTFOLIO (A=40%, B=30%)")
    print(f"   Strategy A Sharpe:    {sr_a:+7.3f}")
    print(f"   Strategy B Sharpe:    {sr_b:+7.3f}")
    print(f"   Portfolio Sharpe:     {sr_combined:+7.3f}     (informational)")
    print(f"   Portfolio max DD:     {dd_combined*100:6.1f}%     {_check(dd_combined < 0.20)}")
    if len(combined_arr) > 5:
        print(f"   MC 5th pctl Sharpe:   {mc_5_sharpe:+7.3f}     (informational)")
        print(f"   MC 95th pctl max DD:  {mc_95_dd*100:6.1f}%     {_check(mc_95_dd < 0.20)}")

    print(f"\n RISK ENGINE")
    for name, passed in risk_results.items():
        print(f"   {name:30s}: {_check(passed)}")

    # Overall count
    checks = [
        0.1 <= b_trades_per_day <= 3.0,
        abs(pearson) < 0.30,
        abs(spearman) < 0.30,
        dd_combined < 0.20,
    ]
    checks.extend(risk_results.values())
    if len(combined_arr) > 5:
        checks.append(mc_95_dd < 0.20)

    n_passed = sum(checks)
    n_total = len(checks)

    print(f"\n OVERALL: {n_passed}/{n_total} PASSED")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--bar-size", default="5min", choices=["1min", "5min"])
    args = parser.parse_args()
    main(use_synthetic=args.synthetic, bar_size=args.bar_size)
