"""
08 — Risk Engine Validation: Validate circuit breaker, position sizing,
     Kelly criterion, and weekly limits with simulated scenarios.

Usage:
    uv run python notebooks/08_risk_engine_validation.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from risk.circuit_breaker import CircuitBreaker, CircuitBreakerState
from risk.risk_manager import RiskManager
from risk.dynamic_stops import DynamicStopCalculator


def _pass_fail(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def test_circuit_breaker_transitions():
    """TEST 1: Circuit Breaker State Transitions."""
    print("\n  TEST 1: Circuit Breaker State Transitions")
    print("  " + "-" * 50)

    capital = 20_000.0
    cb = CircuitBreaker(capital=capital)

    trade_pnls = [+50, -30, -80, -50, -100, +20, -60]
    cumulative = 0.0
    all_correct = True

    for i, pnl in enumerate(trade_pnls):
        cumulative += pnl
        state = cb.on_trade_update(pnl)
        loss_pct = cb.get_daily_loss_pct()
        print(f"    Trade {i+1}: PnL=${pnl:+.0f}, Cumulative=${cumulative:+.0f}, "
              f"Loss%={loss_pct*100:.2f}%, State={state.value}")

        # Verify state based on cumulative loss
        if loss_pct >= 0.03:
            if state != CircuitBreakerState.HALTED:
                all_correct = False
        elif loss_pct >= 0.02:
            if state not in (CircuitBreakerState.NO_NEW_ENTRIES, CircuitBreakerState.HALTED):
                all_correct = False
        elif loss_pct >= 0.01:
            if state not in (CircuitBreakerState.REDUCED, CircuitBreakerState.NO_NEW_ENTRIES, CircuitBreakerState.HALTED):
                all_correct = False

    # Push to Level 3
    cb2 = CircuitBreaker(capital=capital)
    cb2.on_trade_update(-200.0)  # 1% -> REDUCED
    assert cb2.state == CircuitBreakerState.REDUCED
    cb2.on_trade_update(-200.0)  # 2% -> NO_NEW_ENTRIES
    assert cb2.state == CircuitBreakerState.NO_NEW_ENTRIES
    cb2.on_trade_update(-200.0)  # 3% -> HALTED
    assert cb2.state == CircuitBreakerState.HALTED
    assert cb2.should_flatten_all()

    print(f"  Result: {_pass_fail(all_correct)}")
    return all_correct


def test_position_sizing_consistency():
    """TEST 2: Position Sizing Consistency."""
    print("\n  TEST 2: Position Sizing Consistency")
    print("  " + "-" * 50)

    calc = DynamicStopCalculator()
    rng = np.random.RandomState(42)
    all_correct = True
    n_tests = 100

    capital = 20_000.0
    risk_pct = 0.01

    for i in range(n_tests):
        entry = 5400 + rng.uniform(-50, 50)
        atr = rng.uniform(2.0, 15.0)
        mult = rng.uniform(1.5, 4.0)

        result = calc.compute_volatility_adjusted_size(
            capital=capital, risk_pct=risk_pct, entry=entry, atr=atr,
            side="LONG", multiplier=mult,
        )

        # Check margin never exceeds 50%
        if result["margin_pct"] > 0.501:
            all_correct = False
            print(f"    FAIL: margin_pct={result['margin_pct']:.3f} > 0.50")

        # Check dollar risk is reasonable
        if result["contracts"] > 0:
            actual_risk = result["stop_distance"] * 5.0 * result["contracts"]
            target_risk = capital * risk_pct
            # Due to floor(), actual risk should be <= target risk
            if actual_risk > target_risk * 1.5:
                all_correct = False
                print(f"    FAIL: actual_risk=${actual_risk:.0f} >> target=${target_risk:.0f}")

    # Verify wider stops -> fewer contracts
    narrow = calc.compute_volatility_adjusted_size(capital, risk_pct, 5420, 3.0, "LONG", 2.0)
    wide = calc.compute_volatility_adjusted_size(capital, risk_pct, 5420, 10.0, "LONG", 2.0)
    wider_has_fewer = wide["contracts"] <= narrow["contracts"]
    if not wider_has_fewer:
        all_correct = False
        print(f"    FAIL: wider stop has more contracts ({wide['contracts']} > {narrow['contracts']})")

    print(f"  Tested {n_tests} random (entry, ATR) pairs")
    print(f"  Wider stops -> fewer contracts: {_pass_fail(wider_has_fewer)}")
    print(f"  Result: {_pass_fail(all_correct)}")
    return all_correct


def test_kelly_criterion():
    """TEST 3: Kelly Criterion."""
    print("\n  TEST 3: Kelly Criterion")
    print("  " + "-" * 50)

    rm = RiskManager(total_capital=20_000.0)
    all_correct = True

    # Case 1: Strong edge
    qk1 = rm.quarter_kelly(0.55, 50.0, 30.0)
    # b = 50/30 = 1.667, full_kelly = (0.55*1.667 - 0.45)/1.667 = 0.280
    # quarter = 0.070 -> clamp to 0.02
    case1 = qk1 == 0.02
    print(f"  Strong edge (WR=55%, W/L=50/30): qK={qk1:.4f} {'PASS' if case1 else 'FAIL'}")
    if not case1:
        all_correct = False

    # Case 2: No edge
    qk2 = rm.quarter_kelly(0.45, 40.0, 50.0)
    case2 = qk2 == 0.0
    print(f"  No edge (WR=45%, W/L=40/50):     qK={qk2:.4f} {'PASS' if case2 else 'FAIL'}")
    if not case2:
        all_correct = False

    # Case 3: Small edge
    qk3 = rm.quarter_kelly(0.52, 35.0, 30.0)
    case3 = 0.005 <= qk3 <= 0.02
    print(f"  Small edge (WR=52%, W/L=35/30):  qK={qk3:.4f} {'PASS' if case3 else 'FAIL'}")
    if not case3:
        all_correct = False

    print(f"  Result: {_pass_fail(all_correct)}")
    return all_correct


def test_weekly_limit():
    """TEST 4: Weekly Limit."""
    print("\n  TEST 4: Weekly Limit")
    print("  " + "-" * 50)

    cb = CircuitBreaker(capital=20_000.0, weekly_limit_pct=0.05)
    all_correct = True

    daily_losses = [-300, -200, -200, -160, -200]
    cumulative_week = 0.0

    for day, loss in enumerate(daily_losses, 1):
        cb.on_trade_update(loss)
        cumulative_week += loss
        weekly_pct = cb.get_weekly_loss_pct()
        print(f"    Day {day}: loss=${loss:+.0f}, weekly=${cumulative_week:+.0f}, "
              f"weekly%={weekly_pct*100:.1f}%, state={cb.state.value}")
        cb.on_new_day()

    # After 5 days of losses totaling -$1060 = 5.3% -> weekly limit
    weekly_triggered = cb._weekly_size_override == 0.5
    print(f"  Weekly size override: {cb._weekly_size_override} {'PASS' if weekly_triggered else 'FAIL'}")
    if not weekly_triggered:
        all_correct = False

    # New week should reset
    cb.on_new_week()
    weekly_reset = cb._weekly_size_override == 1.0 and cb.state == CircuitBreakerState.NORMAL
    print(f"  After new week: override={cb._weekly_size_override}, state={cb.state.value} "
          f"{'PASS' if weekly_reset else 'FAIL'}")
    if not weekly_reset:
        all_correct = False

    print(f"  Result: {_pass_fail(all_correct)}")
    return all_correct


def test_exposure_limit():
    """TEST 5: Cross-Strategy Exposure."""
    print("\n  TEST 5: Cross-Strategy Exposure")
    print("  " + "-" * 50)

    rm = RiskManager(total_capital=20_000.0, max_portfolio_exposure=0.60)

    # Strategy A opens 3 MES contracts
    rm.register_position("strategy_a", 3, 5420.0, 5.0)
    exp_a = rm.get_total_exposure()
    print(f"  After Strategy A (3 contracts): exposure={exp_a*100:.1f}%")

    can_b = rm.can_open_position("strategy_b")
    print(f"  Strategy B can_open: {can_b}")

    # The exposure is 3 * 5420 * 5 / 20000 = 4.065 = 406.5% -> blocks
    blocked = not can_b
    print(f"  Correctly blocked: {_pass_fail(blocked)}")

    # Close Strategy A
    rm.close_position("strategy_a", 0)
    exp_after = rm.get_total_exposure()
    can_b_after = rm.can_open_position("strategy_b")
    print(f"  After closing 1 position: exposure={exp_after*100:.1f}%, can_open={can_b_after}")

    return blocked


def main():
    print("=" * 70)
    print("  Risk Engine Validation")
    print("=" * 70)

    results = []
    results.append(("Circuit Breaker Transitions", test_circuit_breaker_transitions()))
    results.append(("Position Sizing Consistency", test_position_sizing_consistency()))
    results.append(("Kelly Criterion", test_kelly_criterion()))
    results.append(("Weekly Limit", test_weekly_limit()))
    results.append(("Cross-Strategy Exposure", test_exposure_limit()))

    print(f"\n{'='*70}")
    print("  SUMMARY")
    print("=" * 70)
    passed = sum(1 for _, r in results if r)
    total = len(results)
    for name, result in results:
        print(f"  {name:35s}: {_pass_fail(result)}")
    print(f"\n  {passed}/{total} PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
