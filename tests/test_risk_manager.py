"""Tests for RiskManager."""

import pytest
from risk.risk_manager import RiskManager


class TestPositionSizing:
    """Position sizing math tests."""

    def test_basic_sizing(self):
        """contracts = floor(dollar_risk / (stop_dist * point_value)) * scale."""
        rm = RiskManager(total_capital=20_000.0)
        # dollar_risk = 20000 * 0.01 = 200
        # stop_dist = |5420 - 5412| = 8 pts, point_value = 5.0
        # raw = floor(200 / (8 * 5.0)) = floor(5.0) = 5
        # margin_cap = floor(20000 * 0.5 / 2455) = 4
        contracts = rm.position_size(20_000.0, 0.01, 5420.0, 5412.0, 5.0, 2455.0)
        assert contracts == 4  # margin capped

    def test_sizing_with_scale(self):
        rm = RiskManager(total_capital=20_000.0)
        # Same but scale=0.5: floor(5.0 * 0.5) = 2
        contracts = rm.position_size(20_000.0, 0.01, 5420.0, 5412.0, 5.0, 2455.0, scale=0.5)
        assert contracts == 2

    def test_zero_stop_returns_zero(self):
        rm = RiskManager(total_capital=20_000.0)
        assert rm.position_size(20_000.0, 0.01, 5420.0, 5420.0) == 0

    def test_margin_cap_enforced(self):
        """Contracts cannot exceed 50% margin capacity."""
        rm = RiskManager(total_capital=10_000.0)
        # margin_cap = 10000 * 0.5 / 2455 = 2
        # With tiny stop → huge raw contracts
        contracts = rm.position_size(10_000.0, 0.01, 5420.0, 5419.99, 5.0, 2455.0)
        assert contracts <= 2


class TestQuarterKelly:
    """Quarter Kelly criterion tests."""

    def test_positive_edge(self):
        """win_rate=0.55, avg_win=$50, avg_loss=$30."""
        rm = RiskManager(total_capital=20_000.0)
        # b = 50/30 = 1.667
        # full_kelly = (0.55 * 1.667 - 0.45) / 1.667 = 0.280
        # quarter_kelly = 0.070 → clamp to 0.02
        qk = rm.quarter_kelly(0.55, 50.0, 30.0)
        assert qk == 0.02  # clamped to max

    def test_no_edge(self):
        """win_rate=0.45, avg_win=$40, avg_loss=$50 → no edge."""
        rm = RiskManager(total_capital=20_000.0)
        # b = 40/50 = 0.8
        # full_kelly = (0.45 * 0.8 - 0.55) / 0.8 = -0.2375 → negative
        qk = rm.quarter_kelly(0.45, 40.0, 50.0)
        assert qk == 0.0

    def test_moderate_edge(self):
        """Moderate edge should produce value in [0.005, 0.02] range."""
        rm = RiskManager(total_capital=20_000.0)
        qk = rm.quarter_kelly(0.52, 40.0, 35.0)
        assert 0.005 <= qk <= 0.02

    def test_zero_loss_returns_zero(self):
        rm = RiskManager(total_capital=20_000.0)
        assert rm.quarter_kelly(0.5, 50.0, 0.0) == 0.0


class TestStrategyCapital:
    """Strategy capital allocation tests."""

    def test_default_allocations(self):
        rm = RiskManager(total_capital=100_000.0)
        assert rm.get_strategy_capital("strategy_a") == pytest.approx(40_000.0)
        assert rm.get_strategy_capital("strategy_b") == pytest.approx(30_000.0)
        assert rm.get_strategy_capital("strategy_c") == pytest.approx(30_000.0)

    def test_custom_allocations(self):
        rm = RiskManager(
            total_capital=100_000.0,
            strategy_allocations={"strat_a": 0.60, "strat_b": 0.40},
        )
        assert rm.get_strategy_capital("strat_a") == pytest.approx(60_000.0)
        assert rm.get_strategy_capital("strat_b") == pytest.approx(40_000.0)

    def test_unknown_strategy(self):
        rm = RiskManager(total_capital=100_000.0)
        assert rm.get_strategy_capital("unknown") == 0.0


class TestExposureTracking:
    """Exposure limit enforcement."""

    def test_exposure_calculation(self):
        rm = RiskManager(total_capital=20_000.0)
        rm.register_position("strat_a", 2, 5420.0, 5.0)
        # notional = 2 * 5420.0 * 5.0 = 54,200
        # exposure = 54200 / 20000 = 2.71
        assert rm.get_total_exposure() > 0

    def test_can_open_blocked_by_exposure(self):
        """Should block when exposure exceeds limit."""
        rm = RiskManager(total_capital=20_000.0, max_portfolio_exposure=0.60)
        # Register huge position
        rm.register_position("strat_a", 100, 5420.0, 5.0)
        assert rm.can_open_position("strat_b") is False

    def test_can_open_allowed_when_no_positions(self):
        rm = RiskManager(total_capital=20_000.0)
        assert rm.can_open_position("strat_a") is True

    def test_close_position_reduces_exposure(self):
        rm = RiskManager(total_capital=20_000.0)
        rm.register_position("strat_a", 2, 5420.0, 5.0)
        exp_before = rm.get_total_exposure()
        rm.close_position("strat_a", 0)
        exp_after = rm.get_total_exposure()
        assert exp_after < exp_before


class TestCircuitBreakerIntegration:
    """Risk manager + circuit breaker integration."""

    def test_pnl_update_triggers_cb(self):
        rm = RiskManager(total_capital=20_000.0)
        rm.update_pnl("strat_a", -600.0)  # 3% → HALTED
        assert rm.circuit_breaker.state.value == "halted"
        assert rm.can_open_position("strat_a") is False

    def test_new_day_resets(self):
        rm = RiskManager(total_capital=20_000.0)
        rm.update_pnl("strat_a", -300.0)
        rm.on_new_day()
        assert rm.circuit_breaker.state.value == "normal"
        assert rm.get_strategy_pnl("strat_a") == 0.0
