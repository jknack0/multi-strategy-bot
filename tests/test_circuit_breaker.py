"""Tests for CircuitBreaker risk management."""

import pytest
from risk.circuit_breaker import CircuitBreaker, CircuitBreakerState


class TestCircuitBreakerTransitions:
    """Test state transitions at correct P&L levels."""

    def test_starts_normal(self):
        cb = CircuitBreaker(capital=20_000.0)
        assert cb.state == CircuitBreakerState.NORMAL

    def test_level_1_at_1_pct(self):
        """1% daily loss triggers REDUCED."""
        cb = CircuitBreaker(capital=20_000.0)
        # Lose $200 = 1% of $20K
        cb.on_trade_update(-200.0)
        assert cb.state == CircuitBreakerState.REDUCED

    def test_level_2_at_2_pct(self):
        """2% daily loss triggers NO_NEW_ENTRIES."""
        cb = CircuitBreaker(capital=20_000.0)
        cb.on_trade_update(-200.0)  # 1% → REDUCED
        cb.on_trade_update(-200.0)  # 2% → NO_NEW_ENTRIES
        assert cb.state == CircuitBreakerState.NO_NEW_ENTRIES

    def test_level_3_at_3_pct(self):
        """3% daily loss triggers HALTED."""
        cb = CircuitBreaker(capital=20_000.0)
        cb.on_trade_update(-600.0)  # 3% at once → HALTED
        assert cb.state == CircuitBreakerState.HALTED

    def test_never_de_escalates_intraday(self):
        """Winning trade after loss shouldn't lower the circuit breaker level."""
        cb = CircuitBreaker(capital=20_000.0)
        cb.on_trade_update(-200.0)  # → REDUCED
        assert cb.state == CircuitBreakerState.REDUCED
        cb.on_trade_update(100.0)  # Win $100 — daily loss now $100
        assert cb.state == CircuitBreakerState.REDUCED  # Still REDUCED

    def test_gradual_escalation(self):
        """Small losses accumulate through levels."""
        cb = CircuitBreaker(capital=20_000.0)
        # -$50 each trade, need 4 for L1, 8 for L2, 12 for L3
        for _ in range(3):
            cb.on_trade_update(-50.0)
        assert cb.state == CircuitBreakerState.NORMAL  # $150 = 0.75%

        cb.on_trade_update(-50.0)  # $200 = 1.0%
        assert cb.state == CircuitBreakerState.REDUCED

        for _ in range(4):
            cb.on_trade_update(-50.0)  # $400 = 2.0%
        assert cb.state == CircuitBreakerState.NO_NEW_ENTRIES

        for _ in range(4):
            cb.on_trade_update(-50.0)  # $600 = 3.0%
        assert cb.state == CircuitBreakerState.HALTED


class TestHaltedFlatten:
    """HALTED state triggers flatten signal."""

    def test_should_flatten_on_halt(self):
        cb = CircuitBreaker(capital=20_000.0)
        cb.on_trade_update(-600.0)
        assert cb.should_flatten_all() is True

    def test_flatten_resets_after_check(self):
        """should_flatten_all only true once per HALTED transition."""
        cb = CircuitBreaker(capital=20_000.0)
        cb.on_trade_update(-600.0)
        assert cb.should_flatten_all() is True
        # Another trade update shouldn't re-trigger
        cb.on_trade_update(-10.0)
        assert cb.should_flatten_all() is False


class TestDailyReset:
    """Daily reset restores NORMAL."""

    def test_daily_reset_to_normal(self):
        cb = CircuitBreaker(capital=20_000.0)
        cb.on_trade_update(-300.0)  # → REDUCED
        assert cb.state == CircuitBreakerState.REDUCED
        cb.on_new_day()
        assert cb.state == CircuitBreakerState.NORMAL
        assert cb.get_daily_loss_pct() == 0.0

    def test_daily_reset_clears_pnl(self):
        cb = CircuitBreaker(capital=20_000.0)
        cb.on_trade_update(-100.0)
        cb.on_new_day()
        assert cb.get_daily_loss_pct() == 0.0


class TestWeeklyLimit:
    """Weekly limit persists across daily resets."""

    def test_weekly_limit_reduces_size(self):
        """After 5% weekly loss, next day stays REDUCED."""
        cb = CircuitBreaker(capital=20_000.0, weekly_limit_pct=0.05)
        # Simulate 5 days of losses
        for _ in range(5):
            cb.on_trade_update(-200.0)  # -$200/day
            cb.on_new_day()
        # Weekly P&L = -$1000 = 5% of $20K
        assert cb.state == CircuitBreakerState.REDUCED
        assert cb._weekly_size_override == 0.5

    def test_weekly_reset_clears_limit(self):
        cb = CircuitBreaker(capital=20_000.0)
        cb.on_trade_update(-1000.0)
        cb.on_new_day()  # triggers weekly limit check
        cb.on_new_week()
        assert cb.state == CircuitBreakerState.NORMAL
        assert cb._weekly_size_override == 1.0


class TestSizeMultiplier:
    """Size multiplier returns correct value per state."""

    def test_normal_multiplier(self):
        cb = CircuitBreaker(capital=20_000.0)
        assert cb.get_size_multiplier() == 1.0

    def test_reduced_multiplier(self):
        cb = CircuitBreaker(capital=20_000.0)
        cb.on_trade_update(-200.0)
        assert cb.get_size_multiplier() == 0.5

    def test_no_new_entries_multiplier(self):
        cb = CircuitBreaker(capital=20_000.0)
        cb.on_trade_update(-400.0)
        assert cb.get_size_multiplier() == 0.0

    def test_halted_multiplier(self):
        cb = CircuitBreaker(capital=20_000.0)
        cb.on_trade_update(-600.0)
        assert cb.get_size_multiplier() == 0.0

    def test_weekly_override_stacks(self):
        """Weekly override 0.5 × REDUCED 0.5 = 0.25."""
        cb = CircuitBreaker(capital=20_000.0)
        cb._weekly_size_override = 0.5
        cb._state = CircuitBreakerState.REDUCED
        assert cb.get_size_multiplier() == 0.25


class TestCanOpenPosition:
    """can_open_position blocks appropriately."""

    def test_normal_allows(self):
        cb = CircuitBreaker(capital=20_000.0)
        assert cb.can_open_position() is True

    def test_reduced_allows(self):
        cb = CircuitBreaker(capital=20_000.0)
        cb.on_trade_update(-200.0)
        assert cb.can_open_position() is True

    def test_no_new_entries_blocks(self):
        cb = CircuitBreaker(capital=20_000.0)
        cb.on_trade_update(-400.0)
        assert cb.can_open_position() is False

    def test_halted_blocks(self):
        cb = CircuitBreaker(capital=20_000.0)
        cb.on_trade_update(-600.0)
        assert cb.can_open_position() is False


class TestStateHistory:
    """State history logging."""

    def test_records_transitions(self):
        cb = CircuitBreaker(capital=20_000.0)
        cb.on_trade_update(-200.0)  # → REDUCED
        cb.on_trade_update(-200.0)  # → NO_NEW_ENTRIES
        assert len(cb.state_history) == 2
        assert cb.state_history[0]["from"] == "normal"
        assert cb.state_history[0]["to"] == "reduced"
        assert cb.state_history[1]["from"] == "reduced"
        assert cb.state_history[1]["to"] == "no_new"

    def test_no_duplicate_transitions(self):
        """Same state shouldn't create a history entry."""
        cb = CircuitBreaker(capital=20_000.0)
        cb.on_trade_update(-200.0)  # → REDUCED
        cb.on_trade_update(-10.0)   # still REDUCED
        assert len(cb.state_history) == 1
