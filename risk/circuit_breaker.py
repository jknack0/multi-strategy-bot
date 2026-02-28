"""
Three-level circuit breaker system to prevent catastrophic daily losses.

Level 1 (1% daily loss): REDUCED — position sizes cut 50%
Level 2 (2% daily loss): NO_NEW_ENTRIES — only manage existing positions
Level 3 (3% daily loss): HALTED — flatten all, no trading rest of day

Weekly limit (5%): reduce all sizes 50% for the next week.
"""

import logging
from datetime import datetime
from enum import Enum
from typing import List, Optional

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    NORMAL = "normal"
    REDUCED = "reduced"
    NO_NEW_ENTRIES = "no_new"
    HALTED = "halted"


class CircuitBreaker:
    """Three-level circuit breaker with weekly limit."""

    def __init__(
        self,
        capital: float,
        level_1_pct: float = 0.01,
        level_2_pct: float = 0.02,
        level_3_pct: float = 0.03,
        weekly_limit_pct: float = 0.05,
    ) -> None:
        if capital <= 0:
            raise ValueError("capital must be > 0")
        self.capital = capital
        self.level_1 = level_1_pct
        self.level_2 = level_2_pct
        self.level_3 = level_3_pct
        self.weekly_limit = weekly_limit_pct

        self._state = CircuitBreakerState.NORMAL
        self._daily_pnl: float = 0.0
        self._weekly_pnl: float = 0.0
        self._weekly_size_override: float = 1.0
        self._state_history: List[dict] = []
        self._just_halted: bool = False

    def _transition(self, new_state: CircuitBreakerState) -> None:
        """Log and execute state transition."""
        if new_state == self._state:
            return
        old = self._state
        self._state = new_state
        entry = {
            "from": old.value,
            "to": new_state.value,
            "daily_pnl": self._daily_pnl,
            "daily_loss_pct": self.get_daily_loss_pct(),
        }
        self._state_history.append(entry)
        logger.warning(
            "Circuit breaker: %s -> %s (daily loss: %.2f%%)",
            old.value, new_state.value, self.get_daily_loss_pct() * 100,
        )

    def on_trade_update(self, pnl_change: float) -> CircuitBreakerState:
        """Called after every fill or position close. Updates P&L and escalates state.

        Returns the (possibly new) state.
        """
        self._daily_pnl += pnl_change
        self._weekly_pnl += pnl_change
        self._just_halted = False

        daily_loss_pct = self.get_daily_loss_pct()

        # Escalate — never de-escalate within a day
        if daily_loss_pct >= self.level_3 and self._state != CircuitBreakerState.HALTED:
            self._transition(CircuitBreakerState.HALTED)
            self._just_halted = True
        elif daily_loss_pct >= self.level_2 and self._state in (
            CircuitBreakerState.NORMAL, CircuitBreakerState.REDUCED,
        ):
            self._transition(CircuitBreakerState.NO_NEW_ENTRIES)
        elif daily_loss_pct >= self.level_1 and self._state == CircuitBreakerState.NORMAL:
            self._transition(CircuitBreakerState.REDUCED)

        return self._state

    def on_new_day(self) -> None:
        """Called at session open. Reset daily P&L and state (unless weekly limit)."""
        self._daily_pnl = 0.0
        self._just_halted = False

        weekly_loss_pct = max(0, -self._weekly_pnl / self.capital)
        if weekly_loss_pct >= self.weekly_limit:
            self._weekly_size_override = 0.5
            self._transition(CircuitBreakerState.REDUCED)
        else:
            self._weekly_size_override = 1.0
            if self._state != CircuitBreakerState.NORMAL:
                self._transition(CircuitBreakerState.NORMAL)

    def on_new_week(self) -> None:
        """Called Monday at session open. Reset weekly P&L."""
        self._weekly_pnl = 0.0
        self._weekly_size_override = 1.0
        self._daily_pnl = 0.0
        self._just_halted = False
        if self._state != CircuitBreakerState.NORMAL:
            self._transition(CircuitBreakerState.NORMAL)

    def can_open_position(self) -> bool:
        """True if state allows new entries."""
        return self._state in (CircuitBreakerState.NORMAL, CircuitBreakerState.REDUCED)

    def get_size_multiplier(self) -> float:
        """Position size multiplier based on current state."""
        if self._state == CircuitBreakerState.NORMAL:
            return 1.0 * self._weekly_size_override
        if self._state == CircuitBreakerState.REDUCED:
            return 0.5 * self._weekly_size_override
        return 0.0  # NO_NEW_ENTRIES or HALTED

    def should_flatten_all(self) -> bool:
        """True if state just transitioned to HALTED."""
        return self._just_halted

    @property
    def state(self) -> CircuitBreakerState:
        return self._state

    def get_daily_loss_pct(self) -> float:
        """Current daily loss as positive percentage of capital."""
        return max(0.0, -self._daily_pnl / self.capital)

    def get_weekly_loss_pct(self) -> float:
        """Current weekly loss as positive percentage of capital."""
        return max(0.0, -self._weekly_pnl / self.capital)

    @property
    def state_history(self) -> List[dict]:
        return self._state_history
