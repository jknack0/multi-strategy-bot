"""
Shared risk manager used by ALL strategies.

Handles position sizing (quarter-Kelly, volatility-adjusted),
pre-trade risk checks, capital allocation, and exposure tracking.
"""

import math
import logging
from typing import Dict, List, Optional

from risk.circuit_breaker import CircuitBreaker, CircuitBreakerState

logger = logging.getLogger(__name__)


class RiskManager:
    """Portfolio-level risk manager for multi-strategy operation."""

    DEFAULT_ALLOCATIONS = {
        "strategy_a": 0.40,
        "strategy_b": 0.30,
        "strategy_c": 0.30,
    }

    def __init__(
        self,
        total_capital: float,
        strategy_allocations: Optional[Dict[str, float]] = None,
        max_portfolio_exposure: float = 0.60,
    ) -> None:
        if total_capital <= 0:
            raise ValueError("total_capital must be > 0")
        self.total_capital = total_capital
        self.allocations = strategy_allocations or dict(self.DEFAULT_ALLOCATIONS)
        self.max_portfolio_exposure = max_portfolio_exposure

        self.circuit_breaker = CircuitBreaker(total_capital)
        self._open_positions: Dict[str, list] = {}  # strategy_id -> list of position dicts
        self._daily_pnl: Dict[str, float] = {}
        self._total_daily_pnl: float = 0.0

    def get_strategy_capital(self, strategy_id: str) -> float:
        """Return allocated capital for a strategy."""
        return self.total_capital * self.allocations.get(strategy_id, 0.0)

    def position_size(
        self,
        capital: float,
        risk_pct: float,
        entry: float,
        stop: float,
        point_value: float = 5.0,
        margin_per_contract: float = 2455.0,
        scale: float = 1.0,
    ) -> int:
        """Volatility-adjusted position sizing.

        contracts = floor(dollar_risk / (|entry - stop| * point_value)) * scale
        Capped at floor(capital * 0.5 / margin_per_contract).
        Minimum 1 if raw > 0.
        """
        stop_distance = abs(entry - stop)
        if stop_distance <= 0 or capital <= 0:
            return 0

        dollar_risk = capital * risk_pct
        stop_dollars = stop_distance * point_value
        raw = math.floor(dollar_risk / stop_dollars * scale) if stop_dollars > 0 else 0

        margin_cap = int(capital * 0.5 / margin_per_contract) if margin_per_contract > 0 else 0

        if raw <= 0 or margin_cap <= 0:
            return 0
        return min(raw, margin_cap)

    def quarter_kelly(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """Compute quarter-Kelly optimal risk fraction.

        full_kelly = (p * b - q) / b
        quarter_kelly = full_kelly / 4
        Clamped to [0.005, 0.02] range.
        Returns 0 if no edge.
        """
        if avg_loss <= 0 or win_rate <= 0:
            return 0.0
        b = avg_win / avg_loss  # reward/risk ratio
        p = win_rate
        q = 1.0 - p
        full_kelly = (p * b - q) / b
        if full_kelly <= 0:
            return 0.0
        qk = full_kelly / 4.0
        return max(0.005, min(qk, 0.02))

    def can_open_position(self, strategy_id: str) -> bool:
        """Pre-trade check.

        Returns False if:
        - Circuit breaker blocks new entries
        - Total portfolio exposure exceeds max_portfolio_exposure
        """
        if not self.circuit_breaker.can_open_position():
            return False
        if self.get_total_exposure() >= self.max_portfolio_exposure:
            return False
        return True

    def register_position(
        self,
        strategy_id: str,
        contracts: int,
        entry_price: float,
        point_value: float = 5.0,
    ) -> None:
        """Register a new open position for exposure tracking."""
        if strategy_id not in self._open_positions:
            self._open_positions[strategy_id] = []
        notional = contracts * entry_price * point_value
        self._open_positions[strategy_id].append({
            "contracts": contracts,
            "entry_price": entry_price,
            "notional": notional,
        })

    def close_position(self, strategy_id: str, index: int = 0) -> None:
        """Remove a position from tracking."""
        if strategy_id in self._open_positions and self._open_positions[strategy_id]:
            if index < len(self._open_positions[strategy_id]):
                self._open_positions[strategy_id].pop(index)

    def update_pnl(self, strategy_id: str, pnl_change: float) -> None:
        """Update daily P&L for a strategy. Triggers circuit breaker checks."""
        self._daily_pnl[strategy_id] = self._daily_pnl.get(strategy_id, 0.0) + pnl_change
        self._total_daily_pnl += pnl_change
        self.circuit_breaker.on_trade_update(pnl_change)

    def on_new_day(self) -> None:
        """Reset daily P&L and circuit breaker."""
        self._daily_pnl.clear()
        self._total_daily_pnl = 0.0
        self.circuit_breaker.on_new_day()

    def on_new_week(self) -> None:
        """Reset weekly P&L."""
        self._daily_pnl.clear()
        self._total_daily_pnl = 0.0
        self.circuit_breaker.on_new_week()

    def get_total_exposure(self) -> float:
        """Sum of all open position notional values as fraction of capital."""
        total_notional = 0.0
        for positions in self._open_positions.values():
            for pos in positions:
                total_notional += pos["notional"]
        if self.total_capital <= 0:
            return 0.0
        return total_notional / self.total_capital

    def get_strategy_pnl(self, strategy_id: str) -> float:
        """Return daily P&L for a strategy."""
        return self._daily_pnl.get(strategy_id, 0.0)
