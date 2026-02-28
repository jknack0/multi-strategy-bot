"""
ATR-based dynamic stop loss calculator.

Auto-adjusts stop distance with volatility so dollar risk stays constant.
Uses Wilder's ATR as the volatility measure.
"""

import math
from typing import Dict


class DynamicStopCalculator:
    """Compute ATR-based stops and volatility-adjusted position sizes."""

    def compute_stop(
        self,
        entry: float,
        side: str,
        atr: float,
        multiplier: float = 2.0,
        min_ticks: int = 4,
        tick_size: float = 0.25,
    ) -> float:
        """Compute stop price.

        stop_distance = max(multiplier * ATR, min_ticks * tick_size)
        LONG:  stop = entry - stop_distance
        SHORT: stop = entry + stop_distance

        Args:
            entry: Entry price
            side: 'LONG' or 'SHORT'
            atr: Current ATR value
            multiplier: ATR multiplier for stop distance
            min_ticks: Minimum ticks for stop (prevents bid-ask bounce stops)
            tick_size: Instrument tick size

        Returns:
            Stop price
        """
        stop_distance = max(multiplier * atr, min_ticks * tick_size)
        if side.upper() == "LONG":
            return entry - stop_distance
        return entry + stop_distance

    def compute_volatility_adjusted_size(
        self,
        capital: float,
        risk_pct: float,
        entry: float,
        atr: float,
        side: str = "LONG",
        multiplier: float = 2.0,
        point_value: float = 5.0,
        margin: float = 2455.0,
        min_ticks: int = 4,
        tick_size: float = 0.25,
        max_margin_pct: float = 0.50,
    ) -> Dict:
        """Compute both stop and position size together.

        Wider stop -> fewer contracts -> same dollar risk.

        Returns:
            Dict with stop_distance, stop_price, contracts, dollar_risk,
            margin_used, margin_pct.
        """
        stop_distance = max(multiplier * atr, min_ticks * tick_size)
        stop_price = self.compute_stop(
            entry, side, atr, multiplier, min_ticks, tick_size,
        )

        dollar_risk = capital * risk_pct
        stop_dollars = stop_distance * point_value
        if stop_dollars <= 0:
            raw_contracts = 0
        else:
            raw_contracts = math.floor(dollar_risk / stop_dollars)

        # Margin cap
        margin_cap = int(capital * max_margin_pct / margin) if margin > 0 else 0
        contracts = max(1, min(raw_contracts, margin_cap)) if raw_contracts > 0 else 0

        margin_used = contracts * margin
        margin_pct = margin_used / capital if capital > 0 else 0.0

        return {
            "stop_distance": stop_distance,
            "stop_price": stop_price,
            "contracts": contracts,
            "dollar_risk": dollar_risk,
            "margin_used": margin_used,
            "margin_pct": margin_pct,
        }
