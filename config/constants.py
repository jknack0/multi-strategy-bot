"""
Static instrument specifications and market session constants.

These values rarely change and are not user-configurable.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict


@dataclass(frozen=True)
class InstrumentSpec:
    """Specification for a tradeable instrument."""
    symbol: str
    tick_size: float
    tick_value: float       # tick_size * point_value
    point_value: float
    commission_per_side: float
    commission_round_trip: float
    slippage_model_ticks: float
    total_backtest_cost_per_trade: float
    margin: float           # IB intraday margin
    overnight_margin: float
    exchange: str
    currency: str
    sec_type: str


# ── Instrument Definitions ───────────────────────────────────────────────────

MES = InstrumentSpec(
    symbol="MES",
    tick_size=0.25,
    tick_value=1.25,                    # $5.00 * 0.25
    point_value=5.0,
    commission_per_side=0.62,
    commission_round_trip=1.24,
    slippage_model_ticks=0.25,          # conservative: 0.25 ticks per side
    total_backtest_cost_per_trade=1.87,  # (0.25*1.25*2) + (0.62*2) = $0.625 + $1.24
    margin=2455.0,                      # IB intraday margin
    overnight_margin=2455.0,
    exchange="CME",
    currency="USD",
    sec_type="FUT",
)

ES = InstrumentSpec(
    symbol="ES",
    tick_size=0.25,
    tick_value=12.50,                   # $50.00 * 0.25
    point_value=50.0,
    commission_per_side=0.62,
    commission_round_trip=1.24,
    slippage_model_ticks=0.25,
    total_backtest_cost_per_trade=1.87,
    margin=15600.0,
    overnight_margin=15600.0,
    exchange="CME",
    currency="USD",
    sec_type="FUT",
)

MNQ = InstrumentSpec(
    symbol="MNQ",
    tick_size=0.25,
    tick_value=0.50,                    # $2.00 * 0.25
    point_value=2.0,
    commission_per_side=0.62,
    commission_round_trip=1.24,
    slippage_model_ticks=0.25,
    total_backtest_cost_per_trade=1.87,
    margin=1800.0,
    overnight_margin=1800.0,
    exchange="CME",
    currency="USD",
    sec_type="FUT",
)

INSTRUMENTS: Dict[str, InstrumentSpec] = {
    "MES": MES,
    "ES": ES,
    "MNQ": MNQ,
}


# ── Market Sessions (all times in US/Eastern) ───────────────────────────────

FUTURES_SESSION_OPEN_HOUR: int = 18   # 6:00 PM ET (previous day)
FUTURES_SESSION_OPEN_MINUTE: int = 0
FUTURES_SESSION_CLOSE_HOUR: int = 17  # 5:00 PM ET
FUTURES_SESSION_CLOSE_MINUTE: int = 0

RTH_OPEN_HOUR: int = 9
RTH_OPEN_MINUTE: int = 30
RTH_CLOSE_HOUR: int = 16
RTH_CLOSE_MINUTE: int = 0

VWAP_RESET_HOUR: int = 9
VWAP_RESET_MINUTE: int = 30

# String-format session times for configuration
SESSION_TIMES = {
    "session_open_et": "09:30",
    "session_close_et": "16:00",
    "globex_open_et": "18:00",
    "globex_close_et": "17:00",
    "maintenance_break_start_et": "17:00",
    "maintenance_break_end_et": "18:00",
    "timezone": "US/Eastern",
}

# Strategy-specific time windows
STRATEGY_TIMES = {
    "trade_start": "10:00",      # no entries before this (skip opening chaos)
    "trade_end": "14:00",        # no entries after this
    "flatten_time": "15:55",     # close ALL positions before this
    "vwap_reset_time": "09:30",  # VWAP resets here (RTH open)
}


# ── VIX Regime Thresholds ────────────────────────────────────────────────────

class VIXRegime(Enum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


VIX_THRESHOLDS: Dict[VIXRegime, tuple] = {
    VIXRegime.LOW: (0.0, 15.0),
    VIXRegime.NORMAL: (15.0, 25.0),
    VIXRegime.HIGH: (25.0, 35.0),
    VIXRegime.EXTREME: (35.0, float("inf")),
}

# Parameter adjustments by regime (multipliers for stop/target sizing)
VIX_REGIME_ADJUSTMENTS: Dict[VIXRegime, Dict[str, float]] = {
    VIXRegime.LOW: {"stop_mult": 1.0, "target_mult": 1.0, "size_mult": 1.0},
    VIXRegime.NORMAL: {"stop_mult": 1.2, "target_mult": 1.1, "size_mult": 0.9},
    VIXRegime.HIGH: {"stop_mult": 1.5, "target_mult": 1.3, "size_mult": 0.6},
    VIXRegime.EXTREME: {"stop_mult": 2.0, "target_mult": 1.5, "size_mult": 0.3},
}


# ── Backtest Constants ───────────────────────────────────────────────────────

TRADING_DAYS_PER_YEAR: int = 252
MINUTES_PER_RTH_SESSION: int = 390  # 9:30-16:00
