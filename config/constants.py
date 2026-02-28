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
    point_value: float
    margin: float
    exchange: str
    currency: str
    sec_type: str


# ── Instrument Definitions ───────────────────────────────────────────────────

MES = InstrumentSpec(
    symbol="MES",
    tick_size=0.25,
    point_value=5.0,
    margin=50.0,
    exchange="CME",
    currency="USD",
    sec_type="FUT",
)

ES = InstrumentSpec(
    symbol="ES",
    tick_size=0.25,
    point_value=50.0,
    margin=500.0,
    exchange="CME",
    currency="USD",
    sec_type="FUT",
)

MNQ = InstrumentSpec(
    symbol="MNQ",
    tick_size=0.25,
    point_value=2.0,
    margin=50.0,
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
