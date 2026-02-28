"""
VectorBT-based backtesting engine.

Loads data from QuestDB (or DataFrames), applies indicator-generated signals,
and runs vectorized backtests with configurable slippage and commissions.
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt

from config.constants import INSTRUMENTS, InstrumentSpec
from config.settings import (
    BACKTEST_INITIAL_CAPITAL,
    COMMISSION_PER_SIDE,
    SLIPPAGE_TICKS,
)
from backtesting.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


class BacktestEngine:
    """VectorBT-based backtesting harness.

    Usage:
        engine = BacktestEngine(symbol='MES')
        engine.load_data(df)
        engine.set_signals(entries, exits)
        results = engine.run()
    """

    def __init__(
        self,
        symbol: str = "MES",
        initial_capital: float = BACKTEST_INITIAL_CAPITAL,
        commission_per_side: float = COMMISSION_PER_SIDE,
        slippage_ticks: float = SLIPPAGE_TICKS,
    ) -> None:
        self.symbol = symbol
        self.spec: InstrumentSpec = INSTRUMENTS.get(symbol, INSTRUMENTS["MES"])
        self.initial_capital = initial_capital
        self.commission_per_side = commission_per_side
        self.slippage_ticks = slippage_ticks
        self.slippage_points = slippage_ticks * self.spec.tick_size

        self._data: Optional[pd.DataFrame] = None
        self._entries: Optional[pd.Series] = None
        self._exits: Optional[pd.Series] = None
        self._portfolio: Optional[vbt.Portfolio] = None

    @property
    def slippage_per_contract(self) -> float:
        """Slippage cost in dollars per contract per side."""
        return self.slippage_points * self.spec.point_value

    @property
    def total_cost_per_side(self) -> float:
        """Total cost per side (commission + slippage) in dollars."""
        return self.commission_per_side + self.slippage_per_contract

    def load_data(self, df: pd.DataFrame) -> None:
        """Load OHLCV data for backtesting.

        DataFrame must have columns: open, high, low, close, volume
        and a DatetimeIndex or a 'timestamp' column.
        """
        if "timestamp" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index("timestamp")
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        self._data = df.copy()
        logger.info("Loaded %d bars for backtesting", len(df))

    def load_from_questdb(
        self,
        questdb_client,
        bar_size: str = "1min",
        start=None,
        end=None,
    ) -> None:
        """Load data from QuestDB."""
        df = questdb_client.get_bars(self.symbol, bar_size, start=start, end=end)
        self.load_data(df)

    def set_signals(
        self,
        entries: pd.Series,
        exits: pd.Series,
    ) -> None:
        """Set entry and exit signal arrays.

        Args:
            entries: Boolean Series aligned with data index (True = enter long)
            exits: Boolean Series aligned with data index (True = exit position)
        """
        self._entries = entries.astype(bool)
        self._exits = exits.astype(bool)

    def run(
        self,
        direction: str = "longonly",
        size: float = 1.0,
        freq: Optional[str] = None,
    ) -> Dict:
        """Execute the backtest and return results.

        Args:
            direction: 'longonly', 'shortonly', or 'both'
            size: Number of contracts per trade
            freq: Data frequency for VectorBT (auto-detected if None)

        Returns:
            Dict with portfolio stats and custom metrics
        """
        if self._data is None:
            raise RuntimeError("No data loaded. Call load_data() first.")
        if self._entries is None or self._exits is None:
            raise RuntimeError("No signals set. Call set_signals() first.")

        close = self._data["close"]

        # Convert slippage to fraction of price for VectorBT
        avg_price = close.mean()
        slippage_frac = self.slippage_points / avg_price if avg_price > 0 else 0.0

        # Fixed fees: commission per side * 2 (entry + exit)
        fees_per_trade = self.commission_per_side * 2.0

        self._portfolio = vbt.Portfolio.from_signals(
            close=close,
            entries=self._entries,
            exits=self._exits,
            init_cash=self.initial_capital,
            size=size,
            slippage=slippage_frac,
            fees=fees_per_trade / self.initial_capital,  # as fraction
            direction=direction,
            freq=freq,
        )

        # Extract results
        pf = self._portfolio
        daily_returns = pf.returns().values
        equity = pf.value().values
        trades_pnl = pf.trades.pnl.values if len(pf.trades.records_readable) > 0 else np.array([])

        metrics = compute_all_metrics(
            daily_returns=daily_returns,
            trades=trades_pnl,
            equity_curve=equity,
        )

        metrics["initial_capital"] = self.initial_capital
        metrics["final_equity"] = float(equity[-1]) if len(equity) > 0 else self.initial_capital

        logger.info(
            "Backtest complete: %d trades, Sharpe=%.2f, MaxDD=%.1f%%",
            metrics["total_trades"],
            metrics["sharpe"],
            metrics["max_drawdown"] * 100,
        )

        return metrics

    @property
    def portfolio(self) -> Optional[vbt.Portfolio]:
        """Access the underlying VectorBT portfolio for custom analysis."""
        return self._portfolio

    @property
    def data(self) -> Optional[pd.DataFrame]:
        return self._data
