"""
Monte Carlo simulation via bootstrap resampling.

Resamples trade returns to build 1000 equity paths and outputs
percentile statistics (5th, 25th, 50th, 75th, 95th).
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backtesting.metrics import annualized_sharpe, max_drawdown

logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    """Bootstrap Monte Carlo simulation of trade returns.

    Usage:
        mc = MonteCarloSimulator(n_paths=1000)
        results = mc.simulate(trade_returns, initial_capital=10000)
    """

    def __init__(
        self,
        n_paths: int = 1000,
        seed: Optional[int] = None,
    ) -> None:
        self.n_paths = n_paths
        self.rng = np.random.RandomState(seed)

    def simulate(
        self,
        trade_returns: np.ndarray,
        initial_capital: float = 10000.0,
    ) -> Dict:
        """Run bootstrap Monte Carlo simulation.

        Args:
            trade_returns: Array of individual trade P&L values (in dollars)
            initial_capital: Starting capital

        Returns:
            Dict with keys:
                equity_paths: ndarray of shape (n_paths, n_trades+1)
                final_equity_pcts: percentile dict (5, 25, 50, 75, 95)
                max_drawdown_pcts: percentile dict for max drawdowns
                sharpe_pcts: percentile dict for Sharpe ratios
                summary: DataFrame with percentile summary
        """
        n_trades = len(trade_returns)
        if n_trades == 0:
            return self._empty_result(initial_capital)

        logger.info(
            "Running Monte Carlo: %d paths, %d trades", self.n_paths, n_trades
        )

        equity_paths = np.zeros((self.n_paths, n_trades + 1))
        equity_paths[:, 0] = initial_capital

        final_equities = np.zeros(self.n_paths)
        max_drawdowns = np.zeros(self.n_paths)
        sharpes = np.zeros(self.n_paths)

        for i in range(self.n_paths):
            # Bootstrap resample trade returns (with replacement)
            resampled = self.rng.choice(trade_returns, size=n_trades, replace=True)

            # Build equity curve
            equity = np.empty(n_trades + 1)
            equity[0] = initial_capital
            for j in range(n_trades):
                equity[j + 1] = equity[j] + resampled[j]

            equity_paths[i] = equity
            final_equities[i] = equity[-1]

            # For drawdown/Sharpe, clamp equity floor to avoid div-by-zero
            equity_clamped = np.maximum(equity, 1.0)
            max_drawdowns[i] = max_drawdown(equity_clamped)

            # Convert to daily-like returns for Sharpe
            returns = np.diff(equity_clamped) / equity_clamped[:-1]
            returns = returns[np.isfinite(returns)]
            sharpes[i] = annualized_sharpe(returns) if len(returns) > 1 else 0.0

        # Compute percentiles
        pct_levels = [5, 25, 50, 75, 95]
        final_eq_pcts = {p: float(np.percentile(final_equities, p)) for p in pct_levels}
        mdd_pcts = {p: float(np.percentile(max_drawdowns, p)) for p in pct_levels}
        sharpe_pcts = {p: float(np.percentile(sharpes, p)) for p in pct_levels}

        # Summary table
        summary = pd.DataFrame({
            "percentile": pct_levels,
            "final_equity": [final_eq_pcts[p] for p in pct_levels],
            "max_drawdown": [mdd_pcts[p] for p in pct_levels],
            "sharpe": [sharpe_pcts[p] for p in pct_levels],
        })

        logger.info(
            "Monte Carlo complete. Median final equity: $%.2f, "
            "Median max DD: %.1f%%, Median Sharpe: %.2f",
            final_eq_pcts[50],
            mdd_pcts[50] * 100,
            sharpe_pcts[50],
        )

        prob_positive = float(np.sum(final_equities > initial_capital) / self.n_paths)

        return {
            "equity_paths": equity_paths,
            "final_equity_pcts": final_eq_pcts,
            "max_drawdown_pcts": mdd_pcts,
            "sharpe_pcts": sharpe_pcts,
            "prob_positive": prob_positive,
            "summary": summary,
            "n_paths": self.n_paths,
            "n_trades": n_trades,
        }

    def _empty_result(self, initial_capital: float) -> Dict:
        pct_levels = [5, 25, 50, 75, 95]
        empty_pcts = {p: initial_capital for p in pct_levels}
        zero_pcts = {p: 0.0 for p in pct_levels}
        return {
            "equity_paths": np.full((1, 1), initial_capital),
            "final_equity_pcts": empty_pcts,
            "max_drawdown_pcts": zero_pcts,
            "sharpe_pcts": zero_pcts,
            "prob_positive": 0.0,
            "summary": pd.DataFrame({
                "percentile": pct_levels,
                "final_equity": [initial_capital] * 5,
                "max_drawdown": [0.0] * 5,
                "sharpe": [0.0] * 5,
            }),
            "n_paths": 0,
            "n_trades": 0,
        }
