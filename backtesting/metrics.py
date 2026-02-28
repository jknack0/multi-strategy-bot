"""
Pure-function performance metrics for backtesting.

All functions are stateless and operate on numpy arrays or pandas Series.
"""

from typing import Optional

import numpy as np
import pandas as pd

from config.constants import TRADING_DAYS_PER_YEAR


def annualized_sharpe(
    daily_returns: np.ndarray,
    rf: float = 0.05,
) -> float:
    """Annualized Sharpe ratio.

    Args:
        daily_returns: Array of daily returns (not percentage, e.g., 0.01 = 1%)
        rf: Annual risk-free rate (default 5%)

    Returns:
        Annualized Sharpe ratio
    """
    if len(daily_returns) < 2:
        return 0.0
    daily_rf = rf / TRADING_DAYS_PER_YEAR
    excess = daily_returns - daily_rf
    std = np.std(excess, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(
    daily_returns: np.ndarray,
    rf: float = 0.05,
) -> float:
    """Annualized Sortino ratio (penalizes only downside volatility).

    Args:
        daily_returns: Array of daily returns
        rf: Annual risk-free rate

    Returns:
        Annualized Sortino ratio
    """
    if len(daily_returns) < 2:
        return 0.0
    daily_rf = rf / TRADING_DAYS_PER_YEAR
    excess = daily_returns - daily_rf
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf") if np.mean(excess) > 0 else 0.0
    downside_std = np.sqrt(np.mean(downside ** 2))
    if downside_std == 0:
        return 0.0
    return float(np.mean(excess) / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Maximum drawdown from peak as a positive fraction (0 to 1).

    Args:
        equity_curve: Array of portfolio equity values

    Returns:
        Maximum drawdown (e.g., 0.15 means 15% drawdown)
    """
    if len(equity_curve) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    # Handle division by zero if peak is 0
    drawdown = np.where(np.isfinite(drawdown), drawdown, 0.0)
    return float(np.max(drawdown))


def max_drawdown_duration(equity_curve: np.ndarray) -> int:
    """Duration (in bars) of the longest drawdown.

    Args:
        equity_curve: Array of portfolio equity values

    Returns:
        Number of bars in the longest drawdown period
    """
    if len(equity_curve) < 2:
        return 0
    peak = np.maximum.accumulate(equity_curve)
    in_drawdown = equity_curve < peak
    max_dur = 0
    current_dur = 0
    for dd in in_drawdown:
        if dd:
            current_dur += 1
            max_dur = max(max_dur, current_dur)
        else:
            current_dur = 0
    return max_dur


def profit_factor(trades: np.ndarray) -> float:
    """Profit factor = gross profit / gross loss.

    Args:
        trades: Array of individual trade P&L values

    Returns:
        Profit factor (>1 is profitable)
    """
    if len(trades) == 0:
        return 0.0
    gross_profit = np.sum(trades[trades > 0])
    gross_loss = np.abs(np.sum(trades[trades < 0]))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def calmar_ratio(
    daily_returns: np.ndarray,
    equity_curve: Optional[np.ndarray] = None,
) -> float:
    """Calmar ratio = annualized return / max drawdown.

    Args:
        daily_returns: Array of daily returns
        equity_curve: Optional equity curve (computed from returns if not given)

    Returns:
        Calmar ratio
    """
    if len(daily_returns) < 2:
        return 0.0
    ann_return = np.mean(daily_returns) * TRADING_DAYS_PER_YEAR
    if equity_curve is None:
        equity_curve = np.cumprod(1 + daily_returns) * 10000
    mdd = max_drawdown(equity_curve)
    if mdd == 0:
        return float("inf") if ann_return > 0 else 0.0
    return float(ann_return / mdd)


def win_rate(trades: np.ndarray) -> float:
    """Win rate = fraction of profitable trades.

    Args:
        trades: Array of individual trade P&L values

    Returns:
        Win rate (0 to 1)
    """
    if len(trades) == 0:
        return 0.0
    return float(np.sum(trades > 0) / len(trades))


def expectancy(trades: np.ndarray) -> float:
    """Average expected P&L per trade.

    Args:
        trades: Array of individual trade P&L values

    Returns:
        Average trade P&L
    """
    if len(trades) == 0:
        return 0.0
    return float(np.mean(trades))


def compute_all_metrics(
    daily_returns: np.ndarray,
    trades: np.ndarray,
    equity_curve: Optional[np.ndarray] = None,
    rf: float = 0.05,
) -> dict:
    """Compute all performance metrics at once.

    Args:
        daily_returns: Array of daily returns
        trades: Array of individual trade P&L values
        equity_curve: Optional equity curve
        rf: Annual risk-free rate

    Returns:
        Dict of all metric names and values
    """
    if equity_curve is None and len(daily_returns) > 0:
        equity_curve = np.cumprod(1 + daily_returns) * 10000

    return {
        "sharpe": annualized_sharpe(daily_returns, rf),
        "sortino": sortino_ratio(daily_returns, rf),
        "max_drawdown": max_drawdown(equity_curve) if equity_curve is not None else 0.0,
        "max_drawdown_duration": max_drawdown_duration(equity_curve) if equity_curve is not None else 0,
        "profit_factor": profit_factor(trades),
        "calmar": calmar_ratio(daily_returns, equity_curve),
        "win_rate": win_rate(trades),
        "expectancy": expectancy(trades),
        "total_trades": len(trades),
        "total_return": float(equity_curve[-1] / equity_curve[0] - 1) if equity_curve is not None and len(equity_curve) > 0 else 0.0,
    }
