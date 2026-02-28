"""Tests for backtesting metrics with known-answer verification."""

import numpy as np
import pytest

from backtesting.metrics import (
    annualized_sharpe,
    avg_trade_pnl,
    calmar_ratio,
    compute_all_metrics,
    expectancy,
    max_drawdown,
    max_drawdown_detail,
    max_drawdown_duration,
    profit_factor,
    sortino_ratio,
    win_rate,
)


class TestSharpeRatio:
    """Verify Sharpe ratio computation."""

    def test_constant_positive_returns(self):
        """Constant positive returns should give very high Sharpe (near-zero variance)."""
        returns = np.array([0.01] * 100)
        sharpe = annualized_sharpe(returns, rf=0.0)
        # All identical returns → std(ddof=1) ≈ 0 from float precision
        # This produces an extremely large Sharpe (effectively infinite)
        assert sharpe > 1000  # near-infinite Sharpe

    def test_zero_returns(self):
        """All zero returns should give Sharpe of 0."""
        returns = np.zeros(100)
        sharpe = annualized_sharpe(returns, rf=0.0)
        assert sharpe == 0.0

    def test_positive_sharpe(self):
        """Slightly positive mean with some variance should give positive Sharpe."""
        rng = np.random.RandomState(42)
        returns = rng.normal(0.001, 0.01, 252)  # small positive drift
        sharpe = annualized_sharpe(returns, rf=0.0)
        assert sharpe > 0

    def test_negative_sharpe(self):
        """Negative mean returns should give negative Sharpe."""
        rng = np.random.RandomState(42)
        returns = rng.normal(-0.005, 0.01, 252)
        sharpe = annualized_sharpe(returns, rf=0.0)
        assert sharpe < 0

    def test_too_few_returns(self):
        """Less than 2 returns should give 0."""
        assert annualized_sharpe(np.array([0.01])) == 0.0
        assert annualized_sharpe(np.array([])) == 0.0


class TestSortinoRatio:
    """Verify Sortino ratio computation."""

    def test_no_downside(self):
        """All positive returns should give infinite Sortino."""
        returns = np.array([0.01, 0.02, 0.015, 0.008, 0.012])
        sortino = sortino_ratio(returns, rf=0.0)
        assert sortino == float("inf")

    def test_positive_sortino(self):
        """Mixed returns with positive mean should give positive Sortino."""
        rng = np.random.RandomState(42)
        returns = rng.normal(0.001, 0.01, 252)
        s = sortino_ratio(returns, rf=0.0)
        assert s > 0


class TestMaxDrawdown:
    """Verify max drawdown computation."""

    def test_known_drawdown(self):
        """Equity [100, 110, 90, 95, 120] → DD = (110-90)/110 = 18.18%."""
        equity = np.array([100.0, 110.0, 90.0, 95.0, 120.0])
        dd = max_drawdown(equity)
        expected = (110 - 90) / 110  # 0.18181...
        assert abs(dd - expected) < 0.001

    def test_no_drawdown(self):
        """Monotonically increasing equity → 0% drawdown."""
        equity = np.array([100.0, 110.0, 120.0, 130.0])
        assert max_drawdown(equity) == 0.0

    def test_full_drawdown(self):
        """Equity dropping to near zero → ~100% drawdown."""
        equity = np.array([100.0, 50.0, 1.0])
        dd = max_drawdown(equity)
        assert dd > 0.98

    def test_drawdown_detail(self):
        """max_drawdown_detail returns dict with all fields."""
        equity = np.array([100.0, 110.0, 90.0, 95.0, 120.0])
        detail = max_drawdown_detail(equity)
        assert abs(detail["max_dd_pct"] - (110 - 90) / 110) < 0.001
        assert abs(detail["max_dd_dollars"] - 20.0) < 0.01
        assert detail["peak_idx"] == 1
        assert detail["trough_idx"] == 2
        assert detail["recovery_needed"] > 0


class TestProfitFactor:
    """Verify profit factor computation."""

    def test_known_profit_factor(self):
        """Trades [+10, -5, +10, -5] → PF = 20/10 = 2.0."""
        trades = np.array([10.0, -5.0, 10.0, -5.0])
        pf = profit_factor(trades)
        assert abs(pf - 2.0) < 0.001

    def test_all_winners(self):
        """All winning trades → infinite profit factor."""
        trades = np.array([10.0, 20.0, 5.0])
        pf = profit_factor(trades)
        assert pf == float("inf")

    def test_all_losers(self):
        """All losing trades → profit factor = 0."""
        trades = np.array([-10.0, -20.0, -5.0])
        pf = profit_factor(trades)
        assert pf == 0.0

    def test_empty_trades(self):
        assert profit_factor(np.array([])) == 0.0


class TestWinRate:
    """Verify win rate computation."""

    def test_known_win_rate(self):
        """Trades [+1, -1, +1, +1] → win rate = 75%."""
        trades = np.array([1.0, -1.0, 1.0, 1.0])
        wr = win_rate(trades)
        assert abs(wr - 0.75) < 0.001

    def test_all_winners(self):
        trades = np.array([1.0, 2.0, 3.0])
        assert win_rate(trades) == 1.0

    def test_all_losers(self):
        trades = np.array([-1.0, -2.0, -3.0])
        assert win_rate(trades) == 0.0

    def test_empty(self):
        assert win_rate(np.array([])) == 0.0


class TestAvgTradePnl:
    """Verify avg_trade_pnl breakdown."""

    def test_basic(self):
        trades = np.array([10.0, -5.0, 20.0, -3.0])
        result = avg_trade_pnl(trades)
        assert abs(result["avg_win"] - 15.0) < 0.001   # (10+20)/2
        assert abs(result["avg_loss"] - (-4.0)) < 0.001  # (-5+-3)/2
        assert abs(result["avg_all"] - 5.5) < 0.001     # (10-5+20-3)/4
        assert result["reward_risk_ratio"] > 0

    def test_empty(self):
        result = avg_trade_pnl(np.array([]))
        assert result["avg_win"] == 0.0
        assert result["avg_loss"] == 0.0


class TestCalmarRatio:
    """Verify Calmar ratio computation."""

    def test_positive_calmar(self):
        """Positive returns with some drawdown → positive Calmar."""
        rng = np.random.RandomState(42)
        returns = rng.normal(0.001, 0.01, 252)
        equity = np.cumprod(1 + returns) * 10000
        c = calmar_ratio(returns, equity)
        assert c > 0

    def test_no_drawdown(self):
        """Monotonic returns → infinite Calmar."""
        returns = np.array([0.01, 0.01, 0.01, 0.01])
        equity = np.cumprod(1 + returns) * 10000
        c = calmar_ratio(returns, equity)
        assert c == float("inf")


class TestMaxDrawdownDuration:
    """Verify drawdown duration computation."""

    def test_known_duration(self):
        """Equity [100, 110, 90, 85, 95, 120] → drawdown from idx 1-4 = 3 bars."""
        equity = np.array([100.0, 110.0, 90.0, 85.0, 95.0, 120.0])
        dur = max_drawdown_duration(equity)
        assert dur == 3  # bars 2, 3, 4 are in drawdown


class TestComputeAllMetrics:
    """Verify the master compute_all_metrics function."""

    def test_returns_all_keys(self):
        rng = np.random.RandomState(42)
        returns = rng.normal(0.001, 0.01, 100)
        trades = np.array([10.0, -5.0, 15.0, -3.0, 8.0])
        equity = np.cumprod(1 + returns) * 10000

        result = compute_all_metrics(returns, trades, equity)
        assert "sharpe" in result
        assert "sortino" in result
        assert "max_drawdown" in result
        assert "profit_factor" in result
        assert "calmar" in result
        assert "win_rate" in result
        assert "total_trades" in result
        assert result["total_trades"] == 5
