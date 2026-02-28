"""
Tests for Combinatorial Purged Cross-Validation.

A random (coin-flip) strategy should produce PBO > 0.45 on synthetic data,
validating that CPCV correctly detects lack of genuine edge.
"""

import numpy as np
import pandas as pd
import pytest

from backtesting.cpcv_validator import CPCVValidator
from backtesting.metrics import annualized_sharpe


def _make_synthetic_data(n_bars: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    rng = np.random.RandomState(seed)
    prices = 4000 + np.cumsum(rng.randn(n_bars) * 2.0)
    df = pd.DataFrame({
        "open": prices,
        "high": prices + rng.uniform(0.5, 3.0, n_bars),
        "low": prices - rng.uniform(0.5, 3.0, n_bars),
        "close": prices + rng.randn(n_bars) * 0.5,
        "volume": rng.randint(100, 10000, n_bars),
    })
    return df


def _random_strategy(data: pd.DataFrame) -> np.ndarray:
    """A pure coin-flip strategy: random long/short each bar."""
    rng = np.random.RandomState(hash(len(data)) % 2**31)
    signals = rng.choice([-1, 1], size=len(data))
    returns = data["close"].pct_change().fillna(0).values
    return signals * returns


class TestCPCV:
    """Test CPCV validator."""

    def test_random_strategy_high_pbo(self):
        """A random strategy should have PBO > 0.45 (no genuine edge)."""
        data = _make_synthetic_data(n_bars=5000, seed=42)
        validator = CPCVValidator(n_groups=6, k_test=2, purge_bars=30)

        results = validator.validate(
            data=data,
            strategy_fn=_random_strategy,
        )

        # PBO should indicate overfitting for random strategy
        assert results["pbo"] > 0.45, (
            f"PBO={results['pbo']:.3f} should be > 0.45 for random strategy"
        )
        assert results["n_combos"] == 15  # C(6,2) = 15

    def test_combo_count(self):
        """C(6,2) should produce 15 combinations."""
        validator = CPCVValidator(n_groups=6, k_test=2)
        combos = validator._generate_combos()
        assert len(combos) == 15

    def test_purging(self):
        """Purged train set should be smaller than unpurged."""
        data = _make_synthetic_data(n_bars=600, seed=99)
        n = len(data)

        validator = CPCVValidator(n_groups=6, k_test=2, purge_bars=20)
        groups = validator._split_groups(n)
        combos = validator._generate_combos()

        train_gi, test_gi = combos[0]
        train_mask_purged, _ = validator._get_purged_indices(groups, train_gi, test_gi)
        purged_count = train_mask_purged.sum()

        # Without purging: 4/6 of data = ~400
        unpurged_expected = (n // 6) * 4
        assert purged_count < unpurged_expected

    def test_scores_populated(self):
        """Validate that IS and OOS scores are populated for all combos."""
        data = _make_synthetic_data(n_bars=3000, seed=7)
        validator = CPCVValidator(n_groups=6, k_test=2, purge_bars=10)

        results = validator.validate(data=data, strategy_fn=_random_strategy)

        assert len(results["is_scores"]) == 15
        assert len(results["oos_scores"]) == 15

    def test_k_test_validation(self):
        """k_test >= n_groups should raise ValueError."""
        with pytest.raises(ValueError):
            CPCVValidator(n_groups=6, k_test=6)
        with pytest.raises(ValueError):
            CPCVValidator(n_groups=6, k_test=7)

    def test_different_n_groups(self):
        """Test with different group sizes."""
        data = _make_synthetic_data(n_bars=3000, seed=123)

        for n_groups in [4, 6, 8]:
            validator = CPCVValidator(n_groups=n_groups, k_test=2, purge_bars=10)
            results = validator.validate(data=data, strategy_fn=_random_strategy)
            expected_combos = _comb(n_groups, 2)
            assert results["n_combos"] == expected_combos


def _comb(n: int, k: int) -> int:
    """Compute C(n, k)."""
    from math import factorial
    return factorial(n) // (factorial(k) * factorial(n - k))
