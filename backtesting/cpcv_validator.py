"""
Combinatorial Purged Cross-Validation (CPCV) per Lopez de Prado.

Splits data into N chronological groups, generates C(N, k) train/test
combinations, purges bars near boundaries, and computes the
Probability of Backtest Overfitting (PBO).
"""

import logging
from itertools import combinations
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtesting.metrics import annualized_sharpe

logger = logging.getLogger(__name__)


class CPCVValidator:
    """Combinatorial Purged Cross-Validation.

    Usage:
        validator = CPCVValidator(n_groups=6, k_test=2, purge_bars=60)
        results = validator.validate(data, strategy_fn)
        pbo = results['pbo']
    """

    def __init__(
        self,
        n_groups: int = 6,
        k_test: int = 2,
        purge_bars: int = 60,
    ) -> None:
        if k_test >= n_groups:
            raise ValueError("k_test must be < n_groups")
        self.n_groups = n_groups
        self.k_test = k_test
        self.purge_bars = purge_bars

    def _split_groups(self, n_samples: int) -> List[Tuple[int, int]]:
        """Split data into N chronological groups of roughly equal size.

        Returns:
            List of (start_idx, end_idx) tuples for each group.
        """
        group_size = n_samples // self.n_groups
        groups = []
        for i in range(self.n_groups):
            start = i * group_size
            end = start + group_size if i < self.n_groups - 1 else n_samples
            groups.append((start, end))
        return groups

    def _generate_combos(self) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        """Generate all C(N, k) train/test combinations.

        Returns:
            List of (train_group_indices, test_group_indices) tuples.
        """
        all_indices = list(range(self.n_groups))
        combos = []
        for test_groups in combinations(all_indices, self.k_test):
            train_groups = tuple(i for i in all_indices if i not in test_groups)
            combos.append((train_groups, test_groups))
        return combos

    def _get_purged_indices(
        self,
        groups: List[Tuple[int, int]],
        train_indices: Tuple[int, ...],
        test_indices: Tuple[int, ...],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get train and test sample indices with purging applied.

        Removes `purge_bars` samples from train set that are adjacent
        to any test group boundary.

        Returns:
            (train_mask, test_mask) boolean arrays of length n_samples.
        """
        n_samples = groups[-1][1]
        train_mask = np.zeros(n_samples, dtype=bool)
        test_mask = np.zeros(n_samples, dtype=bool)

        # Set test indices
        for gi in test_indices:
            start, end = groups[gi]
            test_mask[start:end] = True

        # Set train indices
        for gi in train_indices:
            start, end = groups[gi]
            train_mask[start:end] = True

        # Purge: remove train samples near test boundaries
        test_starts = [groups[gi][0] for gi in test_indices]
        test_ends = [groups[gi][1] for gi in test_indices]

        for ts in test_starts:
            purge_start = max(0, ts - self.purge_bars)
            train_mask[purge_start:ts] = False

        for te in test_ends:
            purge_end = min(n_samples, te + self.purge_bars)
            train_mask[te:purge_end] = False

        return train_mask, test_mask

    def validate(
        self,
        data: pd.DataFrame,
        strategy_fn: Callable[[pd.DataFrame], np.ndarray],
        metric_fn: Optional[Callable[[np.ndarray], float]] = None,
    ) -> Dict:
        """Run CPCV validation on a strategy.

        Args:
            data: OHLCV DataFrame
            strategy_fn: Function that takes a DataFrame and returns
                         an array of daily returns.
            metric_fn: Function to evaluate returns (default: Sharpe ratio).
                       Takes array of returns, returns a scalar score.

        Returns:
            Dict with keys:
                pbo: Probability of Backtest Overfitting (0 to 1)
                is_scores: List of in-sample scores per combo
                oos_scores: List of out-of-sample scores per combo
                n_combos: Number of combinations tested
                rank_logits: Array of rank logit values
        """
        if metric_fn is None:
            metric_fn = lambda r: annualized_sharpe(r, rf=0.05)

        n_samples = len(data)
        groups = self._split_groups(n_samples)
        combos = self._generate_combos()

        is_scores: List[float] = []
        oos_scores: List[float] = []

        logger.info(
            "Running CPCV: N=%d, k=%d, purge=%d → %d combinations",
            self.n_groups,
            self.k_test,
            self.purge_bars,
            len(combos),
        )

        for i, (train_gi, test_gi) in enumerate(combos):
            train_mask, test_mask = self._get_purged_indices(groups, train_gi, test_gi)

            train_data = data.iloc[train_mask]
            test_data = data.iloc[test_mask]

            if len(train_data) == 0 or len(test_data) == 0:
                logger.warning("Combo %d: empty split, skipping", i)
                continue

            # Get returns from strategy on train and test
            train_returns = strategy_fn(train_data)
            test_returns = strategy_fn(test_data)

            is_score = metric_fn(train_returns)
            oos_score = metric_fn(test_returns)

            is_scores.append(is_score)
            oos_scores.append(oos_score)

        # Compute PBO: fraction where IS rank doesn't persist OOS
        pbo = self._compute_pbo(is_scores, oos_scores)

        mean_is = float(np.mean(is_scores)) if is_scores else 0.0
        mean_oos = float(np.mean(oos_scores)) if oos_scores else 0.0
        sharpe_degradation = 1.0 - (mean_oos / mean_is) if mean_is != 0 else 0.0

        return {
            "pbo": pbo,
            "is_scores": is_scores,
            "oos_scores": oos_scores,
            "mean_is_sharpe": mean_is,
            "mean_oos_sharpe": mean_oos,
            "sharpe_degradation": sharpe_degradation,
            "n_combos": len(combos),
        }

    @staticmethod
    def _compute_pbo(
        is_scores: List[float],
        oos_scores: List[float],
    ) -> float:
        """Compute Probability of Backtest Overfitting.

        PBO is the fraction of combinations where the top-ranked
        in-sample configuration performs below median out-of-sample.

        A PBO > 0.5 suggests the strategy is likely overfit.
        """
        if len(is_scores) < 2:
            return 0.0

        is_arr = np.array(is_scores)
        oos_arr = np.array(oos_scores)

        # Rank IS scores (higher is better → rank descending)
        is_ranks = np.argsort(np.argsort(-is_arr))
        # Rank OOS scores
        oos_ranks = np.argsort(np.argsort(-oos_arr))

        n = len(is_scores)
        median_rank = n / 2.0

        # For each combo where it's top IS rank, check if OOS rank is below median
        overfit_count = 0
        for i in range(n):
            if is_ranks[i] == 0:  # Best IS
                if oos_ranks[i] >= median_rank:  # Below median OOS
                    overfit_count += 1

        # Alternative: use rank correlation approach
        # PBO = fraction of combos where relative IS advantage doesn't persist OOS
        # Use the simpler approach: for each pair, check rank consistency
        degraded = 0
        total_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_pairs += 1
                is_better = is_arr[i] > is_arr[j]
                oos_better = oos_arr[i] > oos_arr[j]
                if is_better != oos_better:
                    degraded += 1

        if total_pairs == 0:
            return 0.0

        pbo = degraded / total_pairs
        return float(pbo)
