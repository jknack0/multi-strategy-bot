"""
VIX regime classifier.

Classifies the current VIX level into regimes (LOW, NORMAL, HIGH, EXTREME)
and returns corresponding parameter adjustments for position sizing and stops.
"""

from typing import Dict

from config.constants import (
    VIXRegime,
    VIX_THRESHOLDS,
    VIX_REGIME_ADJUSTMENTS,
)


class VIXRegimeClassifier:
    """Classify VIX values into market regimes.

    Usage:
        classifier = VIXRegimeClassifier()
        result = classifier.classify(18.5)
        # result = {'regime': VIXRegime.NORMAL, 'stop_mult': 1.2, ...}
    """

    def __init__(self) -> None:
        self._current_regime: VIXRegime = VIXRegime.NORMAL
        self._current_vix: float = 0.0
        self._history: list = []

    def classify(self, vix_value: float) -> Dict:
        """Classify a VIX value into a regime and return adjustments.

        Args:
            vix_value: Current VIX index value

        Returns:
            Dict with keys: regime, vix, stop_mult, target_mult, size_mult
        """
        self._current_vix = vix_value

        regime = VIXRegime.EXTREME  # Default for values above all thresholds
        for r, (low, high) in VIX_THRESHOLDS.items():
            if low <= vix_value < high:
                regime = r
                break

        self._current_regime = regime
        adjustments = VIX_REGIME_ADJUSTMENTS[regime]

        self._history.append({"vix": vix_value, "regime": regime})
        # Keep last 1000 entries
        if len(self._history) > 1000:
            self._history = self._history[-1000:]

        return {
            "regime": regime,
            "vix": vix_value,
            "stop_mult": adjustments["stop_mult"],
            "target_mult": adjustments["target_mult"],
            "size_mult": adjustments["size_mult"],
        }

    @property
    def current_regime(self) -> VIXRegime:
        return self._current_regime

    @property
    def current_vix(self) -> float:
        return self._current_vix

    def regime_summary(self) -> Dict[str, int]:
        """Count of bars spent in each regime from history."""
        counts: Dict[str, int] = {r.value: 0 for r in VIXRegime}
        for entry in self._history:
            counts[entry["regime"].value] += 1
        return counts
