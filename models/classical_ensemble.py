"""Classical ML ensemble scaffolding for Stage 5."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassicalEnsembleConfig:
    estimators: tuple[str, ...] = ("logistic_regression", "random_forest", "svm")
    use_combat_features: bool = True


def build_classical_ensemble(config: ClassicalEnsembleConfig):
    """Declare the classical ensemble interface for Stage 5."""

    raise NotImplementedError("Classical robustness ensemble is implemented in Part 6/Stage 5.")
