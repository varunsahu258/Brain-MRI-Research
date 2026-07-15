"""SHAP radiomics explainability scaffolding for Part 7."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SHAPRadiomicsConfig:
    max_background_samples: int = 100
    random_state: int = 42


def explain_radiomics_with_shap(*, model, feature_table, config: SHAPRadiomicsConfig, output_path: str | Path):
    """Declare the SHAP radiomics interface for Part 7."""

    raise NotImplementedError("Radiomics SHAP explainability is implemented in Part 7.")
