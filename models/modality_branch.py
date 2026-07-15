"""UCSF auxiliary modality branch scaffolding."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModalityBranchConfig:
    optional_modalities: tuple[str, ...] = ("DWI", "ASL")
    feature_dim: int = 128
    dropout: float = 0.3


def build_modality_branch(config: ModalityBranchConfig):
    """Declare the optional-modality branch interface for later stages."""

    raise NotImplementedError("UCSF auxiliary modality branch is implemented in Part 4/Stage 3.")
