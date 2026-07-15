"""SSL pretraining scaffolding for Stage 1."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SSLPretrainConfig:
    method: str = "masked_autoencoding"
    epochs: int = 100
    learning_rate: float = 1e-4


def build_ssl_pretrainer(config: SSLPretrainConfig):
    """Declare the SSL pretrainer interface for Stage 1."""

    raise NotImplementedError("SSL pretraining is implemented in Part 2/Stage 1, not Part 0.")
