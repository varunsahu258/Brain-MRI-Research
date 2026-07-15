"""Site-adversarial regularization scaffolding."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SiteAdversarialConfig:
    num_sites: int = 3
    lambda_site: float = 0.1
    hidden_dim: int = 128


def build_site_discriminator(config: SiteAdversarialConfig):
    """Declare the site-discriminator interface for Stage 2."""

    raise NotImplementedError("Site-adversarial discriminator is implemented in Part 3/Stage 2.")
