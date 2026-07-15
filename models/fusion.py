"""Fusion modules for deep and radiomics features."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class FusionConfig:
    """Configuration for Stage 0 gated fusion."""

    fusion_type: str = "gated"
    deep_feature_dim: int = 128
    radiomics_feature_dim: int = 16
    fused_feature_dim: int = 128
    num_heads: int = 4


class GatedFusion(nn.Module):
    """Learned gate combining deep and radiomics representations."""

    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        self.config = config
        self.deep_projection = nn.Linear(config.deep_feature_dim, config.fused_feature_dim)
        self.radiomics_projection = nn.Linear(config.radiomics_feature_dim, config.fused_feature_dim)
        self.gate = nn.Sequential(nn.Linear(config.fused_feature_dim * 2, config.fused_feature_dim), nn.Sigmoid())

    def forward(self, deep_features: torch.Tensor, radiomics_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fused features and gate values.

        The fused representation is ``gate * deep + (1 - gate) * radiomics`` so
        both inputs necessarily affect the output unless the learned gate fully
        saturates.
        """

        if deep_features.ndim != 2 or radiomics_features.ndim != 2:
            raise ValueError("Fusion inputs must be rank-2 tensors (batch, features)")
        deep = torch.relu(self.deep_projection(deep_features))
        rad = torch.relu(self.radiomics_projection(radiomics_features))
        gate = self.gate(torch.cat([deep, rad], dim=1))
        return gate * deep + (1.0 - gate) * rad, gate


def build_fusion_module(config: FusionConfig) -> GatedFusion:
    """Build a fusion module.

    Stage 0 supports only ``fusion_type='gated'``. The factory keeps the mode
    switch explicit so Part 5 can add ``cross_attention`` without changing the
    call site.
    """

    if config.fusion_type != "gated":
        raise NotImplementedError("Only gated fusion is implemented in Stage 0; cross_attention is reserved for Part 5.")
    return GatedFusion(config)
