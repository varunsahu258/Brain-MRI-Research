"""3D backbone for Stage 0 and later staged models."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class BackboneConfig:
    """Configuration for a compact 3D imaging backbone.

    Parameters
    ----------
    in_channels
        Number of input MRI channels, expected to be the four normalized core
        sequences for Stage 0.
    feature_dim
        Output feature dimension consumed by fusion modules.
    dropout
        Dropout probability before the projection layer.
    architecture
        Backbone family label. Stage 0 ships a compact ResNet-like encoder;
        the field is retained so later parts can swap architectures by config.
    pretrained_checkpoint
        Optional MedicalNet/SSL checkpoint path. Missing paths are ignored only
        when set to ``None``; explicit missing paths raise an error.
    """

    in_channels: int = 4
    feature_dim: int = 128
    dropout: float = 0.2
    architecture: str = "compact_resnet3d"
    pretrained_checkpoint: str | None = None


class ConvBlock3D(nn.Module):
    """Small Conv-BN-ReLU block used by the compact Stage 0 encoder."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block to a 5D tensor ``(N, C, D, H, W)``."""

        return self.block(x)


class Stage0Backbone(nn.Module):
    """Compact 3D encoder with a checkpoint hook for later SSL weights."""

    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            ConvBlock3D(config.in_channels, 16, stride=1),
            ConvBlock3D(16, 32, stride=2),
            ConvBlock3D(32, 64, stride=2),
            nn.AdaptiveAvgPool3d(1),
        )
        self.projector = nn.Sequential(nn.Flatten(), nn.Dropout(config.dropout), nn.Linear(64, config.feature_dim), nn.ReLU(inplace=True))
        if config.pretrained_checkpoint:
            self.load_pretrained(config.pretrained_checkpoint)

    def load_pretrained(self, weights_path: str | Path, *, strict: bool = False) -> None:
        """Load locally available pretrained weights.

        The method accepts checkpoints stored as either a raw state dict or a
        dictionary containing ``state_dict``. This hook is intentionally stable so
        Stage 1 can swap in SSL-pretrained weights without changing callers.
        """

        path = Path(weights_path)
        if not path.exists():
            raise FileNotFoundError(f"Pretrained backbone checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.load_state_dict(state_dict, strict=strict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode 4-sequence MRI volumes into deep features."""

        if x.ndim != 5:
            raise ValueError("Stage0Backbone expects input shape (batch, channels, depth, height, width)")
        return self.projector(self.encoder(x))


def build_backbone(config: BackboneConfig) -> Stage0Backbone:
    """Build the Stage 0 3D backbone from config."""

    if config.architecture != "compact_resnet3d":
        raise ValueError(f"Unsupported Stage 0 backbone architecture: {config.architecture}")
    return Stage0Backbone(config)
