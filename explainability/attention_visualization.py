"""Attention visualization scaffolding for Part 7."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AttentionVisualizationConfig:
    overlay_dpi: int = 600
    export_tiff: bool = False


def visualize_attention(*, attention_maps, reference_volume, config: AttentionVisualizationConfig, output_path: str | Path):
    """Declare the attention visualization interface for Part 7."""

    raise NotImplementedError("Attention visualizations are implemented in Part 7.")
