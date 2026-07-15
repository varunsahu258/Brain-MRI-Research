"""3D Grad-CAM scaffolding for Part 7."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GradCAM3DConfig:
    target_layer: str
    overlay_dpi: int = 600
    export_tiff: bool = False


def generate_gradcam3d(*, model, volume, target_class: int, config: GradCAM3DConfig, output_path: str | Path):
    """Declare the 3D Grad-CAM interface for explainability outputs."""

    raise NotImplementedError("3D Grad-CAM explainability is implemented in Part 7.")
