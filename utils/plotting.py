"""Centralized high-resolution plotting utilities."""
from __future__ import annotations
from pathlib import Path
import matplotlib as mpl
mpl.rcParams.update({"figure.dpi":150,"savefig.dpi":300,"font.family":"DejaVu Sans","axes.linewidth":1.0,"lines.linewidth":2.0})
def save_figure(fig, path: str|Path, kind: str="both", dpi: int=300, overlay: bool=False, tiff: bool=False) -> list[Path]:
    """Save a Matplotlib figure with project export standards."""
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True); saved=[]; raster_dpi=max(dpi,600 if overlay else 300)
    if kind in ("raster","both"):
        png=path.with_suffix(".png"); fig.savefig(png,dpi=raster_dpi,bbox_inches="tight"); saved.append(png)
        if overlay and tiff:
            tif=path.with_suffix(".tiff"); fig.savefig(tif,dpi=raster_dpi,bbox_inches="tight"); saved.append(tif)
    if kind in ("vector","both") and not overlay:
        svg=path.with_suffix(".svg"); fig.savefig(svg,bbox_inches="tight"); saved.append(svg)
    if kind not in ("raster","vector","both"): raise ValueError("kind must be raster, vector, or both")
    return saved
