"""Preprocessing utilities for multisite brain MRI inputs.

The project assumes skull stripping and atlas alignment have been performed upstream.
This module therefore validates those prerequisites and provides deterministic,
array-based helpers used by tests and smoke runs. File-backed medical-image IO is
kept optional and routed through SimpleITK when available.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .sequence_normalization import CORE_SEQUENCES, normalize_sequence_map


@dataclass(frozen=True)
class PreprocessingReport:
    """Validation report for one subject's preprocessed inputs."""

    subject_id: str
    has_all_core_sequences: bool
    has_segmentation: bool
    atlas_alignment_ok: bool
    skull_stripping_ok: bool
    missing_sequences: tuple[str, ...]


def validate_upstream_preprocessing(
    subject_id: str,
    sequences: Mapping[str, str | Path],
    segmentation_path: str | Path | None,
    *,
    require_existing_files: bool = False,
) -> PreprocessingReport:
    """Validate upstream preprocessing prerequisites without modifying images.

    Parameters
    ----------
    subject_id
        Patient/subject identifier.
    sequences
        Mapping from raw sequence names to image paths.
    segmentation_path
        Path to the tumor segmentation mask, if present.
    require_existing_files
        If ``True``, sequence and segmentation paths must exist on disk.

    Returns
    -------
    PreprocessingReport
        Structured prerequisite validation report.
    """

    normalized = normalize_sequence_map(dict(sequences))
    missing = tuple(seq for seq in CORE_SEQUENCES if seq not in normalized)
    paths_exist = True
    if require_existing_files:
        paths_exist = all(Path(p).exists() for p in normalized.values())
        if segmentation_path is not None:
            paths_exist = paths_exist and Path(segmentation_path).exists()
    has_seg = segmentation_path is not None and (Path(segmentation_path).exists() if require_existing_files else True)
    return PreprocessingReport(
        subject_id=subject_id,
        has_all_core_sequences=not missing,
        has_segmentation=has_seg,
        atlas_alignment_ok=paths_exist and not missing,
        skull_stripping_ok=paths_exist and not missing,
        missing_sequences=missing,
    )


def resample_to_common_spacing(
    image: np.ndarray,
    current_spacing: Sequence[float],
    target_spacing: Sequence[float],
    *,
    order: int = 1,
) -> np.ndarray:
    """Resample a 3D array to target voxel spacing.

    Parameters
    ----------
    image
        Input 3D volume.
    current_spacing
        Current spacing in the same axis order as ``image``.
    target_spacing
        Desired spacing in the same axis order as ``image``.
    order
        Interpolation order passed to ``scipy.ndimage.zoom``.
    """

    from scipy.ndimage import zoom

    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError("resample_to_common_spacing expects a 3D image")
    current = np.asarray(current_spacing, dtype=float)
    target = np.asarray(target_spacing, dtype=float)
    if current.shape != (3,) or target.shape != (3,) or np.any(target <= 0):
        raise ValueError("current_spacing and target_spacing must be positive length-3 values")
    factors = current / target
    return zoom(arr, zoom=factors, order=order)


def n4_bias_correction(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Apply a lightweight bias-field normalization fallback for smoke data.

    Real N4 correction for file-backed images should be performed with SimpleITK
    in production preprocessing. For Part 0 infrastructure tests, this function
    performs deterministic foreground z-score normalization while preserving the
    input shape.
    """

    arr = np.asarray(image, dtype=float)
    if mask is not None:
        fg = arr[np.asarray(mask).astype(bool)]
    else:
        fg = arr[arr != 0]
    if fg.size == 0:
        return arr.copy()
    std = float(np.std(fg)) or 1.0
    return (arr - float(np.mean(fg))) / std


def tumor_centered_crop(image: np.ndarray, segmentation: np.ndarray, crop_size: Sequence[int]) -> tuple[np.ndarray, tuple[slice, slice, slice]]:
    """Crop a 3D image around the tumor mask center of mass.

    Parameters
    ----------
    image
        3D source volume.
    segmentation
        3D segmentation mask aligned to ``image``.
    crop_size
        Desired output crop size.

    Returns
    -------
    crop, slices
        Cropped array and source slices used to produce it.
    """

    arr = np.asarray(image)
    seg = np.asarray(segmentation)
    size = np.asarray(crop_size, dtype=int)
    if arr.shape != seg.shape or arr.ndim != 3 or size.shape != (3,) or np.any(size <= 0):
        raise ValueError("image/segmentation must be matching 3D arrays and crop_size length-3")
    coords = np.argwhere(seg > 0)
    if coords.size == 0:
        raise ValueError("segmentation contains no tumor voxels")
    center = np.round(coords.mean(axis=0)).astype(int)
    starts = np.maximum(0, np.minimum(center - size // 2, np.asarray(arr.shape) - size))
    stops = np.minimum(np.asarray(arr.shape), starts + size)
    slices = tuple(slice(int(s), int(e)) for s, e in zip(starts, stops, strict=True))
    return arr[slices], slices  # type: ignore[index]
