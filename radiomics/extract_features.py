"""Radiomic feature extraction helpers.

Part 0 provides a deterministic NumPy extractor for tests/smoke runs and a
PyRadiomics adapter for later file-backed extraction runs.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from data.sequence_normalization import CORE_SEQUENCES, normalize_sequence_map


def extract_array_features(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Extract simple first-order features from one image/mask pair."""

    arr = np.asarray(image, dtype=float)
    m = np.asarray(mask).astype(bool)
    if arr.shape != m.shape:
        raise ValueError("image and mask must have matching shapes")
    if not np.any(m):
        raise ValueError("mask contains no foreground voxels")
    values = arr[m]
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "volume_voxels": float(values.size),
    }


def extract_subject_array_features(
    subject_id: str,
    site: str,
    sequences: Mapping[str, np.ndarray],
    subregion_masks: Mapping[str, np.ndarray],
) -> dict[str, float | str]:
    """Extract per-sequence, per-subregion first-order features for one subject."""

    normalized = normalize_sequence_map(dict(sequences))
    missing = [seq for seq in CORE_SEQUENCES if seq not in normalized]
    if missing:
        raise ValueError(f"Subject {subject_id} missing core sequences: {missing}")
    row: dict[str, float | str] = {"subject_id": subject_id, "site": site}
    for seq in CORE_SEQUENCES:
        for region, mask in subregion_masks.items():
            feats = extract_array_features(np.asarray(normalized[seq]), mask)
            for name, value in feats.items():
                row[f"{seq}_{region}_{name}"] = value
    return row


def extract_feature_table_from_arrays(subjects: list[dict]) -> pd.DataFrame:
    """Build a radiomics-like feature table from in-memory subject dictionaries."""

    rows = [
        extract_subject_array_features(
            subject_id=str(s["subject_id"]),
            site=str(s["site"]),
            sequences=s["sequences"],
            subregion_masks=s["subregion_masks"],
        )
        for s in subjects
    ]
    return pd.DataFrame(rows)


def extract_pyradiomics_features(image_path: str | Path, mask_path: str | Path, params_path: str | Path | None = None) -> dict[str, float]:
    """Extract PyRadiomics features for one image/mask pair."""

    try:
        from radiomics import featureextractor
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise ImportError("PyRadiomics is required for file-backed feature extraction") from exc
    extractor = featureextractor.RadiomicsFeatureExtractor(str(params_path)) if params_path else featureextractor.RadiomicsFeatureExtractor()
    result = extractor.execute(str(image_path), str(mask_path))
    return {k: float(v) for k, v in result.items() if isinstance(v, (int, float, np.integer, np.floating))}
