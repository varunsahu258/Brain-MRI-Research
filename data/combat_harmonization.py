"""ComBat harmonization wrappers for radiomics and intensity features."""
from __future__ import annotations

from typing import Iterable

import pandas as pd


def _numeric_feature_columns(feature_table: pd.DataFrame, protected_columns: Iterable[str]) -> list[str]:
    protected = set(protected_columns)
    return [c for c in feature_table.select_dtypes("number").columns if c not in protected]


def apply_combat(
    feature_table: pd.DataFrame,
    site_column: str = "site",
    covariates: list[str] | None = None,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Apply ComBat harmonization to a tabular feature matrix.

    Parameters
    ----------
    feature_table
        DataFrame containing one row per subject.
    site_column
        Batch/site column used by ComBat.
    covariates
        Optional biological covariates to preserve during harmonization.
    feature_columns
        Numeric feature columns to harmonize. If omitted, all numeric columns
        except protected covariates are used.
    """

    if site_column not in feature_table:
        raise ValueError(f"Missing site column: {site_column}")
    covariates = covariates or []
    missing_covars = [c for c in covariates if c not in feature_table]
    if missing_covars:
        raise ValueError(f"Missing covariate columns: {missing_covars}")
    protected = [site_column, *covariates]
    feature_columns = feature_columns or _numeric_feature_columns(feature_table, protected)
    if not feature_columns:
        raise ValueError("No numeric feature columns available for ComBat harmonization")

    try:
        from neuroCombat import neuroCombat
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise ImportError("Install neuroCombat to run ComBat harmonization") from exc

    covars = feature_table[protected].copy()
    data = feature_table[feature_columns].T
    harmonized = neuroCombat(dat=data, covars=covars, batch_col=site_column)["data"].T
    out = feature_table.copy()
    out.loc[:, feature_columns] = harmonized.to_numpy() if hasattr(harmonized, "to_numpy") else harmonized
    return out


def apply_location_scale_harmonization(
    feature_table: pd.DataFrame,
    site_column: str = "site",
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Deterministic site-wise location/scale harmonization for smoke tests.

    This is not a replacement for ComBat in experiments; it is an explicit helper
    for pre-CNN intensity statistics and unit tests where neuroCombat may be
    unavailable.
    """

    if site_column not in feature_table:
        raise ValueError(f"Missing site column: {site_column}")
    feature_columns = feature_columns or _numeric_feature_columns(feature_table, [site_column])
    out = feature_table.copy()
    global_mean = out[feature_columns].mean()
    global_std = out[feature_columns].std(ddof=0).replace(0, 1.0)
    for _, idx in out.groupby(site_column).groups.items():
        site_values = out.loc[idx, feature_columns]
        site_std = site_values.std(ddof=0).replace(0, 1.0)
        out.loc[idx, feature_columns] = ((site_values - site_values.mean()) / site_std) * global_std + global_mean
    return out
