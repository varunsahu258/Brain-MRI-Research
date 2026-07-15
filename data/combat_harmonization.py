"""ComBat harmonization wrappers for radiomics/intensity features."""
from __future__ import annotations
import pandas as pd
def apply_combat(feature_table: pd.DataFrame, site_column: str="site", covariates: list[str]|None=None) -> pd.DataFrame:
    """Apply neuroCombat/neuroHarmonize ComBat to a feature table."""
    if site_column not in feature_table: raise ValueError(f"Missing site column: {site_column}")
    try:
        from neuroCombat import neuroCombat
    except Exception as e:
        raise ImportError("Install neuroCombat or neuroHarmonize to run ComBat harmonization") from e
    covars=feature_table[[site_column]+(covariates or [])]
    numeric=feature_table.select_dtypes("number")
    result=neuroCombat(dat=numeric.T, covars=covars, batch_col=site_column)["data"].T
    out=feature_table.copy(); out[numeric.columns]=result; return out
