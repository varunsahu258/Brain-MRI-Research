import pandas as pd
import pytest
from data.combat_harmonization import apply_location_scale_harmonization, apply_combat


def test_location_scale_harmonization_preserves_shape_and_columns():
    df = pd.DataFrame({"subject_id":["a","b","c","d"], "site":["A","A","B","B"], "feat1":[1.0,2.0,10.0,12.0], "feat2":[2.0,4.0,20.0,24.0]})
    out = apply_location_scale_harmonization(df)
    assert out.shape == df.shape
    assert list(out.columns) == list(df.columns)
    assert out[["feat1", "feat2"]].isna().sum().sum() == 0


def test_apply_combat_validates_site_column():
    with pytest.raises(ValueError):
        apply_combat(pd.DataFrame({"feat": [1.0, 2.0]}))
