import numpy as np
import pandas as pd
import pytest
from radiomics.extract_features import extract_array_features, extract_subject_array_features, extract_feature_table_from_arrays


def test_extract_array_features_known_values():
    image = np.arange(8).reshape(2, 2, 2)
    mask = image > 3
    feats = extract_array_features(image, mask)
    assert feats["mean"] == 5.5
    assert feats["volume_voxels"] == 4.0


def test_extract_subject_feature_table_with_t1gd_alias():
    image = np.ones((2, 2, 2))
    mask = np.ones((2, 2, 2))
    row = extract_subject_array_features("s1", "UPENN", {"T1":image, "T1GD":image*2, "T2":image*3, "FLAIR":image*4}, {"whole": mask})
    assert row["subject_id"] == "s1"
    assert row["T1CE_whole_mean"] == 2.0
    table = extract_feature_table_from_arrays([{"subject_id":"s1", "site":"UPENN", "sequences":{"T1":image, "T1CE":image, "T2":image, "FLAIR":image}, "subregion_masks":{"whole":mask}}])
    assert isinstance(table, pd.DataFrame)
    assert table.shape[0] == 1


def test_extract_array_features_rejects_empty_mask():
    with pytest.raises(ValueError):
        extract_array_features(np.ones((2,2,2)), np.zeros((2,2,2)))
