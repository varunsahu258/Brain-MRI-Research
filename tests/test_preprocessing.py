import numpy as np
import pytest
from data.preprocessing import validate_upstream_preprocessing, resample_to_common_spacing, n4_bias_correction, tumor_centered_crop


def test_validate_upstream_preprocessing_normalizes_sequences():
    report = validate_upstream_preprocessing("s1", {"T1":"a", "T1GD":"b", "T2":"c", "FLAIR":"d"}, "seg")
    assert report.has_all_core_sequences is True
    assert report.has_segmentation is True
    assert report.missing_sequences == ()


def test_resample_bias_and_tumor_crop_smoke():
    img = np.arange(27, dtype=float).reshape(3, 3, 3)
    resampled = resample_to_common_spacing(img, (2, 2, 2), (1, 1, 1), order=0)
    assert resampled.shape == (6, 6, 6)
    mask = np.zeros((3, 3, 3), dtype=int); mask[1, 1, 1] = 1
    corrected = n4_bias_correction(img, mask)
    assert corrected.shape == img.shape
    crop, slices = tumor_centered_crop(img, mask, (2, 2, 2))
    assert crop.shape == (2, 2, 2)
    assert len(slices) == 3


def test_tumor_crop_rejects_empty_mask():
    with pytest.raises(ValueError):
        tumor_centered_crop(np.zeros((3, 3, 3)), np.zeros((3, 3, 3)), (2, 2, 2))
