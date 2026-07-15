"""Preprocessing validation and transform skeletons."""
from __future__ import annotations
def validate_upstream_preprocessing(*args, **kwargs) -> bool:
    """Validate expected atlas/skull-stripping prerequisites."""; return True
def resample_to_common_spacing(*args, **kwargs): raise NotImplementedError("Image resampling will be implemented with real image IO in a later part.")
def n4_bias_correction(*args, **kwargs): raise NotImplementedError("N4 correction requires SimpleITK image inputs and is not wired in Part 0.")
def tumor_centered_crop(*args, **kwargs): raise NotImplementedError("Tumor-centered crop requires segmentation masks and is not wired in Part 0.")
