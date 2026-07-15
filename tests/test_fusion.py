import torch
import pytest
from models.fusion import FusionConfig, build_fusion_module


def test_gated_fusion_output_shape_and_gate_range():
    module = build_fusion_module(FusionConfig(deep_feature_dim=4, radiomics_feature_dim=3, fused_feature_dim=5))
    fused, gate = module(torch.ones(2, 4), torch.ones(2, 3))
    assert fused.shape == (2, 5)
    assert gate.shape == (2, 5)
    assert torch.all((gate >= 0) & (gate <= 1))


def test_gated_fusion_uses_deep_and_radiomics_inputs():
    torch.manual_seed(1)
    module = build_fusion_module(FusionConfig(deep_feature_dim=4, radiomics_feature_dim=3, fused_feature_dim=5))
    deep = torch.randn(2, 4)
    rad = torch.randn(2, 3)
    out, _ = module(deep, rad)
    out_changed_deep, _ = module(deep + 1.0, rad)
    out_changed_rad, _ = module(deep, rad + 1.0)
    assert not torch.allclose(out, out_changed_deep)
    assert not torch.allclose(out, out_changed_rad)


def test_cross_attention_mode_reserved_for_later_stage():
    with pytest.raises(NotImplementedError):
        build_fusion_module(FusionConfig(fusion_type="cross_attention"))
