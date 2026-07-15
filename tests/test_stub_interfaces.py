import pytest
import torch
from models.backbone import BackboneConfig, build_backbone
from models.ssl_pretrain import SSLPretrainConfig, build_ssl_pretrainer
from models.modality_branch import ModalityBranchConfig, build_modality_branch
from models.site_adversarial import SiteAdversarialConfig, build_site_discriminator
from models.heads import TaskHeadConfig, build_task_head, build_multitask_heads
from models.classical_ensemble import ClassicalEnsembleConfig, build_classical_ensemble
from explainability.gradcam3d import GradCAM3DConfig, generate_gradcam3d
from explainability.shap_radiomics import SHAPRadiomicsConfig, explain_radiomics_with_shap
from explainability.attention_visualization import AttentionVisualizationConfig, visualize_attention


def test_stage0_model_interfaces_now_build():
    backbone = build_backbone(BackboneConfig(feature_dim=8))
    features = backbone(torch.randn(2, 4, 8, 8, 8))
    assert features.shape == (2, 8)
    head = build_task_head(TaskHeadConfig(task_name="MGMT", input_dim=8))
    assert head(features).shape == (2,)
    heads = build_multitask_heads(TaskHeadConfig(task_name="MGMT", input_dim=8), TaskHeadConfig(task_name="IDH", input_dim=8))
    assert set(heads(features)) == {"MGMT", "IDH"}


def test_later_stage_model_scaffolds_still_raise_clear_errors():
    with pytest.raises(NotImplementedError): build_ssl_pretrainer(SSLPretrainConfig())
    with pytest.raises(NotImplementedError): build_modality_branch(ModalityBranchConfig())
    with pytest.raises(NotImplementedError): build_site_discriminator(SiteAdversarialConfig())
    with pytest.raises(NotImplementedError): build_classical_ensemble(ClassicalEnsembleConfig())


def test_explainability_scaffold_configs_and_clear_errors():
    with pytest.raises(NotImplementedError): generate_gradcam3d(model=None, volume=None, target_class=1, config=GradCAM3DConfig("layer4"), output_path="x.png")
    with pytest.raises(NotImplementedError): explain_radiomics_with_shap(model=None, feature_table=None, config=SHAPRadiomicsConfig(), output_path="x.svg")
    with pytest.raises(NotImplementedError): visualize_attention(attention_maps=None, reference_volume=None, config=AttentionVisualizationConfig(), output_path="x.png")
