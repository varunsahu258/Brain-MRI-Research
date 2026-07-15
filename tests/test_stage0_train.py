from pathlib import Path
import yaml
import pandas as pd
from training.train_stage import train_stage


def test_stage0_training_smoke_exports_probabilities(tmp_path):
    cfg = yaml.safe_load(Path("configs/stage0_baseline.yaml").read_text())
    base = yaml.safe_load(Path("configs/base.yaml").read_text())
    base["paths"]["results_root"] = str(tmp_path / "results")
    base["logging"]["mlflow_tracking_uri"] = str(tmp_path / "mlruns")
    base["cv"]["n_splits"] = 2
    cfg.update({k: v for k, v in base.items() if k not in cfg})
    cfg["extends"] = None
    cfg["training"]["synthetic_smoke_subjects_per_site"] = 4
    cfg["training"]["epochs"] = 1
    config_path = tmp_path / "stage0.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    summary = train_stage(0, config_path)
    results = Path(summary["results_dir"])
    pred_path = results / "metrics" / "predictions_probabilities.csv"
    assert pred_path.exists()
    preds = pd.read_csv(pred_path)
    assert {"y_true", "y_pred", "y_prob", "fold_kind"}.issubset(preds.columns)
    assert preds["y_prob"].between(0, 1).all()
    assert (results / "plots" / "roc_curve_overlay.png").exists()
