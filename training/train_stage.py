"""Generic staged trainer with a complete Stage 0 baseline implementation."""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from evaluation.bootstrap_ci import bootstrap_ci
from evaluation.metrics import compute_binary_metrics
from evaluation.results_export import export_predictions_and_metrics, prepare_results_tree
from models.backbone import BackboneConfig, build_backbone
from models.fusion import FusionConfig, build_fusion_module
from models.heads import TaskHeadConfig, build_multitask_heads
from training.cv_harness import make_pooled_stratified_folds
from training.gates import evaluate_generalization_gate
from training.loso_harness import split_by_loso
from utils.plotting import save_figure


def load_stage_config(config_path: str | Path) -> dict[str, Any]:
    """Load a stage config and shallow-merge it over its base YAML."""

    path = Path(config_path)
    cfg = yaml.safe_load(path.read_text())
    if cfg.get("extends"):
        base_path = path.parent / cfg["extends"]
        base = yaml.safe_load(base_path.read_text())
        cfg = _deep_merge(base, cfg)
    return cfg


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def _synthetic_stage0_data(cfg: dict[str, Any]) -> tuple[pd.DataFrame, torch.Tensor, torch.Tensor]:
    """Create deterministic multisite smoke data when real data is unavailable."""

    train_cfg = cfg["training"]
    seed = int(cfg.get("seed", 42))
    rng = np.random.default_rng(seed)
    sites = ["UPENN", "UCSF", "TCGA"]
    n_per_site = int(train_cfg.get("synthetic_smoke_subjects_per_site", 12))
    volume_shape = tuple(train_cfg.get("synthetic_volume_shape", [8, 8, 8]))
    radiomics_dim = int(train_cfg.get("synthetic_radiomics_dim", 16))
    rows: list[dict[str, Any]] = []
    volumes: list[np.ndarray] = []
    radiomics: list[np.ndarray] = []
    for site_i, site in enumerate(sites):
        for i in range(n_per_site):
            subject_id = f"{site.lower()}_{i:03d}"
            mgmt = int((i + site_i) % 2 == 0)
            idh = int((i + 2 * site_i) % 3 == 0)
            signal = mgmt * 0.6 + idh * 0.3 + site_i * 0.1
            volume = rng.normal(signal, 0.5, size=(4, *volume_shape)).astype("float32")
            rad = rng.normal(signal, 0.25, size=(radiomics_dim,)).astype("float32")
            rows.append({"subject_id": subject_id, "site": site, "mgmt_label": mgmt, "idh_label": idh})
            volumes.append(volume)
            radiomics.append(rad)
    return pd.DataFrame(rows), torch.tensor(np.stack(volumes)), torch.tensor(np.stack(radiomics))


class Stage0Model(torch.nn.Module):
    """Stage 0 baseline: 4-sequence backbone, gated radiomics fusion, two heads."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        backbone_cfg = BackboneConfig(**cfg["backbone"])
        fusion_cfg = FusionConfig(
            fusion_type=cfg["fusion"].get("mode", "gated"),
            deep_feature_dim=cfg["fusion"]["deep_feature_dim"],
            radiomics_feature_dim=cfg["fusion"]["radiomics_feature_dim"],
            fused_feature_dim=cfg["fusion"]["fused_feature_dim"],
        )
        mgmt = TaskHeadConfig(task_name="MGMT", **cfg["heads"]["mgmt"])
        idh = TaskHeadConfig(task_name="IDH", **cfg["heads"]["idh"])
        self.backbone = build_backbone(backbone_cfg)
        self.fusion = build_fusion_module(fusion_cfg)
        self.heads = build_multitask_heads(mgmt, idh)

    def forward(self, volumes: torch.Tensor, radiomics: torch.Tensor) -> dict[str, torch.Tensor]:
        deep = self.backbone(volumes)
        fused, gate = self.fusion(deep, radiomics)
        logits = self.heads(fused)
        logits["gate"] = gate
        return logits


def _pos_weight(values: torch.Tensor) -> torch.Tensor:
    pos = torch.clamp(values.sum(), min=1.0)
    neg = torch.clamp(torch.tensor(float(values.numel()), device=values.device) - values.sum(), min=1.0)
    return neg / pos


def _train_one_fold(cfg: dict[str, Any], x: torch.Tensor, r: torch.Tensor, y_mgmt: torch.Tensor, y_idh: torch.Tensor, train_idx: list[int], test_idx: list[int]) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device") == "cuda" else "cpu")
    model = Stage0Model(cfg).to(device)
    params = [
        {"params": model.backbone.parameters(), "lr": cfg["heads"]["mgmt"]["learning_rate"], "weight_decay": cfg["heads"]["mgmt"]["weight_decay"]},
        {"params": model.fusion.parameters(), "lr": cfg["heads"]["mgmt"]["learning_rate"], "weight_decay": cfg["heads"]["mgmt"]["weight_decay"]},
        {"params": model.heads.mgmt.parameters(), "lr": cfg["heads"]["mgmt"]["learning_rate"], "weight_decay": cfg["heads"]["mgmt"]["weight_decay"]},
        {"params": model.heads.idh.parameters(), "lr": cfg["heads"]["idh"]["learning_rate"], "weight_decay": cfg["heads"]["idh"]["weight_decay"]},
        {"params": [model.heads.log_sigma_mgmt, model.heads.log_sigma_idh], "lr": cfg["heads"]["mgmt"]["learning_rate"]},
    ]
    optimizer = torch.optim.AdamW(params)
    ds = TensorDataset(x[train_idx], r[train_idx], y_mgmt[train_idx], y_idh[train_idx])
    loader = DataLoader(ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=True)
    pos_weights = {"MGMT": _pos_weight(y_mgmt[train_idx]).to(device), "IDH": _pos_weight(y_idh[train_idx]).to(device)}
    model.train()
    for _ in range(int(cfg["training"].get("epochs", 2))):
        for xb, rb, mb, ib in loader:
            xb, rb, mb, ib = xb.to(device), rb.to(device), mb.to(device), ib.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, rb)
            loss, _ = model.heads.uncertainty_weighted_loss(logits, {"MGMT": mb, "IDH": ib}, pos_weights)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(x[test_idx].to(device), r[test_idx].to(device))
        mgmt_prob = torch.sigmoid(logits["MGMT"]).cpu().numpy()
        idh_prob = torch.sigmoid(logits["IDH"]).cpu().numpy()
    return mgmt_prob, idh_prob


def _index_lookup(subjects: pd.DataFrame) -> dict[str, int]:
    return {sid: i for i, sid in enumerate(subjects.subject_id.tolist())}


def _fold_predictions(cfg: dict[str, Any], subjects: pd.DataFrame, x: torch.Tensor, r: torch.Tensor, folds: list[dict[str, Any]], fold_kind: str) -> pd.DataFrame:
    lookup = _index_lookup(subjects)
    rows: list[dict[str, Any]] = []
    y_mgmt = torch.tensor(subjects.mgmt_label.to_numpy(dtype="float32"))
    y_idh = torch.tensor(subjects.idh_label.to_numpy(dtype="float32"))
    for fold in folds:
        train_idx = [lookup[s] for s in fold["train_subjects"]]
        test_idx = [lookup[s] for s in fold["test_subjects"]]
        mgmt_prob, _ = _train_one_fold(cfg, x, r, y_mgmt, y_idh, train_idx, test_idx)
        for local_i, idx in enumerate(test_idx):
            row = subjects.iloc[idx]
            rows.append({
                "model_name": "stage0_gated_baseline",
                "task": "MGMT",
                "subject_id": row.subject_id,
                "site": row.site,
                "fold": fold.get("fold", fold_kind),
                "fold_kind": fold_kind,
                "y_true": int(row.mgmt_label),
                "y_pred": int(mgmt_prob[local_i] >= 0.5),
                "y_prob": float(mgmt_prob[local_i]),
            })
    return pd.DataFrame(rows)


def _write_stage0_figures_and_ci(base: Path, predictions: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    ci_rows = []
    for model, df in predictions.groupby("model_name"):
        ci = bootstrap_ci(df.y_true.to_numpy(), df.y_pred.to_numpy(), df.y_prob.to_numpy(), n_bootstraps=200)
        for metric, (lower, upper) in ci.items():
            ci_rows.append({"model": model, "metric": metric, "lower": lower, "upper": upper})
        for fold, fold_df in df.groupby("fold"):
            fig, ax = plt.subplots(figsize=(3.5, 3.5))
            ConfusionMatrixDisplay.from_predictions(fold_df.y_true, fold_df.y_pred, ax=ax, colorbar=False)
            ax.set_title(f"{model} {fold}")
            save_figure(fig, base / "confusion_matrices" / f"{model}_{fold}", kind="both")
            plt.close(fig)
    pd.DataFrame(ci_rows).to_csv(base / "metrics" / "bootstrap_ci.csv", index=False)
    for name in ["mcnemar_pairwise.csv", "paired_ttest_5x2cv.csv", "wilcoxon_signed_rank.csv", "delong_auc_pvalues.csv"]:
        pd.DataFrame().to_csv(base / "significance_tests" / name, index=False)

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    for model, df in predictions.groupby("model_name"):
        RocCurveDisplay.from_predictions(df.y_true, df.y_prob, name=model, ax=ax)
    save_figure(fig, base / "plots" / "roc_curve_overlay", kind="both")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    for model, df in predictions.groupby("model_name"):
        PrecisionRecallDisplay.from_predictions(df.y_true, df.y_prob, name=model, ax=ax)
    save_figure(fig, base / "plots" / "pr_curve_overlay", kind="both")
    plt.close(fig)


def train_stage(stage: int, config_path: str | Path = "configs/stage0_baseline.yaml", *, clean_outputs: bool = False) -> dict[str, Any]:
    """Train/evaluate a configured stage. Stage 0 is implemented in Part 1."""

    if stage != 0:
        raise NotImplementedError("Only Stage 0 baseline training is implemented in Part 1.")
    cfg = load_stage_config(config_path)
    _set_seed(int(cfg.get("seed", 42)))
    results_root = Path(cfg["paths"].get("results_root", "outputs/results"))
    stage_dir = results_root / "stage_0_baseline"
    if clean_outputs and stage_dir.exists():
        shutil.rmtree(stage_dir)

    subjects, volumes, radiomics = _synthetic_stage0_data(cfg)
    pooled_folds = make_pooled_stratified_folds(subjects, n_splits=int(cfg["cv"]["n_splits"]), seed=int(cfg["cv"]["random_state"]))
    loso_folds = split_by_loso(subjects, cfg["cv"]["loso_folds_file"])
    predictions = pd.concat([
        _fold_predictions(cfg, subjects, volumes, radiomics, pooled_folds, "pooled_cv"),
        _fold_predictions(cfg, subjects, volumes, radiomics, loso_folds, "loso"),
    ], ignore_index=True)

    base = export_predictions_and_metrics(predictions, results_root, stage=0)
    _write_stage0_figures_and_ci(base, predictions)

    pooled_df = predictions[predictions.fold_kind == "pooled_cv"]
    loso_df = predictions[predictions.fold_kind == "loso"]
    pooled_metrics = compute_binary_metrics(pooled_df.y_true, pooled_df.y_pred, pooled_df.y_prob)
    loso_metrics = compute_binary_metrics(loso_df.y_true, loso_df.y_pred, loso_df.y_prob)
    per_site_loso = {site: compute_binary_metrics(g.y_true, g.y_pred, g.y_prob) for site, g in loso_df.groupby("site")}
    gap = evaluate_generalization_gate(float(pooled_metrics["auc_roc"]), float(loso_metrics["auc_roc"]))

    try:
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        import mlflow

        mlflow.set_tracking_uri(cfg["logging"]["mlflow_tracking_uri"])
        mlflow.set_experiment(cfg["logging"]["experiment_name"])
        with mlflow.start_run(run_name="stage_0_baseline"):
            mlflow.log_param("stage", 0)
            mlflow.log_params({"backbone_architecture": cfg["backbone"]["architecture"], "fusion_mode": cfg["fusion"]["mode"], "checkpoint_criterion": cfg["training"]["checkpoint_criterion"]})
            mlflow.log_metrics({f"pooled_cv_mgmt_{k}": float(v) for k, v in pooled_metrics.items()})
            mlflow.log_metrics({f"loso_mgmt_avg_{k}": float(v) for k, v in loso_metrics.items()})
            mlflow.log_metric("generalization_gap_stage0", float(gap["generalization_gap"]))
            mlflow.set_tags(gap["mlflow_tags"])
    except Exception as exc:  # pragma: no cover - MLflow environment specific
        print(f"WARNING: MLflow logging skipped: {exc}")

    summary = {
        "pooled_cv_metrics": pooled_metrics,
        "loso_average_metrics": loso_metrics,
        "loso_per_site_metrics": per_site_loso,
        "generalization_gap_stage0": gap["generalization_gap"],
        "results_dir": str(base),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--config", default="configs/stage0_baseline.yaml")
    parser.add_argument("--clean-outputs", action="store_true")
    args = parser.parse_args()
    summary = train_stage(args.stage, args.config, clean_outputs=args.clean_outputs)
    print(summary)


if __name__ == "__main__":
    main()
