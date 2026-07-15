from __future__ import annotations
from pathlib import Path
import pandas as pd
from .metrics import compute_binary_metrics
STAGE_DIRS={0:"stage_0_baseline",1:"stage_1_ssl_pretrain",2:"stage_2_site_adversarial",3:"stage_3_ucsf_auxiliary",4:"stage_4_cross_attention",5:"stage_5_classical_ensemble"}
def prepare_results_tree(root: str|Path, stage:int) -> Path:
    base=Path(root)/STAGE_DIRS[stage]
    for sub in ["metrics","confusion_matrices","significance_tests","plots"]: (base/sub).mkdir(parents=True,exist_ok=True)
    return base
def export_predictions_and_metrics(predictions: pd.DataFrame, root: str|Path, stage:int) -> Path:
    required={"model_name","subject_id","fold","y_true","y_pred","y_prob"}
    missing=required-set(predictions.columns)
    if missing: raise ValueError(f"Predicted probabilities and required columns missing: {sorted(missing)}")
    if predictions["y_prob"].isna().any(): raise ValueError("Predicted probabilities are required for every row")
    base=prepare_results_tree(root,stage); pred_path=base/"metrics"/"predictions_probabilities.csv"
    if pred_path.exists(): raise FileExistsError(f"Refusing to overwrite existing results: {pred_path}")
    predictions.to_csv(pred_path,index=False)
    rows=[]
    for model,df in predictions.groupby("model_name"):
        m=compute_binary_metrics(df.y_true,df.y_pred,df.y_prob); m["model"]=model; rows.append(m)
    pd.DataFrame(rows).to_csv(base/"metrics"/"per_model_metrics.csv",index=False)
    return base
