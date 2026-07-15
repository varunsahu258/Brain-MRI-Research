from __future__ import annotations
from pathlib import Path
import pandas as pd
from .metrics import compute_binary_metrics
def build_ablation_table(predictions_csv: str|Path) -> pd.DataFrame:
    df=pd.read_csv(predictions_csv); rows=[]
    if "y_prob" not in df: raise ValueError("predictions_probabilities.csv must contain y_prob")
    for model,g in df.groupby("model_name"):
        m=compute_binary_metrics(g.y_true,g.y_pred,g.y_prob); rows.append({"Model":model,"Accuracy":m["accuracy"],"Precision":m["precision"],"Recall":m["recall_sensitivity"],"Specificity":m["specificity"],"F1":m["f1"],"Balanced Acc":m["balanced_accuracy"],"Cohen's kappa":m["cohen_kappa"],"AUC-ROC":m["auc_roc"],"AUC-PR":m["auc_pr"]})
    return pd.DataFrame(rows)
