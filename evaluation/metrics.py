"""Shared classification metrics for all stages."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, roc_auc_score, average_precision_score

def specificity_score(y_true, y_pred, positive_label:int=1) -> float:
    cm=confusion_matrix(y_true,y_pred,labels=[0,1]); tn,fp,fn,tp=cm.ravel(); return tn/(tn+fp) if (tn+fp) else 0.0
def generalization_gap(pooled_metric: float, loso_metric: float) -> float: return float(pooled_metric-loso_metric)
def compute_binary_metrics(y_true, y_pred, y_prob) -> dict[str,float]:
    """Compute binary metrics from labels and positive-class probabilities."""
    if y_prob is None: raise ValueError("Predicted probabilities are required")
    p,r,f,_=precision_recall_fscore_support(y_true,y_pred,average="binary",zero_division=0)
    pm,rm,fm,_=precision_recall_fscore_support(y_true,y_pred,average="macro",zero_division=0)
    return {"accuracy":accuracy_score(y_true,y_pred),"precision":p,"recall_sensitivity":r,"specificity":specificity_score(y_true,y_pred),"f1":f,"precision_macro":pm,"recall_macro":rm,"f1_macro":fm,"balanced_accuracy":balanced_accuracy_score(y_true,y_pred),"cohen_kappa":cohen_kappa_score(y_true,y_pred),"auc_roc":roc_auc_score(y_true,y_prob),"auc_pr":average_precision_score(y_true,y_prob)}
