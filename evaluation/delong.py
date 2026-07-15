"""Paired ROC-AUC comparison using a bootstrap approximation to DeLong p-values."""
from __future__ import annotations
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
def delong_roc_test(y_true, pred_a, pred_b) -> float:
    """Return two-sided p-value comparing paired AUCs.

    This self-contained implementation estimates the paired AUC difference variance by stratified bootstrap.
    """
    y=np.asarray(y_true); a=np.asarray(pred_a); b=np.asarray(pred_b); rng=np.random.default_rng(123); diffs=[]
    pos=np.where(y==1)[0]; neg=np.where(y==0)[0]
    for _ in range(1000):
        idx=np.r_[rng.choice(pos,len(pos),True), rng.choice(neg,len(neg),True)]
        diffs.append(roc_auc_score(y[idx],a[idx])-roc_auc_score(y[idx],b[idx]))
    diff=roc_auc_score(y,a)-roc_auc_score(y,b); se=np.std(diffs,ddof=1)
    return 1.0 if se==0 else float(2*(1-norm.cdf(abs(diff)/se)))
