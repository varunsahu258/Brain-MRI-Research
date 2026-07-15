from __future__ import annotations
import numpy as np
from .metrics import compute_binary_metrics
def bootstrap_ci(y_true, y_pred, y_prob, metrics=("accuracy","auc_roc","auc_pr"), n_bootstraps:int=1000, seed:int=42) -> dict[str, tuple[float,float]]:
    rng=np.random.default_rng(seed); y_true=np.asarray(y_true); y_pred=np.asarray(y_pred); y_prob=np.asarray(y_prob); vals={m:[] for m in metrics}
    for _ in range(n_bootstraps):
        idx=rng.integers(0,len(y_true),len(y_true))
        if len(np.unique(y_true[idx]))<2: continue
        m=compute_binary_metrics(y_true[idx],y_pred[idx],y_prob[idx])
        for k in vals: vals[k].append(m[k])
    return {k:(float(np.percentile(v,2.5)),float(np.percentile(v,97.5))) for k,v in vals.items()}
