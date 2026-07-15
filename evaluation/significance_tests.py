from __future__ import annotations
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
def mcnemar_pairwise(y_true, pred_a, pred_b) -> dict[str,float]:
    y=np.asarray(y_true); a=np.asarray(pred_a)==y; b=np.asarray(pred_b)==y
    table=[[int(np.sum(a & b)), int(np.sum(a & ~b))],[int(np.sum(~a & b)), int(np.sum(~a & ~b))]]
    exact= any(c<25 for row in table for c in row); res=mcnemar(table, exact=exact, correction=not exact)
    return {"statistic":float(res.statistic),"pvalue":float(res.pvalue),"exact":exact}
def wilcoxon_signed_rank(scores_a, scores_b) -> dict[str,float]:
    s,p=wilcoxon(scores_a,scores_b); return {"statistic":float(s),"pvalue":float(p)}

def paired_ttest_5x2cv(estimator1, estimator2, X, y, scoring: str="roc_auc", random_seed: int=42) -> dict[str,float]:
    """Run mlxtend 5x2cv paired t-test for two estimators."""
    try:
        from mlxtend.evaluate import paired_ttest_5x2cv as _paired
    except Exception as e:
        raise ImportError("mlxtend is required for the 5x2cv paired t-test") from e
    t, p = _paired(estimator1=estimator1, estimator2=estimator2, X=X, y=y, scoring=scoring, random_seed=random_seed)
    return {"t_statistic": float(t), "pvalue": float(p)}
