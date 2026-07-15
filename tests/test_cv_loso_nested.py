import pandas as pd
from training.cv_harness import make_pooled_stratified_folds
from training.loso_harness import split_by_loso
from training.nested_cv import make_nested_folds

def test_cv_loso_nested_smoke(tmp_path):
    df=pd.DataFrame({"subject_id":[f"s{i}" for i in range(30)],"site":["UPENN"]*10+["UCSF"]*10+["TCGA"]*10,"mgmt_label":[0,1]*15,"idh_label":[0,1]*15})
    folds=make_pooled_stratified_folds(df,n_splits=2); assert len(folds)==2
    loso=split_by_loso(df); assert len(loso)==3
    nested=make_nested_folds(df, folds, inner_splits=2)
    for f in nested:
        outer_test=set(f["test_subjects"])
        for inner in f["inner_folds"]: assert not outer_test & set(inner["tune_valid_subjects"])
