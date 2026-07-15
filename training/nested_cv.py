from __future__ import annotations
import pandas as pd
from sklearn.model_selection import StratifiedKFold
def make_nested_folds(subjects: pd.DataFrame, outer_folds: list[dict], label_col: str="idh_label", inner_splits:int=3, seed:int=42, enable_inner: bool=True) -> list[dict]:
    out=[]
    for outer in outer_folds:
        test=set(outer["test_subjects"]); train_df=subjects[subjects.subject_id.isin(outer["train_subjects"])].reset_index(drop=True)
        inners=[]
        if enable_inner:
            skf=StratifiedKFold(inner_splits,shuffle=True,random_state=seed)
            for tr,va in skf.split(train_df.subject_id, train_df[label_col]):
                assert not (set(train_df.subject_id.iloc[va]) & test)
                inners.append({"tune_train_subjects":train_df.subject_id.iloc[tr].tolist(),"tune_valid_subjects":train_df.subject_id.iloc[va].tolist()})
        out.append({**outer,"inner_folds":inners})
    return out
