from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold
def make_pooled_stratified_folds(subjects: pd.DataFrame, n_splits:int=5, seed:int=42, save_path: str|Path|None=None) -> list[dict[str,list[str]]]:
    df=subjects[subjects.site.isin(["UPENN","UCSF"])].reset_index(drop=True)
    skf=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed); folds=[]
    for i,(tr,te) in enumerate(skf.split(df.subject_id,df.mgmt_label)): folds.append({"fold":f"fold_{i}","train_subjects":df.subject_id.iloc[tr].tolist(),"test_subjects":df.subject_id.iloc[te].tolist()})
    if save_path: Path(save_path).write_text(json.dumps({"folds":folds},indent=2))
    return folds
