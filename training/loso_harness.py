from __future__ import annotations
import json
from pathlib import Path
SITES=["UPENN","UCSF","TCGA"]
def generate_loso_folds() -> dict:
    return {"version":"v1","folds":[{"fold":f"loso_{s.lower()}","held_out_site":s,"train_sites":[x for x in SITES if x!=s]} for s in SITES]}
def load_loso_folds(path: str|Path="configs/loso_folds_v1.json") -> dict: return json.loads(Path(path).read_text())
def assert_loso_folds_match(current: dict, saved_path: str|Path="configs/loso_folds_v1.json") -> None:
    saved=load_loso_folds(saved_path)
    if current != saved: raise ValueError("LOSO fold assignments drifted from saved versioned file")
def split_by_loso(subjects, folds_path: str|Path="configs/loso_folds_v1.json"):
    folds=load_loso_folds(folds_path); out=[]
    for f in folds["folds"]:
        train=subjects[subjects.site.isin(f["train_sites"])] ; test=subjects[subjects.site==f["held_out_site"]]
        out.append({**f,"train_subjects":train.subject_id.tolist(),"test_subjects":test.subject_id.tolist()})
    return out
