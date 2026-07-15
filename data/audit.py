"""Audit configured multisite MRI dataset roots."""
from __future__ import annotations
import argparse, re
from pathlib import Path
from typing import Any
import pandas as pd, yaml
from .sequence_normalization import CORE_SEQUENCES, normalize_sequence_name
SITE_KEYS={"UPENN":"upenn_root","UCSF":"ucsf_pdgm_root","TCGA":"tcga_gbm_root"}
BONUS=("DWI","ASL")
def _files(root: Path): return [p for p in root.rglob("*") if p.is_file()]
def _subject_id(path: Path, root: Path) -> str: return path.relative_to(root).parts[0] if len(path.relative_to(root).parts)>1 else path.stem.split("_")[0]
def audit_site(site: str, root: str|Path) -> pd.DataFrame:
    root=Path(root)
    if not root.exists(): raise FileNotFoundError(f"Configured {site} root does not exist: {root}")
    files=_files(root)
    if not files: raise ValueError(f"Configured {site} root is empty: {root}")
    rows: dict[str, dict[str, Any]]={}
    for f in files:
        sid=_subject_id(f, root); row=rows.setdefault(sid,{"subject_id":sid,"site":site, **{f"has_{s}":False for s in CORE_SEQUENCES}, "has_segmentation":False,"mgmt_label_present":False,"idh_label_present":False,"mgmt_label":None,"idh_label":None, **{f"has_{b}":False for b in BONUS}})
        up=f.name.upper()
        for token in ("T1GD","T1CE","T1","T2","FLAIR"):
            if re.search(rf"(^|[_\-.]){token}([_\-.]|$)", up): row[f"has_{normalize_sequence_name(token)}"]=True
        if any(x in up for x in ("SEG","MASK","LABEL")): row["has_segmentation"]=True
        for b in BONUS:
            if b in up: row[f"has_{b}"]=True
        if "MGMT" in up: row["mgmt_label_present"]=True
        if "IDH" in up: row["idh_label_present"]=True
    return pd.DataFrame(rows.values()).sort_values("subject_id")
def run_audit(config_path: str|Path="configs/base.yaml") -> dict[str, pd.DataFrame]:
    cfg=yaml.safe_load(Path(config_path).read_text()); out=Path(cfg["paths"].get("audit_output","outputs/audit")); out.mkdir(parents=True, exist_ok=True)
    result={}
    for site,key in SITE_KEYS.items():
        df=audit_site(site,cfg["paths"][key]); df.to_csv(out/f"{site.lower()}_audit.csv", index=False); result[site]=df
    return result
def summarize(df: pd.DataFrame) -> dict[str, Any]:
    return {"subjects":len(df),"mgmt_labels":int(df.mgmt_label_present.sum()),"idh_labels":int(df.idh_label_present.sum()),"missing_core":{s:int((~df[f"has_{s}"]).sum()) for s in CORE_SEQUENCES}}
def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/base.yaml"); args=ap.parse_args()
    for site,df in run_audit(args.config).items(): print(site, summarize(df))
if __name__=="__main__": main()
