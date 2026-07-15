from __future__ import annotations
import hashlib, json, logging
from pathlib import Path
from evaluation.metrics import generalization_gap
from .loso_harness import assert_loso_folds_match
GAP_THRESHOLD=0.10
def evaluate_generalization_gate(pooled_mgmt_auroc: float, loso_mgmt_auroc: float) -> dict[str, object]:
    gap=generalization_gap(pooled_mgmt_auroc, loso_mgmt_auroc); exceeded=gap>GAP_THRESHOLD
    if exceeded: logging.warning("generalization_gap_exceeded: %.3f", gap)
    return {"generalization_gap":gap,"generalization_gap_exceeded":exceeded,"mlflow_tags":{"generalization_gap_exceeded":str(exceeded).lower()}}
def validate_loso_folds(current: dict, saved_path: str="configs/loso_folds_v1.json") -> None: assert_loso_folds_match(current,saved_path)
def config_hash(config: dict) -> str: return hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
def assert_external_test_allowed(config: dict, freeze_confirmed: bool, lock_path: str|Path="outputs/external_test_lock.json") -> str:
    if not freeze_confirmed: raise PermissionError("External testing requires explicit --freeze-confirmed")
    h=config_hash(config); p=Path(lock_path); p.parent.mkdir(parents=True,exist_ok=True)
    locks=json.loads(p.read_text()) if p.exists() else {}
    if h in locks: raise RuntimeError(f"External test already executed for frozen config hash {h}")
    locks[h]={"config_hash":h}; p.write_text(json.dumps(locks,indent=2)); return h
