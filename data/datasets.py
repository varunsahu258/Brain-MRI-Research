"""Dataset abstractions for multisite radiogenomics MRI data."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
try:
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    class Dataset: pass
from .sequence_normalization import CORE_SEQUENCES, normalize_sequence_map

@dataclass(frozen=True)
class SubjectRecord:
    subject_id: str
    site: str
    sequences: dict[str, str]
    label_mgmt: int | None = None
    label_idh: int | None = None
    bonus_modalities: dict[str, str] | None = None

class SiteDataset(Dataset):
    """Patient-level dataset for one site."""
    site_name = "SITE"
    supports_bonus_modalities = False
    def __init__(self, records: Sequence[SubjectRecord], require_core: bool = True) -> None:
        self.records = list(records)
        self.require_core = require_core
        for r in self.records:
            if r.site != self.site_name: raise ValueError(f"Record {r.subject_id} has site {r.site}, expected {self.site_name}")
            norm = normalize_sequence_map(r.sequences)
            if require_core and any(s not in norm for s in CORE_SEQUENCES): raise ValueError(f"{r.subject_id} missing core sequence")
    def __len__(self) -> int: return len(self.records)
    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self.records[idx]
        return {"subject_id": r.subject_id, "site": r.site, "sequences": normalize_sequence_map(r.sequences), "labels": {"MGMT": r.label_mgmt, "IDH": r.label_idh}, "bonus_modalities": r.bonus_modalities or {}}
    @classmethod
    def from_audit_csv(cls, path: str | Path) -> "SiteDataset":
        import pandas as pd
        df = pd.read_csv(path); recs=[]
        for _, row in df.iterrows():
            seqs={s: f"present:{s}" for s in CORE_SEQUENCES if bool(row.get(f"has_{s}", False))}
            recs.append(SubjectRecord(str(row.subject_id), cls.site_name, seqs, row.get("mgmt_label"), row.get("idh_label")))
        return cls(recs, require_core=False)
class UPENNDataset(SiteDataset): site_name = "UPENN"
class UCSFDataset(SiteDataset):
    site_name = "UCSF"; supports_bonus_modalities = True
class TCGADataset(SiteDataset): site_name = "TCGA"
class PooledDataset(Dataset):
    """Combine site datasets while preserving subject site labels."""
    def __init__(self, datasets: Sequence[SiteDataset]) -> None:
        self.datasets=list(datasets); self.index=[]
        for d_i, d in enumerate(self.datasets): self.index += [(d_i, i) for i in range(len(d))]
    def __len__(self) -> int: return len(self.index)
    def __getitem__(self, idx: int) -> dict[str, Any]:
        d_i, i = self.index[idx]; return self.datasets[d_i][i]
