"""Sequence-name normalization for multisite brain MRI inputs."""
from __future__ import annotations

CORE_SEQUENCES = ("T1", "T1CE", "T2", "FLAIR")
ALIASES = {"T1GD": "T1CE", "T1C": "T1CE", "T1CE": "T1CE", "T1": "T1", "T2": "T2", "FLAIR": "FLAIR"}

def normalize_sequence_name(name: str) -> str:
    """Return canonical sequence name.

    Parameters
    ----------
    name : str
        Raw sequence label from a dataset.
    """
    key = name.upper().replace("-", "").replace("_", "")
    if key not in ALIASES:
        raise ValueError(f"Unknown MRI sequence name: {name}")
    return ALIASES[key]

def normalize_sequence_map(paths: dict[str, str]) -> dict[str, str]:
    """Normalize sequence-keyed path mapping to T1/T1CE/T2/FLAIR schema."""
    out: dict[str, str] = {}
    for key, value in paths.items():
        norm = normalize_sequence_name(key)
        if norm in out and out[norm] != value:
            raise ValueError(f"Duplicate normalized sequence {norm}")
        out[norm] = value
    return out

def missing_core_sequences(paths: dict[str, str]) -> list[str]:
    """List canonical core sequences absent from a mapping."""
    normalized = normalize_sequence_map(paths)
    return [seq for seq in CORE_SEQUENCES if seq not in normalized]
