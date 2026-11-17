from __future__ import annotations

import hashlib
from typing import Optional

from core.dag import DAG


def stable_seed_from_dag(dag: DAG, master_seed: int, bits: int = 32) -> int:
    """
    Stable per-DAG seed derived from (master_seed, dag_id).
    """
    dag_id = getattr(dag, "dag_id", "")
    h = hashlib.sha256(dag_id.encode("utf-8")).digest()
    x = int.from_bytes(h[:8], "little")
    mask = (1 << bits) - 1
    return (master_seed ^ x) & mask


def tie_seed_for(
    dag: DAG,
    *,
    mode: str,
    master_seed: int,
    fixed_seed: Optional[int] = None,
    trial_idx: Optional[int] = None,
) -> int:
    """
    Unified tie-seed generator.
      - fixed: fixed_seed required
      - per_dag: stable hash(dag_id)
      - per_trial: master_seed + trial_idx required
    """
    if mode == "fixed":
        if fixed_seed is None:
            raise ValueError("fixed_seed must be provided when mode='fixed'")
        return int(fixed_seed) & 0xFFFFFFFF

    if mode == "per_dag":
        return stable_seed_from_dag(dag, master_seed)

    if mode == "per_trial":
        if trial_idx is None:
            raise ValueError("trial_idx must be provided when mode='per_trial'")
        return int(master_seed + trial_idx) & 0xFFFFFFFF

    raise ValueError(f"Unknown tie seed mode: {mode}")
