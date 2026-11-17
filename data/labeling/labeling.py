"""Scenario labels: 0 for a static certificate, 1 for a witnessed anomaly,
and undefined when neither condition holds."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
from collections import Counter
import hashlib
import json
import math
import random
from core.dag import DAG
from analysis.static_analysis import anomaly_test_scenario
from schedule.long_makespan import simulate_long_makespan
from schedule.runtime_sim import simulate_fixed_priority_runtime
from utils.tie_seed import tie_seed_for

@dataclass(frozen=True)
class LabelingConfig:
    num_cores: int
    n_trials: int = 100_000
    min_scale: float = 0.5
    max_scale: float = 1.0
    quantize: Optional[float] = None
    master_seed: int = 123
    early_stop: bool = True
    tie_break: str = "deterministic"
    tie_seed_mode: str = "per_trial"
    tie_seed_fixed: int = 1
    debug: bool = False
    debug_dag_limit: int = 3
    debug_print_trials: int = 0

@dataclass(frozen=True)
class LabelResult:
    label: str
    static_pass: bool
    makespan_long: float
    best_makespan_seen: float
    anomaly_found: bool
    trial_found: int
    n_trials_run: int
    witness_json: str = ""


def _stable_seed(master_seed: int, dag_id: str) -> int:
    h = hashlib.sha256(dag_id.encode('utf-8')).digest()
    return (master_seed ^ int.from_bytes(h[:8], 'little')) & 0xffffffffffffffff


def _sample_exec_times(dag: DAG, rng: random.Random, cfg: LabelingConfig) -> Dict[str, float]:
    out = {}
    for v in dag.node_ids():
        scale = rng.uniform(cfg.min_scale, cfg.max_scale)
        if cfg.quantize is not None:
            scale = round(scale / cfg.quantize) * cfg.quantize
            scale = min(max(scale, cfg.min_scale), cfg.max_scale)
        out[v] = dag.get_node(v).wcet * scale
    return out


def label_dag(dag: DAG, cfg: LabelingConfig,
              forced_exec_times: Optional[Dict[str, float]] = None,
              forced_tie_seed: Optional[int] = None) -> LabelResult:
    if not (0 < cfg.min_scale <= cfg.max_scale <= 1):
        raise ValueError("require 0 < min_scale <= max_scale <= 1")
    if cfg.quantize is not None and cfg.quantize <= 0:
        raise ValueError("quantize must be positive")
    _, _, ms_long = simulate_long_makespan(dag, cfg.num_cores)
    # A varying randomized tie policy changes the scheduling scenario. It cannot
    # use the long baseline to prove safety, and no 0 label is emitted.
    ties = len(set(n.priority for n in dag.nodes.values())) != len(dag.nodes)
    static_pass = False
    if cfg.tie_break == "deterministic" and not ties:
        static_pass, _ = anomaly_test_scenario(dag, cfg.num_cores, min_fraction=0.0)
    if static_pass:
        return LabelResult('0', True, ms_long, ms_long, False, -1, 0)
    if cfg.n_trials <= 0 and forced_exec_times is None:
        return LabelResult('undefined', False, ms_long, ms_long, False, -1, 0)
    rng = random.Random(_stable_seed(cfg.master_seed, dag.dag_id))
    max_ms = ms_long
    n_run = 0
    found = -1
    witness = ""
    for trial in ([1] if forced_exec_times is not None else range(1, cfg.n_trials + 1)):
        et = forced_exec_times if forced_exec_times is not None else _sample_exec_times(dag, rng, cfg)
        seed = (forced_tie_seed if forced_tie_seed is not None else
                tie_seed_for(dag, mode=cfg.tie_seed_mode,
                    master_seed=cfg.master_seed, fixed_seed=cfg.tie_seed_fixed, trial_idx=trial))
        _, _, ms = simulate_fixed_priority_runtime(dag, cfg.num_cores, et,
                            seed=seed, tie_break=cfg.tie_break)
        max_ms = max(max_ms, ms)
        n_run += 1
        if ms > ms_long + 1e-8*max(1, ms_long):
            if found == -1:
                found = trial
                witness = json.dumps({'exec_times': et, 'tie_seed': seed,
                                      'tie_break': cfg.tie_break, 'makespan': ms}, sort_keys=True)
            if cfg.early_stop:
                break
    return LabelResult('1' if found != -1 else 'undefined', False,
                       ms_long, max_ms, found != -1, found, n_run, witness)
