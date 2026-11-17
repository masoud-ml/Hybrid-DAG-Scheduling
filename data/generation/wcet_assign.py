"""Reproducible WCET budget assignment, including source and sink."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import math
import random
from core.dag import DAG

@dataclass(frozen=True)
class WcetAssignConfig:
    min_node_ms: float = 1e-5
    include_src_sink: bool = True
    src_sink_ms: float = 1.0


def assign_wcets_from_budget(dag: DAG, total_budget_ms: float,
                             seed: Optional[int] = None,
                             cfg: WcetAssignConfig = WcetAssignConfig()) -> Dict[str, float]:
    """Assign strictly positive WCETs whose total equals budget (numerical tolerance)."""
    ids = sorted(dag.node_ids())
    if not ids or not math.isfinite(total_budget_ms) or total_budget_ms <= 0:
        raise ValueError("positive finite budget and nonempty DAG required")
    rng = random.Random(seed)
    fixed = set() if cfg.include_src_sink else set(ids) & {"src", "sink"}
    if cfg.min_node_ms <= 0 or (fixed and cfg.src_sink_ms <= 0):
        raise ValueError("minimum node and source/sink WCET must be positive")
    budget_fixed = cfg.src_sink_ms * len(fixed)
    variable = [v for v in ids if v not in fixed]
    min_total = budget_fixed + len(variable) * cfg.min_node_ms
    if not variable or min_total > total_budget_ms:
        raise ValueError(f"WCET budget {total_budget_ms} insufficient, minimum is {min_total}")
    for v in fixed:
        dag.get_node(v).wcet = cfg.src_sink_ms
    extra = total_budget_ms - min_total
    weights = [rng.expovariate(1.0) for _ in variable]
    denom = sum(weights)
    for v, w in zip(variable, weights):
        dag.get_node(v).wcet = cfg.min_node_ms + extra*w/denom
    # Correct rounding on last variable node for strict budget conservation.
    last = variable[-1]
    dag.get_node(last).wcet += total_budget_ms - dag.wcet_total()
    if not math.isclose(dag.wcet_total(), total_budget_ms, rel_tol=1e-11, abs_tol=1e-7):
        raise AssertionError("budget conservation violated")
    return {v: dag.get_node(v).wcet for v in ids}
