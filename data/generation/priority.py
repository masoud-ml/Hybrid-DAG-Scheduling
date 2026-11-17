from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from core.dag import DAG


@dataclass(frozen=True)
class PriorityAssignConfig:
    base_priority: int = 10_000
    tie_breaker: str = "task_id"  


def assign_priorities_front_layer_highest_wcet(
    dag: DAG,
    cfg: PriorityAssignConfig = PriorityAssignConfig(),
) -> Dict[str, int]:
    layers = _compute_layers(dag)

    items: List[Tuple[int, float, str]] = []
    for nid in dag.node_ids():
        n = dag.get_node(nid)
        items.append((layers[nid], float(n.wcet), nid))

    # front layer first; within layer highest WCET first
    if cfg.tie_breaker == "task_id":
        items.sort(key=lambda x: (x[0], -x[1], x[2]))
    else:
        items.sort(key=lambda x: (x[0], -x[1]))

    p = cfg.base_priority
    out: Dict[str, int] = {}
    for _, _, nid in items:
        out[nid] = p
        dag.get_node(nid).priority = p
        p -= 1
    return out


def _compute_layers(dag: DAG) -> Dict[str, int]:
    topo = dag.topological_order()
    layer: Dict[str, int] = {}
    for nid in topo:
        preds = dag.get_node(nid).predecessors
        if not preds:
            layer[nid] = 0
        else:
            layer[nid] = 1 + max(layer[p] for p in preds)
    return layer
