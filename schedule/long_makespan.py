"""WCET schedule using the runtime fixed-priority dispatcher."""
from __future__ import annotations

from typing import Dict, Tuple
from core.dag import DAG
from schedule.runtime_sim import simulate_fixed_priority_runtime


def simulate_long_makespan(dag: DAG, num_cores: int) -> Tuple[Dict[str, float], Dict[str, float], float]:
    wcet = {v: dag.get_node(v).wcet for v in dag.node_ids()}
    return simulate_fixed_priority_runtime(dag, num_cores, wcet, tie_break="deterministic")


def simulate_algorithm1(dag: DAG, num_cores: int) -> Tuple[Dict[str, float], Dict[str, float], float]:
    """Simulate a WCET schedule with priority-based fallback tie resolution.
    
    Equal predecessor finish times are resolved by descending priority,
    then node ID."""
    if num_cores <= 0: raise ValueError('num_cores must be positive')
    cores = [(0.0, i) for i in range(num_cores)]
    queue = set(dag.node_ids())
    starts = {}
    finishes = {}
    while queue:
        cores.sort()
        load, core=cores[0]
        ready = [v for v in queue if dag.pre(v).issubset(finishes)]
        ready.sort(key=lambda v:(-dag.get_node(v).priority,v))
        if not ready: raise ValueError('DAG is cyclic')
        node=next((v for v in ready if load >= dag.f_pre(finishes,v)), None)
        if node is None:
            node=min(ready,key=lambda v:(dag.f_pre(finishes,v),
                                               -dag.get_node(v).priority,v))
        start=max(load,dag.f_pre(finishes,node))
        end=start+dag.get_node(node).wcet
        cores[0]=(end,core)
        starts[node]=start
        finishes[node]=end
        queue.remove(node)
    return starts,finishes,max(finishes.values(),default=0.0)
