"""Global work-conserving, nonpreemptive fixed-priority DAG simulator.

Simultaneous completions are processed before dispatch. Priority ties
use node-ID order by default; randomized ties are diagnostic only."""
from __future__ import annotations

import heapq
import math
import random
from typing import Dict, Tuple
from core.dag import DAG


def simulate_fixed_priority_runtime(
    dag: DAG,
    num_cores: int,
    exec_times: Dict[str, float],
    seed: int = 123,
    *,
    tie_break: str = "deterministic",
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    if not isinstance(num_cores, int) or isinstance(num_cores, bool) or num_cores < 1:
        raise ValueError("num_cores must be >= 1")
    if tie_break not in ("deterministic", "seeded_random"):
        raise ValueError("tie_break must be 'deterministic' or 'seeded_random'")
    nodes = dag.node_ids()
    if not nodes:
        raise ValueError("cannot schedule an empty DAG")
    for v in nodes:
        if v not in exec_times or not math.isfinite(exec_times[v]) or not (0 <= exec_times[v] <= dag.get_node(v).wcet) or (exec_times[v] == 0 and not dag.get_node(v).dummy):
            raise ValueError(f"execution time must satisfy 0 < ET <= WCET for {v}")
    for v in nodes:
        node = dag.get_node(v)
        if not math.isfinite(node.wcet) or node.wcet < 0:
            raise ValueError('WCET must be finite and nonnegative')
        if node.dummy and (node.wcet != 0 or (dag.pre(v) and dag.suc(v))):
            raise ValueError('Only zero-cost source/sink dummy nodes are supported')
    dag.topological_order()  
    rng = random.Random(seed)
    start: Dict[str, float] = {}
    finish: Dict[str, float] = {}
    free = list(range(num_cores))
    heapq.heapify(free)
    running: list[tuple[float, int, str]] = []
    now = 0.0
    while len(finish) < len(nodes):
        # Dummy boundary nodes consume no core; close zero-time dependencies first.
        while True:
            zeros = [v for v in nodes if v not in start and dag.get_node(v).dummy
                     and dag.pre(v).issubset(finish)]
            if not zeros:
                break
            for v in zeros:
                start[v] = finish[v] = now
        if len(finish) == len(nodes):
            break
        ready = [v for v in nodes if v not in start and dag.pre(v).issubset(finish)]
        if tie_break == "deterministic":
            ready.sort(key=lambda v: (-dag.get_node(v).priority, v))
        else:
            ready.sort()
            rng.shuffle(ready)
            ready.sort(key=lambda v: -dag.get_node(v).priority)  # stable shuffle within priority
        while free and ready:
            v = ready.pop(0)
            core = heapq.heappop(free)
            start[v] = now
            heapq.heappush(running, (now + exec_times[v], core, v))
        if not running:
            raise RuntimeError("invalid DAG: no runnable or running nodes")
        now = running[0][0]
        # Exact simultaneous completions; avoid epsilon grouping distinct times.
        while running and running[0][0] == now:
            ft, core, v = heapq.heappop(running)
            finish[v] = ft
            heapq.heappush(free, core)
    return start, finish, max(finish.values())
