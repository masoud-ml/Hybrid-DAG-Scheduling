from __future__ import annotations
from dataclasses import dataclass
from statistics import median
from core.dag import DAG
from schedule.long_makespan import simulate_long_makespan
from schedule.runtime_sim import simulate_fixed_priority_runtime
from analysis.graph_metrics import critical_path, antichain_width

FEATURE_COLUMNS = [
    "max_makespan", "min_makespan", "max_workload", "min_workload",
    "max_et", "min_et", "median_et", "n_nodes", "max_parallel",
    "critical_path", "avg_indeg", "avg_outdeg", "num_cores",
]

@dataclass(frozen=True)
class ScenarioFeatures:
    min_workload: float
    max_workload: float
    min_makespan: float
    max_makespan: float
    min_et: float
    max_et: float
    median_et: float
    n_nodes: int
    max_parallel: int
    critical_path: float
    avg_indeg: float
    avg_outdeg: float
    num_cores: int


def compute_features(dag: DAG, num_cores: int, *, tie_seed_mode: str = "per_dag",
                     tie_master_seed: int = 123, tie_seed_fixed: int = 1,
                     tie_trial_idx: int | None = None) -> ScenarioFeatures:
    """Compute scenario features from WCET and half-WCET executions.
    
    Workload is the sum of node execution times. ET statistics use WCETs."""
    wcet = {v: dag.get_node(v).wcet for v in dag.node_ids()}
    half = {v: 0.5*c for v, c in wcet.items()}
    _, _, max_ms = simulate_long_makespan(dag, num_cores)
    _, _, min_ms = simulate_fixed_priority_runtime(dag, num_cores, half)
    ets = list(wcet.values())
    n = len(ets)
    edges = sum(len(dag.pre(v)) for v in wcet)
    return ScenarioFeatures(
        min_workload=sum(half.values()), max_workload=sum(wcet.values()),
        min_makespan=min_ms, max_makespan=max_ms,
        min_et=min(ets), max_et=max(ets), median_et=median(ets),
        n_nodes=n, max_parallel=antichain_width(dag),
        critical_path=critical_path(dag), avg_indeg=edges/n, avg_outdeg=edges/n,
        num_cores=num_cores,
    )
