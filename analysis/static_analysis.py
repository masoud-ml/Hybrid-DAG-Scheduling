"""Sufficient test for timing-anomaly freedom.

Certification requires deterministic nonpreemptive dispatch and unique
priorities. A failed test leaves the scenario unresolved."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
from core.dag import DAG
from schedule.long_makespan import simulate_long_makespan
from analysis.graph_metrics import antichain_width, earliest_starts


@dataclass(frozen=True)
class NodeAnomalyReport:
    node_id: str
    passes: bool
    reason: str
    eta: Set[str]
    ib_remove: Set[str]
    ib_potential: Set[str]
    ib_final: Set[str]


def eta_set(dag: DAG, vj: str) -> Set[str]:
    return {vx for vx in dag.node_ids() if vx != vj and dag.pre(vj) <= dag.pre(vx)}


def ib_remove_set(dag: DAG, vj: str) -> Set[str]:
    eta = eta_set(dag, vj)
    pj = dag.get_node(vj).priority
    out = set()
    for vx in dag.node_ids():
        if (dag.get_node(vx).priority < pj and
            (vx in eta or bool(dag.ancestors(vx) & eta))):
            out.add(vx)
            out |= dag.descendants(vx)
    return out


def ib_potential_set(dag: DAG, vj: str) -> Set[str]:
    return dag.concurrent_nodes(vj) - ib_remove_set(dag, vj)


def ib_final_set(dag: DAG, vj: str, start_times_long: Dict[str, float],
                 finish_times_long: Dict[str, float]) -> Set[str]:
    sj = start_times_long[vj]
    return {x for x in ib_potential_set(dag, vj) if finish_times_long[x] > sj}


def anomaly_test_node(dag: DAG, vj: str, num_cores: int,
                      start_times_long: Dict[str, float],
                      finish_times_long: Dict[str, float],
                      finish_times_seen: Dict[str, float] | None = None,
                      *, min_fraction: float = 0.0) -> NodeAnomalyReport:
    eta = eta_set(dag, vj)
    rem = ib_remove_set(dag, vj)
    pot = dag.concurrent_nodes(vj) - rem
    final = {x for x in pot if finish_times_long[x] > start_times_long[vj]}
    has_ties = len({dag.get_node(x).priority for x in dag.node_ids()}) != len(dag.nodes)
    if has_ties:
        return NodeAnomalyReport(vj, False, "Tied priorities: Tied-priority behavior unspecified by proof; conservative refusal", eta, rem, pot, final)
    if not final:
        return NodeAnomalyReport(vj, True, "Eq. 5: IBfinal empty", eta, rem, pot, final)
    if len(final) < num_cores:
        return NodeAnomalyReport(vj, True, "Lemma 3: fewer candidates than cores", eta, rem, pot, final)
    if antichain_width(dag, final) < num_cores:
        return NodeAnomalyReport(vj, True, "Lemma 3: maximum antichain smaller than cores", eta, rem, pot, final)
    if not any(dag.get_node(x).priority > dag.get_node(vj).priority for x in final):
        return NodeAnomalyReport(vj, True, "Lemma 5: no higher-priority candidate", eta, rem, pot, final)
    # A lower-priority blocker must start before the target becomes ready.
    # f'_pre(j) <= f_pre_long(j) (by preceding-node induction).
    fpre_long = dag.f_pre(finish_times_long, vj)
    lower = earliest_starts(dag, min_fraction=min_fraction)
    feasible = {x for x in final if dag.get_node(x).priority > dag.get_node(vj).priority
                or lower[x] < fpre_long}
    if len(feasible) < num_cores or antichain_width(dag, feasible) < num_cores:
        return NodeAnomalyReport(vj, True, "Lemma 4: too few eligible candidates", eta, rem, pot, final)
    # The core-filling antichain must contain a higher-priority node.
    high = {x for x in feasible if dag.get_node(x).priority > dag.get_node(vj).priority}
    if not any(1 + antichain_width(dag, feasible & dag.concurrent_nodes(h)) >= num_cores
               for h in high):
        return NodeAnomalyReport(vj, True, "Lemmas 3+5: no core-filling antichain containing a higher-priority node", eta, rem, pot, final)
    return NodeAnomalyReport(vj, False, "Necessary overrun conditions not ruled out (unknown)", eta, rem, pot, final)


def anomaly_test_scenario(dag: DAG, num_cores: int, *, min_fraction: float = 0.0
                         ) -> Tuple[bool, List[NodeAnomalyReport]]:
    """Test anomaly freedom with a guaranteed ET/WCET lower bound.
    
    min_fraction=0 covers any positive execution time; 0.5 requires every
    release to execute for at least half its WCET."""
    if not 0.0 <= min_fraction <= 1.0:
        raise ValueError("min_fraction must be within [0,1]")
    dag.source(); dag.sink()
    starts, finishes, _ = simulate_long_makespan(dag, num_cores)
    order = sorted(dag.node_ids(), key=lambda v: (starts[v], -dag.get_node(v).priority, v))
    reports = []
    for v in order:
        report = anomaly_test_node(dag, v, num_cores, starts, finishes,
                                   min_fraction=min_fraction)
        reports.append(report)
        if not report.passes:
            # A failed test breaks the induction for subsequent start times.
            break
    return all(r.passes for r in reports), reports
