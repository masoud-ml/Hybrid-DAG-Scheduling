"""Response-time bounds for a single isolated nonpreemptive DAG."""
from __future__ import annotations
import math
from core.dag import DAG
from analysis.graph_metrics import critical_path, antichain_width
from analysis.static_analysis import ib_potential_set


def serial_bound(dag: DAG) -> float:
    return dag.wcet_total()


def generic_list_bound(dag: DAG, num_cores: int) -> float:
    """Graham bound L+(W-L)/m for a greedy work-conserving list schedule."""
    if num_cores <= 0: raise ValueError('num_cores >0')
    l=critical_path(dag)
    return l + (dag.wcet_total()-l)/num_cores


def chen2023_single_dag_bound(dag: DAG, num_cores: int, *,
                             ceil_delay: bool=True, return_details: bool=False):
    """Compute a priority-aware response-time bound for an isolated DAG.
    
    Requires unique priorities, one source, one sink, homogeneous cores,
    and global nonpreemptive scheduling. ceil_delay rounds interference
    delays up to whole input time units; disabling it retains fractional delays."""
    if num_cores <= 0: raise ValueError('num_cores > 0')
    if len({n.priority for n in dag.nodes.values()}) != len(dag.nodes):
        raise ValueError('Chen2023 baseline requires strict priority ranking')
    dag.source(); dag.sink()
    interference: dict[str,set[str]] = {}
    finish: dict[str,float] = {}
    for v in dag.topological_order():
        potential=ib_potential_set(dag,v)
        if antichain_width(dag,potential)<num_cores:
            interference[v]=set()
        else:
            high={x for x in potential if dag.get_node(x).priority>dag.get_node(v).priority}
            low={x for x in potential if dag.get_node(x).priority<dag.get_node(v).priority}
            low_max=set(sorted(low,key=lambda x:(-dag.get_node(x).wcet,x))[:max(0,num_cores-1)])
            low_pre=(set().union(*(dag.ancestors(x) for x in low_max)) if low_max else set()) & (low-low_max)
            interference[v]=high|low_max|low_pre
        pre=dag.pre(v)
        if not pre:
            start=0.0
        else:
            def pred_bound(p):
                work=math.fsum(dag.get_node(x).wcet for x in sorted(interference[v]-interference[p]))
                delay=work/num_cores
                return finish[p]+(math.ceil(delay) if ceil_delay else delay)
            start=max(pred_bound(p) for p in pre)
        finish[v]=start+dag.get_node(v).wcet
    bound=max(finish.values())
    return (bound,dict(finish),interference) if return_details else bound


def serrano2016_single_dag_bound(dag: DAG, num_cores: int) -> float:
    """Compute the response-time bound for an isolated unconditional DAG.
    
    On dedicated cores, other-task interference vanishes and the bound is
    L + (W-L)/m."""
    dag.source(); dag.sink()
    return generic_list_bound(dag, num_cores)
