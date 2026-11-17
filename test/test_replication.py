import math
import random
from dataclasses import asdict
import pytest
from core.dag import DAG,Node
from data.generation.dag_generator import DagGenConfig,generate_layered_dag
from data.generation.priority import assign_priorities_front_layer_highest_wcet
from data.generation.wcet_assign import assign_wcets_from_budget,WcetAssignConfig
from data.feature.features import compute_features,FEATURE_COLUMNS
from schedule.runtime_sim import simulate_fixed_priority_runtime
from schedule.long_makespan import simulate_long_makespan,simulate_algorithm1
from analysis.static_analysis import anomaly_test_scenario,eta_set
from analysis.wcrt_bounds import chen2023_single_dag_bound,generic_list_bound
from analysis.graph_metrics import antichain_width,critical_path
from data.labeling.labeling import LabelingConfig,label_dag


def example1():
    d=DAG('paper_fig1')
    for v,c,p in [('v1',1,5),('v2',2,4),('v3',3,3),('v4',2,2),('v6',2,1)]:
        d.add_node(Node(v,c,p))
    for v in ['v2','v3','v4']:
        d.add_edge('v1',v);d.add_edge(v,'v6')
    return d


def test_paper_fig1_makespan():
    d=example1()
    starts,finish,ms=simulate_long_makespan(d,2)
    assert ms==7 and starts['v4']==3 and finish['v6']==7
    assert simulate_algorithm1(d,2)[2]==7


def test_ties_disable_static_certificate():
    d=DAG('ties')
    d.add_nodes([Node('s',1,8),Node('a',10,2),Node('b',1,2),Node('c',1,2),Node('z',1,1)])
    for v in ['a','b','c']:
        d.add_edge('s',v);d.add_edge(v,'z')
    assert not anomaly_test_scenario(d,2)[0]
    assert not label_dag(d,LabelingConfig(2,n_trials=0)).static_pass


def test_features_paper_values():
    d=example1();f=asdict(compute_features(d,2))
    assert list(FEATURE_COLUMNS)==list(FEATURE_COLUMNS)
    assert set(f)==set(FEATURE_COLUMNS)
    assert f['max_workload']==10
    assert f['min_workload']==5
    assert f['max_makespan']==7
    assert f['min_makespan']==3.5
    assert f['min_et']==1 and f['max_et']==3
    assert f['median_et']==2
    assert f['critical_path']==6
    assert f['max_parallel']==3


def test_wcet_conservation_including_source_sink():
    d=generate_layered_dag('budget',DagGenConfig(5,4),seed=4)
    a=assign_wcets_from_budget(d,421.4,seed=3)
    assert math.isclose(sum(a.values()),421.4,abs_tol=1e-7)
    assign_wcets_from_budget(d,421.4,seed=3,cfg=WcetAssignConfig(include_src_sink=False))
    assert math.isclose(d.wcet_total(),421.4,abs_tol=1e-7)


def test_analytical_bound_and_paper_algorithm_regression():
    rng=random.Random(123)
    for i in range(60):
        d=generate_layered_dag(f'prop{i}',DagGenConfig(rng.randint(3,6),rng.randint(2,5),
                            fixed_layer_width=False),seed=rng.getrandbits(64))
        assign_wcets_from_budget(d,120,seed=rng.getrandbits(64))
        assign_priorities_front_layer_highest_wcet(d)
        m=rng.randint(1,4)
        baseline=simulate_long_makespan(d,m)[2]
        eq=simulate_algorithm1(d,m)[2]
        assert math.isclose(baseline,eq,abs_tol=1e-7), (i,m,baseline,eq)
        assert chen2023_single_dag_bound(d,m)+1e-7>=baseline
        assert generic_list_bound(d,m)+1e-7>=baseline
        safe,_=anomaly_test_scenario(d,m)
        for trial in range(12):
            ets={v:d.get_node(v).wcet*rng.uniform(0.5,1) for v in d.node_ids()}
            ms=simulate_fixed_priority_runtime(d,m,ets)[2]
            assert ms<=chen2023_single_dag_bound(d,m)+1e-7
            if safe:assert ms<=baseline+1e-7


def test_concurrent_width_beyond_layer_width():
    d=example1()
    assert antichain_width(d)==3
    assert eta_set(d,'v2')=={'v3','v4'}


def paper_fig2():
    d=DAG('paper_fig2')
    nodes=[('v1',1,7),('v2',3,6),('v3',2,5),('v4',2,4),('v5',2,3),
           ('v6',5,2),('v7',1,1)]
    d.add_nodes([Node(v,w,p) for v,w,p in nodes])
    d.add_edges([('v1','v2'),('v1','v3'),('v2','v4'),('v2','v5'),
                 ('v3','v6'),('v4','v7'),('v5','v7'),('v6','v7')])
    return d


def test_paper_figure2_timing_anomaly_is_detected():
    d=paper_fig2()
    assert simulate_long_makespan(d,2)[2] == 9
    reduced={v: d.get_node(v).wcet for v in d.node_ids()}
    reduced['v2']=1.0
    assert simulate_fixed_priority_runtime(d,2,reduced)[2] == 10
    safe,_=anomaly_test_scenario(d,2)
    assert not safe
    result=label_dag(d,LabelingConfig(num_cores=2,n_trials=0),forced_exec_times=reduced)
    assert result.label=='1' and result.best_makespan_seen==10
