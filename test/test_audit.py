import itertools
import json
import random
from dataclasses import asdict
import numpy as np
import pandas as pd
import pytest
from core.dag import DAG, Node
from analysis.graph_metrics import antichain_width
from analysis.static_analysis import anomaly_test_scenario, ib_final_set
from analysis.wcrt_bounds import chen2023_single_dag_bound, serrano2016_single_dag_bound
from schedule.long_makespan import simulate_algorithm1, simulate_long_makespan
from schedule.runtime_sim import simulate_fixed_priority_runtime
from data.labeling.labeling import label_dag, LabelingConfig
from data.feature.features import FEATURE_COLUMNS
from pipeline.dataset_generation import generate_dataset
from pipeline.dataset_labeling_sp import label_dataset
from pipeline.dataset_labeling_mp import label_dataset_parallel
from pipeline.train_model import main, SemiSupTrainConfig
from learning.model import RFModelConfig
from evaluation.evaluate import paper_summaries
from utils.dag_saver import save_dag_json
from utils.dag_loader import load_dag_json
from test.test_replication import paper_fig2, example1


def test_nonfinite_and_invalid_runtime_inputs():
    for x in (float('nan'),float('inf'),-1,0):
        with pytest.raises(ValueError): Node('a',x,1)
    d=example1(); et={v:n.wcet for v,n in d.nodes.items()}
    et['v1'] += 1e-9
    with pytest.raises(ValueError): simulate_fixed_priority_runtime(d,2,et)
    with pytest.raises(ValueError): simulate_fixed_priority_runtime(d,2.5,et)


def test_dummy_source_sink_roundtrip(tmp_path):
    d=DAG('dummy')
    d.add_nodes([Node('s',0,10,dummy=True),Node('a',3,3),Node('b',2,2),Node('z',0,0,dummy=True)])
    d.add_edges([('s','a'),('s','b'),('a','z'),('b','z')])
    assert simulate_long_makespan(d,2)[2]==3
    assert simulate_algorithm1(d,2)[2]==3
    assert anomaly_test_scenario(d,2)[0]
    f=tmp_path/'d.json';save_dag_json(d,str(f))
    assert simulate_long_makespan(load_dag_json(str(f)),2)[2]==3


def test_eq5_does_not_remove_short_positive_overlap():
    d=DAG('tiny');d.add_nodes([Node('a',1,2),Node('b',1,3)])
    assert ib_final_set(d,'a',{'a':1.0,'b':0.0},{'a':2.,'b':1.+1e-12})=={'b'}


def test_lemma5_high_priority_must_belong_to_full_antichain():
    # Construct a poset where width=3, but every antichain including h has size<=2.
    d=DAG('joint');d.add_nodes([Node(x,1,i) for i,x in enumerate(['s','a','b','c','h','t'])])
    d.add_edges([('s','a'),('s','b'),('s','c'),('a','h'),('b','h'),('h','t'),('c','t')])
    candidates={'a','b','c','h'}
    assert antichain_width(d,candidates)==3
    assert 1+antichain_width(d,candidates & d.concurrent_nodes('h'))==2


def test_exhaustive_binary_execution_grid():
    # 120 different small posets/priorities, all 2^5 internal duration choices.
    # This is finite-grid validation, never a continuous-time proof.
    rng=random.Random(98421)
    for case in range(120):
        d=DAG(f'exhaustive{case}')
        ranks=list(range(7));rng.shuffle(ranks)
        d.add_nodes([Node(str(i),rng.randint(1,4),ranks[i]) for i in range(7)])
        for i in range(1,6):
            for j in range(i+1,6):
                if rng.random()<.3:d.add_edge(str(i),str(j))
        for i in range(1,6):
            if not d.pre(str(i)):d.add_edge('0',str(i))
            if not d.suc(str(i)):d.add_edge(str(i),'6')
        m=1+case%4
        long=simulate_long_makespan(d,m)
        assert simulate_algorithm1(d,m)==long
        safe,reports=anomaly_test_scenario(d,m)
        bound=chen2023_single_dag_bound(d,m)
        b4=serrano2016_single_dag_bound(d,m)
        for bits in itertools.product((.5,1.),repeat=5):
            et={v:n.wcet for v,n in d.nodes.items()}
            et.update({str(i):d.get_node(str(i)).wcet*bits[i-1] for i in range(1,6)})
            ms=simulate_fixed_priority_runtime(d,m,et)[2]
            assert ms<=bound+1e-9
            assert ms<=b4+1e-9
            if safe:assert ms<=long[2]+1e-9


def test_witness_replays_and_first_trial_preserved():
    d=paper_fig2();et={v:n.wcet for v,n in d.nodes.items()};et['v2']=1
    r=label_dag(d,LabelingConfig(2,n_trials=0),forced_exec_times=et)
    w=json.loads(r.witness_json)
    assert simulate_fixed_priority_runtime(d,2,w['exec_times'],seed=w['tie_seed'],tie_break=w['tie_break'])[2]==w['makespan']==10


def test_resume_rejects_changed_parameters_and_sp_mp_agree(tmp_path):
    root=tmp_path/'gen'
    kwargs=dict(length_min=4,length_max=4,parallel_min=3,parallel_max=3,core_min=2,core_max=2,per_setting=3)
    assert generate_dataset(root,**kwargs)==3
    assert generate_dataset(root,**kwargs)==0
    with pytest.raises(ValueError):generate_dataset(root,**{**kwargs,'master_seed':99})
    a=root/'sp.csv';b=root/'mp.csv'
    label_dataset(root/'index.csv',a,n_trials=4)
    label_dataset_parallel(root/'index.csv',b,n_trials=4,workers=2)
    pd.testing.assert_frame_equal(pd.read_csv(a),pd.read_csv(b))
    with pytest.raises(ValueError):label_dataset(root/'index.csv',a,n_trials=5)


def test_table3_fixed_slices_and_missing_baseline():
    rows=[]
    for i,(m,l,p) in enumerate([(10,8,10),(10,9,10),(12,9,12)]):
        rows.append(dict(dag_id=str(i),num_cores=m,length=l,parallelism=p,static_pass=True,
                         label='0',long_makespan=8.,chen2023_bound=10.,bound_4=12.,bound_28=np.nan))
    table,dist=paper_summaries(pd.DataFrame(rows))
    a=table[(table.panel=='a')&(table.value==10)].iloc[0]
    assert a.Counts==1 and a.MeanShorter_8_pct==20
    assert np.isnan(a.MeanShorter_28_pct)
    assert dist['a'].loc[10,'0']==1


def test_final_holdout_never_fitted_and_unseen_disjoint(tmp_path,monkeypatch):
    import pipeline.train_model as training
    rng=np.random.default_rng(5)
    X=rng.normal(size=(100,13)); X[:,0]=np.arange(100) # identity marker
    labels=[0]*30+[1]*30+['undefined']*40
    frame=pd.DataFrame(X,columns=FEATURE_COLUMNS);frame['dag_id']=[f'd{i}' for i in range(100)];frame['label']=labels
    csv=tmp_path/'labels.csv';frame.to_csv(csv,index=False)
    seen=[];metric_calls=[]
    original=training._new_model;metrics=training._metrics
    def factory(cfg):
        model=original(cfg);fit=model.fit
        def record(X,y):seen.extend(X[:,0].astype(int).tolist());return fit(X,y)
        model.fit=record
        return model
    # Local callbacks are not serializable.
    monkeypatch.setattr(training,'_new_model',factory)
    monkeypatch.setattr(training.joblib,'dump',lambda *a,**kw:None)
    def measure(model,X,y):metric_calls.append(X[:,0].astype(int).tolist());return metrics(model,X,y)
    monkeypatch.setattr(training,'_metrics',measure)
    main(SemiSupTrainConfig(labels_csv=str(csv),out_dir=str(tmp_path/'model'),n_labeled_per_class=25,
         n_undef=20,n_unseen=20,max_cycles=1,make_plots=False,rf_cfg=RFModelConfig(n_estimators=5,n_jobs=1)))
    manifest=pd.read_csv(tmp_path/'model'/'split_manifest.csv')
    held=set(manifest.loc[manifest.partition=='validation','row'])
    unseen=set(manifest.loc[manifest.partition=='unseen','row'])
    assert not held.intersection(seen) and not unseen.intersection(seen)
    assert len(metric_calls)==1 and set(metric_calls[0])==held
    assert len(unseen)==20


def test_paper_training_never_silently_downsamples(tmp_path):
    frame=pd.DataFrame(np.ones((20,13)),columns=FEATURE_COLUMNS)
    frame['label']=[0]*10+[1]*10;csv=tmp_path/'x.csv';frame.to_csv(csv,index=False)
    with pytest.raises(ValueError,match='requested labeled count'):
        main(SemiSupTrainConfig(labels_csv=str(csv),out_dir=str(tmp_path/'model'),strict_counts=True))


def test_self_training_strict_confidence_and_retraining(monkeypatch):
    import pipeline.train_model as training
    class Fake:
        sklearn_model=None
        classes_=np.array([0,1])
        def __init__(self):self.sklearn_model=self
        def fit(self,X,y):return self
        def predict_proba(self,X):return np.tile([.9,.1],(len(X),1))
    monkeypatch.setattr(training,'_new_model',lambda cfg:Fake())
    X=np.array([[0.],[1.]]);y=np.array([0,1]);pool=np.array([[2.],[3.]])
    cfg=SemiSupTrainConfig(confidence_threshold=.9)
    rounds,history,_=training.fit_self_training(X,y,pool,cfg,cycles=1)
    assert len(rounds)==1 # Confidence must strictly exceed 90%.
    cfg=SemiSupTrainConfig(confidence_threshold=.89)
    rounds,history,_=training.fit_self_training(X,y,pool,cfg,cycles=1)
    assert len(rounds)==2 and history[1]['added']==2


def test_paper_figure3_equations_2_3_4():
    from analysis.static_analysis import eta_set,ib_remove_set,ib_potential_set
    d=DAG('figure3')
    d.add_nodes([Node(v,1,p) for v,p in [('v1',9),('v2',2),('v3',4),('v4',6),('v5',1),('v6',1),('vj',3),('v7',0)]])
    d.add_edges([('v1','v2'),('v1','v3'),('v1','vj'),('v2','v4'),('v3','v5'),('v3','v6'),
                 ('v4','v7'),('v5','v7'),('v6','v7'),('vj','v7')])
    assert eta_set(d,'vj')=={'v2','v3'}
    assert ib_remove_set(d,'vj') & d.concurrent_nodes('vj')=={'v2','v4','v5','v6'}
    assert ib_potential_set(d,'vj')=={'v3'}


def test_paper_figure4_equation5(monkeypatch):
    import analysis.static_analysis as sa
    d=DAG('figure4_set_example')
    d.add_nodes([Node(v,1,i) for i,v in enumerate(['v2','v4','v5','v6','v7','vj'])])
    monkeypatch.setattr(sa,'ib_potential_set',lambda *_:{'v2','v4','v5','v6','v7'})
    assert sa.ib_final_set(d,'vj',{'vj':5},{'v2':2,'v4':4,'v5':5,'v6':9,'v7':7})=={'v6','v7'}


def test_joint_antichain_condition_in_node_test(monkeypatch):
    import analysis.static_analysis as sa
    d=DAG('joint_node')
    d.add_nodes([Node(v,1,p) for v,p in [('s',10),('j',8),('a',1),('b',2),('c',3),('h',9),('t',0)]])
    d.add_edges([('s','j'),('s','a'),('s','b'),('s','c'),('a','h'),('b','h'),('h','t'),('c','t')])
    # Test the conjunction independently of earlier blocker pruning.
    monkeypatch.setattr(sa,'ib_remove_set',lambda *_:set())
    starts={v:0 for v in d.node_ids()};starts['j']=1
    finishes={'s':1,'j':2,'a':2,'b':2,'c':2,'h':3,'t':1}
    report=sa.anomaly_test_node(d,'j',3,starts,finishes)
    assert report.passes and 'no core-filling antichain' in report.reason


def test_gantt_rejects_impossible_core_reconstruction():
    from utils.gantt import build_gantt_tasks
    with pytest.raises(ValueError,match='overlap'):
        build_gantt_tasks({'a':0,'b':0},{'a':2,'b':2},num_cores=1)
