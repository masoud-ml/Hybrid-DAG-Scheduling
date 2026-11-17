"""Offline hybrid analysis: static certification and RF prediction.

Only statically certified scenarios use the WCET-simulation bound.
RF predictions do not certify anomaly freedom."""
from __future__ import annotations
import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import joblib
import numpy as np
from analysis.static_analysis import anomaly_test_scenario
from analysis.wcrt_bounds import chen2023_single_dag_bound, generic_list_bound
from data.feature.features import compute_features, FEATURE_COLUMNS
from schedule.long_makespan import simulate_long_makespan
from utils.dag_loader import load_dag_json

@dataclass(frozen=True)
class HybridResult:
    dag_id: str
    num_cores: int
    static_anomaly_free: bool
    long_makespan: float
    analytic_wcrt: float
    safe_selected_wcrt: float
    bound_method: str
    ml_predicted_label: int | None
    ml_confidence: float | None
    ml_is_safety_certified: bool


def apply_hybrid(dag, num_cores: int, model_bundle=None, *,
                 analytic_method='chen2023') -> HybridResult:
    if len({n.priority for n in dag.nodes.values()})!=len(dag.nodes):
        raise ValueError('Strict unique priorities required for certified hybrid bounds')
    _,_,long_ms=simulate_long_makespan(dag,num_cores)
    passed,_=anomaly_test_scenario(dag,num_cores)
    if analytic_method=='chen2023':
        analytic=chen2023_single_dag_bound(dag,num_cores)
    elif analytic_method=='generic':
        analytic=generic_list_bound(dag,num_cores)
    else:
        raise ValueError('analytic_method must be chen2023 or generic')
    label=None
    conf=None
    if model_bundle is not None:
        features=asdict(compute_features(dag,num_cores))
        columns=model_bundle['feature_cols']
        if any(c not in features for c in columns):
            raise ValueError('Model feature schema incompatible with generator')
        raw=model_bundle['model'].predict_proba(np.array([[features[c] for c in columns]],dtype=float))[0]
        classes=model_bundle['model'].sklearn_model.classes_
        ix=int(raw.argmax())
        label=int(classes[ix]);conf=float(raw[ix])
    if analytic+1e-7 < long_ms:
        raise AssertionError('Analytic bound below observed WCET makespan; check [8] implementation')
    bound=long_ms if passed else analytic
    return HybridResult(dag.dag_id,num_cores,passed,long_ms,analytic,bound,
                        'long_makespan_certified' if passed else analytic_method,
                        label,conf,False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dag',required=True)
    p.add_argument('--cores',type=int,required=True)
    p.add_argument('--model')
    p.add_argument('--analytic',choices=['chen2023','generic'],default='chen2023')
    p.add_argument('--out')
    a=p.parse_args()
    dag=load_dag_json(a.dag)
    bundle=joblib.load(a.model) if a.model else None
    output=asdict(apply_hybrid(dag,a.cores,bundle,analytic_method=a.analytic))
    data=json.dumps(output,indent=2)
    if a.out:
        Path(a.out).write_text(data,encoding='utf-8')
    print(data)

if __name__=='__main__':main()
