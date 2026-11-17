"""Parallel labeling workers; main process owns all CSV writes."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from data.labeling.labeling import label_dag, LabelingConfig
from data.feature.features import compute_features
from utils.provenance import label_guard
from utils.dag_loader import load_dag_json


def _work(args):
    row, root, trials, seed = args
    path = Path(row['json_path'])
    if not path.exists():
        path = Path(root) / 'dags' / (row['dag_id']+'.json')
    dag = load_dag_json(str(path))
    cores = int(row['num_cores'])
    label = label_dag(dag, LabelingConfig(num_cores=cores,n_trials=trials,master_seed=seed))
    features = compute_features(dag,cores)
    return {**row, **asdict(features), **asdict(label)}


def label_dataset_parallel(index_csv='out/index.csv', labels_csv='out/labels.csv', *,
                           n_trials=100000, workers=2, seed=123, max_scenarios=None,
                           resume=True):
    idx = Path(index_csv)
    dest = Path(labels_csv)
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and not resume:
        raise FileExistsError(f'{dest}: use a new output or resume')
    label_guard(idx, dest, n_trials, seed)
    known = set()
    expected = None
    if dest.exists():
        with dest.open(newline='',encoding='utf-8') as f:
            rd=csv.DictReader(f)
            expected=rd.fieldnames
            known={row['dag_id'] for row in rd}
    def jobs():
        with idx.open(newline='',encoding='utf-8') as f:
            rows=(r for r in csv.DictReader(f) if r['dag_id'] not in known)
            if max_scenarios is not None:
                rows=islice(rows,max_scenarios)
            for r in rows:
                yield (r,str(idx.parent),n_trials,seed)
    count=0
    with dest.open('a',newline='',encoding='utf-8') as dst, ProcessPoolExecutor(max_workers=workers) as pool:
        writer=None
        iterator=iter(jobs())
        while True:
            batch=list(islice(iterator, max(16, workers*4)))
            if not batch:
                break
            for row in pool.map(_work,batch,chunksize=1):
                if writer is None:
                    if expected is not None and list(row)!=expected:
                        raise ValueError('Existing labels.csv is from an incompatible schema')
                    writer=csv.DictWriter(dst,fieldnames=list(row))
                    if dst.tell()==0: writer.writeheader()
                writer.writerow(row)
                count+=1
                if count % 10==0: dst.flush()
    return count


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',default='out/index.csv')
    p.add_argument('--out',default='out/labels.csv')
    p.add_argument('--trials',type=int,default=100000)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--max-scenarios',type=int)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    count=label_dataset_parallel(a.index,a.out,n_trials=a.trials,workers=a.workers,
                                  seed=a.seed,max_scenarios=a.max_scenarios,resume=a.resume)
    print(f'Processed {count} scenarios')

if __name__=='__main__': main()
