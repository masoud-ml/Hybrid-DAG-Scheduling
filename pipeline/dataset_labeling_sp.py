"""Streaming scenario labeling and feature extraction with resume support."""
from __future__ import annotations
from dataclasses import asdict, replace
from pathlib import Path
import argparse
import csv
from data.feature.features import compute_features
from data.labeling.labeling import LabelingConfig, label_dag
from utils.provenance import label_guard
from utils.dag_loader import load_dag_json


def label_dataset(index_csv='out/index.csv', labels_csv='out/labels.csv', *,
                 n_trials=100_000, seed=123, max_scenarios=None,
                 min_scale=0.5, max_scale=1.0, resume=True):
    idx = Path(index_csv)
    if not idx.exists():
        raise FileNotFoundError(idx)
    outfile = Path(labels_csv)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    if outfile.exists() and not resume:
        raise FileExistsError(f"{outfile} exists: choose new output file or resume")
    label_guard(idx, outfile, n_trials, seed, min_scale, max_scale)
    completed = set()
    fields = None
    if outfile.exists():
        with outfile.open(newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            completed = {r['dag_id'] for r in reader}
    n = 0
    with idx.open(newline='', encoding='utf-8') as source, outfile.open('a', newline='', encoding='utf-8') as dst:
        writer = None
        for item in csv.DictReader(source):
            if item['dag_id'] in completed:
                continue
            path = Path(item['json_path'])
            if not path.exists():
                path = idx.parent / 'dags' / f"{item['dag_id']}.json"
            dag = load_dag_json(str(path))
            cores = int(item['num_cores'])
            result = label_dag(dag, LabelingConfig(num_cores=cores, n_trials=n_trials,
                         master_seed=seed, min_scale=min_scale, max_scale=max_scale))
            features = compute_features(dag, cores)
            row = {**item, **asdict(features), **asdict(result)}
            if writer is None:
                if fields is not None and list(row) != fields:
                    raise ValueError('Existing labels.csv has obsolete fields; use new output file')
                writer = csv.DictWriter(dst, fieldnames=list(row))
                if dst.tell() == 0:
                    writer.writeheader()
            writer.writerow(row)
            n += 1
            if n % 10 == 0:
                dst.flush()
            if max_scenarios is not None and n >= max_scenarios:
                break
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', default='out/index.csv')
    p.add_argument('--out', default='out/labels.csv')
    p.add_argument('--trials', type=int, default=100000)
    p.add_argument('--max-scenarios', type=int)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    n = label_dataset(a.index,a.out,n_trials=a.trials,max_scenarios=a.max_scenarios,
                      seed=a.seed, resume=a.resume)
    print(f'Processed {n} scenarios into {a.out}')

if __name__ == '__main__':
    main()
