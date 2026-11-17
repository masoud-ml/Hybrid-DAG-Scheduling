"""Generate deterministic scenario sweeps with resume support.

The default sweep uses lengths 4–15, parallelism 4–15, and 2–16 cores,
with 1,000 scenarios per setting."""
from __future__ import annotations
import argparse
import csv
import os
import random
from pathlib import Path
from data.generation.dag_generator import DagGenConfig, generate_layered_dag
from data.generation.wcet_assign import assign_wcets_from_budget
from data.generation.priority import assign_priorities_front_layer_highest_wcet
from utils.provenance import guard_resume
from utils.dag_saver import save_dag_json, compute_dag_summary


def bucketize_priorities(dag, n_levels: int) -> None:
    """Quantize priorities for diagnostic runs; tied priorities disable certification."""
    if n_levels < 1:
        raise ValueError("n_levels must be positive")
    ids = sorted(dag.node_ids(), key=lambda v: dag.get_node(v).priority)
    for i, v in enumerate(ids):
        dag.get_node(v).priority = min(n_levels - 1, i*n_levels//len(ids)) + 1


def generate_dataset(out_dir="out", *, length_min=4, length_max=15,
                     parallel_min=4, parallel_max=15, core_min=2, core_max=16,
                     per_setting=1000, master_seed=7, dep_prob=0.5,
                     fixed_layer_width=True, resume=True, max_scenarios=None):
    output = Path(out_dir)
    dags_dir = output / 'dags'
    dags_dir.mkdir(parents=True, exist_ok=True)
    index_path = output / 'index.csv'
    guard_resume(index_path, dict(stage='generate', length_min=length_min, length_max=length_max,
        parallel_min=parallel_min, parallel_max=parallel_max, core_min=core_min, core_max=core_max,
        per_setting=per_setting, master_seed=master_seed, dep_prob=dep_prob,
        fixed_layer_width=fixed_layer_width))
    done = set()
    if index_path.exists() and resume:
        with index_path.open(newline='', encoding='utf-8') as f:
            done = {r['dag_id'] for r in csv.DictReader(f)}
    if index_path.exists() and not resume:
        raise FileExistsError(f"{index_path} exists; use a fresh output folder or --resume")
    columns = ["dag_id", "json_path", "num_cores", "length", "parallelism", "period_ms",
               "utilization", "total_budget_ms", "n_nodes", "n_edges", "depth", "max_width",
               "avg_indeg", "avg_outdeg", "max_indeg", "max_outdeg", "wcet_sum",
               "wcet_min", "wcet_max", "wcet_avg"]
    count = 0
    with index_path.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if f.tell() == 0:
            writer.writeheader()
        sid = 0
        for cores in range(core_min, core_max + 1):
            for L in range(length_min, length_max + 1):
                for P in range(parallel_min, parallel_max + 1):
                    for trial in range(per_setting):
                        sid += 1
                        name = f"sc{sid:07d}"
                        if name in done:
                            continue
                        rng = random.Random(master_seed + 104729*sid)
                        period = rng.randint(1000, 3000)
                        budget = 0.5 * period
                        dag = generate_layered_dag(name, DagGenConfig(L, P, dep_prob,
                                           fixed_layer_width), seed=rng.getrandbits(64))
                        assign_wcets_from_budget(dag, budget, seed=rng.getrandbits(64))
                        assign_priorities_front_layer_highest_wcet(dag)
                        filename = dags_dir / f"{name}.json"
                        save_dag_json(dag, str(filename), meta={"length":L,"parallelism":P,
                            "num_cores":cores,"period_ms":period,"utilization":0.5,
                            "total_budget_ms":budget,"generator_seed":master_seed})
                        row = dict(dag_id=name, json_path=str(filename.resolve()), num_cores=cores,
                                   length=L, parallelism=P, period_ms=period, utilization=0.5,
                                   total_budget_ms=budget, **{k:v for k,v in compute_dag_summary(dag).items() if k != "dag_id"})
                        writer.writerow(row)
                        count += 1
                        if count % 100 == 0:
                            f.flush()
                        if max_scenarios is not None and count >= max_scenarios:
                            f.flush()
                            return count
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='out')
    parser.add_argument('--per-setting', type=int, default=1000)
    parser.add_argument('--length-min', type=int, default=4)
    parser.add_argument('--length-max', type=int, default=15)
    parser.add_argument('--parallel-min', type=int, default=4)
    parser.add_argument('--parallel-max', type=int, default=15)
    parser.add_argument('--core-min', type=int, default=2)
    parser.add_argument('--core-max', type=int, default=16)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--max-scenarios', type=int)
    parser.add_argument('--variable-width', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    count = generate_dataset(args.out, length_min=args.length_min, length_max=args.length_max,
        parallel_min=args.parallel_min, parallel_max=args.parallel_max,
        core_min=args.core_min, core_max=args.core_max, per_setting=args.per_setting,
        master_seed=args.seed, fixed_layer_width=not args.variable_width,
        resume=args.resume, max_scenarios=args.max_scenarios)
    print(f"Generated {count} DAGs in {args.out}")

if __name__ == '__main__':
    main()
