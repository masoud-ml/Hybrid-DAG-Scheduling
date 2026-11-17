"""Run dataset generation, labeling, training and evaluation."""
from __future__ import annotations
import argparse
import json
import platform
from datetime import datetime,timezone
from pathlib import Path
from pipeline.dataset_generation import generate_dataset
from pipeline.dataset_labeling_sp import label_dataset
from pipeline.dataset_labeling_mp import label_dataset_parallel
from pipeline.train_model import main as train, SemiSupTrainConfig
from evaluation.evaluate import evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['quick','paper'],default='quick')
    p.add_argument('--out',default='out/reproduction')
    p.add_argument('--phase',choices=['all','generate','label','train','evaluate'],default='all')
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--max-scenarios',type=int,default=None)
    p.add_argument('--trials',type=int,default=None)
    p.add_argument('--per-setting',type=int,default=None)
    p.add_argument('--other-bounds-csv',default=None)
    a=p.parse_args()
    folder=Path(a.out)
    folder.mkdir(parents=True,exist_ok=True)
    quick=a.mode=='quick'
    per_setting=a.per_setting if a.per_setting is not None else (3 if quick else 1000)
    trials=a.trials if a.trials is not None else (64 if quick else 100000)
    settings=dict(length_min=4,length_max=5 if quick else 15,
         parallel_min=3 if quick else 4,parallel_max=4 if quick else 15,
         core_min=2,core_max=3 if quick else 16,per_setting=per_setting,
         master_seed=a.seed, resume=True, max_scenarios=a.max_scenarios)
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),mode=a.mode,
          phase=a.phase,python=platform.python_version(),num_cores='2..3' if quick else '2..16',
          length='4..5' if quick else '4..15',parallelism='3..4' if quick else '4..15',
          per_setting=per_setting,randomized_trials=trials,seed=a.seed,
          wcet_scaling='Uniform(0.5,1.0) continuous',
          priority='front-layer-first, high-WCET-first, unique',
          static_certificate='unique fixed-priority, Lemmas 3-5, conservative',
          ml='RandomForest n_estimators=100 threshold=0.90 up to 5 cycles',
          reference='DOI:10.1109/TC.2025.3603674',
          disclaimer='Exact numerical replication requires author seeds/data; adapted [28] requires validated external bounds')
    (folder/f'protocol_{a.phase}.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    if a.phase in ('all','generate'):
        n=generate_dataset(folder,**settings)
        print(f'Generation finished: +{n} scenarios')
    if a.phase in ('all','label'):
        args=dict(index_csv=str(folder/'index.csv'),labels_csv=str(folder/'labels.csv'),
                  n_trials=trials,seed=a.seed,resume=True,max_scenarios=a.max_scenarios)
        n=(label_dataset(**args) if a.workers==1 else
           label_dataset_parallel(**args, workers=a.workers))
        print(f'Labeling finished: +{n} labeled or unresolved scenarios')
    if a.phase in ('all','train'):
        if not (folder/'labels.csv').exists():
            raise FileNotFoundError(folder/'labels.csv')
        try:
            train(SemiSupTrainConfig(labels_csv=str(folder/'labels.csv'),
                                    out_dir=str(folder/'learning'), strict_counts=not quick))
        except ValueError as e:
            if a.phase=='train' or not quick:raise
            warning=f'Skipped training because sample does not satisfy ML requirements: {e}'
            (folder/'TRAINING_NOT_READY.txt').write_text(warning,encoding='utf-8')
            print(warning)
    if a.phase in ('all','evaluate'):
        if not (folder/'labels.csv').exists():
            raise FileNotFoundError(folder/'labels.csv')
        result=evaluate(str(folder/'labels.csv'),str(folder/'evaluation'),
                        other_bounds_csv=a.other_bounds_csv)
        print(f'Evaluated {len(result)} scenarios')

if __name__=='__main__':main()
