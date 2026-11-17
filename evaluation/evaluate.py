"""Evaluate response-time bounds across fixed-parameter slices.

Optional comparison bounds are loaded from an external CSV.
Missing baseline values remain missing."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.wcrt_bounds import chen2023_single_dag_bound, serrano2016_single_dag_bound
from analysis.static_analysis import anomaly_test_scenario
from schedule.long_makespan import simulate_long_makespan
from utils.dag_loader import load_dag_json

SLICES = {
    'a': ('num_cores', {'parallelism': 10, 'length': 8}, list(range(4, 17, 2)), [10,12,14,16]),
    'b': ('parallelism', {'num_cores': 12, 'length': 8}, list(range(9,16)), list(range(9,14))),
    'c': ('length', {'num_cores': 12, 'parallelism': 12}, list(range(9,16)), list(range(9,16))),
}


def paper_summaries(df):
    tables = []
    distributions = {}
    for panel, (axis, fixed, values, table_values) in SLICES.items():
        sub = df
        for key, value in fixed.items(): sub = sub[sub[key] == value]
        sub = sub[sub[axis].isin(values)]
        dist = pd.crosstab(sub[axis], sub.label.astype(str)).reindex(
            index=values, columns=['0','1','undefined'], fill_value=0).fillna(0).astype(int)
        distributions[panel] = dist
        for value in table_values:
            group = sub[sub[axis] == value]
            certified = group[group.static_pass]
            row = dict(panel=panel, parameter=axis, value=value, **fixed,
                       scenarios=len(group), Counts=len(certified))
            for ref, col in [('8','chen2023_bound'), ('4','bound_4'), ('28','bound_28')]:
                gains = (certified[col] - certified.long_makespan) * 100 / certified[col]
                row[f'available_{ref}'] = int(gains.notna().sum())
                complete = len(certified) > 0 and gains.notna().all()
                row[f'MeanShorter_{ref}_pct'] = gains.mean() if complete else np.nan
                row[f'MaxShorter_{ref}_pct'] = gains.max() if complete else np.nan
            tables.append(row)
    return pd.DataFrame(tables), distributions


def evaluate(labels_csv, out_dir, *, max_scenarios=None, other_bounds_csv=None):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(labels_csv, keep_default_na=False)
    if max_scenarios is not None: labels = labels.head(max_scenarios)
    rows = []
    for _, r in labels.iterrows():
        path = Path(str(r['json_path']))
        if not path.exists(): path = Path(labels_csv).parent / 'dags' / (str(r['dag_id'])+'.json')
        dag = load_dag_json(str(path)); cores = int(r['num_cores'])
        baseline = simulate_long_makespan(dag, cores)[2]
        passed, _ = anomaly_test_scenario(dag, cores)
        recorded = str(r['static_pass']).lower() in ('true','1')
        if passed != recorded or not np.isclose(baseline, float(r['makespan_long']), rtol=1e-12):
            raise ValueError(f'Stale labels for {r["dag_id"]}; regenerate with current analysis')
        if passed != (str(r['label']) in ('0','0.0')):
            raise ValueError('Static certificate/label mismatch')
        bound = chen2023_single_dag_bound(dag, cores)
        b4 = serrano2016_single_dag_bound(dag, cores)
        if baseline > min(bound, b4) + 1e-8 * max(1, baseline):
            raise AssertionError(f'Analytic bound smaller than WCET makespan: {r["dag_id"]}')
        rows.append(dict(dag_id=r['dag_id'], num_cores=cores, length=int(r['length']),
            parallelism=int(r['parallelism']), static_pass=passed, label=str(r['label']).removesuffix('.0'),
            long_makespan=baseline, chen2023_bound=bound, bound_4=b4, bound_28=np.nan,
            generic_list_bound=b4, safe_hybrid_bound=baseline if passed else bound,
            improvement_over_chen2023_pct=(bound-baseline)*100/bound if passed else np.nan))
    df = pd.DataFrame(rows)
    if df.empty:
        df.to_csv(out/'bounds_by_scenario.csv', index=False)
        return df
    if other_bounds_csv:
        other = pd.read_csv(other_bounds_csv)
        if not {'dag_id','bound_28'} <= set(other.columns):
            raise ValueError('External CSV requires dag_id,bound_28')
        supplied = [c for c in ('bound_4','bound_28') if c in other]
        df = df.drop(columns=supplied).merge(other[['dag_id']+supplied], on='dag_id', how='left', validate='one_to_one')
        for col in supplied:
            known = df[col].notna()
            if (~np.isfinite(df.loc[known,col])).any() or (df.loc[known,col] < df.loc[known,'long_makespan']-1e-8).any():
                raise ValueError(f'Invalid external baseline {col}')
    df.to_csv(out/'bounds_by_scenario.csv', index=False)
    table, distributions = paper_summaries(df)
    table.to_csv(out/'table3.csv', index=False)
    df.groupby(['num_cores','length','parallelism']).agg(
        scenarios=('dag_id','count'), certified=('static_pass','sum'),
        avg_chen2023_gain_pct=('improvement_over_chen2023_pct','mean'),
        max_chen2023_gain_pct=('improvement_over_chen2023_pct','max')).to_csv(out/'all_settings_summary.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for panel, dist in distributions.items():
        dist.to_csv(out/f'fig11{panel}_counts.csv')
        if dist.to_numpy().sum() == 0: continue
        ax = dist.rename(columns={'0':'Anomaly-free','1':'Anomaly','undefined':'Undefined'}).plot.bar(stacked=True)
        ax.set_xlabel(SLICES[panel][0]); ax.set_ylabel('Number of DAG scenarios')
        ax.set_title(', '.join(f'{k}={v}' for k,v in SLICES[panel][1].items()))
        plt.tight_layout(); plt.savefig(out/f'fig11{panel}.png', dpi=150); plt.close()
    (out/'LIMITATIONS.txt').write_text(
        'Table 3 uses only statically certified scenarios. Missing settings have Counts=0 and no gain.\n'
        '[4] is the isolated-DAG Eq.4 specialization; [28] requires validated external adapted bounds.\n'
        'Panel c fixes Core=Parallelism=12 (caption/body); the original table heading has a typo.\n')
    return df

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', default='out/labels.csv'); p.add_argument('--out', default='out/evaluation')
    p.add_argument('--max-scenarios', type=int); p.add_argument('--other-bounds-csv')
    a = p.parse_args()
    d = evaluate(a.labels, a.out, max_scenarios=a.max_scenarios, other_bounds_csv=a.other_bounds_csv)
    print(f'Evaluated {len(d)} scenarios')
