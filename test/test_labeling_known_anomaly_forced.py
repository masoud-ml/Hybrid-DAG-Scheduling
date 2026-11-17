from core.dag import DAG, Node
from data.labeling.labeling import LabelingConfig, label_dag


def build_known_anomaly_dag() -> DAG:
    dag = DAG("known_anomaly_rnd_8_0.35")
    priorities = {'v1': 1, 'v2': 1, 'v3': 4, 'v4': 4, 'v5': 3, 'v6': 2, 'v7': 3, 'v8': 2}
    wcet = {'v1': 4, 'v2': 5, 'v3': 5, 'v4': 4, 'v5': 5, 'v6': 2, 'v7': 5, 'v8': 2}
    dag.add_nodes([Node(n, wcet=wcet[n], priority=priorities[n]) for n in wcet.keys()])
    dag.add_edges([
        ('v1', 'v3'), ('v1', 'v5'), ('v1', 'v6'), ('v1', 'v8'),
        ('v2', 'v6'), ('v3', 'v5'), ('v3', 'v7'), ('v3', 'v8'),
    ])
    return dag


def test_label_forced_is_one():
    dag = build_known_anomaly_dag()

    cfg = LabelingConfig(
        num_cores=2,
        n_trials=0,  # ignored when forced_exec_times is provided
        min_scale=0.5,
        max_scale=1.0,
        master_seed=123,
        early_stop=True,
        tie_break="seeded_random",
        tie_seed_mode="fixed",
        tie_seed_fixed=1,
    )

    scales = {'v1': 0.5, 'v2': 0.5, 'v3': 1.0, 'v4': 1.0, 'v5': 1.0, 'v6': 0.5, 'v7': 1.0, 'v8': 1.0}
    forced_exec_times = {v: dag.get_node(v).wcet * scales[v] for v in scales}

    res = label_dag(dag, cfg, forced_exec_times=forced_exec_times, forced_tie_seed=1)
    print(res)
    assert res.label == "1"
    assert res.anomaly_found is True


if __name__ == "__main__":
    test_label_forced_is_one()
    print("OK")
