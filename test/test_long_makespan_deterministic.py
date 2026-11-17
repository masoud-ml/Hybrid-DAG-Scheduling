from core.dag import DAG, Node
from schedule.long_makespan import simulate_long_makespan


def build_known_anomaly_dag() -> DAG:
    dag = DAG("known_anomaly_rnd_8_0.35")
    priorities = {'v1': 1, 'v2': 1, 'v3': 4, 'v4': 4, 'v5': 3, 'v6': 2, 'v7': 3, 'v8': 2}
    wcet = {'v1': 4, 'v2': 5, 'v3': 5, 'v4': 4, 'v5': 5, 'v6': 2, 'v7': 5, 'v8': 2}
    dag.add_nodes([Node(n, wcet=wcet[n], priority=priorities[n]) for n in wcet.keys()])
    dag.add_edges([
        ('v1', 'v3'),
        ('v1', 'v5'),
        ('v1', 'v6'),
        ('v1', 'v8'),
        ('v2', 'v6'),
        ('v3', 'v5'),
        ('v3', 'v7'),
        ('v3', 'v8'),
    ])
    return dag


def test_long_makespan_is_deterministic():
    dag = build_known_anomaly_dag()
    ms_values = []
    for _ in range(50):
        start, finish, ms = simulate_long_makespan(dag, num_cores=2)
        ms_values.append(ms)

    print("\nms values:", ms_values)
    assert len(set(ms_values)) == 1, "Long-makespan is not deterministic."


if __name__ == "__main__":
    test_long_makespan_is_deterministic()
    print("OK")
