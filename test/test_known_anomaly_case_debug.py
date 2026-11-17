from core.dag import DAG, Node
from schedule.long_makespan import simulate_long_makespan
from schedule.runtime_sim import simulate_fixed_priority_runtime


def extract_edges(dag):
    out = []
    for v in dag.node_ids():
        for p in dag.get_node(v).predecessors:
            out.append((p, v))
    return sorted(out)


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


def test_known_anomaly_case_debug():
    dag = build_known_anomaly_dag()
    m = 2

    print("\n=== DAG SNAPSHOT ===")
    print("node_ids:", list(dag.node_ids()))
    pr = {v: dag.get_node(v).priority for v in dag.node_ids()}
    wc = {v: dag.get_node(v).wcet for v in dag.node_ids()}
    print("priorities(actual):", pr)
    print("wcet(actual):", wc)
    print("edges(actual):", extract_edges(dag))

    sL, fL, msL = simulate_long_makespan(dag, num_cores=m)

    scales = {'v1': 0.5, 'v2': 0.5, 'v3': 1.0, 'v4': 1.0, 'v5': 1.0, 'v6': 0.5, 'v7': 1.0, 'v8': 1.0}
    exec_times = {v: dag.get_node(v).wcet * scales[v] for v in scales.keys()}

    sR, fR, msR = simulate_fixed_priority_runtime(dag, num_cores=m, exec_times=exec_times, seed=1)

    print("\n=== KNOWN ANOMALY CASE (DEBUG) ===")
    print("ms_long   :", msL)
    print("ms_runtime:", msR)

    print("\nLong schedule (start order):")
    ordL = sorted(sL.items(), key=lambda x: (x[1], -dag.get_node(x[0]).priority, x[0]))
    for nid, st in ordL:
        print(nid, "t=", st, "f=", fL[nid], "P=", dag.get_node(nid).priority, "C=", dag.get_node(nid).wcet)

    print("\nRuntime schedule (start order):")
    ordR = sorted(sR.items(), key=lambda x: (x[1], -dag.get_node(x[0]).priority, x[0]))
    for nid, st in ordR:
        print(nid, "t=", st, "f=", fR[nid], "P=", dag.get_node(nid).priority, "C'=", exec_times[nid])


if __name__ == "__main__":
    test_known_anomaly_case_debug()
