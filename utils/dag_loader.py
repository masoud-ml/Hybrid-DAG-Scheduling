from __future__ import annotations

import json

from typing import Any, Dict, Optional, Tuple

from core.dag import DAG, Node, Edge


def load_dag_json(
    json_path: str,
    *,
    return_meta: bool = False,
) -> DAG | Tuple[DAG, Dict[str, Any]]:
    """Load a DAG from JSON, optionally returning its metadata."""
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    dag_id = payload.get("dag_id")
    if not dag_id:
        raise ValueError(f"Invalid DAG JSON (missing dag_id): {json_path}")

    dag = DAG(dag_id=dag_id)

    nodes = payload.get("nodes", [])
    if not nodes:
        raise ValueError(f"DAG JSON has no nodes: {json_path}")

    for n in nodes:
        if "id" not in n:
            raise ValueError(f"Node entry missing 'id' in {json_path}")
        dag.add_node(
            Node(
                id=n["id"],
                wcet=float(n.get("wcet", 0.0)),
                priority=int(n.get("priority", 0)),
                dummy=bool(n.get("dummy", False)),
            )
        )

    edges = payload.get("edges", [])
    for e in edges:
        if "u" not in e or "v" not in e:
            raise ValueError(f"Edge entry malformed in {json_path}: {e}")
        dag.add_edge(e["u"], e["v"])

    meta = payload.get("meta", {})

    if return_meta:
        return dag, meta
    return dag
