from __future__ import annotations

import os
import csv
import json

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from core.dag import DAG


def dag_to_dict(dag: DAG, *, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Serialize DAG to a JSON-friendly dict.

    Output schema:
      {
        "dag_id": str,
        "nodes": [
            { "id": str, "wcet": float, "priority": int, "predecessors": [str] }
        ],
        "edges": [ {"u": str, "v": str} ],
        "meta": { ... }   # optional
      }
    """
    nodes_out: List[Dict[str, Any]] = []
    edges: List[Tuple[str, str]] = []

    for nid in sorted(dag.node_ids()):
        n = dag.get_node(nid)
        preds = sorted(getattr(n, "predecessors", []))
        nodes_out.append(
            {
                "id": nid,
                "wcet": float(n.wcet),
                "priority": int(n.priority),
                "dummy": n.dummy,
                "predecessors": preds,
            }
        )
        for p in preds:
            edges.append((p, nid))

    edges = sorted(set(edges))

    out: Dict[str, Any] = {
        "dag_id": dag.dag_id,
        "nodes": nodes_out,
        "edges": [{"u": u, "v": v} for (u, v) in edges],
    }

    if meta is not None:
        out["meta"] = _jsonify(meta)

    return out


def save_dag_json(
    dag: DAG,
    out_path: str,
    *,
    meta: Optional[Dict[str, Any]] = None,
    indent: int = 2,
) -> str:
    """
    Save DAG as JSON to out_path.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload = dag_to_dict(dag, meta=meta)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=indent)
    return out_path


def compute_dag_summary(dag: DAG) -> Dict[str, Any]:
    """
    Compute a compact, CSV-friendly summary of DAG structure and WCET stats.
    """
    node_ids = list(dag.node_ids())
    n_nodes = len(node_ids)

    edges = set()
    indeg: Dict[str, int] = {}
    outdeg: Dict[str, int] = {}

    for nid in node_ids:
        preds = list(getattr(dag.get_node(nid), "predecessors", []))
        indeg[nid] = len(preds)
        for p in preds:
            edges.add((p, nid))

    for u, v in edges:
        outdeg[u] = outdeg.get(u, 0) + 1
    for nid in node_ids:
        outdeg.setdefault(nid, 0)

    n_edges = len(edges)

    # Layer/depth/width
    layers = _compute_layers(dag)
    depth = max(layers.values()) + 1 if layers else 0
    width_by_layer: Dict[int, int] = {}
    for lv in layers.values():
        width_by_layer[lv] = width_by_layer.get(lv, 0) + 1
    max_width = max(width_by_layer.values()) if width_by_layer else 0

    wcets = [float(dag.get_node(nid).wcet) for nid in node_ids] if node_ids else [0.0]

    return {
        "dag_id": dag.dag_id,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "depth": depth,
        "max_width": max_width,
        "avg_indeg": sum(indeg.values()) / n_nodes if n_nodes else 0.0,
        "avg_outdeg": sum(outdeg.values()) / n_nodes if n_nodes else 0.0,
        "max_indeg": max(indeg.values()) if indeg else 0,
        "max_outdeg": max(outdeg.values()) if outdeg else 0,
        "wcet_sum": sum(wcets),
        "wcet_min": min(wcets),
        "wcet_max": max(wcets),
        "wcet_avg": sum(wcets) / n_nodes if n_nodes else 0.0,
    }


def write_index_csv(out_csv: str, rows: Iterable[Dict[str, Any]]) -> str:
    """
    Write index.csv (flat rows).
    """
    rows = list(rows)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    keys = sorted({k for r in rows for k in r.keys()})
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

    return out_csv


def _compute_layers(dag: DAG) -> Dict[str, int]:
    topo = dag.topological_order()
    layer: Dict[str, int] = {}
    for nid in topo:
        preds = getattr(dag.get_node(nid), "predecessors", [])
        layer[nid] = 0 if not preds else 1 + max(layer[p] for p in preds)
    return layer


def _jsonify(x: Any) -> Any:
    if is_dataclass(x):
        return _jsonify(asdict(x))
    if isinstance(x, dict):
        return {str(k): _jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonify(v) for v in x]
    return x
