"""Generate a layered DAG.

Edges between adjacent internal layers are sampled independently. Nodes
without predecessors or successors connect to the source or sink."""
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Optional
from core.dag import DAG, Node

@dataclass(frozen=True)
class DagGenConfig:
    length: int
    parallelism: int
    dep_prob: float=0.5
    fixed_layer_width: bool=True


def generate_layered_dag(dag_id: str,cfg: DagGenConfig,seed: Optional[int]=None)->DAG:
    if cfg.length<3 or cfg.parallelism<1 or not 0<=cfg.dep_prob<=1:
        raise ValueError('length >=3, parallelism >=1 and 0<=dep_prob<=1 required')
    rng=random.Random(seed)
    layers=[['src']]
    for i in range(1,cfg.length-1):
        width=cfg.parallelism if cfg.fixed_layer_width else rng.randint(1,cfg.parallelism)
        layers.append([f'l{i}_n{k}' for k in range(width)])
    layers.append(['sink'])
    dag=DAG(dag_id)
    for layer in layers:
        for v in layer:
            dag.add_node(Node(v,1.0,0))
    for i in range(1,len(layers)-2):
        for u in layers[i]:
            for v in layers[i+1]:
                if rng.random()<cfg.dep_prob:
                    dag.add_edge(u,v)
    for v in dag.node_ids():
        if v not in ('src','sink') and not dag.pre(v):dag.add_edge('src',v)
    for v in dag.node_ids():
        if v not in ('src','sink') and not dag.suc(v):dag.add_edge(v,'sink')
    return dag
