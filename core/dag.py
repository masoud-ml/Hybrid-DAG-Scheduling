from __future__ import annotations

import math
import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union


@dataclass(frozen=True)
class Edge:
    """Directed edge u -> v."""
    u: str
    v: str


@dataclass
class Node:
    """DAG node with WCET and fixed priority; larger values have higher priority."""
    id: str
    wcet: float
    priority: int

    dummy: bool = False

    # Graph relations are maintained by DAG; these are cached adjacency sets.
    predecessors: Set[str] = field(default_factory=set, repr=False)
    successors: Set[str] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.wcet) or self.wcet < 0 or (self.wcet == 0 and not self.dummy):
            raise ValueError(f"Node {self.id}: wcet must be finite and positive (zero only for explicit dummy), got {self.wcet}")


class DAG:
    """Acyclic graph with adjacency and transitive dependency queries."""

    def __init__(self, dag_id: str):
        self.dag_id = dag_id
        self._nodes: Dict[str, Node] = {}

    def add_node(self, node: Node) -> None:
        if node.id in self._nodes:
            raise ValueError(f"Duplicate node id: {node.id}")
        self._nodes[node.id] = node

    def add_nodes(self, nodes: Iterable[Node]) -> None:
        for n in nodes:
            self.add_node(n)

    def add_edge(self, u: str, v: str) -> None:
        if u not in self._nodes or v not in self._nodes:
            raise KeyError(f"Edge references missing node(s): {u}->{v}")
        if u == v:
            raise ValueError("Self-loop is not allowed in a DAG.")

        self._nodes[u].successors.add(v)
        self._nodes[v].predecessors.add(u)

        if self._has_cycle():
            self._nodes[u].successors.remove(v)
            self._nodes[v].predecessors.remove(u)
            raise ValueError(f"Adding edge {u}->{v} creates a cycle.")

    def add_edges(self, edges: Iterable[Union[Edge, Tuple[str, str]]]) -> None:
        for e in edges:
            if isinstance(e, Edge):
                self.add_edge(e.u, e.v)
            else:
                u, v = e
                self.add_edge(u, v)

    @property
    def nodes(self) -> Dict[str, Node]:
        return self._nodes

    def node_ids(self) -> List[str]:
        return list(self._nodes.keys())

    def get_node(self, node_id: str) -> Node:
        return self._nodes[node_id]

    def wcet_total(self) -> float:
        return sum(n.wcet for n in self._nodes.values())

    def pre(self, v: str) -> Set[str]:
        """pre(v): direct predecessors of v."""
        return set(self.get_node(v).predecessors)

    def suc(self, v: str) -> Set[str]:
        """suc(v): direct successors of v."""
        return set(self.get_node(v).successors)

    def ancestors(self, v: str) -> Set[str]:
        """anc(v): all (transitive) predecessors."""
        visited: Set[str] = set()
        q = deque(self.pre(v))
        while q:
            x = q.popleft()
            if x in visited:
                continue
            visited.add(x)
            q.extend(self.pre(x))
        return visited

    def descendants(self, v: str) -> Set[str]:
        """des(v): all (transitive) successors."""
        visited: Set[str] = set()
        q = deque(self.suc(v))
        while q:
            x = q.popleft()
            if x in visited:
                continue
            visited.add(x)
            q.extend(self.suc(x))
        return visited

    def concurrent_nodes(self, v: str) -> Set[str]:
        """Return nodes unrelated to v by ancestry or descent, excluding v itself."""
        anc = self.ancestors(v)
        des = self.descendants(v)
        all_nodes = set(self.node_ids())
        return (all_nodes - anc - des) - {v}

    def f_pre(self, finish_times: Dict[str, float], v: str) -> float:
        """
        f_pre(v): time when v becomes eligible to execute
        = max finish time among predecessors.
        """
        preds = self.pre(v)
        if not preds:
            return 0.0
        return max(finish_times[p] for p in preds)

    def sources(self) -> List[Node]:
        return [n for n in self._nodes.values() if not n.predecessors]

    def sinks(self) -> List[Node]:
        return [n for n in self._nodes.values() if not n.successors]

    def source(self) -> Node:
        src = self.sources()
        if len(src) != 1:
            raise ValueError(f"Expected exactly 1 source, found {len(src)}")
        return src[0]

    def sink(self) -> Node:
        sk = self.sinks()
        if len(sk) != 1:
            raise ValueError(f"Expected exactly 1 sink, found {len(sk)}")
        return sk[0]

    def topological_order(self) -> List[str]:
        in_deg = {nid: len(n.predecessors) for nid, n in self._nodes.items()}
        ready = [nid for nid, d in in_deg.items() if d == 0]
        heapq.heapify(ready)
        order: List[str] = []

        while ready:
            nid = heapq.heappop(ready)
            order.append(nid)
            for succ in self._nodes[nid].successors:
                in_deg[succ] -= 1
                if in_deg[succ] == 0:
                    heapq.heappush(ready, succ)

        if len(order) != len(self._nodes):
            raise ValueError("Graph is not a DAG (cycle detected).")
        return order

    def _has_cycle(self) -> bool:
        try:
            _ = self.topological_order()
            return False
        except ValueError:
            return True
