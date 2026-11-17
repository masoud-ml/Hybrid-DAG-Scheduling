"""Exact DAG graph metrics: weighted critical path and maximum antichain.

Maximum simultaneous structural parallelism is the maximum antichain width.
"""
from __future__ import annotations
from collections import deque
from core.dag import DAG


def critical_path(dag: DAG) -> float:
    longest: dict[str, float] = {}
    for v in dag.topological_order():
        longest[v] = dag.get_node(v).wcet + max((longest[p] for p in dag.pre(v)), default=0.0)
    return max(longest.values(), default=0.0)


def earliest_starts(dag: DAG, min_fraction: float = 0.0) -> dict[str, float]:
    """Optimistic lower bound on start time from precedence alone."""
    starts: dict[str, float] = {}
    for v in dag.topological_order():
        starts[v] = max((starts[p] + min_fraction * dag.get_node(p).wcet for p in dag.pre(v)), default=0.0)
    return starts


def antichain_width(dag: DAG, subset=None) -> int:
    """Size of maximum set of pairwise incomparable nodes (exact)."""
    vertices = sorted(set(dag.node_ids()) if subset is None else set(subset))
    if not vertices:
        return 0
    adj = {u: sorted(dag.descendants(u).intersection(vertices)) for u in vertices}
    dist: dict[str, int] = {}
    left: dict[str, str | None] = {u: None for u in vertices}
    right: dict[str, str | None] = {u: None for u in vertices}

    def bfs():
        q = deque()
        has_path = False
        for u in vertices:
            if left[u] is None:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = -1
        while q:
            u = q.popleft()
            for v in adj[u]:
                m = right[v]
                if m is None:
                    has_path = True
                elif dist[m] == -1:
                    dist[m] = dist[u] + 1
                    q.append(m)
        return has_path

    def dfs(u):
        for v in adj[u]:
            m = right[v]
            if m is None or (dist[m] == dist[u] + 1 and dfs(m)):
                left[u] = v
                right[v] = u
                return True
        dist[u] = -1
        return False

    matching = 0
    while bfs():
        for u in vertices:
            if left[u] is None and dfs(u):
                matching += 1
    return len(vertices) - matching
