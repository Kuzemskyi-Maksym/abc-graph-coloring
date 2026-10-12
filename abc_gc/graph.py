"""Graph I/O and coloring checks for DIMACS .col files."""
from __future__ import annotations


class Graph:
    def __init__(self, n: int, edges: list[tuple[int, int]], name: str = ""):
        self.n = n
        self.name = name
        adj = [set() for _ in range(n)]
        for u, v in edges:
            if u != v:
                adj[u].add(v)
                adj[v].add(u)
        self.adjset = adj
        self.adj = [sorted(a) for a in adj]
        self.deg = [len(a) for a in self.adj]
        self.m = sum(self.deg) // 2

    @classmethod
    def from_dimacs(cls, path: str) -> "Graph":
        n = 0
        edges = []
        with open(path, encoding="latin-1") as f:
            for line in f:
                if line.startswith("p"):
                    n = int(line.split()[2])
                elif line.startswith("e"):
                    _, a, b = line.split()[:3]
                    edges.append((int(a) - 1, int(b) - 1))
        name = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        return cls(n, edges, name)


def num_colors(col: list[int]) -> int:
    return len(set(col))


def is_proper(g: Graph, col: list[int]) -> bool:
    if len(col) != g.n or any(c is None or c < 0 for c in col):
        return False
    return all(col[u] != col[v] for u in range(g.n) for v in g.adj[u] if u < v)


def normalize(col: list[int]) -> list[int]:
    """Relabel colors to 0..k-1 in order of first appearance."""
    mp = {}
    return [mp.setdefault(c, len(mp)) for c in col]
