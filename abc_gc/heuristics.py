"""Classic constructive heuristics: greedy first-fit, DSATUR, RLF."""
from __future__ import annotations
import random
from .graph import Graph


def greedy(g: Graph, order: list[int]) -> list[int]:
    """First-fit: each vertex gets the smallest color not used by its colored neighbours."""
    col = [-1] * g.n
    for v in order:
        used = {col[u] for u in g.adj[v]}
        c = 0
        while c in used:
            c += 1
        col[v] = c
    return col


def dsatur(g: Graph, rng: random.Random | None = None) -> list[int]:
    """DSATUR (Brelaz 1979): pick the vertex with max saturation, ties by degree, then random/index."""
    n = g.n
    col = [-1] * n
    sat = [set() for _ in range(n)]
    uncolored = set(range(n))
    while uncolored:
        if rng is None:
            v = max(uncolored, key=lambda x: (len(sat[x]), g.deg[x], -x))
        else:
            best = max((len(sat[x]), g.deg[x]) for x in uncolored)
            cand = [x for x in uncolored if (len(sat[x]), g.deg[x]) == best]
            v = rng.choice(cand)
        c = 0
        while c in sat[v]:
            c += 1
        col[v] = c
        uncolored.discard(v)
        for u in g.adj[v]:
            sat[u].add(c)
    return col


def rlf(g: Graph) -> list[int]:
    """Recursive Largest First (Leighton 1979): build one colour class at a time."""
    col = [-1] * g.n
    remaining = set(range(g.n))
    c = 0
    while remaining:
        # degree inside the uncoloured subgraph
        cand = set(remaining)
        blocked = set()
        v = max(cand, key=lambda x: (len(g.adjset[x] & remaining), -x))
        while True:
            col[v] = c
            remaining.discard(v)
            cand.discard(v)
            nb = g.adjset[v] & cand
            cand -= nb
            blocked |= nb
            if not cand:
                break
            # next: candidate with most neighbours among blocked (uncolourable for c)
            v = max(cand, key=lambda x: (len(g.adjset[x] & blocked), -len(g.adjset[x] & cand), -x))
        c += 1
    return col
