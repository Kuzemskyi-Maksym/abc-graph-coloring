"""Exact graph colouring: DSATUR-based backtracking (branch and bound), Brélaz 1979 style.

Heuristics inside the search (the ones used throughout the project):
  * vertex choice  - maximum saturation (number of distinct colours among neighbours),
                     ties broken by degree in the uncoloured subgraph (degree heuristic);
  * colour priority - order in which feasible colours are tried at a node:
                     'first'     -> lowest colour index first,
                     'most_used' -> colour with the largest class first;
                     a new colour is tried last and only if it can still beat the upper bound.
Bounds:
  * upper bound UB from heuristic DSATUR, improved whenever a better complete colouring is found;
  * lower bound LB = size of a clique found greedily (any colouring needs >= LB colours).
If the search finishes, UB is the chromatic number (proven). With a time limit it may stop
early; then UB is the best colouring found and only LB <= chi <= UB is known.
"""
from __future__ import annotations
import sys
import time
from dataclasses import dataclass

from .graph import Graph, normalize
from .heuristics import dsatur


@dataclass
class ExactResult:
    k: int                 # best number of colours found (= chi if proven)
    col: list[int]
    lb: int                # clique lower bound
    proven: bool           # search completed (or UB == LB): k is the chromatic number
    nodes: int
    time_to_best: float
    time: float


def greedy_clique(g: Graph) -> list[int]:
    """Largest clique among greedy clique constructions started from every vertex."""
    best = []
    order_by_deg = sorted(range(g.n), key=lambda v: -g.deg[v])
    for s in order_by_deg:
        if g.deg[s] + 1 <= len(best):
            break
        clique = [s]
        cand = set(g.adjset[s])
        while cand:
            v = max(cand, key=lambda x: (len(g.adjset[x] & cand), g.deg[x]))
            clique.append(v)
            cand &= g.adjset[v]
        if len(clique) > len(best):
            best = clique
    return best


class _Abort(Exception):
    pass


def exact_coloring(g: Graph, time_limit: float = 60.0, color_order: str = "first") -> ExactResult:
    t0 = time.perf_counter()
    n = g.n
    best_col = normalize(dsatur(g))
    UB = max(best_col) + 1
    clique = greedy_clique(g)
    LB = len(clique)
    t_best = time.perf_counter() - t0
    if UB == LB or n == 0:
        return ExactResult(UB, best_col, LB, True, 0, t_best, t_best)

    col = [-1] * n
    # cnt[v][c] = number of neighbours of v with colour c (c < UB_initial)
    K = UB
    cnt = [[0] * K for _ in range(n)]
    sat = [0] * n                       # number of distinct neighbour colours
    unc_deg = list(g.deg)               # degree inside the uncoloured subgraph
    size = [0] * K                      # colour class sizes
    uncolored = set(range(n))
    nodes = 0
    state = {"UB": UB, "col": best_col, "t": t_best}
    sys.setrecursionlimit(max(10000, 4 * n))

    # pre-colour the clique: w.l.o.g. its vertices get distinct colours (symmetry breaking)
    def assign(v, c):
        col[v] = c
        size[c] += 1
        uncolored.discard(v)
        for u in g.adj[v]:
            cu = cnt[u]
            if cu[c] == 0:
                sat[u] += 1
            cu[c] += 1
            unc_deg[u] -= 1

    def unassign(v, c):
        col[v] = -1
        size[c] -= 1
        uncolored.add(v)
        for u in g.adj[v]:
            cu = cnt[u]
            cu[c] -= 1
            if cu[c] == 0:
                sat[u] -= 1
            unc_deg[u] += 1

    for i, v in enumerate(clique):
        assign(v, i)

    def search(k_used: int):
        nonlocal nodes
        nodes += 1
        if (nodes & 1023) == 0 and time.perf_counter() - t0 > time_limit:
            raise _Abort
        if not uncolored:
            state["UB"] = k_used
            state["col"] = list(col)
            state["t"] = time.perf_counter() - t0
            if k_used == LB:
                raise StopIteration
            return
        # vertex choice: max saturation, then max degree in the uncoloured subgraph
        v = max(uncolored, key=lambda x: (sat[x], unc_deg[x]))
        cv = cnt[v]
        feasible = [c for c in range(k_used) if cv[c] == 0]
        if color_order == "most_used":
            feasible.sort(key=lambda c: -size[c])
        for c in feasible:
            assign(v, c)
            search(k_used)
            unassign(v, c)
            if k_used >= state["UB"]:
                return
        if k_used + 1 < state["UB"]:          # open a new colour only if it can still improve
            assign(v, k_used)
            search(k_used + 1)
            unassign(v, k_used)

    proven = True
    try:
        search(LB)
    except StopIteration:
        pass
    except _Abort:
        proven = False
    t = time.perf_counter() - t0
    k = state["UB"]
    if k == LB:
        proven = True
    return ExactResult(k, normalize(state["col"]), LB, proven, nodes, round(state["t"], 3), round(t, 2))
