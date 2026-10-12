"""Bee-colony algorithms for the graph colouring problem (minimise the number of colours).

All algorithms share one interface:  algo(g, time_limit, seed, **params) -> Result
and always return a *proper* colouring (checked by the runner).

Implemented:
  vertex_bee   - "vertex = food source" constructive bee algorithm, after the lecture /
                 Chahkandi & Mirzaei (nectar = vertex degree, onlookers colour neighbours).
                 Switches: select rule, colour rule.
  site_bee     - lecture version 1: food source = a whole colouring; foragers try to empty
                 the last colour class by recolouring / one-swap with a neighbour.
  habc         - Fister Jr., Fister & Brest (2012) style hybrid ABC: real weight vector ->
                 vertex order -> greedy decoder, ABC update, RWDE local search instead of scouts.
                 (Adapted from fixed-k 3-colouring to minimising k; see README.)
  tabucol      - classic TabuCol-like min-conflict search on fixed k (non-bee reference).
  own_abc      - our modification: stage 1 vertex_bee with saturation-based selection and
                 most-used-colour rule; stage 2 ABC on fixed k-1 where employed/onlooker bees
                 make tabu min-conflict moves and scouts restart from the best proper colouring.
"""
from __future__ import annotations
import math
import random
import time
from dataclasses import dataclass, field

from .graph import Graph, normalize, num_colors
from .heuristics import greedy, dsatur


@dataclass
class Result:
    k: int
    col: list[int]
    time_to_best: float
    trace: list[tuple[float, int]] = field(default_factory=list)  # (seconds, k) improvements
    iters: int = 0


class _Best:
    def __init__(self, t0: float):
        self.t0 = t0
        self.k = math.inf
        self.col = None
        self.t = 0.0
        self.trace = []

    def offer(self, col):
        k = num_colors(col)
        if k < self.k:
            self.k, self.col = k, normalize(col)
            self.t = time.perf_counter() - self.t0
            self.trace.append((round(self.t, 4), k))
            return True
        return False


# ---------------------------------------------------------------- colour choice rules

def _choose_color(g, col, size, u, rule, rng):
    """Pick a colour for u among currently used colours that no coloured neighbour has.
    rule: 'first' (lowest index), 'most_used' (largest class), 'random'. New colour if none fits."""
    forb = {col[w] for w in g.adj[u] if col[w] >= 0}
    used = [c for c, s in size.items() if s > 0 and c not in forb]
    if used:
        if rule == "first":
            return min(used)
        if rule == "most_used":
            best = max(size[c] for c in used)
            return min(c for c in used if size[c] == best)
        return rng.choice(used)
    c = 0
    while size.get(c, 0) > 0 or c in forb:
        c += 1
    return c


def _set(col, size, u, c):
    if col[u] >= 0:
        size[col[u]] -= 1
    col[u] = c
    if c >= 0:
        size[c] = size.get(c, 0) + 1


# ---------------------------------------------------------------- vertex = food source

def vertex_bee_construct(g: Graph, rng: random.Random, n_employed: int, n_onlookers: int,
                         select: str = "random", color_rule: str = "first") -> list[int]:
    """One full pass of the constructive bee algorithm.

    Employed bees each pick an unlocked vertex (food source) and annul its colour.
    Nectar of a vertex = its degree. Onlookers are split between the chosen vertices
    proportionally to nectar (at most deg(v) per vertex) and each one colours one
    unlocked neighbour with a permissible colour. Then the vertex itself is coloured
    and locked (nectar 0). Repeat until every vertex is locked.
    select: 'random' | 'degree' (roulette by degree) | 'saturation' (DSATUR-like).
    """
    n = g.n
    col = [-1] * n
    size: dict[int, int] = {}
    unlocked = set(range(n))
    while unlocked:
        nb = min(n_employed, len(unlocked))
        pool = list(unlocked)
        if select == "random":
            chosen = rng.sample(pool, nb)
        elif select == "degree":
            chosen, w = [], [g.deg[v] + 1 for v in pool]
            for _ in range(nb):
                i = rng.choices(range(len(pool)), weights=w)[0]
                chosen.append(pool[i])
                pool.pop(i), w.pop(i)
        elif select == "saturation":
            key = {v: (len({col[u] for u in g.adj[v] if col[u] >= 0}), g.deg[v], rng.random()) for v in pool}
            chosen = sorted(pool, key=key.__getitem__, reverse=True)[:nb]
        else:
            raise ValueError(select)
        for v in chosen:                      # employed bees annul the colour
            _set(col, size, v, -1)
        nectar = [max(g.deg[v], 0) for v in chosen]
        tot = sum(nectar) or 1
        for v, nec in zip(chosen, nectar):
            k_on = min(g.deg[v], round(n_onlookers * nec / tot))
            if g.deg[v] == 0:
                _set(col, size, v, _choose_color(g, col, size, v, "first", rng))
                unlocked.discard(v)
                continue
            # onlookers: uncoloured neighbours first (high degree first), then coloured unlocked ones
            nbrs = [u for u in g.adj[v] if u in unlocked and u not in chosen]
            nbrs.sort(key=lambda u: (col[u] >= 0, -g.deg[u], rng.random()))
            for u in nbrs[:k_on]:
                c = _choose_color(g, col, size, u, color_rule, rng)
                if col[u] < 0 or c != col[u]:
                    _set(col, size, u, c)
            _set(col, size, v, _choose_color(g, col, size, v, color_rule, rng))
            unlocked.discard(v)
    return col


def vertex_bee(g: Graph, time_limit: float, seed: int, n_employed=None, n_onlookers=None,
               select="random", color_rule="first", **_):
    """Restart the constructive pass until the time limit; keep the best colouring."""
    rng = random.Random(seed)
    t0 = time.perf_counter()
    ne = n_employed or max(1, min(10, g.n // 2))
    no = n_onlookers or 3 * ne
    best = _Best(t0)
    it = 0
    while True:
        best.offer(vertex_bee_construct(g, rng, ne, no, select, color_rule))
        it += 1
        if time.perf_counter() - t0 > time_limit:
            break
    return Result(best.k, best.col, best.t, best.trace, it)


# ---------------------------------------------------------------- site = whole colouring

def _site_search(g, col, foragers, rng):
    """Foragers try to empty the last colour class. One forager = one move attempt:
    (a) move a last-class vertex to a lower colour, or
    (b) give it the colour c of a neighbour u that is its only c-neighbour,
        and move u to another lower colour. Vertices taken by max degree (priority queue)."""
    col = normalize(col)
    k = max(col) + 1
    last = k - 1
    queue = [v for v in range(g.n) if col[v] == last]
    queue.sort(key=lambda v: (-g.deg[v], rng.random()))
    for v in queue:
        if foragers <= 0:
            break
        forb = {col[u] for u in g.adj[v]}
        foragers -= 1
        opts = [c for c in range(last) if c not in forb]
        if opts:
            col[v] = rng.choice(opts) if rng.random() < 0.3 else opts[0]
            continue
        nbrs = [u for u in g.adj[v] if col[u] < last]
        rng.shuffle(nbrs)
        for u in nbrs:
            if foragers <= 0:
                break
            foragers -= 1
            c = col[u]
            if sum(1 for w in g.adj[v] if col[w] == c) != 1:
                continue
            fu = {col[w] for w in g.adj[u] if w != v}
            opts = [c2 for c2 in range(last) if c2 != c and c2 not in fu]
            if opts:
                col[u] = rng.choice(opts)
                col[v] = c
                break
    return normalize(col)


def _site_key(col):
    sizes = {}
    for c in col:
        sizes[c] = sizes.get(c, 0) + 1
    return (len(sizes), -sum(s * s for s in sizes.values()))   # fewer colours, then uneven classes


def site_bee(g: Graph, time_limit: float, seed: int, n_sites=10, n_selected=4, n_foragers=None,
             n_scouts=2, **_):
    rng = random.Random(seed)
    t0 = time.perf_counter()
    best = _Best(t0)
    nf = n_foragers or max(10, max(g.deg))

    def scout():
        order = list(range(g.n))
        rng.shuffle(order)
        return greedy(g, order)

    sites = [dsatur(g)] + [scout() for _ in range(n_sites - 1)]
    for s in sites:
        best.offer(s)
    it = 0
    while time.perf_counter() - t0 < time_limit:
        it += 1
        sites.sort(key=_site_key)
        chosen = [0] + rng.sample(range(1, n_sites), n_selected - 1)
        w = [n_selected - r for r in range(n_selected)]           # best site gets most foragers
        for r, i in enumerate(chosen):
            f = max(1, round(nf * w[r] / sum(w)))
            cand = _site_search(g, sites[i], f, rng)
            if _site_key(cand) <= _site_key(sites[i]):
                sites[i] = cand
                best.offer(cand)
        sites.sort(key=_site_key)
        for j in range(n_scouts):                                  # scouts replace the worst sites
            sites[-1 - j] = scout()
            best.offer(sites[-1 - j])
    return Result(best.k, best.col, best.t, best.trace, it)


# ---------------------------------------------------------------- HABC (Fister et al. 2012, adapted)

def habc(g: Graph, time_limit: float, seed: int, sn=50, limit=100, lam=0.1, **_):
    """Food source = weight vector w in [0,1]^n. Decoder: sort vertices by weight (desc),
    greedy first-fit. Objective: (k, -sum |C_i|^2). Employed: classic ABC one-dimension update
    v_j = w_j + phi (w_j - w_kj). Onlookers: roulette on quality. Scouts replaced by RWDE:
    w' = w + lam * U (random unit direction) around the exhausted source (as in the paper)."""
    rng = random.Random(seed)
    t0 = time.perf_counter()
    best = _Best(t0)
    n = g.n

    def decode(w):
        order = sorted(range(n), key=lambda v: -w[v])
        return greedy(g, order)

    def score(col):
        k, s2 = _site_key(col)
        return k + 1.0 / (1.0 - s2)        # lexicographic-ish: fractional tie-break < 1

    def new_source():
        w = [rng.random() for _ in range(n)]
        col = decode(w)
        best.offer(col)
        return w, score(col)

    src = [new_source() for _ in range(sn)]
    trial = [0] * sn
    it = 0

    def try_update(i):
        w, f = src[i]
        k = rng.randrange(sn - 1)
        k += k >= i
        j = rng.randrange(n)
        w2 = list(w)
        w2[j] = min(1.0, max(0.0, w[j] + rng.uniform(-1, 1) * (w[j] - src[k][0][j])))
        col = decode(w2)
        f2 = score(col)
        best.offer(col)
        if f2 <= f:
            src[i] = (w2, f2)
            trial[i] = 0 if f2 < f else trial[i] + 1
        else:
            trial[i] += 1

    while time.perf_counter() - t0 < time_limit:
        it += 1
        for i in range(sn):
            try_update(i)
        q = [1.0 / s[1] for s in src]
        for _ in range(sn):
            if time.perf_counter() - t0 > time_limit:
                break
            try_update(rng.choices(range(sn), weights=q)[0])
        for i in range(sn):
            if trial[i] > limit:                       # RWDE instead of a random scout
                w = src[i][0]
                u = [rng.gauss(0, 1) for _ in range(n)]
                nu = math.sqrt(sum(x * x for x in u)) or 1
                w2 = [min(1.0, max(0.0, a + lam * b / nu * math.sqrt(n))) for a, b in zip(w, u)]
                col = decode(w2)
                best.offer(col)
                src[i], trial[i] = (w2, score(col)), 0
    return Result(best.k, best.col, best.t, best.trace, it)


# ---------------------------------------------------------------- fixed-k conflict search

class _Source:
    """A k-colouring (possibly with conflicts) with incremental bookkeeping (TabuCol style)."""

    def __init__(self, g, col, k, rng):
        self.g, self.k, self.rng = g, k, rng
        self.col = list(col)
        self.gamma = [[0] * k for _ in range(g.n)]
        for v in range(g.n):
            for u in g.adj[v]:
                self.gamma[v][self.col[u]] += 1
        self.conf = {v for v in range(g.n) if self.gamma[v][self.col[v]] > 0}
        self.f = sum(self.gamma[v][self.col[v]] for v in range(g.n)) // 2
        self.tabu: dict[tuple[int, int], int] = {}
        self.it = 0
        self.best_f = self.f

    def move(self, v, c):
        a = self.col[v]
        self.f += self.gamma[v][c] - self.gamma[v][a]
        self.col[v] = c
        for u in self.g.adj[v]:
            gu = self.gamma[u]
            gu[a] -= 1
            gu[c] += 1
            if gu[self.col[u]] > 0:
                self.conf.add(u)
            else:
                self.conf.discard(u)
        if self.gamma[v][c] > 0:
            self.conf.add(v)
        else:
            self.conf.discard(v)

    def walk(self, steps):
        """Tabu min-conflict moves: random conflicting vertex -> best non-tabu colour."""
        rng = self.rng
        for _ in range(steps):
            if self.f == 0:
                return True
            self.it += 1
            v = rng.choice(tuple(self.conf))
            gv, a = self.gamma[v], self.col[v]
            bestd, cands = None, []
            for c in range(self.k):
                if c == a:
                    continue
                d = gv[c] - gv[a]
                if self.tabu.get((v, c), 0) > self.it and self.f + d >= self.best_f:
                    continue                                 # tabu unless aspiration
                if bestd is None or d < bestd:
                    bestd, cands = d, [c]
                elif d == bestd:
                    cands.append(c)
            if not cands:
                continue
            c = rng.choice(cands)
            self.move(v, c)
            self.tabu[(v, a)] = self.it + rng.randint(0, 9) + int(0.6 * len(self.conf))
            if self.f < self.best_f:
                self.best_f = self.f
        return self.f == 0


def _reduce_init(g, proper, k, rng, which="smallest"):
    """From a proper colouring with k+1 colours, drop one class and reassign its vertices
    to the least-conflicting colour -> start point for the k-colour conflict search."""
    col = normalize(proper)
    sizes = {}
    for c in col:
        sizes[c] = sizes.get(c, 0) + 1
    drop = min(sizes, key=lambda c: (sizes[c], rng.random())) if which == "smallest" else rng.choice(list(sizes))
    remap = {c: i for i, c in enumerate(c for c in sorted(sizes) if c != drop)}
    new = [remap.get(c, -1) for c in col]
    for v in range(g.n):
        if new[v] < 0:
            cnt = [0] * k
            for u in g.adj[v]:
                if new[u] >= 0:
                    cnt[new[u]] += 1
            m = min(cnt)
            new[v] = rng.choice([c for c in range(k) if cnt[c] == m])
    return new


def tabucol(g: Graph, time_limit: float, seed: int, **_):
    """Non-bee reference: start from DSATUR, then repeatedly search for a (k-1)-colouring."""
    rng = random.Random(seed)
    t0 = time.perf_counter()
    best = _Best(t0)
    best.offer(dsatur(g))
    while time.perf_counter() - t0 < time_limit:
        k = best.k - 1
        if k < 1:
            break
        s = _Source(g, _reduce_init(g, best.col, k, rng), k, rng)
        while time.perf_counter() - t0 < time_limit:
            if s.walk(500):
                best.offer(s.col)
                break
    return Result(best.k, best.col, best.t, best.trace, 0)


# ---------------------------------------------------------------- our modification

def own_abc(g: Graph, time_limit: float, seed: int, stage1_share=0.15, stage1_select="saturation",
            stage1_color="most_used", sn=6, steps=200, limit=15, stage2=True, **_):
    """Stage 1: constructive vertex-bee (saturation selection + most-used colour rule).
    Stage 2 (ABC on fixed k = best-1):
      food source  = a k-colouring with conflicts, quality = number of conflicting edges;
      employed bee = `steps` tabu min-conflict moves on its source;
      onlookers    = extra `steps` on sources chosen by roulette on 1/(1+conflicts);
      scout        = a source without improvement for `limit` cycles is rebuilt from the
                     best proper colouring by dropping a random colour class.
    When any source reaches 0 conflicts, k decreases and all sources are rebuilt."""
    rng = random.Random(seed)
    t0 = time.perf_counter()
    best = _Best(t0)
    ne = max(1, min(10, g.n // 2))
    t1 = time_limit * (stage1_share if stage2 else 1.0)
    while True:
        best.offer(vertex_bee_construct(g, rng, ne, 3 * ne, stage1_select, stage1_color))
        if time.perf_counter() - t0 > t1:
            break
    if not stage2:
        return Result(best.k, best.col, best.t, best.trace, 0)
    cycles = 0
    while time.perf_counter() - t0 < time_limit and best.k > 1:
        k = best.k - 1
        srcs = [_Source(g, _reduce_init(g, best.col, k, rng, "smallest" if i == 0 else "random"), k, rng)
                for i in range(sn)]
        trial = [0] * sn
        done = False
        while not done and time.perf_counter() - t0 < time_limit:
            cycles += 1
            for i, s in enumerate(srcs):                          # employed bees
                before = s.best_f
                if s.walk(steps):
                    done = True
                    best.offer(s.col)
                    break
                trial[i] = 0 if s.best_f < before else trial[i] + 1
            if done:
                break
            q = [1.0 / (1 + s.f) for s in srcs]                   # onlookers
            for _ in range(sn):
                i = rng.choices(range(sn), weights=q)[0]
                if srcs[i].walk(steps):
                    done = True
                    best.offer(srcs[i].col)
                    break
            for i in range(sn):                                    # scouts
                if trial[i] > limit:
                    srcs[i] = _Source(g, _reduce_init(g, best.col, k, rng, "random"), k, rng)
                    trial[i] = 0
    return Result(best.k, best.col, best.t, best.trace, cycles)


ALGORITHMS = {
    "DSATUR": None,  # handled in runner (deterministic)
    "VertexBee(CM)": lambda g, T, s: vertex_bee(g, T, s, select="random", color_rule="first"),
    "SiteBee(lecture)": lambda g, T, s: site_bee(g, T, s),
    "HABC(Fister)": lambda g, T, s: habc(g, T, s),
    "TabuCol": lambda g, T, s: tabucol(g, T, s),
    "Own-ABC": lambda g, T, s: own_abc(g, T, s),
    # ablations of our modification
    "Own: stage1 only": lambda g, T, s: own_abc(g, T, s, stage2=False),
    "VertexBee+sat": lambda g, T, s: vertex_bee(g, T, s, select="saturation", color_rule="first"),
    "VertexBee+mostused": lambda g, T, s: vertex_bee(g, T, s, select="random", color_rule="most_used"),
    "Own: 1 source": lambda g, T, s: own_abc(g, T, s, sn=1, limit=10**9),
}
