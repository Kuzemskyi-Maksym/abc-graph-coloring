"""Our three modifications of the bee algorithm for graph colouring.

  own_a  - "vertex = food source" constructive bees (lecture / Chahkandi & Mirzaei) improved:
           nectar = number of still-uncoloured neighbours (dynamic), vertices chosen one at a
           time by roulette on nectar, most-used-colour rule; every constructed colouring is a
           food source, foragers then try to empty its last colour class (lecture, version 1),
           onlookers add effort to good sources, scouts build new colourings.
  own_b  - site = colouring, fixed k = best-1, bees make tabu min-conflict moves; sources start
           from DIFFERENT constructions (DSATUR, RLF, randomised greedy, own_a construction),
           scouts restart from a fresh construction instead of the current best.
  own_c  - adaptive rule selection: a food source is a proper colouring; each bee picks one of
           four operators with probabilities that are updated by the operator's success
           (probability matching). Operators: empty-last-class (lecture), Kempe chain swap,
           iterated greedy (Culberson), random recolouring (diversification).
All return a proper colouring and minimise the number of colours.
"""
from __future__ import annotations
import random
import time

from .graph import Graph, normalize
from .heuristics import greedy, dsatur, rlf
from .bees import Result, _Best, _choose_color, _set, _site_search, _site_key, _Source


# ------------------------------------------------------------------ A: improved vertex bees

def construct_a(g: Graph, rng: random.Random, n_onlookers: int = 30, color_rule: str = "most_used",
                select: str = "roulette") -> list[int]:
    """One constructive pass. Each step: one employed bee chooses an unlocked vertex with
    probability ~ (1 + number of uncoloured neighbours) (select='roulette'; 'max' = greedy),
    annuls its colour, onlookers (at most deg) colour its unlocked neighbours, then the vertex
    gets a colour and is locked."""
    n = g.n
    col = [-1] * n
    size: dict[int, int] = {}
    unc = list(g.deg)                    # uncoloured neighbours of each vertex
    unlocked = set(range(n))

    def setc(u, c):
        was, now = col[u] >= 0, c >= 0
        _set(col, size, u, c)
        if was != now:
            d = -1 if now else 1
            for w in g.adj[u]:
                unc[w] += d

    while unlocked:
        pool = list(unlocked)
        if select == "max":
            best = max(unc[v] for v in pool)
            v = rng.choice([x for x in pool if unc[x] == best])
        else:
            v = rng.choices(pool, weights=[1 + unc[x] for x in pool])[0]
        setc(v, -1)
        if g.deg[v]:
            nbrs = [u for u in g.adj[v] if u in unlocked]
            nbrs.sort(key=lambda u: (col[u] >= 0, -unc[u], rng.random()))
            for u in nbrs[:min(g.deg[v], n_onlookers)]:
                c = _choose_color(g, col, size, u, color_rule, rng)
                if col[u] < 0 or c != col[u]:
                    setc(u, c)
        setc(v, _choose_color(g, col, size, v, color_rule, rng))
        unlocked.discard(v)
    return col


def own_a(g: Graph, time_limit: float, seed: int, sn=8, n_foragers=None, limit=20,
          color_rule="most_used", foragers=True, **_):
    rng = random.Random(seed)
    t0 = time.perf_counter()
    best = _Best(t0)
    nf = n_foragers or max(10, max(g.deg))

    def scout():
        col = normalize(construct_a(g, rng, color_rule=color_rule))
        best.offer(col)
        return col

    src = [scout() for _ in range(sn)]
    trial = [0] * sn
    it = 0
    while time.perf_counter() - t0 < time_limit:
        it += 1
        if not foragers:                                   # ablation: constructions only
            scout()
            continue

        def work(i):
            cand = _site_search(g, src[i], nf, rng)
            if _site_key(cand) < _site_key(src[i]):
                src[i], trial[i] = cand, 0
                best.offer(cand)
            else:
                if _site_key(cand) == _site_key(src[i]):
                    src[i] = cand                          # sideways move keeps diversity
                trial[i] += 1

        for i in range(sn):                                 # employed bees
            work(i)
        keys = [_site_key(s) for s in src]                  # onlookers: rank-based roulette
        order = sorted(range(sn), key=lambda i: keys[i])
        w = [0.0] * sn
        for r, i in enumerate(order):
            w[i] = sn - r
        for _ in range(sn):
            work(rng.choices(range(sn), weights=w)[0])
        for i in range(sn):                                 # scouts
            if trial[i] > limit:
                src[i], trial[i] = scout(), 0
    return Result(best.k, best.col, best.t, best.trace, it)


# ------------------------------------------------------------------ B: diverse sites + tabu

def _to_k(g, proper, k, rng):
    """Keep the k largest colour classes of a proper colouring, put the other vertices into the
    least-conflicting of the k colours -> start for the fixed-k conflict search."""
    col = normalize(proper)
    sizes = {}
    for c in col:
        sizes[c] = sizes.get(c, 0) + 1
    keep = sorted(sizes, key=lambda c: (-sizes[c], rng.random()))[:k]
    remap = {c: i for i, c in enumerate(keep)}
    new = [remap.get(c, -1) for c in col]
    rest = [v for v in range(g.n) if new[v] < 0]
    rng.shuffle(rest)
    for v in rest:
        cnt = [0] * k
        for u in g.adj[v]:
            if new[u] >= 0:
                cnt[new[u]] += 1
        m = min(cnt)
        new[v] = rng.choice([c for c in range(k) if cnt[c] == m])
    return new


def own_b(g: Graph, time_limit: float, seed: int, sn=6, steps=200, limit=15, **_):
    rng = random.Random(seed)
    t0 = time.perf_counter()
    best = _Best(t0)

    def construct(i):
        kind = i % 4
        if kind == 0:
            return dsatur(g, rng)
        if kind == 1:
            return rlf(g)
        if kind == 2:
            order = list(range(g.n)); rng.shuffle(order)
            return greedy(g, order)
        return construct_a(g, rng)

    starts = [construct(i) for i in range(sn)]
    for s in starts:
        best.offer(s)
    nscout = 0
    while time.perf_counter() - t0 < time_limit and best.k > 1:
        k = best.k - 1
        srcs = [_Source(g, _to_k(g, starts[i] if i else best.col, k, rng), k, rng) for i in range(sn)]
        trial = [0] * sn
        done = False
        while not done and time.perf_counter() - t0 < time_limit:
            for i, s in enumerate(srcs):                      # employed
                before = s.best_f
                if s.walk(steps):
                    done = True; best.offer(s.col); break
                trial[i] = 0 if s.best_f < before else trial[i] + 1
            if done:
                break
            q = [1.0 / (1 + s.f) for s in srcs]                # onlookers
            for _ in range(sn):
                i = rng.choices(range(sn), weights=q)[0]
                if srcs[i].walk(steps):
                    done = True; best.offer(srcs[i].col); break
            if done:
                break
            for i in range(sn):                                 # scouts: fresh construction
                if trial[i] > limit:
                    nscout += 1
                    srcs[i] = _Source(g, _to_k(g, construct(nscout), k, rng), k, rng)
                    trial[i] = 0
        starts = [construct(i) for i in range(sn)]
    return Result(best.k, best.col, best.t, best.trace, nscout)


# ------------------------------------------------------------------ C: adaptive rule selection

def _op_empty_last(g, col, rng, budget):
    return _site_search(g, col, budget, rng)


def _op_kempe(g, col, rng, budget):
    """Kempe chain interchange: v in the last class (colour a), random colour b; swap a<->b in
    the connected component of the {a,b}-subgraph containing v. Stays proper."""
    col = list(col)
    k = max(col) + 1
    if k < 2:
        return col
    a = k - 1
    for _ in range(max(1, budget // 10)):
        last = [v for v in range(g.n) if col[v] == a]
        if not last:
            break
        v = rng.choice(last)
        b = rng.randrange(a)
        comp, stack = {v}, [v]
        while stack:
            x = stack.pop()
            for y in g.adj[x]:
                if y not in comp and col[y] in (a, b):
                    comp.add(y); stack.append(y)
        for x in comp:
            col[x] = b if col[x] == a else a
    return normalize(col)


def _op_iterated_greedy(g, col, rng, budget):
    """Culberson iterated greedy: list vertices class by class in some class order and recolour
    first-fit. Never uses more colours than before."""
    classes = {}
    for v, c in enumerate(col):
        classes.setdefault(c, []).append(v)
    cls = list(classes.values())
    mode = rng.randrange(4)
    if mode == 0:
        cls.sort(key=len, reverse=True)
    elif mode == 1:
        cls.reverse()
    elif mode == 2:
        rng.shuffle(cls)
    else:
        cls.sort(key=len)
    order = [v for c in cls for v in c]
    return normalize(greedy(g, order))


def _op_random_recolor(g, col, rng, budget):
    col = list(col)
    k = max(col) + 1
    for _ in range(max(1, budget // 10)):
        v = rng.randrange(g.n)
        forb = {col[u] for u in g.adj[v]}
        opts = [c for c in range(k) if c not in forb and c != col[v]]
        if opts:
            col[v] = rng.choice(opts)
    return normalize(col)


OPERATORS = [("empty_last", _op_empty_last), ("kempe", _op_kempe),
             ("iterated_greedy", _op_iterated_greedy), ("random_recolor", _op_random_recolor)]


def own_c(g: Graph, time_limit: float, seed: int, sn=8, limit=30, adaptive=True, alpha=0.2,
          pmin=0.05, n_foragers=None, **_):
    rng = random.Random(seed)
    t0 = time.perf_counter()
    best = _Best(t0)
    nf = n_foragers or max(10, max(g.deg))
    m = len(OPERATORS)
    q = [1.0] * m                     # operator quality estimates
    uses = [0] * m

    def probs():
        if not adaptive:
            return [1.0 / m] * m
        s = sum(q)
        return [pmin + (1 - m * pmin) * x / s for x in q]

    def scout():
        col = normalize(construct_a(g, rng))
        best.offer(col)
        return col

    src = [normalize(dsatur(g, rng))] + [scout() for _ in range(sn - 1)]
    best.offer(src[0])
    trial = [0] * sn
    it = 0

    def work(i):
        j = rng.choices(range(m), weights=probs())[0]
        uses[j] += 1
        cand = OPERATORS[j][1](g, src[i], rng, nf)
        old, new = _site_key(src[i]), _site_key(cand)
        if new[0] < old[0]:
            r = 1.0
        elif new < old:
            r = 0.3
        else:
            r = 0.0
        q[j] = (1 - alpha) * q[j] + alpha * r + 1e-3
        if new <= old:
            src[i] = cand
            best.offer(cand)
        trial[i] = 0 if new < old else trial[i] + 1

    while time.perf_counter() - t0 < time_limit:
        it += 1
        for i in range(sn):
            work(i)
        keys = [_site_key(s) for s in src]
        order = sorted(range(sn), key=lambda i: keys[i])
        w = [0.0] * sn
        for r, i in enumerate(order):
            w[i] = sn - r
        for _ in range(sn):
            work(rng.choices(range(sn), weights=w)[0])
        for i in range(sn):
            if trial[i] > limit:
                src[i], trial[i] = scout(), 0
    res = Result(best.k, best.col, best.t, best.trace, it)
    res.op_usage = dict(zip([o[0] for o in OPERATORS], uses))
    return res
