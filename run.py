"""Experiment runner: every algorithm x instance x seed with the same wall-clock budget.
Usage: python run.py --time 2 --seeds 5 --algs all --out results.csv data/*.col"""
import argparse, csv, sys, time, statistics
from multiprocessing import Pool
from abc_gc.graph import Graph, is_proper, num_colors
from abc_gc.heuristics import dsatur, rlf
from abc_gc.registry import ALGORITHMS
from abc_gc.exact import exact_coloring

# best known number of colours (DIMACS literature; chromatic number where proven)
BKS = {"myciel5": 6, "myciel6": 7, "queen6_6": 7, "queen8_8": 9, "queen10_10": 11,
       "le450_15a": 15, "le450_15c": 15, "le450_25c": 25, "DSJC125.1": 5, "DSJC125.5": 17,
       "DSJC125.9": 44, "DSJC250.1": 8, "DSJC250.5": 28, "school1": 14, "miles250": 8,
       "anna": 11, "games120": 9, "DSJR500.1": 12}


def exact_job(args):
    path, T, order = args
    g = Graph.from_dimacs(path)
    r = exact_coloring(g, T, order)
    assert is_proper(g, r.col) and num_colors(r.col) == r.k
    return dict(instance=g.name, n=g.n, m=g.m, bks=BKS.get(g.name, ""), alg=f"Exact B&B ({order})", seed=0,
                k=r.k, time_to_best=r.time_to_best, wall=r.time, iters=r.nodes, lb=r.lb, proven=r.proven)


def job(args):
    path, alg, seed, T = args
    g = Graph.from_dimacs(path)
    t = time.perf_counter()
    r = ALGORITHMS[alg](g, T, seed)
    wall = time.perf_counter() - t
    assert is_proper(g, r.col) and num_colors(r.col) == r.k, (alg, g.name, seed)
    return dict(instance=g.name, n=g.n, m=g.m, bks=BKS.get(g.name, ""), alg=alg, seed=seed,
                k=r.k, time_to_best=round(r.time_to_best, 3), wall=round(wall, 2), iters=r.iters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--time", type=float, default=2.0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--algs", default="all")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--exact", type=float, default=0, help="also run exact B&B with this time limit (s)")
    a = ap.parse_args()
    algs = list(ALGORITHMS) if a.algs == "all" else a.algs.split(",")
    rows = []
    for p in a.files:                                   # deterministic baselines
        g = Graph.from_dimacs(p)
        for name, f in (("DSATUR", dsatur), ("RLF", rlf)):
            t = time.perf_counter(); col = f(g); w = time.perf_counter() - t
            assert is_proper(g, col)
            rows.append(dict(instance=g.name, n=g.n, m=g.m, bks=BKS.get(g.name, ""), alg=name, seed=0,
                             k=num_colors(col), time_to_best=round(w, 3), wall=round(w, 3), iters=1))
    jobs = [(p, al, s, a.time) for p in a.files for al in algs for s in range(a.seeds)]
    with Pool(a.procs) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.append(r)
            print(f"[{i}/{len(jobs)}] {r['instance']:12s} {r['alg']:20s} seed={r['seed']} k={r['k']}", flush=True)
    if a.exact:
        ej = [(p, a.exact, o) for p in a.files for o in ("first", "most_used")]
        with Pool(a.procs) as pool:
            for r in pool.imap_unordered(exact_job, ej):
                rows.append(r)
                print(f"[exact] {r['instance']:12s} {r['alg']} k={r['k']} lb={r['lb']} proven={r['proven']}", flush=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
