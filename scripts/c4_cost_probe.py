"""Cost probe for suite C4 (ogbn-proteins) on a generated graph; no dataset is downloaded.

Builds a synthetic multi-label graph shaped like ogbn-proteins (132,534 nodes, a
species-like 8-group partition with the OGB split sizes, 8 sparse edge channels in [0, 1],
planted community labels) with ``--edges`` undirected edges, runs every C4 method of
scripts/run_c4.py on ``--tasks`` task views exactly as the CI job does (same Runner), and
writes one JSON summary with per-method wall clock, EDSP iteration counts, the peak of
numpy/Python allocations per method block (tracemalloc) and the process peak RSS.
The numbers are a COST PROBE ONLY: synthetic graphs say nothing about accuracy, and
iteration counts of the propagation methods on the real graph may differ (they depend on
its spectrum), which is why the summary also reports per-iteration costs.

    python scripts/c4_cost_probe.py --edges 5000000 --tasks 1 --out results/runs/c4_cost_probe.json
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_c4 import TRACK_A_BASELINES, WEIGHTED, Runner, git_sha

from edsp.data.graph import GraphData, Split
from edsp.sparse_backend import DirectedPlan, edsp_csr
from edsp.utils.memo import shared_structure

N, SPLIT_SIZES = 132_534, (86_619, 21_236, 24_679)  # ogbn-proteins nodes and species split sizes


def synthetic_proteins(m: int, tasks: int, seed: int = 0) -> GraphData:
    rng = np.random.default_rng(seed)
    species = np.repeat(np.arange(8), -(-N // 8))[:N]  # 8 groups; split by group as OGB does
    rng.shuffle(species)
    comm = rng.integers(0, 50, N)  # planted functional modules
    # 80% of edges inside a module (within its species), 15% inside the species, 5% anywhere
    kind = rng.choice(3, m, p=[0.8, 0.15, 0.05])
    u = rng.integers(0, N, m)
    v = np.empty(m, dtype=np.int64)
    order = np.lexsort((comm, species))
    key_sp = species[order]
    key_mod = species[order] * 1000 + comm[order]
    for mask, keys, own in ((kind == 0, key_mod, species[u] * 1000 + comm[u]), (kind == 1, key_sp, species[u])):
        lo = np.searchsorted(keys, own[mask], "left")
        hi = np.searchsorted(keys, own[mask], "right")
        v[mask] = order[lo + (rng.random(mask.sum()) * (hi - lo)).astype(np.int64)]
    v[kind == 2] = rng.integers(0, N, int((kind == 2).sum()))
    keep = u != v
    e = np.unique(np.stack([np.minimum(u, v), np.maximum(u, v)], 1)[keep], axis=0)
    attr = np.where(rng.random((len(e), 8)) < 0.3, rng.beta(2, 5, (len(e), 8)), 0.0).astype(np.float32)
    pos = rng.random((tasks, 50)) < rng.uniform(0.05, 0.4, (tasks, 1))
    Y = (pos[:, comm].T ^ (rng.random((N, tasks)) < 0.05)).astype(np.int8)
    groups = np.argsort(np.bincount(species))  # assign whole groups to train / val / test
    tr = np.flatnonzero(np.isin(species, groups[:6]))
    va = np.flatnonzero(species == groups[6])
    te = np.flatnonzero(species == groups[7])
    return GraphData("synthetic-proteins", N, e.astype(np.int32), 2, [Split("species", tr, va, te, "synthetic species")],
                     Y, edge_attr=attr, multilabel=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", type=int, default=5_000_000)
    ap.add_argument("--tasks", type=int, default=1)
    ap.add_argument("--iter-probe", type=int, default=20, help="EDSP iterations timed with epsilon = 0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    t0 = time.perf_counter()
    g = synthetic_proteins(args.edges, args.tasks)
    gen = time.perf_counter() - t0
    rss_graph = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    recs: list[dict] = []
    with shared_structure():
        def emit(rec: dict, seconds: float) -> None:
            recs.append(rec | {"seconds": seconds, "rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024})

        tracemalloc.start()
        graph_mib = tracemalloc.get_traced_memory()[0] / 2**20  # 0: the graph was allocated before tracing
        t1 = time.perf_counter()
        r = Runner(g, emit)
        setup = time.perf_counter() - t1
        resident = {"runner (directed plan + noisy-OR weights)": tracemalloc.get_traced_memory()[0] / 2**20}
        blocks = [{"EDSP-const-default", "EDSP-const-tuned"}, {"EDSP-native-default", "EDSP-native-tuned"},
                  *({b} for b in (*TRACK_A_BASELINES, *(f"{x}+w" for x in WEIGHTED)))]
        block_peak: dict[str, float] = {}
        task_seconds = []
        for t in range(args.tasks):
            t2 = time.perf_counter()
            for blk in blocks:
                tracemalloc.reset_peak()
                r.run_task(t, "species", blk)
                key = " + ".join(sorted(blk))
                block_peak[key] = max(block_peak.get(key, 0.0), tracemalloc.get_traced_memory()[1] / 2**20)
            task_seconds.append(time.perf_counter() - t2)
        resident["after one task (+ shared adjacency, weighted adjacency, components, embedding)"] = (
            tracemalloc.get_traced_memory()[0] / 2**20)
        tracemalloc.stop()
        # per-iteration cost of the CSR backend (independent of convergence)
        tr_idx, tr_y = g.task_view(0).vault("species").train_labels()
        prior = np.full((N, 2), 0.5)
        prior[tr_idx] = np.eye(2)[tr_y]
        trust = np.zeros(N)
        trust[tr_idx] = np.inf
        plan = DirectedPlan(N, g.edges)
        p = r.weights
        h = np.zeros(len(g.edges), np.int8)
        a = time.perf_counter()
        edsp_csr(g.edges, p, h, prior, prior_trust=trust, max_iter=1, epsilon=0.0, plan=plan)
        b = time.perf_counter()
        edsp_csr(g.edges, p, h, prior, prior_trust=trust, max_iter=1 + args.iter_probe, epsilon=0.0, plan=plan)
        c = time.perf_counter()
    by = defaultdict(list)
    for x in recs:
        by[x["method"]].append(x)
    methods = {}
    for mth, xs in by.items():
        d = {"seconds_mean": float(np.mean([x["seconds"] for x in xs]))}
        it = [x["diagnostics"].get("iterations") for x in xs if "diagnostics" in x]
        if mth.startswith("EDSP") and "selection" in xs[0]:
            d["seconds_per_config"] = xs[0]["selection"]["seconds_per_config"]
        if any(i is not None for i in it):
            d["iterations"] = it
        if xs[0].get("diagnostics", {}).get("converged") is not None:
            d["converged"] = [x["diagnostics"]["converged"] for x in xs]
        methods[mth] = d
    summary = {
        "kind": "SYNTHETIC COST PROBE (not a result; scripts/c4_cost_probe.py)",
        "git_sha": git_sha(), "runner_type": "agent-session", "nodes": N, "undirected_edges": len(g.edges),
        "tasks_run": args.tasks, "generate_seconds": round(gen, 2), "runner_setup_seconds": round(setup, 2),
        "task_seconds": [round(x, 1) for x in task_seconds],
        "edsp_csr_setup_plus_1_iteration_seconds": round(b - a, 3),
        "edsp_csr_seconds_per_iteration": round((c - b) / args.iter_probe, 4),
        "peak_rss_mib_after_graph": round(rss_graph, 1),
        "graph_arrays_mib": round(float(g.edges.nbytes + g.edge_attr.nbytes + g.num_nodes * g.num_tasks) / 2**20, 1),
        "traced_mib_at_start": round(graph_mib, 1),
        "traced_resident_mib": {k: round(v, 1) for k, v in resident.items()},
        "traced_peak_mib_per_block": {k: round(v, 1) for k, v in block_peak.items()},
        "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "methods": methods,
    }
    text = json.dumps(summary, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
