"""Suite C4 (Master Prompt §3.1 C4, §5.11 H5): P-NATIVE vs P-CONST on native probabilistic graphs.

A C4 graph is multi-label (ogbn-proteins: 112 binary GO tasks); every task t is run as
the binary view ``GraphData.task_view(t)`` (k = 2) on the official split(s), all Track A:

  EDSP-const-default   P-CONST(0.5), H-TRAINONLY(b1), T_L = inf (= run_e1's EDSP-default)
  EDSP-native-default  P-NATIVE noisy-OR, otherwise identical
  EDSP-const-tuned     run_e1's 9-config grid: p0 in P_GRID x H in {TRAINONLY(b1), LEARNED, ALL1},
                       chosen by the validation value of the headline metric (ROC-AUC)
  EDSP-native-tuned    9 configs as well (rule R6, equal budget): P-NATIVE modes NATIVE_MODES x the
                       same 3 H regimes, chosen the same way
  baselines            majority, train_prior, harmonic, harmonic_cmn, lgc, ppr_lp, linbp (unweighted,
                       validation-selected exactly as in run_e1) and the weighted versions
                       harmonic+w, harmonic_cmn+w, lgc+w, ppr_lp+w, linbp+w, whose adjacency is
                       weighted by the same noisy-OR p that EDSP-native-default uses

The two defaults are configurations of the tuned grids; each is run once and its record
reused (``reused_from_grid``), since EDSP is deterministic. EDSP runs on the scipy CSR
backend (edsp.sparse_backend, parity with edsp.core tested), with one directed-edge plan
shared by all runs; label-free operators (adjacency, its components and spectral radius,
the H-LEARNED structural embedding) are built once per job (edsp.utils.memo).

Every run appends one JSON record (schema of run_e1 plus ``task``) to --out/<dataset>.jsonl,
or <dataset>.tasks<a>-<b>.jsonl with --tasks (rule R1). Test labels are read only by
edsp.eval.evaluate, validation labels only by edsp.selection.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_e1 import (
    TRACK_A_BASELINES,
    config_hash,
    default_config,
    edsp_diagnostics,
    git_sha,
    headline,
    tuned_grid,
)

from edsp.baselines import BASELINES, select_and_predict
from edsp.baselines.propagation import WEIGHTED, adjacency
from edsp.data.graph import GraphData
from edsp.data.loaders import load
from edsp.eval.evaluate import evaluate
from edsp.regimes import build_inputs
from edsp.regimes import probability as P
from edsp.regimes.inputs import _polarity
from edsp.selection import validation_score
from edsp.sparse_backend import DirectedPlan, edsp_csr
from edsp.utils.memo import shared_structure

# ogbn-proteins' 8 edge-feature columns are not named by OGB (paper, docs, issues checked
# on 2026-11-09; docs/SPEC_AMBIGUITIES.md "ogbn-proteins edge-feature columns"), so the
# "experimental" and "textmining" channels of Master Prompt §2.2 (ii) cannot be selected
# with evidence; the registered substitutes are the max and the mean over the 8 channels.
NATIVE_MODES = ("noisy-or", "max", "mean")
WEIGHT_MODE = "noisy-or"  # edge weights of the "+w" baselines
BACKEND = "scipy-csr (edsp.sparse_backend)"


def native_default_config() -> dict:
    return default_config(False) | {"p": {"name": "native", "mode": "noisy-or"}}


def native_grid() -> list[dict]:
    """The H regimes of run_e1.tuned_grid in the same order, P-CONST(p0) replaced by P-NATIVE(mode)."""
    hs = list(dict.fromkeys(json.dumps(c["h"], sort_keys=True) for c in tuned_grid(False)))
    return [default_config(False) | {"p": {"name": "native", "mode": m}, "h": json.loads(h)}
            for h in hs for m in NATIVE_MODES]


def parse_range(text: str | None, total: int) -> range:
    if not text:
        return range(total)
    a, b = (int(x) for x in text.split(":"))
    return range(max(a, 0), min(b, total))


def _summary(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if not len(x):
        return {}
    q = np.quantile(x, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    return {"mean": float(x.mean()), "quantiles_0_10_25_50_75_90_100": [float(v) for v in q],
            "fraction_zero": float(np.mean(x == 0))}


def structure_stats(g: GraphData, p_native: dict[str, np.ndarray]) -> dict:
    """Label-free description of the C4 graph and its splits (indices only, no labels)."""
    n, e = g.num_nodes, g.edges
    deg = np.bincount(e.ravel(), minlength=n)
    out: dict = {"num_nodes": n, "num_edges": len(e), "num_tasks": g.num_tasks,
                 "components": int(connected_components(adjacency(g), directed=False)[0]),
                 "degree": _summary(deg), "isolated_nodes": int(np.sum(deg == 0)), "splits": {}}
    for s in g.splits:
        part = np.full(n, 3, dtype=np.int8)  # 0 train, 1 val, 2 test, 3 none
        part[s.train_idx], part[s.val_idx], part[s.test_idx] = 0, 1, 2
        a, b = np.sort(np.stack([part[e[:, 0]], part[e[:, 1]]]), axis=0)
        names = ("train", "val", "test", "none")
        counts = {f"{names[i]}-{names[j]}": int(np.sum((a == i) & (b == j))) for i in range(4) for j in range(i, 4)}
        has_train_nb = np.zeros(n, dtype=bool)
        tr = part == 0
        has_train_nb[e[tr[e[:, 1]], 0]] = True
        has_train_nb[e[tr[e[:, 0]], 1]] = True
        out["splits"][s.split_id] = {"sizes": [len(s.train_idx), len(s.val_idx), len(s.test_idx)], "source": s.source,
                                     "edges_by_part": counts,
                                     "test_nodes_with_train_neighbour": float(has_train_nb[s.test_idx].mean()),
                                     "val_nodes_with_train_neighbour": float(has_train_nb[s.val_idx].mean())}
    out["p_native"] = {m: _summary(p) for m, p in p_native.items()}
    den_nor = np.bincount(e.ravel(), weights=np.repeat(-np.log1p(-np.clip(p_native["noisy-or"], 0, 1 - 1e-12)), 2), minlength=n)
    out["p_esc"] = {"noisy-or": _summary(-np.expm1(-den_nor)),
                    "const-0.5": _summary(-np.expm1(-deg * math.log(2.0)))}
    return out


ARGS: argparse.Namespace


class Runner:
    """Runs all C4 methods on the task views of one multi-label graph."""

    def __init__(self, g: GraphData, emit, metric: str = "roc_auc") -> None:
        self.g, self.emit, self.metric = g, emit, metric
        self.plan = DirectedPlan(g.num_nodes, g.edges)
        self.weights = P.p_native(g, WEIGHT_MODE)  # one object: the "+w" adjacency is built once per job

    def _edsp(self, gt: GraphData, vault, cfg: dict, pol_cache: dict):
        key = json.dumps(cfg["h"], sort_keys=True)
        if key not in pol_cache:
            pol_cache[key] = _polarity(gt, vault, cfg["h"], "A")
        t0 = time.perf_counter()
        inp = build_inputs(gt, vault, cfg, polarity=pol_cache[key])
        res = edsp_csr(inp.edges, inp.probabilities, inp.heterophily, inp.priors,
                       prior_trust=inp.prior_trust, edge_trust=inp.edge_trust, plan=self.plan)
        res = dataclasses.replace(res, weights=np.empty(0))  # per-directed-edge weights: 632 MB each, unused
        diag = {k: v for k, v in inp.diagnostics.items() if k != "priors"}
        del inp
        return {"config": cfg, "res": res, "val": validation_score(vault, res.probabilities, self.metric),
                "seconds": time.perf_counter() - t0, "regime_diagnostics": diag}

    def _edsp_record(self, task: int, sid: str, method: str, run: dict, vault, **extra) -> dict:
        res = run["res"]
        rec = {"task": task, "split": sid, "method": method, "track": "A", "config": run["config"],
               "metrics": evaluate(vault, res.probabilities), "diagnostics": edsp_diagnostics(res) | {"backend": BACKEND},
               "regime_diagnostics": run["regime_diagnostics"], **extra}
        if not res.converged:  # pre-registered secondary column (PREREGISTRATION §2)
            rec["metrics_mean_last_two"] = evaluate(vault, 0.5 * (res.last_iterates[0] + res.last_iterates[1]))
        return rec

    def run_task(self, task: int, split_id: str, methods: set[str] | None = None) -> None:
        gt = self.g.task_view(task)
        vault = gt.vault(split_id)
        want = (lambda m: True) if methods is None else (lambda m: m in methods)
        pol_cache: dict = {}
        for family, grid, default in (("const", tuned_grid(False), default_config(False)),
                                      ("native", native_grid(), native_default_config())):
            if not (want(f"EDSP-{family}-tuned") or want(f"EDSP-{family}-default")):
                continue
            runs = [self._edsp(gt, vault, c, pol_cache) for c in grid]
            di = next(i for i, r in enumerate(runs) if r["config"] == default)
            if want(f"EDSP-{family}-default"):
                self.emit(self._edsp_record(task, split_id, f"EDSP-{family}-default", runs[di], vault,
                                            reused_from_grid={"grid": f"EDSP-{family}-tuned", "index": di}),
                          runs[di]["seconds"])
            if want(f"EDSP-{family}-tuned"):
                vals = [r["val"] for r in runs]
                best = int(np.argmax([s if np.isfinite(s) else -np.inf for s in vals]))
                self.emit(self._edsp_record(task, split_id, f"EDSP-{family}-tuned", runs[best], vault,
                                            selection={"metric": self.metric, "val_scores": vals,
                                                       "grid": [r["config"] for r in runs], "chosen_index": best,
                                                       "seconds_per_config": [round(r["seconds"], 3) for r in runs]}),
                          sum(r["seconds"] for r in runs))
            del runs
        for b in TRACK_A_BASELINES + [f"{x}+w" for x in WEIGHTED]:
            if not want(b):
                continue
            name, weighted = b.removesuffix("+w"), b.endswith("+w")
            t0 = time.perf_counter()
            probs, info = select_and_predict(name, gt, vault, self.metric, weights=self.weights if weighted else None)
            self.emit({"task": task, "split": split_id, "method": b, "group": BASELINES[name][0], "track": "A",
                       "config": {"baseline": name, "chosen": info.get("chosen"),
                                  "weights": f"P-NATIVE {WEIGHT_MODE}" if weighted else None},
                       "metrics": evaluate(vault, probs),
                       "diagnostics": {k: v for k, v in info.items()
                                       if k in ("converged", "iterations", "uncovered_fraction", "solver", "val_scores")}},
                      time.perf_counter() - t0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["ogbn-proteins"])
    ap.add_argument("--tasks", default=None, help="START:STOP — run tasks[START:STOP] (one CI job per range)")
    ap.add_argument("--suite", default="C4")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-splits", type=int, default=None, help="limit splits per dataset (ogbn-proteins has one)")
    ap.add_argument("--methods", default=None, help="comma-separated subset of the method names (default: all)")
    ap.add_argument("--time-budget-min", type=float, default=None,
                    help="stop starting new tasks when the next one would likely exceed this budget; "
                         "the tasks left are recorded as NOT-RUN")
    args = ap.parse_args()
    args.out = args.out or str(ROOT / "results" / "runs" / args.suite.lower())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_id = os.environ.get("GITHUB_RUN_ID")
    runner = f"actions-run-{run_id}" if run_id else os.environ.get("EDSP_ENVIRONMENT", "local")
    sha = git_sha()
    methods = set(args.methods.split(",")) if args.methods else None
    start = time.perf_counter()
    failures = 0
    for name in args.datasets:
        t_load = time.perf_counter()
        g = load(name)
        if not g.multilabel:
            raise SystemExit(f"{name}: suite C4 expects a multi-label graph (use run_e1.py otherwise)")
        dhash = g.content_hash()  # of the graph as loaded (same hash as dataset_stats.py)
        if g.edges.dtype != np.int32 and g.num_nodes < np.iinfo(np.int32).max:
            g.edges = g.edges.astype(np.int32)  # memory: 316 MB instead of 632 MB on ogbn-proteins
        load_seconds = time.perf_counter() - t_load
        metric = headline(name)
        tasks = parse_range(args.tasks, g.num_tasks)
        path = out / (f"{name}.jsonl" if not args.tasks else f"{name}.tasks{tasks.start:03d}-{tasks.stop:03d}.jsonl")

        def emit(rec: dict, seconds: float, path: Path = path, name: str = name, dhash: str = dhash,
                 metric: str = metric) -> None:
            rec = {"suite": args.suite, "dataset": name, "headline_metric": metric, "constructed_benchmark": False,
                   **rec, "run_record": {"git_sha": sha, "runner_type": runner,
                                         "dataset_hash": dhash + (f":task{rec['task']}" if rec.get("task") is not None else ""),
                                         "config_hash": config_hash(rec.get("config")),
                                         "wall_clock_seconds": round(seconds, 3),
                                         "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}}
            with open(path, "a") as fh:
                fh.write(json.dumps(rec, default=float) + "\n")

        with shared_structure():
            t0 = time.perf_counter()
            r = Runner(g, emit, metric)
            if tasks.start == 0 and (methods is None or "DATASET-STATS" in methods):
                stats = structure_stats(g, {m: P.p_native(g, m) for m in NATIVE_MODES})
                emit({"task": None, "split": None, "method": "DATASET-STATS", "cell": "C4",
                      "stats": stats | {"load_seconds": round(load_seconds, 1)}}, time.perf_counter() - t0)
            longest = 0.0
            splits = g.splits[: args.max_splits] if args.max_splits else g.splits
            for t in tasks:
                for split in splits:
                    elapsed = (time.perf_counter() - start) / 60
                    if args.time_budget_min and elapsed + 1.25 * longest / 60 > args.time_budget_min:
                        failures += 1
                        emit({"task": t, "split": split.split_id, "method": "NOT-RUN",
                              "reason": f"time budget: {elapsed:.0f} of {args.time_budget_min:.0f} min used, "
                                        f"longest task so far {longest / 60:.0f} min"}, 0.0)
                        print(f"[not run] {name} task {t}: time budget", file=sys.stderr, flush=True)
                        continue
                    t_task = time.perf_counter()
                    try:
                        r.run_task(t, split.split_id, methods)
                        print(f"{name} task {t} split {split.split_id} done in {time.perf_counter() - t_task:.0f} s", flush=True)
                    except Exception as exc:  # noqa: BLE001
                        failures += 1
                        emit({"task": t, "split": split.split_id, "method": "NOT-RUN",
                              "reason": f"{type(exc).__name__}: {exc}"[:500]}, time.perf_counter() - t_task)
                        print(f"[not run] {name} task {t}: {exc}", file=sys.stderr, flush=True)
                    longest = max(longest, time.perf_counter() - t_task)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
