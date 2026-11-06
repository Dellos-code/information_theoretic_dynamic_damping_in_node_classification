"""Suite E1 (Master Prompt §7): Track-A EDSP vs B0/B1 Track-A baselines on real datasets.

For each dataset and split (all official / registered splits):
  * EDSP-default  — zero-tuning default of §2.5 (Track-A, P-CONST(0.5),
                    H-TRAINONLY(b1), T_L = inf; H-NATIVE on signed graphs,
                    H-RELATION on relation-typed graphs — suite E2).
  * EDSP-tuned    — best of 9 registered configurations by the validation value
                    of the dataset's headline metric (rule R6: same 9-point budget
                    as the tuned baselines).
  * baselines     — majority, train_prior, harmonic, harmonic_cmn, lgc, ppr_lp,
                    linbp (+ signed_vote, signed_lgc on signed graphs), validation-selected.
Pre-registered extras (docs/PREREGISTRATION.md):
  * secondary column for non-converged EDSP runs: mean of the last two iterates;
  * H7 descriptive outcome: convergence, oscillation, and on the first
    ``--h7-splits`` splits the per-node L1 distance between the fixed points of
    EDSP-GS (id order) and EDSP-damped(0.5), both also reported as named variants.

Every run appends one JSON record to --out/<dataset>.jsonl (rule R1).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edsp.baselines import BASELINES, select_and_predict
from edsp.baselines.propagation import adjacency
from edsp.core import edsp
from edsp.data.loaders import load
from edsp.data.perturb import perturb
from edsp.data.registry import dataset_entry, datasets_with_loader
from edsp.data.stats import DEGREE_BINS, degree_bins, label_stats, local_homophily_bins
from edsp.eval.evaluate import evaluate, stratified_accuracy
from edsp.regimes import build_inputs
from edsp.regimes.inputs import _polarity
from edsp.selection import validation_score

TRACK_A_BASELINES = ["majority", "train_prior", "harmonic", "harmonic_cmn", "lgc", "ppr_lp", "linbp"]
P_GRID = (0.1, 0.5, 0.9)
EPS = 1e-6


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def headline(name: str) -> str:
    m = dataset_entry(name)["metric"]
    return "roc_auc" if "roc" in m.lower() else "accuracy"


def _native_h(signed: bool, relation: bool) -> dict:
    if signed:
        return {"name": "native"}
    if relation:
        return {"name": "relation"}
    return {"name": "trainonly", "fill": "b1"}


def default_config(signed: bool, relation: bool = False) -> dict:
    return {"track": "A", "p": {"name": "const", "p0": 0.5}, "h": _native_h(signed, relation), "T_L": math.inf}


def tuned_grid(signed: bool, relation: bool = False) -> list[dict]:
    hs = [_native_h(signed, relation), {"name": "learned"}, {"name": "all1"}]
    return [{"track": "A", "p": {"name": "const", "p0": p0}, "h": h, "T_L": math.inf} for h in hs for p0 in P_GRID]


def config_hash(cfg: dict) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()


def run_core(inputs, variant: str = "EDSP-canonical", **kw):
    return edsp(inputs.edges, inputs.probabilities, inputs.heterophily, inputs.priors,
                prior_trust=inputs.prior_trust, edge_trust=inputs.edge_trust, variant=variant, **kw)


def scored(vault, probs, bins) -> dict:
    return evaluate(vault, probs) | {"stratified": {
        "local_homophily": stratified_accuracy(vault, probs, bins[0], 10),
        "degree": stratified_accuracy(vault, probs, bins[1], DEGREE_BINS)}}


def edsp_diagnostics(res) -> dict:
    prev, last = res.last_iterates
    moving = np.abs(last - prev).sum(axis=1) >= EPS
    return {
        "iterations": res.iterations, "converged": bool(res.converged), "final_residual": float(res.final_residual),
        "oscillation": bool(res.oscillation), "oscillation_amplitude": float(res.oscillation_amplitude),
        "fraction_nodes_still_moving": float(moving.mean()), "phi_residual_l1": float(res.phi_residual_l1),
        "regime_fractions_last": [float(x) for x in res.regime_trace[-1]] if res.regime_trace else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--max-splits", type=int, default=None)
    ap.add_argument("--h7-splits", type=int, default=3)
    ap.add_argument("--gs-max-iter", type=int, default=300)
    ap.add_argument("--suite", default="E1")
    ap.add_argument("--perturb", nargs="*", default=None,
                    help="PERTURBED-REAL mixing specs (edsp.data.perturb), e.g. global:h=0.3 uniform")
    ap.add_argument("--perturb-seeds", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.out = args.out or str(ROOT / "results" / "runs" / args.suite.lower())
    names = args.datasets or [d for d in datasets_with_loader() if d != "ogbn-proteins"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_id = os.environ.get("GITHUB_RUN_ID")
    runner = f"actions-run-{run_id}" if run_id else os.environ.get("EDSP_ENVIRONMENT", "local")
    sha = git_sha()
    failures = 0
    jobs = [(b, None, 0) for b in names] if not args.perturb else [
        (b, spec, s) for b in names for spec in args.perturb for s in range(args.perturb_seeds)]
    loaded: dict = {}
    for base, spec, pseed in jobs:
        if base not in loaded:
            loaded = {base: load(base)}
        g = loaded[base] if spec is None else perturb(loaded[base], spec, pseed)
        name = g.name
        entry = dataset_entry(base)
        metric = headline(base)
        signed = g.edge_sign is not None
        relation = g.edge_relation is not None
        bins = (local_homophily_bins(g, 10), degree_bins(g))
        dhash = g.content_hash()
        path = out / f"{name}.jsonl"
        splits = g.splits[: args.max_splits] if args.max_splits else g.splits

        pert = g.meta["canonicalization"]["perturbation"] if spec is not None else None

        def emit(rec: dict, t0: float, path: Path = path, name: str = name, dhash: str = dhash, metric: str = metric,
                 entry: dict = entry, base: str = base, pert: dict | None = pert) -> None:
            rec |= {"suite": args.suite, "dataset": name, "headline_metric": metric,
                    "constructed_benchmark": bool(entry.get("constructed_benchmark", False)),
                    **({"base_dataset": base, "perturbation": pert, "data_class": "PERTURBED-REAL"} if pert else {}),
                    "run_record": {"git_sha": sha, "runner_type": runner, "dataset_hash": dhash,
                                   "config_hash": config_hash(rec.get("config")),
                                   "wall_clock_seconds": round(time.perf_counter() - t0, 3),
                                   "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}}
            with open(path, "a") as fh:
                fh.write(json.dumps(rec, default=float) + "\n")
                print(f"  {rec['method']} {rec.get('split')} {time.perf_counter() - t0:.0f}s maxrss={__import__('resource').getrusage(0).ru_maxrss // 1024}MB", flush=True)

        # descriptive dataset statistics used to form the empirical mixed cells
        # C3c (local-homophily spread) and C3d (structured compatibility), §3.1
        t0 = time.perf_counter()
        ls = label_stats(g, train_split=None) if not g.multilabel else {}
        emit({"split": None, "method": "DATASET-STATS", "cell": entry.get("cell"),
              "stats": {key: ls.get(key) for key in ("edge_homophily", "adjusted_homophily", "label_informativeness",
                                                    "local_homophily_std", "compatibility_distances")}
              | {"num_classes": g.num_classes, "num_nodes": g.num_nodes, "num_edges": len(g.edges),
                 "components": int(connected_components(adjacency(g), directed=False)[0])}}, t0)

        for si, split in enumerate(splits):
            sid = split.split_id
            try:
                # ---------------- EDSP-default (+ secondary column, H7 diagnostics)
                t0 = time.perf_counter()
                vault = g.vault(sid)
                cfg = default_config(signed, relation)
                inp = build_inputs(g, vault, cfg)
                res = run_core(inp)
                rec = {"split": sid, "method": "EDSP-default", "track": "A", "config": cfg,
                       "metrics": scored(vault, res.probabilities, bins), "diagnostics": edsp_diagnostics(res),
                       "regime_diagnostics": {k: v for k, v in inp.diagnostics.items() if k != "priors"}}
                if not res.converged:
                    rec["metrics_mean_last_two"] = evaluate(vault, 0.5 * (res.last_iterates[0] + res.last_iterates[1]))
                emit(rec, t0)

                # ---------------- H7(c) and named variants on the first splits
                if si < args.h7_splits:
                    t0 = time.perf_counter()
                    gs = run_core(inp, "EDSP-GS", max_iter=args.gs_max_iter)
                    emit({"split": sid, "method": "EDSP-GS", "track": "A", "config": cfg | {"variant": "EDSP-GS", "max_iter": args.gs_max_iter},
                          "metrics": scored(vault, gs.probabilities, bins), "diagnostics": edsp_diagnostics(gs)}, t0)
                    t0 = time.perf_counter()
                    dm = run_core(inp, "EDSP-damped", beta=0.5)
                    dist = np.abs(gs.probabilities - dm.probabilities).sum(axis=1)
                    emit({"split": sid, "method": "EDSP-damped(0.5)", "track": "A", "config": cfg | {"variant": "EDSP-damped", "beta": 0.5},
                          "metrics": scored(vault, dm.probabilities, bins), "diagnostics": edsp_diagnostics(dm),
                          "h7_gs_vs_damped_l1": {"mean": float(dist.mean()), "p95": float(np.quantile(dist, 0.95)),
                                                 "gs_converged": bool(gs.converged), "damped_converged": bool(dm.converged)}}, t0)

                # ---------------- EDSP-tuned (9 configs, validation-selected)
                t0 = time.perf_counter()
                pol_cache: dict[str, object] = {}
                cands = []
                for c in tuned_grid(signed, relation):
                    key = json.dumps(c["h"], sort_keys=True)
                    if key not in pol_cache:
                        pol_cache[key] = _polarity(g, vault, c["h"], "A")
                    i = build_inputs(g, vault, c, polarity=pol_cache[key])  # type: ignore[arg-type]
                    r = run_core(i)
                    cands.append((c, r, validation_score(vault, r.probabilities, metric)))
                best = int(np.argmax([s if np.isfinite(s) else -np.inf for _, _, s in cands]))
                c, r, _ = cands[best]
                emit({"split": sid, "method": "EDSP-tuned", "track": "A", "config": c,
                      "metrics": scored(vault, r.probabilities, bins), "diagnostics": edsp_diagnostics(r),
                      "selection": {"metric": metric, "val_scores": [s for _, _, s in cands],
                                    "grid": [cc for cc, _, _ in cands], "chosen_index": best}}, t0)

                # ---------------- baselines
                for b in TRACK_A_BASELINES + (["signed_vote", "signed_lgc"] if signed else []):
                    t0 = time.perf_counter()
                    probs, info = select_and_predict(b, g, vault, metric)
                    emit({"split": sid, "method": b, "group": BASELINES[b][0], "track": "A",
                          "config": {"baseline": b, "chosen": info.get("chosen")},
                          "metrics": scored(vault, probs, bins),
                          "diagnostics": {k: v for k, v in info.items() if k in ("converged", "iterations", "uncovered_fraction", "solver")}}, t0)
                print(f"{name} split {sid} done", flush=True)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                with open(path, "a") as fh:
                    fh.write(json.dumps({"suite": args.suite, "dataset": name, "split": sid, "method": "NOT-RUN",
                                         "reason": f"{type(exc).__name__}: {exc}"[:500],
                                         "run_record": {"git_sha": sha, "runner_type": runner}}) + "\n")
                print(f"[not run] {name} split {sid}: {exc}", file=sys.stderr, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
