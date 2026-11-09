"""Aggregate suite C4 run records (scripts/run_c4.py) into descriptive tables.

Input: results/runs/c4/*.jsonl. Output (results/aggregates/c4/): C4_SUMMARY.md,
c4_aggregates.json. Per dataset and split: the headline metric per method as mean ± std
over tasks (ogbn-proteins: the mean ROC-AUC over the 112 tasks is the OGB metric), the
paired per-task comparisons P-NATIVE vs P-CONST and weighted vs unweighted baselines
(mean difference, two-sided Wilcoxon signed-rank over tasks, wins / ties / losses),
convergence and the chosen tuned configurations. No hypothesis verdict is computed
here: the H5 decision rule is the one registered in docs/PREREGISTRATION.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edsp.eval.stats_tests import bootstrap_ci, wilcoxon_paired

EDSP = ["EDSP-const-default", "EDSP-native-default", "EDSP-const-tuned", "EDSP-native-tuned"]
BASE = ["majority", "train_prior", "harmonic", "harmonic_cmn", "lgc", "ppr_lp", "linbp"]
WEIGHTED = ["harmonic", "harmonic_cmn", "lgc", "ppr_lp", "linbp"]
METHODS = EDSP + BASE + [f"{b}+w" for b in WEIGHTED]
PAIRS = [("EDSP-native-default", "EDSP-const-default"), ("EDSP-native-tuned", "EDSP-const-tuned")] + [
    (f"{b}+w", b) for b in WEIGHTED] + [("EDSP-native-tuned", f"{b}+w") for b in WEIGHTED]


def load(run_dir: Path, suite: str = "C4") -> list[dict]:
    recs = []
    for f in sorted(run_dir.glob("*.jsonl")):
        if f.name.startswith("ACQUISITION_LOG"):
            continue
        recs += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    return [r for r in recs if r.get("suite") == suite]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="C4")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    s = args.suite.lower()
    recs = load(Path(args.runs or ROOT / "results" / "runs" / s), args.suite)
    out = Path(args.out or ROOT / "results" / "aggregates" / s)
    out.mkdir(parents=True, exist_ok=True)
    not_run = [r for r in recs if r.get("method") == "NOT-RUN"]
    dstats = {r["dataset"]: r["stats"] for r in recs if r.get("method") == "DATASET-STATS"}
    runs = [r for r in recs if r.get("method") not in ("NOT-RUN", "DATASET-STATS")]
    by: dict = defaultdict(dict)  # (dataset, split, method) -> {task: record}
    metric = {}
    for r in runs:
        by[(r["dataset"], r["split"], r["method"])][r["task"]] = r
        metric[r["dataset"]] = r["headline_metric"]
    shas = sorted({r["run_record"]["git_sha"][:12] for r in runs})
    agg: dict = {"git_sha": shas, "datasets": {}, "not_run": not_run, "dataset_stats": dstats}
    md = [f"# {args.suite} — P-NATIVE vs P-CONST on native probabilistic graphs (H5), Track A", "",
          (f"Run records: {len(runs)} (git {', '.join(shas) or '—'}). Mean ± std over tasks (×100) of the headline "
           "metric; 95% bootstrap CI over tasks in brackets. Descriptive only: the H5 decision rule is the one "
           "registered in docs/PREREGISTRATION.md."), ""]
    for ds, sid in sorted({(d, sp) for d, sp, _ in by}):
        key = metric[ds]

        def vals(m: str, ds: str = ds, sid: str = sid, key: str = key) -> dict:
            return {t: r["metrics"][key] for t, r in by.get((ds, sid, m), {}).items()
                    if r["metrics"].get(key) is not None and np.isfinite(r["metrics"][key])}

        entry: dict = {"metric": key, "methods": {}, "pairs": {}}
        md += [f"## {ds} — split {sid} [{key}]", "", "| method | tasks | mean ± std | 95% CI |", "|---|---|---|---|"]
        for m in METHODS:
            v = np.array(list(vals(m).values()))
            if not len(v):
                continue
            lo, hi = bootstrap_ci(100 * v)
            entry["methods"][m] = {"tasks": len(v), "mean": 100 * v.mean(), "std": 100 * v.std(), "ci95": [lo, hi]}
            md.append(f"| {m} | {len(v)} | {100 * v.mean():.2f} ± {100 * v.std():.2f} | [{lo:.2f}, {hi:.2f}] |")
        md += ["", "Paired per-task comparisons (A − B, ×100; Wilcoxon signed-rank over tasks, two-sided, uncorrected):", "",
               "| A | B | tasks | mean Δ | p | A > B | tie | A < B |", "|---|---|---|---|---|---|---|---|"]
        for a, b in PAIRS:
            va, vb = vals(a), vals(b)
            common = sorted(set(va) & set(vb))
            if not common:
                continue
            x, y = np.array([va[t] for t in common]), np.array([vb[t] for t in common])
            d = x - y
            p = wilcoxon_paired(x, y)
            row = {"tasks": len(common), "mean_delta": 100 * float(d.mean()), "p": p,
                   "a_better": int(np.sum(d > 0)), "tie": int(np.sum(d == 0)), "b_better": int(np.sum(d < 0))}
            entry["pairs"][f"{a} vs {b}"] = row
            md.append(f"| {a} | {b} | {len(common)} | {row['mean_delta']:+.2f} | {p:.3g} | {row['a_better']} | "
                      f"{row['tie']} | {row['b_better']} |")
        md += ["", "| EDSP method | non-converged tasks | oscillation flagged | median iterations | chosen configurations (count) |",
               "|---|---|---|---|---|"]
        for m in EDSP:
            rs = list(by.get((ds, sid, m), {}).values())
            if not rs:
                continue
            nc = sum(not r["diagnostics"]["converged"] for r in rs)
            osc = sum(bool(r["diagnostics"]["oscillation"]) for r in rs)
            it = float(np.median([r["diagnostics"]["iterations"] for r in rs]))
            chosen = Counter(json.dumps({"p": r["config"]["p"], "h": r["config"]["h"]}, sort_keys=True) for r in rs)
            entry["methods"].setdefault(m, {}).update({"non_converged": nc, "oscillation": osc,
                                                       "median_iterations": it, "chosen": dict(chosen)})
            ch = "; ".join(f"{c} ×{k}" for c, k in chosen.most_common()) if m.endswith("tuned") else "—"
            md.append(f"| {m} | {nc} / {len(rs)} | {osc} | {it:.0f} | {ch} |")
            if nc:
                sec = [r["metrics_mean_last_two"][key] for r in rs if "metrics_mean_last_two" in r]
                last = [r["metrics"][key] for r in rs if "metrics_mean_last_two" in r]
                md.append(f"| ↳ secondary column (non-converged tasks) | last iterate {100 * np.mean(last):.2f} | "
                          f"mean of last two {100 * np.mean(sec):.2f} | | |")
        md.append("")
        agg["datasets"][f"{ds}|{sid}"] = entry
    if not_run:
        md += ["## Not run", ""] + [f"- {r['dataset']} task {r.get('task')} split {r.get('split')}: {r['reason']}" for r in not_run]
    text = "\n".join(md) + "\n"
    (out / f"{args.suite}_SUMMARY.md").write_text(text)
    (out / f"{s}_aggregates.json").write_text(json.dumps(agg, indent=2, default=float) + "\n")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as fh:
            fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
