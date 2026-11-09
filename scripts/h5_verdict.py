"""H5 verdict on ogbn-proteins, exactly as registered in docs/PREREGISTRATION.md A15.

Primary: EDSP-native-tuned vs EDSP-const-tuned, two-sided Wilcoxon signed-rank over the tasks,
paired by task, on test ROC-AUC. Supported iff p < 0.05 and mean Δ > 0; otherwise not supported.
Tasks without both records (NOT-RUN by the time budget) are left out and counted.

Secondary, descriptive: the same comparison for the two defaults; EDSP-native-tuned against the
best weighted baseline and against the best unweighted baseline ("best" = highest mean ROC-AUC
over the tasks), as per-task win / tie / loss; mean ROC-AUC per method (in C4_SUMMARY.md).

Input: results/runs/c4/*.jsonl (assembled, one record per task and method).
Output: results/aggregates/c4/H5_VERDICT.md and h5_verdict.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edsp.eval.stats_tests import bootstrap_ci, wilcoxon_paired

DATASET, SPLIT, KEY = "ogbn-proteins", "species", "roc_auc"
UNWEIGHTED = ["majority", "train_prior", "harmonic", "harmonic_cmn", "lgc", "ppr_lp", "linbp"]
WEIGHTED = [f"{b}+w" for b in ["harmonic", "harmonic_cmn", "lgc", "ppr_lp", "linbp"]]


def load(run_dir: Path) -> tuple[dict, list, int]:
    by: dict = defaultdict(dict)  # method -> {task: auc}
    not_run, n = [], 0
    for f in sorted(run_dir.glob("*.jsonl")):
        if f.name.startswith("ACQUISITION_LOG"):
            continue
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("suite") != "C4" or r.get("dataset") != DATASET or r.get("method") == "DATASET-STATS":
                continue
            if r.get("method") == "NOT-RUN":
                not_run.append(r)
                continue
            if r.get("split") != SPLIT:
                continue
            v = r["metrics"].get(KEY)
            t = int(r["task"])
            if t in by[r["method"]] and by[r["method"]][t] != v:
                raise SystemExit(f"conflicting duplicate: {r['method']} task {t} — assemble first")
            if v is not None and np.isfinite(v):
                by[r["method"]][t] = float(v)
                n += 1
    return by, not_run, n


def compare(by: dict, a: str, b: str) -> dict:
    common = sorted(set(by.get(a, {})) & set(by.get(b, {})))
    if len(common) < 2:
        return {"tasks": len(common)}
    x, y = np.array([by[a][t] for t in common]), np.array([by[b][t] for t in common])
    d = x - y
    lo, hi = bootstrap_ci(100 * d)
    return {"tasks": len(common), "mean_delta": float(100 * d.mean()), "median_delta": float(100 * np.median(d)),
            "ci95_mean_delta": [float(lo), float(hi)], "p": float(wilcoxon_paired(x, y)),
            "win": int(np.sum(d > 0)), "tie": int(np.sum(d == 0)), "loss": int(np.sum(d < 0))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "results" / "runs" / "c4"))
    ap.add_argument("--out", default=str(ROOT / "results" / "aggregates" / "c4"))
    ap.add_argument("--expected-tasks", type=int, default=112)
    args = ap.parse_args()
    by, not_run, n = load(Path(args.runs))
    means = {m: float(100 * np.mean(list(v.values()))) for m, v in by.items() if v}
    prim = compare(by, "EDSP-native-tuned", "EDSP-const-tuned")
    supported = prim.get("p", 1.0) < 0.05 and prim.get("mean_delta", 0.0) > 0
    best_w = max((m for m in WEIGHTED if m in means), key=means.get, default=None)
    best_u = max((m for m in UNWEIGHTED if m in means), key=means.get, default=None)
    sec = {"defaults": compare(by, "EDSP-native-default", "EDSP-const-default")}
    if best_w:
        sec[f"vs best weighted ({best_w})"] = compare(by, "EDSP-native-tuned", best_w)
    if best_u:
        sec[f"vs best unweighted ({best_u})"] = compare(by, "EDSP-native-tuned", best_u)
    missing = args.expected_tasks - prim["tasks"]
    res = {"primary": prim, "supported": bool(supported), "secondary": sec, "mean_roc_auc": means,
           "records": n, "not_run": [{k: r.get(k) for k in ("task", "baseline", "reason")} for r in not_run],
           "tasks_left_out_of_primary": missing}

    def line(name: str, c: dict) -> str:
        if "p" not in c:
            return f"| {name} | {c['tasks']} | — | — | — | — | — |"
        return (f"| {name} | {c['tasks']} | {c['mean_delta']:+.3f} [{c['ci95_mean_delta'][0]:+.3f}, "
                f"{c['ci95_mean_delta'][1]:+.3f}] | {c['median_delta']:+.3f} | {c['p']:.3g} | "
                f"{c['win']} / {c['tie']} / {c['loss']} | {'yes' if c['p'] < 0.05 else 'no'} |")

    md = ["# H5 — P-NATIVE vs P-CONST on ogbn-proteins (PREREGISTRATION A15)", "",
          (f"**Verdict: {'SUPPORTED' if supported else 'NOT SUPPORTED'}** (supported iff the two-sided Wilcoxon "
           "over tasks gives p < 0.05 and the mean Δ > 0)."), "",
          (f"{n} run records; {prim['tasks']} of {args.expected_tasks} tasks have both primary records "
           f"({missing} left out); {len(not_run)} NOT-RUN records. ROC-AUC ×100; Δ = A − B per task; "
           "95% bootstrap CI of the mean Δ over tasks; win / tie / loss = tasks where A > B / = / <."), "",
          "| comparison (A vs B) | tasks | mean Δ [95% CI] | median Δ | p | win / tie / loss | p < 0.05 |",
          "|---|---|---|---|---|---|---|",
          line("**primary:** EDSP-native-tuned vs EDSP-const-tuned", prim),
          *[line(("EDSP-native-default vs EDSP-const-default" if k == "defaults" else f"EDSP-native-tuned {k}"), c)
            for k, c in sec.items()], "",
          "Mean ROC-AUC over tasks (×100):", "",
          "| method | mean |", "|---|---|",
          *[f"| {m} | {v:.2f} |" for m, v in sorted(means.items(), key=lambda x: -x[1])], ""]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "H5_VERDICT.md").write_text("\n".join(md) + "\n")
    (out / "h5_verdict.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
