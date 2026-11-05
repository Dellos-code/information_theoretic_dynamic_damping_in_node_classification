"""Aggregate E1 run records into tables (Master Prompt §5.9, §6.5).

Input: results/runs/e1/*.jsonl. Output (results/aggregates/e1/):
  E1_SUMMARY.md     headline table, secondary column, win/tie/loss, Friedman/Nemenyi,
                    calibration and H7 descriptive tables
  e1_headline.tex   booktabs LaTeX headline table
  e1_aggregates.json
Real-world aggregates (ranks, win/tie/loss totals) exclude constructed benchmarks
(minesweeper) and the Sampson case study (docs/PREREGISTRATION.md §3).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edsp.eval.stats_tests import bootstrap_ci, friedman_nemenyi, holm, wilcoxon_paired

METHODS = ["EDSP-default", "EDSP-tuned", "majority", "train_prior", "harmonic", "harmonic_cmn", "lgc", "ppr_lp", "linbp", "signed_vote"]
RANKED = ["EDSP-default", "EDSP-tuned", "train_prior", "harmonic", "harmonic_cmn", "lgc", "ppr_lp", "linbp"]
BASELINES = ["harmonic", "harmonic_cmn", "lgc", "ppr_lp", "linbp", "train_prior"]
EXCLUDED_FROM_RANKS = {"sampson"}


def load(run_dir: Path) -> list[dict]:
    recs = []
    for f in sorted(run_dir.glob("*.jsonl")):
        recs += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "results" / "runs" / "e1"))
    ap.add_argument("--out", default=str(ROOT / "results" / "aggregates" / "e1"))
    args = ap.parse_args()
    recs = load(Path(args.runs))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    not_run = [r for r in recs if r.get("method") == "NOT-RUN"]
    recs = [r for r in recs if r.get("method") != "NOT-RUN"]
    by = defaultdict(dict)  # (dataset, method) -> {split: record}
    meta = {}
    for r in recs:
        by[(r["dataset"], r["method"])][r["split"]] = r
        meta[r["dataset"]] = (r["headline_metric"], r.get("constructed_benchmark", False))
    datasets = sorted(meta)
    shas = sorted({r["run_record"]["git_sha"][:12] for r in recs})
    runners = sorted({r["run_record"]["runner_type"] for r in recs})

    def values(ds: str, m: str, key: str | None = None) -> dict:
        key = key or meta[ds][0]
        return {s: rec["metrics"][key] for s, rec in by.get((ds, m), {}).items()}

    agg: dict = {"git_sha": shas, "runners": runners, "datasets": {}, "not_run": not_run}
    md = ["# E1 — Track-A EDSP vs B0/B1 Track-A baselines", "",
          (f"Run records: {len(recs)} (git {', '.join(shas)}; runners {', '.join(runners)}). "
          "Headline metric per dataset in brackets; mean ± std over splits (×100). "
          "Rows marked † are constructed benchmarks or case studies and are excluded from real-world aggregates."), "",
          "| dataset [metric] | splits | " + " | ".join(METHODS) + " |", "|---|---|" + "---|" * len(METHODS)]
    for ds in datasets:
        metric, constructed = meta[ds]
        cells, row = [], {}
        nsplits = len(by.get((ds, "EDSP-default"), {}))
        best = max((np.mean(list(values(ds, m).values())) for m in METHODS if values(ds, m)), default=np.nan)
        for m in METHODS:
            v = np.array([x for x in values(ds, m).values() if x is not None and np.isfinite(x)])
            if len(v) == 0:
                cells.append("—")
                continue
            mu, sd = 100 * v.mean(), 100 * v.std()
            lo, hi = bootstrap_ci(100 * v)
            row[m] = {"mean": mu, "std": sd, "ci95": [lo, hi], "n": len(v)}
            txt = f"{mu:.2f} ± {sd:.2f}"
            cells.append(f"**{txt}**" if np.isclose(v.mean(), best) else txt)
        tag = "†" if constructed or ds in EXCLUDED_FROM_RANKS else ""
        md.append(f"| {ds}{tag} [{metric}] | {nsplits} | " + " | ".join(cells) + " |")
        agg["datasets"][ds] = {"metric": metric, "constructed": constructed, "methods": row}

    # secondary column
    md += ["", "## Pre-registered secondary column: EDSP-default on non-converged runs", "",
           "| dataset | non-converged runs | headline, last iterate | headline, mean of last two iterates |", "|---|---|---|---|"]
    for ds in datasets:
        recs_d = list(by.get((ds, "EDSP-default"), {}).values())
        nc = [r for r in recs_d if not r["diagnostics"]["converged"]]
        if not nc:
            md.append(f"| {ds} | 0 / {len(recs_d)} | — | — |")
            continue
        k = meta[ds][0]
        a = np.mean([r["metrics"][k] for r in nc])
        b = np.mean([r["metrics_mean_last_two"][k] for r in nc])
        md.append(f"| {ds} | {len(nc)} / {len(recs_d)} | {100 * a:.2f} | {100 * b:.2f} |")

    # win / tie / loss (Wilcoxon over splits, Holm over datasets per baseline)
    md += ["", "## EDSP-tuned vs each baseline: win / tie / loss", "",
           ("Per dataset: two-sided Wilcoxon signed-rank over splits on the headline metric; "
           "Holm correction across datasets per baseline; win/loss if adjusted p < 0.05, else tie. "
           "Real-world datasets only."), "",
           "| baseline | win | tie | loss | datasets (W / L) |", "|---|---|---|---|---|"]
    real = [d for d in datasets if not meta[d][1] and d not in EXCLUDED_FROM_RANKS]
    wtl = {}
    for b in BASELINES:
        ps, signs, used = [], [], []
        for ds in real:
            a, bb = values(ds, "EDSP-tuned"), values(ds, b)
            common = sorted(set(a) & set(bb))
            if len(common) < 2:
                continue
            x, y = np.array([a[s] for s in common]), np.array([bb[s] for s in common])
            ps.append(wilcoxon_paired(x, y))
            signs.append(np.sign(np.mean(x - y)))
            used.append(ds)
        adj = holm(ps)
        w = [d for d, p, s in zip(used, adj, signs, strict=True) if np.isfinite(p) and p < 0.05 and s > 0]
        lo = [d for d, p, s in zip(used, adj, signs, strict=True) if np.isfinite(p) and p < 0.05 and s < 0]
        t = len(used) - len(w) - len(lo)
        wtl[b] = {"win": w, "loss": lo, "tie": t}
        md.append(f"| {b} | {len(w)} | {t} | {len(lo)} | W: {', '.join(w) or '—'}; L: {', '.join(lo) or '—'} |")
    agg["win_tie_loss_edsp_tuned"] = wtl

    # Friedman / Nemenyi
    full = [d for d in real if all(values(d, m) for m in RANKED)]
    if len(full) >= 2:
        S = np.array([[np.mean(list(values(d, m).values())) for m in RANKED] for d in full])
        fn = friedman_nemenyi(S)
        agg["friedman_nemenyi"] = fn | {"methods": RANKED, "datasets": full}
        order = np.argsort(fn["average_ranks"])
        md += ["", "## Friedman / Nemenyi (real-world datasets, headline metric)", "",
               f"Datasets: {len(full)}; Friedman p = {fn['friedman_p']:.3g}; Nemenyi CD (α = 0.05) = {fn['nemenyi_cd']:.2f}.", "",
               "| method | average rank |", "|---|---|"]
        md += [f"| {RANKED[i]} | {fn['average_ranks'][i]:.2f} |" for i in order]

    # calibration
    md += ["", "## Calibration (mean over splits; lower is better)", "",
           "| dataset | EDSP-default NLL | EDSP-default ECE | EDSP-tuned NLL | EDSP-tuned ECE | best baseline by NLL (NLL / ECE) |", "|---|---|---|---|---|---|"]
    for ds in datasets:
        def mean_of(m: str, key: str, ds: str = ds) -> float:
            v = list(values(ds, m, key).values())
            return float(np.mean(v)) if v else float("nan")
        bb = min(BASELINES + ["majority"], key=lambda m: mean_of(m, "nll") if np.isfinite(mean_of(m, "nll")) else np.inf)
        md.append(f"| {ds} | {mean_of('EDSP-default', 'nll'):.3f} | {mean_of('EDSP-default', 'ece'):.3f} | "
                  f"{mean_of('EDSP-tuned', 'nll'):.3f} | {mean_of('EDSP-tuned', 'ece'):.3f} | {bb}: {mean_of(bb, 'nll'):.3f} / {mean_of(bb, 'ece'):.3f} |")

    # H7
    md += ["", "## H7 (descriptive): convergence and fixed-point non-uniqueness", "",
           "| dataset | EDSP-default non-converged | oscillation flagged | mean fraction of nodes still moving | GS vs damped L1: mean | p95 | splits |", "|---|---|---|---|---|---|---|"]
    h7 = {}
    for ds in datasets:
        d = [r["diagnostics"] for r in by.get((ds, "EDSP-default"), {}).values()]
        dm = [r["h7_gs_vs_damped_l1"] for r in by.get((ds, "EDSP-damped(0.5)"), {}).values() if "h7_gs_vs_damped_l1" in r]
        h7[ds] = {"non_converged_rate": float(np.mean([not x["converged"] for x in d])) if d else None,
                  "oscillation_rate": float(np.mean([x["oscillation"] for x in d])) if d else None,
                  "moving_fraction": float(np.mean([x["fraction_nodes_still_moving"] for x in d])) if d else None,
                  "gs_damped_l1_mean": float(np.mean([x["mean"] for x in dm])) if dm else None,
                  "gs_damped_l1_p95": float(np.mean([x["p95"] for x in dm])) if dm else None, "h7_splits": len(dm)}
        x = h7[ds]

        def f(v: float | None) -> str:
            return "—" if v is None else f"{v:.3f}"
        md.append(f"| {ds} | {f(x['non_converged_rate'])} | {f(x['oscillation_rate'])} | {f(x['moving_fraction'])} | {f(x['gs_damped_l1_mean'])} | {f(x['gs_damped_l1_p95'])} | {x['h7_splits']} |")
    agg["h7"] = h7
    if not_run:
        md += ["", "## Not run", ""] + [f"- {r['dataset']} split {r['split']}: {r['reason']}" for r in not_run]

    text = "\n".join(md) + "\n"
    (out / "E1_SUMMARY.md").write_text(text)
    (out / "e1_aggregates.json").write_text(json.dumps(agg, indent=2, default=float) + "\n")
    tex = ["\\begin{tabular}{l" + "r" * len(RANKED) + "}", "\\toprule", "Dataset & " + " & ".join(m.replace("_", "\\_") for m in RANKED) + " \\\\", "\\midrule"]
    for ds in datasets:
        r = agg["datasets"][ds]["methods"]
        tex.append(ds + (" (constructed)" if meta[ds][1] else "") + " & " + " & ".join(
            f"{r[m]['mean']:.1f}$\\pm${r[m]['std']:.1f}" if m in r else "--" for m in RANKED) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (out / "e1_headline.tex").write_text("\n".join(tex) + "\n")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as fh:
            fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
