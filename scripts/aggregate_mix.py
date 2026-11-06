"""Aggregate the PERTURBED-REAL mixing sweep (suite MIX; edsp/data/perturb.py).

Input: results/perturbed/runs/*.jsonl. Output (results/perturbed/aggregates/):
  MIX_SUMMARY.md, mix_aggregates.json.
These numbers are PERTURBED-REAL (rule R4): they never enter the real-world
tables, ranks or win/tie/loss totals of E1/E2.

Per base graph and mixing condition: the achieved mixing statistics (mean over
re-wiring seeds), every method's headline metric (mean ± std over seeds × splits),
and EDSP-tuned minus the best baseline (by mean) with a two-sided Wilcoxon test
over the paired (seed, split) runs, Holm-corrected over the conditions of a base graph.
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

from edsp.eval.stats_tests import holm, wilcoxon_paired

METHODS = ["EDSP-default", "EDSP-tuned", "train_prior", "harmonic", "harmonic_cmn", "lgc", "ppr_lp", "linbp"]
BASELINES = ["harmonic", "harmonic_cmn", "lgc", "ppr_lp", "linbp"]


def spec_order(spec: str) -> tuple:
    mode = spec.split(":")[0]
    order = ["global", "beta", "uniform", "logistic", "regional", "community", "permutation"]
    return (order.index(mode) if mode in order else 99, spec)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "results" / "perturbed" / "runs"))
    ap.add_argument("--out", default=str(ROOT / "results" / "perturbed" / "aggregates"))
    args = ap.parse_args()
    recs = []
    for f in sorted(Path(args.runs).glob("*.jsonl")):
        if f.name.startswith("ACQUISITION_LOG"):
            continue
        recs += [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    recs = [r for r in recs if r.get("suite") == "MIX" and r.get("perturbation")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def spec_of(r: dict) -> str:
        p = r["perturbation"]
        return p["mode"] + (":" + ",".join(f"{k}={v:g}" for k, v in p["params"].items()) if p["params"] else "")

    stats = defaultdict(list)  # (base, spec) -> list of stats dicts
    vals = defaultdict(dict)  # (base, spec, method) -> {(seed, split): value}
    metric_of = {}
    not_run = []
    for r in recs:
        key = (r["base_dataset"], spec_of(r))
        if r["method"] == "DATASET-STATS":
            stats[key].append(r["stats"])
        elif r["method"] == "NOT-RUN":
            not_run.append(r)
        else:
            vals[key + (r["method"],)][(r["perturbation"]["seed"], r["split"])] = r["metrics"][r["headline_metric"]]
            metric_of[r["base_dataset"]] = r["headline_metric"]
    shas = sorted({r["run_record"]["git_sha"][:12] for r in recs})
    md = ["# MIX — PERTURBED-REAL mixing sweep (not real-world results)", "",
          (f"Run records: {len(recs)} (git {', '.join(shas)}). Real nodes, labels, features and splits; edges re-wired "
           "by `edsp/data/perturb.py` to the mixing condition in the first column. Achieved statistics are means over "
           "re-wiring seeds (full labels, descriptive). Method cells: headline metric ×100, mean ± std over "
           "(seed, split) runs; best in bold. Δ = EDSP-tuned − best baseline; p = Wilcoxon over paired runs, "
           "Holm over the conditions of the base graph.")]
    agg: dict = {"git_sha": shas, "bases": {}}
    for base in sorted({k[0] for k in stats}):
        specs = sorted({k[1] for k in stats if k[0] == base}, key=spec_order)
        md += ["", f"## {base} [{metric_of.get(base, '?')}]", "",
               "| condition | h_edge | h_adj | LI | std local h | " + " | ".join(METHODS) + " | Δ vs best baseline | p (Holm) |",
               "|---|---|---|---|---|" + "---|" * len(METHODS) + "---|---|"]
        rows, ps, pending = {}, [], []
        for spec in specs:
            st = stats[(base, spec)]
            mean = {k: float(np.mean([s[k] for s in st])) for k in ("edge_homophily", "adjusted_homophily", "label_informativeness", "local_homophily_std")}
            cell = {m: np.array(list(vals[(base, spec, m)].values())) for m in METHODS if vals.get((base, spec, m))}
            means = {m: float(v.mean()) for m, v in cell.items()}
            best_b = max((b for b in BASELINES if b in means), key=lambda b: means[b], default=None)
            delta, p = float("nan"), float("nan")
            if best_b and "EDSP-tuned" in means:
                a, b = vals[(base, spec, "EDSP-tuned")], vals[(base, spec, best_b)]
                common = sorted(set(a) & set(b))
                delta = means["EDSP-tuned"] - means[best_b]
                if len(common) >= 2:
                    p = wilcoxon_paired(np.array([a[c] for c in common]), np.array([b[c] for c in common]))
            ps.append(p)
            pending.append((spec, mean, cell, means, best_b, delta))
            rows[spec] = {"stats": mean, "methods": {m: {"mean": means[m], "std": float(cell[m].std()), "n": len(cell[m])} for m in cell},
                          "best_baseline": best_b, "delta_edsp_tuned": delta, "p_raw": p}
        adj = holm(ps)
        for (spec, mean, cell, means, best_b, delta), p in zip(pending, adj, strict=True):
            top = max(means.values()) if means else np.nan
            cells = [("**{:.1f} ± {:.1f}**" if np.isclose(means[m], top) else "{:.1f} ± {:.1f}").format(100 * means[m], 100 * cell[m].std())
                     if m in means else "—" for m in METHODS]
            rows[spec]["p_holm"] = p
            md.append(f"| {spec} | {mean['edge_homophily']:.2f} | {mean['adjusted_homophily']:+.2f} | {mean['label_informativeness']:.2f} | "
                      f"{mean['local_homophily_std']:.2f} | " + " | ".join(cells)
                      + f" | {100 * delta:+.1f} ({best_b}) | {p:.3g} |")
        agg["bases"][base] = rows
    if not_run:
        md += ["", "## Not run", ""] + [f"- {r['dataset']} {r['split']}: {r['reason']}" for r in not_run]
    text = "\n".join(md) + "\n"
    (out / "MIX_SUMMARY.md").write_text(text)
    (out / "mix_aggregates.json").write_text(json.dumps(agg, indent=2, default=float) + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
