"""A18: EDSP-B as a post-hoc smoother — non-inferiority to the base predictor and harm of the comparators.

The same analysis serves the exploratory phase (the E3 graphs already used) and the confirmatory
phase (the graphs fixed in PREREGISTRATION A18); only the inputs differ.

For every graph g and method m (EDSP-B-tuned, and each post-hoc comparator), d_k = m − base on
split k (headline metric ×100; base = `feature_lr`, the predictor all post-hoc methods refine).

- Non-inferiority (NI) of m: H0 mean(d) ≤ −δ vs H1 mean(d) > −δ, one-sided.
- Harm of m: H0 mean(d) ≥ −δ vs H1 mean(d) < −δ, one-sided.

Primary statistic: Nadeau–Bengio corrected resampled t over the K splits,
t = (mean(d) ± δ) / sqrt((1/K + n_test/n_train) · s²), K − 1 degrees of freedom. Sensitivity:
one-sided Wilcoxon signed-rank on d ± δ. NI of EDSP-B-tuned "on every graph" is an
intersection–union test: every graph at one-sided α = 0.025, no multiplicity adjustment. Harm of
a comparator is Holm-adjusted across the graphs, per comparator. EDSP-B-tuned vs each
comparator: two-sided corrected t, Holm across graphs per comparator, win / tie / loss.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edsp.eval.stats_tests import holm

REF, BASE = "EDSP-tuned", "feature_lr"
COMPARATORS = ["correct_and_smooth", "cs_ref", "cs_2hop", "cs_2hop_ref", "clp", "clp_es", "linbp_b"]


def load(dirs: list[Path], suite: str, exclude: set[tuple[str, str]] = frozenset()) -> tuple[dict, dict, dict]:
    by: dict = defaultdict(dict)  # (dataset, method) -> {split: value×100}
    metric, hadj = {}, {}
    for d in dirs:
        for f in sorted(d.glob("*.jsonl")):
            if f.name.startswith("ACQUISITION_LOG"):
                continue
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("suite") != suite:
                    continue
                if r.get("method") == "DATASET-STATS":
                    h = (r.get("stats") or {}).get("adjusted_homophily")
                    if h is not None:
                        hadj[r["dataset"]] = float(h)
                    continue
                if r.get("method") == "NOT-RUN" or r.get("track") not in (None, "B"):
                    continue
                if (r["dataset"], str(r["split"])) in exclude:
                    continue
                v = r["metrics"].get(r["headline_metric"])
                if v is None or not np.isfinite(v):
                    continue
                key = (r["dataset"], r["method"])
                if r["split"] in by[key] and abs(by[key][r["split"]] - 100 * v) > 1e-9:
                    raise SystemExit(f"conflicting duplicate {key} split {r['split']}: assemble first")
                by[key][r["split"]] = 100 * float(v)
                metric[r["dataset"]] = r["headline_metric"]
    return by, metric, hadj


def corrected_t(d: np.ndarray, shift: float, ratio: float) -> tuple[float, int]:
    """t statistic of mean(d) + shift with the Nadeau–Bengio variance correction."""
    k = len(d)
    s2 = float(np.var(d, ddof=1)) if k > 1 else 0.0
    se = np.sqrt((1.0 / k + ratio) * s2)
    m = float(np.mean(d)) + shift
    if se == 0.0:
        return (np.inf if m > 0 else -np.inf if m < 0 else 0.0), k - 1
    return m / se, k - 1


def one_sided(d: np.ndarray, delta: float, ratio: float, side: str) -> dict:
    """side = 'ni': H1 mean(d) > −δ; side = 'harm': H1 mean(d) < −δ."""
    t, df = corrected_t(d, delta, ratio)
    p = float(stats.t.sf(t, df)) if side == "ni" else float(stats.t.cdf(t, df))
    x = d + delta
    if np.all(x == 0):
        pw = 1.0
    else:
        pw = float(stats.wilcoxon(x, alternative="greater" if side == "ni" else "less", zero_method="wilcox").pvalue)
    return {"t": float(t), "p": p, "p_wilcoxon": pw}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--suite", default="E3")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--delta", type=float, default=1.0, help="NI margin in points of the headline metric (×100)")
    ap.add_argument("--alpha", type=float, default=0.025, help="one-sided level of each NI / harm test")
    ap.add_argument("--ratio", type=float, default=1 / 3, help="n_test / n_train of the split protocol")
    ap.add_argument("--ratio-for", nargs="*", default=[], help="DATASET=RATIO overrides")
    ap.add_argument("--het-threshold", type=float, default=0.2, help="heterophilous iff adjusted homophily < this")
    ap.add_argument("--exclude-split", nargs="*", default=[],
                    help="DATASET:SPLIT pairs left out for every method (e.g. base predictor not identical across runs)")
    ap.add_argument("--label", default="exploratory")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ratios = {k: float(v) for k, v in (x.split("=") for x in args.ratio_for)}
    exclude = {tuple(x.split(":", 1)) for x in args.exclude_split}
    by, metric, hadj = load([Path(x) for x in args.runs], args.suite, exclude)
    graphs = sorted(d for d in metric if (args.datasets is None or d in args.datasets)
                    and by.get((d, REF)) and by.get((d, BASE)))
    methods = [REF] + [c for c in COMPARATORS if any(by.get((d, c)) for d in graphs)]
    res: dict = {"label": args.label, "delta": args.delta, "alpha": args.alpha, "graphs": {}, "methods": methods}
    for g in graphs:
        r = ratios.get(g, args.ratio)
        entry = {"metric": metric[g], "adjusted_homophily": hadj.get(g), "ratio": r,
                 "heterophilous": None if g not in hadj else hadj[g] < args.het_threshold, "vs_base": {}, "vs_ref": {}}
        base = by[(g, BASE)]
        for m in methods:
            v = by.get((g, m), {})
            common = sorted(set(v) & set(base))
            if len(common) < 2:
                continue
            d = np.array([v[s] - base[s] for s in common])
            entry["vs_base"][m] = {"n": len(common), "mean": float(d.mean()), "sd": float(d.std(ddof=1)),
                                   "ni": one_sided(d, args.delta, r, "ni"), "harm": one_sided(d, args.delta, r, "harm")}
            if m != REF:
                e = by[(g, REF)]
                cm = sorted(set(e) & set(v))
                if len(cm) >= 2:
                    dd = np.array([e[s] - v[s] for s in cm])
                    t, df = corrected_t(dd, 0.0, r)
                    p = float(2 * stats.t.sf(abs(t), df)) if np.isfinite(t) else (0.0 if t != 0 else 1.0)
                    entry["vs_ref"][m] = {"n": len(cm), "mean": float(dd.mean()), "p": p}
        res["graphs"][g] = entry
    # NI of EDSP-B (IUT), harm per comparator (Holm), EDSP vs comparator (Holm), per method
    summary: dict = {}
    for m in methods:
        gs = [g for g in graphs if m in res["graphs"][g]["vs_base"]]
        ni_pass = [g for g in gs if res["graphs"][g]["vs_base"][m]["ni"]["p"] < args.alpha]
        hp = holm([res["graphs"][g]["vs_base"][m]["harm"]["p"] for g in gs])
        harmed = [g for g, p in zip(gs, hp, strict=True) if p < args.alpha]
        for g, p in zip(gs, hp, strict=True):
            res["graphs"][g]["vs_base"][m]["harm"]["p_holm"] = float(p)
        summary[m] = {"graphs": len(gs), "ni_pass": ni_pass, "ni_fail": [g for g in gs if g not in ni_pass],
                      "iut_all_ni": len(ni_pass) == len(gs) and len(gs) > 0, "harm_holm": harmed}
        if m != REF:
            gv = [g for g in graphs if m in res["graphs"][g]["vs_ref"]]
            adj = holm([res["graphs"][g]["vs_ref"][m]["p"] for g in gv])
            w, t_, lo = [], [], []
            for g, p in zip(gv, adj, strict=True):
                x = res["graphs"][g]["vs_ref"][m]
                x["p_holm"] = float(p)
                (w if p < 0.05 and x["mean"] > 0 else lo if p < 0.05 and x["mean"] < 0 else t_).append(g)
            summary[m]["edsp_vs"] = {"win": w, "tie": t_, "loss": lo}
    res["summary"] = summary

    def fmt(x: dict | None, key: str) -> str:
        if not x:
            return "—"
        p = x[key]["p"]
        return f"{x['mean']:+.2f} ({x['sd']:.2f}; p {p:.2g})"

    md = [f"# A18 post-hoc smoothers — {args.label}", "",
          (f"Base = `{BASE}`; d = method − base per split (headline metric ×100). NI margin δ = {args.delta}; "
           f"one-sided α = {args.alpha}; Nadeau–Bengio corrected t (default n_test/n_train = {args.ratio:.3g}); "
           f"heterophilous iff adjusted homophily < {args.het_threshold}."
           + (f" Left out: {', '.join(sorted(args.exclude_split))}." if args.exclude_split else "")), "",
          "Mean d (sd; one-sided NI p). NI holds where p < α.", "",
          "| graph [metric] | h_adj | " + " | ".join(methods) + " |", "|---|---|" + "---|" * len(methods)]
    for g in graphs:
        e = res["graphs"][g]
        h = "—" if e["adjusted_homophily"] is None else f"{e['adjusted_homophily']:.2f}"
        md.append(f"| {g} [{e['metric']}] | {h} | " + " | ".join(fmt(e["vs_base"].get(m), "ni") for m in methods) + " |")
    md += ["", "| method | graphs | NI holds | NI fails | significant harm (> δ below base, Holm) | EDSP-B-tuned vs it: win / tie / loss |",
           "|---|---|---|---|---|---|"]
    for m in methods:
        s = summary[m]
        wtl = s.get("edsp_vs")
        md.append(f"| {m} | {s['graphs']} | {len(s['ni_pass'])} | {', '.join(s['ni_fail']) or '—'} | "
                  f"{', '.join(s['harm_holm']) or '—'} | "
                  + ("—" if wtl is None else f"{len(wtl['win'])} / {len(wtl['tie'])} / {len(wtl['loss'])}") + " |")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "A18_SUMMARY.md").write_text("\n".join(md) + "\n")
    (out / "a18.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
