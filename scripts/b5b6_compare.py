"""B5 / B6 "ours vs SOTA" comparison (docs/PREREGISTRATION.md A17).

B5: EDSP-tuned (Track A, records of results/runs/e2) against the signed GNNs SGCN, SNEA, SDGNN
and SiGAT (results/runs/b5). B6: EDSP-B-tuned (Track B; amazon-fraud from results/runs/e3,
yelpchi from results/runs/b6) against CARE-GNN and PC-GNN (results/runs/b6).

Per graph: two-sided Wilcoxon signed-rank over the common splits on the headline metric;
Holm across the ranked graphs of the family, separately per method; win / loss if the adjusted
p < 0.05, else tie. Ranked graphs: the 8 Congress graphs and wiki-editor (signed); amazon-fraud
and yelpchi (fraud). wiki-rfa, wiki-elec and sampson are case studies: their Δ and raw p are
shown, never counted. Context columns (other methods already run on the same splits) are
descriptive only.

Output: results/aggregates/b5b6/B5B6_SUMMARY.md and b5b6.json.
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

from edsp.eval.stats_tests import friedman_nemenyi, holm, wilcoxon_paired

SIGNED_RANKED = [f"congress-{c}-{n}" for c in ("house", "senate") for n in (93, 100, 107, 114)] + ["wiki-editor"]
SIGNED_CASE = ["wiki-rfa", "wiki-elec", "sampson"]
FRAUD_RANKED = ["amazon-fraud", "yelpchi"]
B5 = ["sgcn", "snea", "sdgnn", "sigat"]
B6 = ["care_gnn", "pc_gnn"]
B5_CONTEXT = ["EDSP-default", "signed_vote", "signed_lgc", "lgc", "harmonic_cmn", "linbp"]
B6_CONTEXT = ["EDSP-default", "feature_lr", "correct_and_smooth", "mlp", "bwgnn"]
LABEL = {"EDSP-tuned": "EDSP-tuned", "care_gnn": "CARE-GNN", "pc_gnn": "PC-GNN", "sgcn": "SGCN", "snea": "SNEA",
         "sdgnn": "SDGNN", "sigat": "SiGAT", "correct_and_smooth": "C&S", "feature_lr": "LR", "mlp": "MLP",
         "bwgnn": "BWGNN", "signed_vote": "signed_vote", "signed_lgc": "signed_lgc", "lgc": "LGC",
         "harmonic_cmn": "harmonic+CMN", "linbp": "LinBP", "EDSP-default": "EDSP-default"}


def load(dirs: list[Path], suite: str) -> tuple[dict, dict, list]:
    by: dict = defaultdict(dict)  # (dataset, method) -> {split: record}
    metric, not_run = {}, []
    for d in dirs:
        for f in sorted(d.glob("*.jsonl")):
            if f.name.startswith("ACQUISITION_LOG"):
                continue
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("suite") != suite or r.get("method") == "DATASET-STATS":
                    continue
                if r.get("method") == "NOT-RUN":
                    not_run.append(r)
                    continue
                key = (r["dataset"], r["method"])
                if r["split"] in by[key] and by[key][r["split"]]["metrics"] != r["metrics"]:
                    raise SystemExit(f"conflicting duplicate {key} {r['split']} across {dirs}")
                by[key][r["split"]] = r
                metric[r["dataset"]] = r["headline_metric"]
    return by, metric, not_run


def family(by: dict, metric: dict, ranked: list[str], case: list[str], new: list[str], context: list[str], ref: str) -> dict:
    def vals(d: str, m: str) -> dict:
        return {s: r["metrics"][metric[d]] for s, r in by.get((d, m), {}).items()}

    out: dict = {"graphs": {}, "wtl": {}}
    for d in ranked + case:
        if d not in metric:
            continue
        row = {}
        for m in [ref, *new, *context]:
            v = vals(d, m)
            if v:
                a = np.array(list(v.values()), dtype=float)
                row[m] = {"mean": float(100 * a.mean()), "std": float(100 * a.std()), "n": len(a)}
        tests = {}
        e = vals(d, ref)
        for m in new:
            b = vals(d, m)
            common = sorted(set(e) & set(b))
            if len(common) < 2:
                continue
            x, y = np.array([e[s] for s in common]), np.array([b[s] for s in common])
            tests[m] = {"delta": float(100 * np.mean(x - y)), "p": float(wilcoxon_paired(x, y)), "n": len(common)}
        best = max(row, key=lambda m: row[m]["mean"]) if row else None
        out["graphs"][d] = {"metric": metric[d], "ranked": d in ranked, "methods": row, "tests": tests, "best": best,
                            "ref_rank": 1 + sorted(-row[m]["mean"] for m in row).index(-row[ref]["mean"]) if ref in row else None}
    for m in new:
        used = [d for d in ranked if d in out["graphs"] and m in out["graphs"][d]["tests"]]
        adj = holm([out["graphs"][d]["tests"][m]["p"] for d in used])
        w, lo, t = [], [], []
        for d, p in zip(used, adj, strict=True):
            out["graphs"][d]["tests"][m]["p_holm"] = float(p)
            delta = out["graphs"][d]["tests"][m]["delta"]
            (w if p < 0.05 and delta > 0 else lo if p < 0.05 and delta < 0 else t).append(d)
        out["wtl"][m] = {"win": w, "tie": t, "loss": lo, "graphs": len(used)}
    return out


def table(res: dict, ref: str, new: list[str], context: list[str], refname: str) -> list[str]:
    cols = [ref, *new, *context]
    md = ["| graph [metric] | " + " | ".join(LABEL[c] if c != ref else refname for c in cols) + " | EDSP rank |",
          "|---|" + "---|" * (len(cols) + 1)]
    for d, g in res["graphs"].items():
        cells = []
        for c in cols:
            x = g["methods"].get(c)
            if x is None:
                cells.append("—")
                continue
            s = f"{x['mean']:.2f} ± {x['std']:.2f}"
            cells.append(f"**{s}**" if c == g["best"] else s)
        tag = "" if g["ranked"] else " (case study)"
        md.append(f"| {d}{tag} [{g['metric']}] | " + " | ".join(cells) + f" | {g['ref_rank']} / {len(g['methods'])} |")
    md += ["", "| graph | " + " | ".join(f"Δ vs {LABEL[m]} (p; p Holm; verdict)" for m in new) + " |",
           "|---|" + "---|" * len(new)]
    for d, g in res["graphs"].items():
        cells = []
        for m in new:
            t = g["tests"].get(m)
            if t is None:
                cells.append("—")
                continue
            if "p_holm" in t:
                v = "win" if t["p_holm"] < 0.05 and t["delta"] > 0 else "loss" if t["p_holm"] < 0.05 and t["delta"] < 0 else "tie"
                cells.append(f"{t['delta']:+.2f} ({t['p']:.3g}; {t['p_holm']:.3g}; {v})")
            else:
                cells.append(f"{t['delta']:+.2f} ({t['p']:.3g}; —; case study)")
        md.append(f"| {d} | " + " | ".join(cells) + " |")
    md += ["", f"| {refname} vs | win | tie | loss | win graphs | loss graphs |", "|---|---|---|---|---|---|"]
    for m in new:
        x = res["wtl"][m]
        md.append(f"| {LABEL[m]} | {len(x['win'])} | {len(x['tie'])} | {len(x['loss'])} | {', '.join(x['win']) or '—'} | {', '.join(x['loss']) or '—'} |")
    return md


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "aggregates" / "b5b6"))
    args = ap.parse_args()
    runs = ROOT / "results" / "runs"
    by5, met5, nr5 = load([runs / "e2", runs / "b5"], "E2")
    by6, met6, nr6 = load([runs / "e3", runs / "b6"], "E3")
    met6 = {d: m for d, m in met6.items() if d in FRAUD_RANKED}
    r5 = family(by5, met5, SIGNED_RANKED, SIGNED_CASE, B5, B5_CONTEXT, "EDSP-tuned")
    r6 = family(by6, met6, FRAUD_RANKED, [], B6, B6_CONTEXT, "EDSP-tuned")
    shas = sorted({r["run_record"]["git_sha"][:12] for by in (by5, by6) for (_, m), v in by.items()
                   if m in B5 + B6 for r in v.values()})
    # exploratory (not preregistered; A17 leaves B5 / B6 out of the ranks): average ranks of EDSP-tuned, the signed
    # GNNs and the E2 signed baselines over the ranked signed graphs
    rank_m = ["EDSP-tuned", *B5, "signed_vote", "signed_lgc"]
    rank_g = [d for d in SIGNED_RANKED if d in r5["graphs"] and all(m in r5["graphs"][d]["methods"] for m in rank_m)]
    fn = friedman_nemenyi(np.array([[r5["graphs"][d]["methods"][m]["mean"] for m in rank_m] for d in rank_g]))
    r5["exploratory_ranks"] = {"methods": rank_m, "graphs": rank_g, "friedman_p": float(fn["friedman_p"]),
                               "nemenyi_cd": float(fn["nemenyi_cd"]),
                               "average_ranks": {m: float(x) for m, x in zip(rank_m, fn["average_ranks"], strict=True)}}
    ranks_md = ["### Exploratory: average ranks on the ranked signed graphs (not preregistered)", "",
                (f"{len(rank_g)} graphs; Friedman p = {fn['friedman_p']:.3g}; Nemenyi CD (α = 0.05) = {fn['nemenyi_cd']:.2f}. "
                 + ", ".join(f"{LABEL[rank_m[i]]} {fn['average_ranks'][i]:.2f}" for i in np.argsort(fn["average_ranks"])) + ".")]
    md = ["# B5 / B6 — EDSP against signed GNNs and fraud GNNs (PREREGISTRATION A17)", "",
          (f"New-method records from git {', '.join(shas)}. Mean ± std over splits (×100); bold = best mean among the "
           "columns shown. Δ = mean paired difference EDSP − method (×100); p = two-sided Wilcoxon over splits; "
           "Holm across the ranked graphs of the family, separately per method; win / loss if Holm-adjusted p < 0.05. "
           "'EDSP rank' counts only the columns shown."), "",
          "## B5 — signed graphs (Track A: structure + train labels)", "",
          *table(r5, "EDSP-tuned", B5, B5_CONTEXT, "EDSP-tuned"), "", *ranks_md, "",
          "## B6 — fraud graphs (Track B: node features)", "",
          *table(r6, "EDSP-tuned", B6, B6_CONTEXT, "EDSP-B-tuned"), "",
          (f"NOT-RUN records: B5 {len([r for r in nr5 if r.get('baseline') in B5])}, "
           f"B6 {len([r for r in nr6 if r.get('baseline') in B6 and r['dataset'] in FRAUD_RANKED])}.")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "B5B6_SUMMARY.md").write_text("\n".join(md) + "\n")
    (out / "b5b6.json").write_text(json.dumps({"git_sha": shas, "B5": r5, "B6": r6}, indent=1, sort_keys=True) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
