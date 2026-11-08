"""A11: whose win is it in MIX? EDSP-tuned vs simple signed propagation fed H-LEARNED's polarity.

Input: MIX run records that include the A10 ablation methods (`signed_vote+HL`,
`signed_lgc+HL`; produced by `run_e1.py --suite MIX --learned-sign-baselines`), and
optionally the original A4 / A8 records for the determinism check.

Per (base, condition): EDSP-tuned minus the better (by mean) of the two +HL baselines,
two-sided Wilcoxon over the paired (seed, split) runs, Holm over the conditions of a
base graph within each family (loc = 0.8 and loc = 0 separately). PERTURBED-REAL
(rule R4): never enters the real-world tables.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edsp.eval.stats_tests import holm, wilcoxon_paired

HL = ["signed_vote+HL", "signed_lgc+HL"]


def spec_of(r: dict) -> str:
    p = r["perturbation"]
    return p["mode"] + (":" + ",".join(f"{k}={v:g}" for k, v in p["params"].items()) if p["params"] else "")


def load(dirs: list[str]) -> dict:
    """(base, spec, method) -> {(seed, split): headline value}."""
    vals: dict = defaultdict(dict)
    for d in dirs:
        for f in sorted(Path(d).glob("*.jsonl")):
            if f.name.startswith("ACQUISITION_LOG"):
                continue
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("suite") != "MIX" or not r.get("perturbation") or r["method"] in ("DATASET-STATS", "NOT-RUN"):
                    continue
                vals[(r["base_dataset"], spec_of(r), r["method"])][(r["perturbation"]["seed"], r["split"])] = (
                    r["metrics"][r["headline_metric"]])
    return vals


def family(spec: str) -> str:
    """A8 conditions carry an explicit loc=0; A4 conditions use the default locality 0.8."""
    return "loc = 0" if re.search(r"loc=0(?![.\d])", spec) else "loc = 0.8"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="directories with the A11 run records")
    ap.add_argument("--reference", nargs="*", default=[], help="directories with the original A4 / A8 records")
    ap.add_argument("--out", default=str(ROOT / "results" / "perturbed" / "attribution" / "aggregates"))
    args = ap.parse_args()
    vals = load(args.runs)
    ref = load(args.reference) if args.reference else {}
    cells = sorted({(b, s) for (b, s, m) in vals if m == "EDSP-tuned"})
    rows = []
    for base, spec in cells:
        e = vals[(base, spec, "EDSP-tuned")]
        hl = {m: vals.get((base, spec, m), {}) for m in HL}
        hl = {m: v for m, v in hl.items() if v}
        if not hl:
            rows.append({"base": base, "spec": spec, "family": family(spec), "missing": "no +HL records"})
            continue
        best = max(hl, key=lambda m: np.mean(list(hl[m].values())))
        keys = sorted(set(e) & set(hl[best]))
        a = np.array([e[k] for k in keys])
        b = np.array([hl[best][k] for k in keys])
        row = {"base": base, "spec": spec, "family": family(spec), "pairs": len(keys),
               "edsp_tuned": float(a.mean()), "best_hl": best, "best_hl_mean": float(b.mean()),
               "other_hl_means": {m: float(np.mean(list(v.values()))) for m, v in hl.items()},
               "delta": float(a.mean() - b.mean()), "p_raw": wilcoxon_paired(a, b)}
        r0 = ref.get((base, spec, "EDSP-tuned"))
        if r0:
            common = sorted(set(r0) & set(e))
            row["determinism"] = {"compared": len(common),
                                  "max_abs_diff": float(max((abs(r0[k] - e[k]) for k in common), default=float("nan")))}
        rows.append(row)
    for base in sorted({r["base"] for r in rows}):
        for fam in ("loc = 0.8", "loc = 0"):
            grp = [r for r in rows if r["base"] == base and r["family"] == fam and "p_raw" in r]
            for r, p in zip(grp, holm([r["p_raw"] for r in grp])):
                r["p_holm"] = p
                r["verdict"] = ("tie" if not np.isfinite(p) or p >= 0.05 else ("win" if r["delta"] > 0 else "loss"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "attribution.json").write_text(json.dumps(rows, indent=1))
    md = ["# A11 — whose win is it in MIX? (PERTURBED-REAL, not real-world results)", "",
          ("EDSP-tuned vs the better (by mean) of `signed_vote+HL` / `signed_lgc+HL` on the same re-wired "
           "graphs; ×100, mean over (seed, split) pairs; Wilcoxon over pairs, Holm over the conditions of a "
           "base within each family. Win = EDSP's propagation adds value beyond H-LEARNED's polarity."), "",
          "| base | family | condition | pairs | EDSP-tuned | signed_vote+HL | signed_lgc+HL | Δ | p (Holm) | verdict | determinism (max |Δ| vs A4/A8) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "missing" in r:
            md.append(f"| {r['base']} | {r['family']} | {r['spec']} | — | — | — | — | — | — | NOT-RUN: {r['missing']} | — |")
            continue
        o = r["other_hl_means"]
        det = r.get("determinism")
        md.append(f"| {r['base']} | {r['family']} | {r['spec']} | {r['pairs']} | {100 * r['edsp_tuned']:.2f} | "
                  + " | ".join(f"{100 * o[m]:.2f}" if m in o else "—" for m in HL)
                  + f" | {100 * r['delta']:+.2f} | {r.get('p_holm', float('nan')):.3g} | {r.get('verdict', '—')} | "
                  + (f"{det['max_abs_diff']:.2e} over {det['compared']}" if det else "—") + " |")
    (out / "ATTRIBUTION_SUMMARY.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
