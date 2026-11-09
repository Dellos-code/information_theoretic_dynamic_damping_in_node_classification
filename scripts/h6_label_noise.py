"""H6 (PREREGISTRATION A16): does finite label trust T_L help under label noise?

Reads E6 run records (dataset names ``<graph>~ln-<mode>-<rate>``) and applies A16:
per graph and condition, EDSP-TL vs EDSP-default, two-sided Wilcoxon over splits, Holm
across the 14 graphs within each condition. Primary conditions uniform:0.3 and
pairwise:0.3: supported iff both have >= 5 wins and <= 1 loss; not supported iff both
have <= 1 win; else inconclusive. Secondary: every rate, the clean control
(uniform:0), the T_L chosen, and EDSP-TL against LGC / harmonic under the same noise.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edsp.eval.stats_tests import holm, wilcoxon_paired

GRAPHS = ["amazon-computers", "amazon-photo", "amazon-ratings", "chameleon-filtered", "citeseer", "coauthor-cs",
          "coauthor-physics", "cora", "pubmed", "questions", "roman-empire", "squirrel-filtered", "tolokers", "wikics"]
PRIMARY = [("uniform", 0.3), ("pairwise", 0.3)]
NAME = re.compile(r"^(?P<g>.+)~ln-(?P<m>uniform|pairwise)-(?P<r>[0-9.]+)$")


def load(dirs: list[str]) -> tuple[dict, dict]:
    vals: dict = defaultdict(lambda: defaultdict(dict))  # (graph, mode, rate) -> method -> split -> value
    chosen: dict = defaultdict(Counter)
    for d in dirs:
        for f in sorted(Path(d).rglob("*.jsonl")):
            if f.name.startswith("ACQUISITION_LOG"):
                continue
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                m = NAME.match(str(r.get("dataset", "")))
                if not m or r.get("split") is None or r.get("method") in (None, "DATASET-STATS", "NOT-RUN"):
                    continue
                key = (m["g"], m["m"], float(m["r"]))
                vals[key][r["method"]].setdefault(str(r["split"]), r["metrics"][r["headline_metric"]])
                if r["method"] == "EDSP-TL":
                    chosen[key][str(r["config"]["T_L"])] += 1
    return vals, chosen


def compare(a: dict, b: dict) -> tuple[int, float, float]:
    ks = sorted(set(a) & set(b))
    x, y = np.array([a[k] for k in ks]), np.array([b[k] for k in ks])
    return len(ks), float(x.mean() - y.mean()) if ks else float("nan"), wilcoxon_paired(x, y) if ks else float("nan")


def verdicts(vals: dict, method: str, ref: str) -> dict:
    out = {}
    for cond in sorted({(mo, ra) for (_, mo, ra) in vals}):
        rows = []
        for g in GRAPHS:
            M = vals.get((g, *cond), {})
            if method in M and ref in M:
                rows.append((g, *compare(M[method], M[ref])))
        adj = holm([p for *_, p in rows])
        res = []
        for (g, n, d, p), q in zip(rows, adj, strict=True):
            v = "tie" if not np.isfinite(q) or q >= 0.05 else ("win" if d > 0 else "loss")
            res.append({"graph": g, "pairs": n, "delta": d, "p_holm": q, "verdict": v})
        out[f"{cond[0]}:{cond[1]:g}"] = res
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=str(ROOT / "results" / "aggregates" / "e6"))
    args = ap.parse_args()
    vals, chosen = load(args.runs)
    tl = verdicts(vals, "EDSP-TL", "EDSP-default")
    ctx = {b: verdicts(vals, "EDSP-TL", b) for b in ("lgc", "harmonic")}

    def count(res):
        c = Counter(r["verdict"] for r in res)
        return c["win"], c["tie"], c["loss"], len(res)

    prim = [count(tl.get(f"{mo}:{ra:g}", [])) for mo, ra in PRIMARY]
    complete = all(n == len(GRAPHS) for *_, n in prim)
    if all(w >= 5 and lo <= 1 for w, _, lo, _ in prim):
        verdict = "SUPPORTED"
    elif all(w <= 1 for w, _, _, _ in prim):
        verdict = "NOT SUPPORTED"
    else:
        verdict = "INCONCLUSIVE"
    if not complete:
        verdict += " (provisional: primary conditions incomplete)"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"verdict": verdict, "primary": dict(zip([f"{m}:{r:g}" for m, r in PRIMARY], prim, strict=True)),
               "edsp_tl_vs_default": tl, "context": ctx,
               "chosen_T_L": {f"{g}~{mo}:{ra:g}": dict(c) for (g, mo, ra), c in chosen.items()}}
    (out / "h6.json").write_text(json.dumps(summary, indent=1))
    md = ["# H6 — finite label trust under label noise (PREREGISTRATION A16)", "",
          f"**Verdict: {verdict}.** Primary conditions (EDSP-TL vs EDSP-default, win / tie / loss of "
          f"{len(GRAPHS)} graphs, Holm): " + "; ".join(f"{m}:{r:g} → {w}/{t}/{lo} (n={n})"
                                                       for (m, r), (w, t, lo, n) in zip(PRIMARY, prim, strict=True)), "",
          "## EDSP-TL vs EDSP-default by condition (win / tie / loss; mean Δ ×100 over graphs)", "",
          "| condition | graphs | win | tie | loss | mean Δ | EDSP-TL vs LGC (w/t/l) | EDSP-TL vs harmonic (w/t/l) |",
          "|---|---|---|---|---|---|---|---|"]
    for cond, res in tl.items():
        w, t, lo, n = count(res)
        md.append(f"| {cond} | {n} | {w} | {t} | {lo} | {100 * np.nanmean([r['delta'] for r in res]):+.2f} | "
                  + " | ".join("{}/{}/{}".format(*count(ctx[b].get(cond, []))[:3]) for b in ("lgc", "harmonic")) + " |")
    md += ["", "## Per graph, primary conditions (Δ ×100, Holm p)", "", "| graph | " + " | ".join(
        f"{m}:{r:g}" for m, r in PRIMARY) + " |", "|---|" + "---|" * len(PRIMARY)]
    for g in GRAPHS:
        cells = []
        for m, r in PRIMARY:
            hit = [x for x in tl.get(f"{m}:{r:g}", []) if x["graph"] == g]
            cells.append(f"{100 * hit[0]['delta']:+.2f} (p={hit[0]['p_holm']:.3g}, {hit[0]['verdict']})" if hit else "NOT-RUN")
        md.append(f"| {g} | " + " | ".join(cells) + " |")
    md += ["", "## T_L chosen by validation (EDSP-TL), counts over splits", ""]
    agg: dict = defaultdict(Counter)
    for (_, mo, ra), c in chosen.items():
        agg[f"{mo}:{ra:g}"].update(c)
    md += [f"- {cond}: " + ", ".join(f"T_L={k}: {v}" for k, v in sorted(c.items())) for cond, c in sorted(agg.items())]
    (out / "H6_SUMMARY.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
