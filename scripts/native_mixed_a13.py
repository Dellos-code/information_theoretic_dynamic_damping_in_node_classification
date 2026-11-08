"""A13: native mixing — does the MIX k = 2 advantage appear on real graphs that mix by themselves?

Reads E1 run records of the "native-mixed" registry extension (run with
`--learned-sign-baselines --h7-splits 0`) and applies the rules fixed in
docs/PREREGISTRATION.md A13:

(a) transfer: per dataset, EDSP-tuned vs the best (by mean) sign-blind baseline among
    harmonic, harmonic_cmn, lgc, ppr_lp, linbp; two-sided Wilcoxon over splits; Holm across
    the 13 k = 2 datasets. Supported iff >= 3 wins, wins > losses and wins in >= 2 of the 4
    source families (FB100, twitch, genius, deezer); not supported iff <= 1 win; else
    inconclusive. A dataset without records counts as NOT-RUN (no win).
(b) attribution: on every (a)-win dataset, EDSP-tuned vs the better (by mean) of
    signed_vote+HL / signed_lgc+HL; Wilcoxon over splits; Holm over those datasets.
actor (k = 5) is reported, not tested.
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

TESTED = ["fb100-penn94", "fb100-amherst41", "fb100-cornell5", "fb100-johnshopkins55", "fb100-reed98",
          "genius", "deezer-europe", "twitch-de", "twitch-en", "twitch-es", "twitch-fr", "twitch-pt", "twitch-ru"]
REPORTED = ["actor"]
SIGN_BLIND = ["harmonic", "harmonic_cmn", "lgc", "ppr_lp", "linbp"]
HL = ["signed_vote+HL", "signed_lgc+HL"]
SHOWN = ["EDSP-default", "EDSP-tuned", "majority", "train_prior", *SIGN_BLIND, *HL]


def family(d: str) -> str:
    return "FB100" if d.startswith("fb100-") else "twitch" if d.startswith("twitch-") else d


def load(dirs: list[str]) -> dict:
    vals: dict = defaultdict(lambda: defaultdict(dict))
    metric: dict = {}
    for d in dirs:
        for f in sorted(Path(d).rglob("*.jsonl")):
            if f.name.startswith("ACQUISITION_LOG"):
                continue
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                ds, m = r.get("dataset"), r.get("method")
                if ds not in TESTED + REPORTED or m in (None, "DATASET-STATS", "NOT-RUN") or r.get("split") is None:
                    continue
                v = r["metrics"].get(r["headline_metric"])
                if v is None:
                    continue
                metric[ds] = r["headline_metric"]
                vals[ds][m].setdefault(str(r["split"]), v)
    return vals, metric


def compare(a: dict, b: dict) -> tuple[int, float, float]:
    keys = sorted(set(a) & set(b))
    x = np.array([a[k] for k in keys])
    y = np.array([b[k] for k in keys])
    return len(keys), float(x.mean() - y.mean()), wilcoxon_paired(x, y)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=str(ROOT / "results" / "aggregates" / "native-mixed"))
    args = ap.parse_args()
    vals, metric = load(args.runs)
    rows = {}
    for ds in TESTED + REPORTED:
        M = vals.get(ds, {})
        if "EDSP-tuned" not in M:
            rows[ds] = {"dataset": ds, "not_run": True}
            continue
        means = {m: float(np.mean(list(M[m].values()))) for m in SHOWN if m in M}
        best_sb = max((m for m in SIGN_BLIND if m in M), key=lambda m: means[m])
        n, d, p = compare(M["EDSP-tuned"], M[best_sb])
        row = {"dataset": ds, "family": family(ds), "metric": metric[ds], "splits": len(M["EDSP-tuned"]),
               "means": means, "best_sign_blind": best_sb, "a_pairs": n, "a_delta": d, "a_p_raw": p}
        hl = [m for m in HL if m in M]
        if hl:
            best_hl = max(hl, key=lambda m: means[m])
            n2, d2, p2 = compare(M["EDSP-tuned"], M[best_hl])
            row.update({"best_hl": best_hl, "b_pairs": n2, "b_delta": d2, "b_p_raw": p2})
        rows[ds] = row
    tested = [rows[d] for d in TESTED if not rows[d].get("not_run")]
    for r, p in zip(tested, holm([r["a_p_raw"] for r in tested])):
        r["a_p_holm"] = p
        r["a_verdict"] = "tie" if not np.isfinite(p) or p >= 0.05 else ("win" if r["a_delta"] > 0 else "loss")
    wins = [r for r in tested if r["a_verdict"] == "win"]
    for r, p in zip(wins, holm([r.get("b_p_raw", float("nan")) for r in wins])):
        r["b_p_holm"] = p
        r["b_verdict"] = "tie" if not np.isfinite(p) or p >= 0.05 else ("win" if r["b_delta"] > 0 else "loss")
    n_win = len(wins)
    n_loss = sum(r["a_verdict"] == "loss" for r in tested)
    fams = sorted({r["family"] for r in wins})
    not_run = [d for d in TESTED if rows[d].get("not_run")]
    if n_win >= 3 and n_win > n_loss and len(fams) >= 2:
        verdict = "SUPPORTED"
    elif n_win <= 1:
        verdict = "NOT SUPPORTED"
    else:
        verdict = "INCONCLUSIVE"
    if not_run and verdict != "NOT SUPPORTED":
        verdict += f" (provisional: {len(not_run)} of 13 datasets NOT-RUN so far)"
    summary = {"verdict": verdict, "wins": n_win, "losses": n_loss, "ties": len(tested) - n_win - n_loss,
               "win_families": fams, "not_run": not_run, "rows": rows}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "a13.json").write_text(json.dumps(summary, indent=1))
    md = ["# A13 — native mixing (real graphs; E1 Track A, `--learned-sign-baselines`, `--h7-splits 0`)", "",
          f"**(a) transfer verdict: {verdict}** — {n_win} wins, {summary['ties']} ties, {n_loss} losses "
          f"over {len(tested)} tested datasets; win families: {', '.join(fams) or '—'}"
          + (f"; NOT-RUN: {', '.join(not_run)}" if not_run else "") + ".", "",
          ("Means ×100 over splits (headline metric per dataset). (a): EDSP-tuned vs the best sign-blind "
           "baseline (by mean), Wilcoxon over splits, Holm across the tested datasets. (b) only for (a) wins: "
           "EDSP-tuned vs the better of the two +HL baselines, Holm over those datasets."), "",
          "| dataset | metric | splits | EDSP-def | EDSP-tuned | majority | " + " | ".join(SIGN_BLIND + HL)
          + " | best sign-blind | Δ (a) | p Holm (a) | (a) | Δ (b) | p Holm (b) | (b) |",
          "|---|---|---|" + "---|" * (3 + len(SIGN_BLIND) + len(HL)) + "---|---|---|---|---|---|---|"]
    for ds in TESTED + REPORTED:
        r = rows[ds]
        if r.get("not_run"):
            md.append(f"| {ds} | NOT-RUN |" + " |" * (19))
            continue
        mm = r["means"]

        def f(m, mm=mm):
            return f"{100 * mm[m]:.2f}" if m in mm else "—"

        tested_row = ds in TESTED
        md.append(
            f"| {ds}{'' if tested_row else ' (not tested)'} | {r['metric']} | {r['splits']} | {f('EDSP-default')} | "
            f"{f('EDSP-tuned')} | {f('majority')} | " + " | ".join(f(m) for m in SIGN_BLIND + HL)
            + f" | {r['best_sign_blind']} | {100 * r['a_delta']:+.2f} | "
            + (f"{r['a_p_holm']:.3g} | {r['a_verdict']}" if tested_row else f"{r['a_p_raw']:.3g} (raw) | —")
            + (f" | {100 * r['b_delta']:+.2f} | {r['b_p_holm']:.3g} | {r['b_verdict']} |" if "b_verdict" in r else " | — | — | — |"))
    (out / "A13_SUMMARY.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
