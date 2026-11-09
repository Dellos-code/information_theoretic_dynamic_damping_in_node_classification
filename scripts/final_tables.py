"""LaTeX tables for the final report, generated only from committed aggregate files.

Writes docs/final/*.tex and prints a plain-text digest of every number used, so the report
text can be checked against it. No number is typed by hand here: each table reads a
results/aggregates/**.json (or results/perturbed/**.json) file.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGG = ROOT / "results" / "aggregates"
OUT = ROOT / "docs" / "final"

NAMES = {"harmonic": "Harmonic", "harmonic_cmn": "Harmonic+CMN", "lgc": "LGC", "ppr_lp": "PPR-LP",
         "linbp": "LinBP", "train_prior": "Train prior", "signed_vote": "Signed vote", "signed_lgc": "Signed LGC",
         "feature_lr": "LogReg (base)", "correct_and_smooth": "C\\&S", "mlp": "MLP", "gcn": "GCN+", "sage": "SAGE+",
         "gat": "GAT+", "gprgnn": "GPR-GNN", "bwgnn": "BWGNN", "EDSP-tuned": "EDSP-tuned",
         "EDSP-default": "EDSP-default", "majority": "Majority"}


def tex_escape(s: str) -> str:
    return s.replace("_", "\\_").replace("&", "\\&") if "\\&" not in s else s.replace("_", "\\_")


def load(path: Path) -> dict | list:
    return json.loads(path.read_text())


def wtl_table(suites: list[tuple[str, str, str]]) -> tuple[str, list[str]]:
    """suites: (label, aggregate json path relative to AGG, caption note)."""
    rows, digest = [], []
    for label, rel, _ in suites:
        agg = load(AGG / rel)
        for b, v in agg["win_tie_loss_edsp_tuned"].items():
            w, t, lo = len(v["win"]), v["tie"], len(v["loss"])
            if w + t + lo == 0:
                continue
            rows.append(f"{label} & {NAMES.get(b, tex_escape(b))} & {w} & {t} & {lo} \\\\")
            digest.append(f"WTL {label} {b}: {w}/{t}/{lo}")
    tex = ["\\begin{tabular}{llrrr}", "\\toprule",
           "Suite & Baseline & Win & Tie & Loss \\\\", "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"]
    return "\n".join(tex) + "\n", digest


def ranks_table(suites: list[tuple[str, str]]) -> tuple[str, list[str]]:
    blocks, digest = [], []
    for label, rel in suites:
        fn = load(AGG / rel).get("friedman_nemenyi")
        if not fn:
            continue
        pairs = sorted(zip(fn["methods"], fn["average_ranks"], strict=True), key=lambda x: x[1])
        cells = ", ".join(f"{NAMES.get(m, tex_escape(m))} {r:.2f}" for m, r in pairs)
        blocks.append(f"{label} ({len(fn['datasets'])} graphs, Friedman $p={fn['friedman_p']:.2g}$) & {cells} \\\\")
        digest += [f"RANK {label} {m}: {r:.2f}" for m, r in pairs]
        digest.append(f"RANK {label} datasets={len(fn['datasets'])} friedman_p={fn['friedman_p']:.3g}")
    tex = ["\\begin{tabular}{lp{11cm}}", "\\toprule", "Suite & Average rank (lower is better) \\\\", "\\midrule",
           *blocks, "\\bottomrule", "\\end{tabular}"]
    return "\n".join(tex) + "\n", digest


def h4_table() -> tuple[str, list[str]]:
    rows, digest = [], []
    for mode, rel in (("GAT+(h64)", "e3/e3_aggregates.json"), ("GAT+ omitted", "e3-gat-omitted/e3_aggregates.json")):
        h4 = load(AGG / rel)["H4"]
        for d, bg, d1, _p1, d2, _p2 in h4["rows"]:
            digest.append(f"H4 [{mode}] {d}: best={bg} EDSP-C&S={100 * d1:+.2f} GNN-EDSP={100 * d2:+.2f}")
        digest.append(f"H4 [{mode}] same_as_cs={h4['same_as_cs']}/{h4['n']} gnn_better={h4['gnn_better']}/{h4['n']}")
    h4 = load(AGG / "e3/e3_aggregates.json")["H4"]
    for d, bg, d1, _p1, d2, _p2 in h4["rows"]:
        rows.append(f"{tex_escape(d)} & {NAMES.get(bg, bg)} & {100 * d1:+.2f} & {100 * d2:+.2f} \\\\")
    tex = ["\\begin{tabular}{llrr}", "\\toprule",
           "Graph (C2) & Best GNN & EDSP$-$C\\&S & GNN$-$EDSP \\\\", "\\midrule", *rows, "\\bottomrule",
           "\\end{tabular}"]
    return "\n".join(tex) + "\n", digest


def a13_table() -> tuple[str, list[str]]:
    a = load(AGG / "native-mixed" / "a13.json")
    rows, digest = [], [f"A13 verdict: {a['verdict']} wins={a['wins']} ties={a['ties']} losses={a['losses']}"]
    for d, r in a["rows"].items():
        if r.get("not_run") or "a_verdict" not in r:
            continue
        m = r["means"]
        b = f"{100 * r['b_delta']:+.2f} ({r['b_verdict']})" if "b_verdict" in r else "--"
        rows.append(f"{tex_escape(d)} & {100 * m['EDSP-tuned']:.2f} & {NAMES.get(r['best_sign_blind'])} "
                    f"{100 * m[r['best_sign_blind']]:.2f} & {100 * r['a_delta']:+.2f} ({r['a_verdict']}) & {b} \\\\")
        digest.append(f"A13 {d}: EDSP={100 * m['EDSP-tuned']:.2f} a={100 * r['a_delta']:+.2f} {r['a_verdict']} b={b}")
    tex = ["\\begin{tabular}{lrlrr}", "\\toprule",
           "Graph & EDSP-tuned & Best sign-blind & $\\Delta$ (a) & $\\Delta$ vs +HL (b) \\\\", "\\midrule", *rows,
           "\\bottomrule", "\\end{tabular}"]
    return "\n".join(tex) + "\n", digest


def a11_table() -> tuple[str, list[str]]:
    rows_in = load(ROOT / "results" / "perturbed" / "attribution" / "aggregates" / "attribution.json")
    rows, digest = [], []
    count = {}
    for r in rows_in:
        if "verdict" not in r:
            continue
        count[(r["family"], r["verdict"])] = count.get((r["family"], r["verdict"]), 0) + 1
        rows.append(f"{tex_escape(r['base'])} & {r['family']} & {tex_escape(r['spec'])} & {100 * r['delta']:+.2f} & {r['verdict']} \\\\")
    digest += [f"A11 {fam} {v}: {n}" for (fam, v), n in sorted(count.items())]
    tex = ["\\begin{tabular}{lllrl}", "\\toprule", "Base & Locality & Condition & $\\Delta$ vs best +HL & Verdict \\\\",
           "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"]
    return "\n".join(tex) + "\n", digest


def b5b6_table() -> tuple[str, list[str]]:
    """A17: EDSP-tuned (B5, Track A) / EDSP-B-tuned (B6, Track B) vs the signed and fraud GNNs."""
    path = AGG / "b5b6" / "b5b6.json"
    if not path.exists():
        return "", []
    a = load(path)
    names = {"sgcn": "SGCN", "snea": "SNEA", "sdgnn": "SDGNN", "sigat": "SiGAT", "care_gnn": "CARE-GNN",
             "pc_gnn": "PC-GNN"}
    rows, digest = [], []
    for fam, label in (("B5", "B5 signed (9)"), ("B6", "B6 fraud (2)")):
        for m, v in a[fam]["wtl"].items():
            w, t, lo = len(v["win"]), len(v["tie"]), len(v["loss"])
            rows.append(f"{label} & {names[m]} & {w} & {t} & {lo} \\\\")
            digest.append(f"WTL {fam} {m}: {w}/{t}/{lo}")
        for d, g in a[fam]["graphs"].items():
            if g["ranked"]:
                digest.append(f"{fam} {d}: EDSP={g['methods']['EDSP-tuned']['mean']:.2f} best={g['best']} "
                              f"{g['methods'][g['best']]['mean']:.2f} rank={g['ref_rank']}/{len(g['methods'])}")
    er = a["B5"].get("exploratory_ranks")
    if er:
        digest.append(f"B5 exploratory ranks friedman_p={er['friedman_p']:.3g}: "
                      + ", ".join(f"{m} {r:.2f}" for m, r in sorted(er["average_ranks"].items(), key=lambda x: x[1])))
    tex = ["\\begin{tabular}{llrrr}", "\\toprule", "Family & Method & Win & Tie & Loss \\\\", "\\midrule", *rows,
           "\\bottomrule", "\\end{tabular}"]
    return "\n".join(tex) + "\n", digest


def h5_table() -> tuple[str, list[str]]:
    """A15: H5 on ogbn-proteins (primary and secondary comparisons over the 112 tasks)."""
    path = AGG / "c4" / "h5_verdict.json"
    if not path.exists():
        return "", []
    a = load(path)
    comps = [("EDSP-native-tuned vs EDSP-const-tuned", a["primary"])] + [
        (("EDSP-native-default vs EDSP-const-default" if k == "defaults" else f"EDSP-native-tuned {k}"), c)
        for k, c in a["secondary"].items()]
    rows, digest = [], [f"H5 supported={a['supported']} tasks={a['primary']['tasks']} not_run={len(a['not_run'])}"]
    for name, c in comps:
        lo, hi = c["ci95_mean_delta"]
        rows.append(f"{name} & {c['tasks']} & {c['mean_delta']:+.2f} [{lo:+.2f}, {hi:+.2f}] & {c['p']:.2g} & "
                    f"{c['win']} / {c['loss']} \\\\")
        digest.append(f"H5 {name}: delta={c['mean_delta']:+.3f} ci=[{lo:+.3f},{hi:+.3f}] p={c['p']:.3g} "
                      f"win={c['win']} tie={c['tie']} loss={c['loss']}")
    digest += [f"H5 mean {m}: {v:.2f}" for m, v in sorted(a["mean_roc_auc"].items(), key=lambda x: -x[1])]
    tex = ["\\begin{tabular}{lrrrr}", "\\toprule",
           "Comparison & Tasks & Mean $\\Delta$ AUC [95\\% CI] & $p$ & Better / worse \\\\", "\\midrule", *rows,
           "\\bottomrule", "\\end{tabular}"]
    return "\n".join(tex) + "\n", digest


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    digest: list[str] = []
    suites = [("E1", "e1/e1_aggregates.json", ""), ("E2", "e2/e2_aggregates.json", ""),
              ("E3", "e3/e3_aggregates.json", "")]
    for name, (tex, dg) in {
        "tab_wtl.tex": wtl_table(suites),
        "tab_ranks.tex": ranks_table([("E1", "e1/e1_aggregates.json"), ("E2", "e2/e2_aggregates.json"),
                                      ("E3", "e3/e3_aggregates.json")]),
        "tab_h4.tex": h4_table(),
        "tab_a13.tex": a13_table(),
        "tab_a11.tex": a11_table(),
        "tab_b5b6.tex": b5b6_table(),
        "tab_h5.tex": h5_table(),
    }.items():
        if tex:
            (OUT / name).write_text(tex)
        digest += dg
    for suite in ("e1", "e2", "e2u", "e3", "e5"):
        src = AGG / suite / f"{suite}_headline.tex"
        if src.exists():
            (OUT / f"tab_headline_{suite}.tex").write_text(src.read_text())
    (OUT / "DIGEST.txt").write_text("\n".join(digest) + "\n")
    print("\n".join(digest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
