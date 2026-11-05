"""Compute §3.2 dataset statistics and check them against published values.

Writes, under --out (default results/dataset_stats):
  <dataset>.json      full statistics plus a run record (rule R1)
  summary.json        one row per dataset + reference check
  SUMMARY.md          Markdown table (also appended to $GITHUB_STEP_SUMMARY)
  table.tex           booktabs LaTeX table
Datasets that fail to load are listed under "not_run" with the reason (rule R7).
Exit status is 1 if any requested dataset failed to load.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edsp.data.loaders import load
from edsp.data.registry import datasets_with_loader
from edsp.data.stats import dataset_stats

DEFAULT_EXCLUDE = {"ogbn-proteins"}  # large; opt in with --include-large
CHECKED = ("edge_homophily", "adjusted_homophily", "label_informativeness")


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def compare(name: str, labels: dict, published: dict, tol: float) -> dict:
    ref = published.get("datasets", {}).get(name)
    if ref is None:
        return {"status": "NO-REFERENCE"}
    rows, worst = {}, 0.0
    for key in CHECKED:
        if key in ref:
            ours = labels[key]
            diff = abs(round(ours, 2) - ref[key])
            worst = max(worst, diff)
            rows[key] = {"computed": ours, "reported": ref[key], "abs_diff_rounded": round(diff, 4)}
    status = "MATCH" if worst <= tol + 1e-9 else "DISCREPANCY"
    return {"status": status, "source": ref.get("source"), "source_status": ref.get("source_status"), "values": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--include-large", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "dataset_stats"))
    args = ap.parse_args()

    names = args.datasets or [d for d in datasets_with_loader() if args.include_large or d not in DEFAULT_EXCLUDE]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    published = json.loads((ROOT / "data" / "PUBLISHED_STATS.json").read_text())
    tol = float(published.get("tolerance", 0.01))
    env = os.environ.get("GITHUB_RUN_ID")
    runner = f"actions-run-{env}" if env else os.environ.get("EDSP_ENVIRONMENT", "local")

    rows, not_run = [], []
    for name in names:
        t0 = time.perf_counter()
        try:
            g = load(name)
            stats = dataset_stats(g)
        except Exception as exc:  # noqa: BLE001
            not_run.append({"dataset": name, "reason": f"{type(exc).__name__}: {exc}"[:500]})
            print(f"[not run] {name}: {exc}", file=sys.stderr)
            continue
        wall = time.perf_counter() - t0
        stats["run_record"] = {
            "git_sha": git_sha(),
            "runner_type": runner,
            "python": platform.python_version(),
            "wall_clock_seconds": round(wall, 3),
            "peak_rss_mib": round(peak_rss_mib(), 1),
            "dataset_hash": stats["dataset_hash"],
        }
        lab = stats["labels"]
        stats["reference_check"] = compare(name, lab, published, tol) if "edge_homophily" in lab else {"status": "MULTILABEL"}
        (out / f"{name}.json").write_text(json.dumps(stats, indent=2) + "\n")
        s = stats["structure"]
        rows.append({
            "dataset": name,
            "n": s["num_nodes"], "m": s["num_undirected_edges"], "k": s["num_classes"],
            "isolated": s["isolated_nodes"], "components": s["components"],
            "h_edge": lab.get("edge_homophily"), "h_node": lab.get("node_homophily"),
            "h_adj": lab.get("adjusted_homophily"), "LI": lab.get("label_informativeness"),
            "local_h_std": lab.get("local_homophily_std"),
            "d_I": lab.get("compatibility_distances", {}).get("to_identity"),
            "d_unifhet": lab.get("compatibility_distances", {}).get("to_uniform_heterophily"),
            "check": stats["reference_check"]["status"],
            "dataset_hash": stats["dataset_hash"],
        })

    summary = {"run": {"git_sha": git_sha(), "runner_type": runner}, "rows": rows, "not_run": not_run}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def f(x: float | None) -> str:
        return "—" if x is None else f"{x:.3f}"

    md = ["| dataset | n | m | k | isolated | comps | h_edge | h_node | h_adj | LI | std(local h) | ‖H−I‖ | ‖H−U‖ | reference |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['dataset']} | {r['n']} | {r['m']} | {r['k']} | {r['isolated']} | {r['components']} | "
                  f"{f(r['h_edge'])} | {f(r['h_node'])} | {f(r['h_adj'])} | {f(r['LI'])} | {f(r['local_h_std'])} | "
                  f"{f(r['d_I'])} | {f(r['d_unifhet'])} | {r['check']} |")
    if not_run:
        md += ["", "**Not run**", ""] + [f"- {x['dataset']}: {x['reason']}" for x in not_run]
    md += ["", f"git `{summary['run']['git_sha'][:12]}`, runner `{runner}`. ‖H−U‖ is the distance to (J−I)/(k−1)."]
    text = "\n".join(md) + "\n"
    (out / "SUMMARY.md").write_text(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as fh:
            fh.write("## Dataset statistics (PR-2)\n\n" + text)

    tex = ["\\begin{tabular}{lrrrrrrr}", "\\toprule",
           "Dataset & $n$ & $|E|$ & $k$ & $h_{\\mathrm{edge}}$ & $h_{\\mathrm{adj}}$ & LI & std$(h_v)$ \\\\", "\\midrule"]
    for r in rows:
        tex.append(f"{r['dataset']} & {r['n']} & {r['m']} & {r['k']} & {f(r['h_edge'])} & {f(r['h_adj'])} & {f(r['LI'])} & {f(r['local_h_std'])} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (out / "table.tex").write_text("\n".join(tex) + "\n")
    print(text)
    return 1 if not_run else 0


if __name__ == "__main__":
    raise SystemExit(main())
