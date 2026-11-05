"""PR-5 Definition of Done: B0/B1 baselines and their parity check.

Runs every applicable B0/B1 baseline on the Planetoid public split and on the
first random 60/20/20 split, scores them through edsp.eval.evaluate, and compares
the zero-tuning harmonic LP with the REPORTED values in
data/PUBLISHED_BASELINES.json (parity within 1 point, else 'parity failed').
These are baseline numbers, not EDSP results.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edsp.baselines import BASELINES, select_and_predict
from edsp.baselines.propagation import BASELINES as _B
from edsp.data.loaders import load
from edsp.eval.evaluate import evaluate


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["cora", "citeseer", "pubmed"])
    ap.add_argument("--splits", nargs="*", default=["public", "rand0"])
    ap.add_argument("--out", default=str(ROOT / "results" / "baseline_parity"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ref = json.loads((ROOT / "data" / "PUBLISHED_BASELINES.json").read_text())
    tol = float(ref["tolerance_points"])
    run_id = os.environ.get("GITHUB_RUN_ID")
    runner = f"actions-run-{run_id}" if run_id else os.environ.get("EDSP_ENVIRONMENT", "local")
    rows, parity, not_run = [], [], []
    for name in args.datasets:
        g = load(name)
        for split in args.splits:
            for bname, (group, track, _) in BASELINES.items():
                if bname == "signed_vote" and g.edge_sign is None:
                    continue
                t0 = time.perf_counter()
                try:
                    vault = g.vault(split)
                    probs, info = select_and_predict(bname, g, vault)
                    metrics = evaluate(vault, probs)
                except Exception as exc:  # noqa: BLE001
                    not_run.append({"dataset": name, "split": split, "baseline": bname, "reason": f"{type(exc).__name__}: {exc}"[:300]})
                    continue
                rows.append({"dataset": name, "split": split, "baseline": bname, "group": group, "track": track,
                             "accuracy": metrics["accuracy"], "nll": metrics["nll"], "chosen": info.get("chosen"),
                             "converged": info.get("converged", info.get("smooth", {}).get("converged")),
                             "run_record": {"git_sha": git_sha(), "runner_type": runner, "dataset_hash": g.content_hash(),
                                            "wall_clock_seconds": round(time.perf_counter() - t0, 3),
                                            "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}})
        for e in ref["entries"]:
            if e["dataset"] != name:
                continue
            vault = g.vault(e["split"])
            probs, info = _B[e["baseline"]][2](g, vault)
            acc = 100 * evaluate(vault, probs)["accuracy"]
            parity.append({"baseline": e["baseline"], "dataset": name, "split": e["split"], "reproduced": round(acc, 2),
                           "reported": e["accuracy"], "abs_diff": round(abs(acc - e["accuracy"]), 2),
                           "status": "PARITY" if abs(acc - e["accuracy"]) <= tol else "PARITY FAILED (documented)",
                           "source_status": e["source_status"], "converged": info.get("converged")})
    (out / "baseline_parity.json").write_text(json.dumps({"rows": rows, "parity": parity, "not_run": not_run}, indent=2) + "\n")
    md = ["### Parity (REPORTED vs REPRODUCED, accuracy %)", "", "| baseline | dataset | split | reproduced | reported | |diff| | status | source |", "|---|---|---|---|---|---|---|---|"]
    md += [f"| {p['baseline']} | {p['dataset']} | {p['split']} | {p['reproduced']:.2f} | {p['reported']:.1f} | {p['abs_diff']:.2f} | {p['status']} | {p['source_status']} |" for p in parity]
    md += ["", "### All B0/B1 baselines (test accuracy %, validation-selected hyper-parameters)", "", "| dataset | split | baseline | group | track | accuracy | NLL | chosen |", "|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['dataset']} | {r['split']} | {r['baseline']} | {r['group']} | {r['track']} | {100 * r['accuracy']:.2f} | {r['nll']:.3f} | {r['chosen'] or '—'} |" for r in rows]
    if not_run:
        md += ["", "**Not run**", ""] + [f"- {x}" for x in not_run]
    md += ["", f"git `{git_sha()[:12]}`; runner `{runner}`. Baseline numbers, not EDSP results."]
    text = "\n".join(md) + "\n"
    (out / "SUMMARY.md").write_text(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as fh:
            fh.write("## B0/B1 baselines and parity (PR-5)\n\n" + text)
    print(text)
    return 1 if not_run else 0


if __name__ == "__main__":
    raise SystemExit(main())
