"""PR-4 Definition of Done: log H-LEARNED edge-classifier AUCs (and H-TRAINONLY coverage).

For every single-label dataset with a loader (ogbn-proteins excluded: multi-label,
handled per task in E2) and each track (B only if node features exist), fit the
H-LEARNED classifier on train-train edges of split 0, record the validation-edge
ROC-AUC used for selection, and score test-test edges through
edsp.eval.evaluate.edge_polarity_auc (the only module allowed to read test labels).
These are diagnostics of a polarity regime, not EDSP results.
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edsp.data.loaders import load
from edsp.data.registry import default_datasets
from edsp.eval.evaluate import edge_polarity_auc
from edsp.regimes.polarity import h_learned, h_trainonly
from edsp.selection import select_threshold_by_val_edge_accuracy


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "edge_classifier"))
    args = ap.parse_args()
    names = args.datasets or [d for d in default_datasets() if d != "ogbn-proteins"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_id = os.environ.get("GITHUB_RUN_ID")
    runner = f"actions-run-{run_id}" if run_id else os.environ.get("EDSP_ENVIRONMENT", "local")
    rows, not_run = [], []
    for name in names:
        try:
            g = load(name)
        except Exception as exc:  # noqa: BLE001
            not_run.append({"dataset": name, "reason": f"load: {exc}"[:300]})
            continue
        cov = h_trainonly(g, g.vault(args.split), "b1").diagnostics
        for track in ("A", "B"):
            if track == "B" and g.features is None:
                continue
            t0 = time.perf_counter()
            try:
                vault = g.vault(args.split)
                pol = h_learned(g, vault, track)  # default decision: prior-corrected
                test, extra = {}, {}
                if pol.score is not None:
                    test = edge_polarity_auc(vault, g.edges, pol.score)
                    grid = tuple(np.round(np.linspace(0.05, 0.95, 19), 2))
                    t_val, _ = select_threshold_by_val_edge_accuracy(vault, g.edges, pol.score, grid)
                    extra = {
                        "test_edge_accuracy_balanced": edge_polarity_auc(vault, g.edges, pol.raw_score)["test_edge_accuracy"],
                        "val_threshold": t_val,
                        "test_edge_accuracy_val_threshold": edge_polarity_auc(vault, g.edges, pol.score, t_val)["test_edge_accuracy"],
                    }
            except Exception as exc:  # noqa: BLE001
                not_run.append({"dataset": name, "track": track, "reason": f"{type(exc).__name__}: {exc}"[:300]})
                continue
            d = pol.diagnostics
            best = d.get("val_edge_auc", [None])[d["l2_grid"].index(d["chosen_l2"])] if "chosen_l2" in d else None
            rows.append({
                "dataset": name, "track": track, "split": g.splits[args.split].split_id,
                "train_edges": d.get("train_edges"), "train_edges_used": d.get("train_edges_used"),
                "chosen_l2": d.get("chosen_l2"), "val_edge_auc": best, "val_edge_auc_grid": d.get("val_edge_auc"),
                "predicted_het_fraction": d.get("predicted_heterophilic_fraction"), "status": d.get("status", "ok"),
                "trainonly_coverage": cov["coverage"], "train_edge_heterophily": cov["train_edge_heterophily"],
                **test, **extra,
                "run_record": {"git_sha": git_sha(), "runner_type": runner, "dataset_hash": g.content_hash(),
                               "wall_clock_seconds": round(time.perf_counter() - t0, 3),
                               "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)},
            })
            print(f"{name:20s} {track} val_auc={best} test_auc={test.get('test_edge_auc')}", flush=True)
    (out / "edge_classifier.json").write_text(json.dumps({"rows": rows, "not_run": not_run}, indent=2) + "\n")

    def f(x):
        return "—" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))

    md = ["| dataset | track | train–train edges | H-TRAINONLY coverage | train-edge het. | val-edge AUC | test-edge AUC | test acc. balanced-0.5 | test acc. prior-corrected | test acc. val-threshold | test acc. H-ALL1 | test acc. H-ALL0 | predicted het. fraction |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        het = r.get("test_edge_heterophily")
        md.append(f"| {r['dataset']} | {r['track']} | {r['train_edges']} | {f(r['trainonly_coverage'])} | {f(r['train_edge_heterophily'])} | "
                  f"{f(r['val_edge_auc'])} | {f(r.get('test_edge_auc'))} | {f(r.get('test_edge_accuracy_balanced'))} | {f(r.get('test_edge_accuracy'))} | "
                  f"{f(r.get('test_edge_accuracy_val_threshold'))} | {f(het)} | {f(None if het is None else 1 - het)} | {f(r['predicted_het_fraction'])} |")
    if not_run:
        md += ["", "**Not run**", ""] + [f"- {x}" for x in not_run]
    md += ["", f"split index {args.split}; git `{git_sha()[:12]}`; runner `{runner}`. Diagnostics of the H-LEARNED polarity regime, not EDSP results."]
    text = "\n".join(md) + "\n"
    (out / "SUMMARY.md").write_text(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as fh:
            fh.write("## H-LEARNED edge-classifier diagnostics (PR-4)\n\n" + text)
    print(text)
    return 1 if not_run else 0


if __name__ == "__main__":
    raise SystemExit(main())
