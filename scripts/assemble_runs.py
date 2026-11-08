"""Assemble run records spread over many results branches into one directory.

Large suites run as many workflow dispatches (one per split or per method group), each
pushing its records to its own branch `results/<suite>-<run_id>`. This script merges
them for the aggregators, without changing any record:

- key = (dataset, split, method); identical duplicates (same headline value) are kept once;
- conflicting duplicates (same key, different value) keep the record from the
  `--prefer` ref if it has one, otherwise the one from the earliest run, and are listed;
- NOT-RUN records are kept only for (dataset, split) pairs with no scored record of the
  missing kind; all of them are listed in ASSEMBLY.md;
- DATASET-STATS: one per dataset (they are label-free descriptive statistics).

Usage:
  python scripts/assemble_runs.py --subdir results/runs/e3 --out results/runs/e3 \
      --refs main 'origin/results/e3-*' --prefer main --exclude 42474871330 ...
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def expand(patterns: list[str]) -> list[str]:
    refs = ["main"] if "main" in patterns else []
    remote = git("for-each-ref", "--format=%(refname:short)", "refs/remotes").split()
    for p in patterns:
        if p != "main":
            refs += sorted(r for r in remote if fnmatch.fnmatch(r, p))
    return refs


def run_id(ref: str) -> int:
    tail = ref.rsplit("-", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subdir", required=True, help="path of the records inside each ref, e.g. results/runs/e3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--refs", nargs="+", required=True, help="'main' (working tree) and/or remote ref globs")
    ap.add_argument("--prefer", default=None, help="ref whose record wins a conflict")
    ap.add_argument("--exclude", nargs="*", default=[], help="run ids whose branches are skipped")
    ap.add_argument("--datasets", nargs="*", default=None, help="keep only these datasets")
    ap.add_argument("--relabel-hidden", nargs="*", default=[], metavar="DATASET:METHOD:HIDDEN",
                    help="records of METHOD on DATASET whose config.hidden is HIDDEN become METHOD(hHIDDEN), "
                         "e.g. tolokers:gat:64 (A14); the original name is kept in relabelled_from")
    args = ap.parse_args()

    refs = [r for r in expand(args.refs) if str(run_id(r)) not in args.exclude]
    scored: dict = {}
    sources: dict = defaultdict(set)
    kept_from: dict = {}
    conflicts: list = []
    stats: dict = {}
    not_run: list = []
    acq: dict = {}
    for ref in sorted(refs, key=lambda r: (r != args.prefer, run_id(r))):
        if ref == "main":
            files = {str(p): p.read_text() for p in Path(args.subdir).glob("*.jsonl")}
        else:
            names = [n for n in git("ls-tree", "-r", "--name-only", ref, "--", args.subdir).split() if n.endswith(".jsonl")]
            files = {n: git("show", f"{ref}:{n}") for n in names}
        for name, text in files.items():
            if Path(name).name.startswith("ACQUISITION_LOG"):
                acq.setdefault(Path(name).name, text)
                continue
            for line in text.splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                d = r.get("dataset")
                if args.datasets and d not in args.datasets:
                    continue
                m = r.get("method")
                hid = (r.get("config") or {}).get("hidden")
                if hid is not None and f"{d}:{m}:{hid}" in args.relabel_hidden:
                    r = {**r, "method": f"{m}(h{hid})", "relabelled_from": m}
                    m = r["method"]
                if m == "DATASET-STATS":
                    stats.setdefault(d, r)
                    continue
                if m == "NOT-RUN":
                    not_run.append((ref, r))
                    continue
                key = (d, str(r.get("split")), m)
                value = r["metrics"][r["headline_metric"]]
                if key in scored:
                    old = scored[key]
                    if abs(old["metrics"][old["headline_metric"]] - value) > 1e-12:
                        conflicts.append((key, kept_from[key], ref, old["metrics"][old["headline_metric"]], value))
                    sources[key].add(ref)
                    continue
                scored[key] = r
                kept_from[key] = ref
                sources[key].add(ref)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.jsonl"):
        old.unlink()
    by_ds: dict = defaultdict(list)
    for (d, _s, _m), r in sorted(scored.items()):
        by_ds[d].append(r)
    have = {(d, s) for (d, s, _m) in scored}
    kept_nr = [(ref, r) for ref, r in not_run if (r.get("dataset"), str(r.get("split"))) not in have]
    for ref, r in kept_nr:
        by_ds[r["dataset"]].append(r)
    for d, recs in by_ds.items():
        lines = ([json.dumps(stats[d])] if d in stats else []) + [json.dumps(r) for r in recs]
        (out / f"{d}.jsonl").write_text("\n".join(lines) + "\n")
    for name, text in acq.items():
        (out / name).write_text(text)

    md = ["# Assembly of run records", "",
          f"Sources ({len(refs)} refs): " + ", ".join(refs), "",
          (f"Scored records: {len(scored)}; datasets: {len(by_ds)}; conflicting duplicates: {len(conflicts)}; "
           f"NOT-RUN records kept: {len(kept_nr)} (of {len(not_run)}; the rest were superseded by a scored re-run)."), ""]
    counts = defaultdict(lambda: defaultdict(set))
    for (d, s, m) in scored:
        counts[d][m].add(s)
    md += ["| dataset | method: splits |", "|---|---|"]
    for d in sorted(counts):
        md.append(f"| {d} | " + ", ".join(f"{m}: {len(v)}" for m, v in sorted(counts[d].items())) + " |")
    if conflicts:
        md += ["", "## Conflicting duplicates (first kept)", "", "| dataset | split | method | kept from | dropped from | kept | dropped |",
               "|---|---|---|---|---|---|---|"]
        md += [f"| {k[0]} | {k[1]} | {k[2]} | {a} | {b} | {x:.6f} | {y:.6f} |" for k, a, b, x, y in conflicts]
    if not_run:
        md += ["", "## NOT-RUN records", "", "| ref | dataset | split | kept | reason |", "|---|---|---|---|---|"]
        md += [f"| {ref} | {r.get('dataset')} | {r.get('split')} | {'yes' if (ref, r) in kept_nr else 'superseded'} | "
               f"{str(r.get('reason', ''))[:120]} |" for ref, r in not_run]
    (out / "ASSEMBLY.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:6]), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
