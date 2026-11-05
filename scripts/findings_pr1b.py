from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

import numpy as np

from edsp import canonical_phi, edsp


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _cases() -> list[dict]:
    return [
        {
            "id": "a-p0.5",
            "label": "(a) one homophilic edge, p=0.5",
            "edges": [(0, 1)],
            "probabilities": [0.5],
            "heterophily": [0],
            "priors": [[0.9, 0.1], [0.1, 0.9]],
        },
        {
            "id": "a-p0.9",
            "label": "(a) one homophilic edge, p=0.9",
            "edges": [(0, 1)],
            "probabilities": [0.9],
            "heterophily": [0],
            "priors": [[0.9, 0.1], [0.1, 0.9]],
        },
        {
            "id": "b-p0.3",
            "label": "(b) heterophilic 4-cycle, p=0.3",
            "edges": [(0, 1), (1, 2), (2, 3), (3, 0)],
            "probabilities": [0.3] * 4,
            "heterophily": [1] * 4,
            "priors": [[0.9, 0.1]] * 4,
        },
        {
            "id": "b-p0.5",
            "label": "(b) heterophilic 4-cycle, p=0.5",
            "edges": [(0, 1), (1, 2), (2, 3), (3, 0)],
            "probabilities": [0.5] * 4,
            "heterophily": [1] * 4,
            "priors": [[0.9, 0.1]] * 4,
        },
        {
            "id": "b-p0.7",
            "label": "(b) heterophilic 4-cycle, p=0.7",
            "edges": [(0, 1), (1, 2), (2, 3), (3, 0)],
            "probabilities": [0.7] * 4,
            "heterophily": [1] * 4,
            "priors": [[0.9, 0.1]] * 4,
        },
        {
            "id": "b-p0.9",
            "label": "(b) heterophilic 4-cycle, p=0.9",
            "edges": [(0, 1), (1, 2), (2, 3), (3, 0)],
            "probabilities": [0.9] * 4,
            "heterophily": [1] * 4,
            "priors": [[0.9, 0.1]] * 4,
        },
        {
            "id": "c-p0.9",
            "label": "(c) one-hot priors, one homophilic edge, p=0.9",
            "edges": [(0, 1)],
            "probabilities": [0.9],
            "heterophily": [0],
            "priors": [[1.0, 0.0], [0.0, 1.0]],
        },
        {
            "id": "e7-p0.9",
            "label": "(e7) unlabelled heterophilic 4-cycle, one non-uniform start, p=0.9",
            "edges": [(0, 1), (1, 2), (2, 3), (3, 0)],
            "probabilities": [0.9] * 4,
            "heterophily": [1] * 4,
            "priors": [[0.9, 0.1], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
        },
    ]


def _variants() -> list[tuple[str, dict]]:
    return [
        ("EDSP-canonical", {"variant": "EDSP-canonical"}),
        ("EDSP-GS (id)", {"variant": "EDSP-GS", "order": "id"}),
        ("EDSP-GS (seeded random)", {"variant": "EDSP-GS", "order": "random", "seed": 0}),
        ("EDSP-damped(beta=0.5)", {"variant": "EDSP-damped", "beta": 0.5}),
    ]


def _format_matrix(values: np.ndarray) -> str:
    return np.array2string(values, precision=4, suppress_small=False, separator=", ", max_line_width=10000).replace(
        "\n", " "
    )


def _markdown_table(records: list[dict]) -> str:
    grouped: dict[str, dict[str, dict]] = {}
    labels: dict[str, str] = {}
    for record in records:
        grouped.setdefault(record["case_id"], {})[record["variant"]] = record
        labels[record["case_id"]] = record["case_label"]
    lines = [
        "| Case | EDSP-canonical | EDSP-GS (id) | EDSP-GS (seeded random, seed=0) | EDSP-damped(beta=0.5) |",
        "|---|---|---|---|---|",
    ]
    for case_id in [case["id"] for case in _cases()]:
        row = [labels[case_id]]
        for variant in [name for name, _ in _variants()]:
            r = grouped[case_id][variant]
            status = f"{'converged' if r['converged'] else 'max_iter'} in {r['iterations']}"
            row.append(
                f"{status}; osc={r['oscillation']}; amp={r['oscillation_amplitude']:.10f}; "
                f"||Φ(V)-V||₁={r['phi_residual_l1']:.3e}; V={_format_matrix(np.array(r['final_v']))}"
            )
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def generate_records(max_iter: int) -> list[dict]:
    sha = _git_sha()
    now = dt.datetime.now(dt.UTC).isoformat()
    records: list[dict] = []
    for case in _cases():
        for variant_name, variant_kwargs in _variants():
            run = edsp(
                case["edges"],
                case["probabilities"],
                case["heterophily"],
                case["priors"],
                max_iter=max_iter,
                **variant_kwargs,
            )
            phi = canonical_phi(
                case["edges"], case["probabilities"], case["heterophily"], np.asarray(case["priors"]), run.probabilities
            )
            records.append(
                {
                    "timestamp_utc": now,
                    "git_sha": sha,
                    "case_id": case["id"],
                    "case_label": case["label"],
                    "variant": variant_name,
                    "seed": variant_kwargs.get("seed"),
                    "iterations": run.iterations,
                    "converged": run.converged,
                    "oscillation": run.oscillation,
                    "oscillation_amplitude": run.oscillation_amplitude,
                    "final_v": run.probabilities.tolist(),
                    "phi_residual_l1": float(np.max(np.sum(np.abs(phi - run.probabilities), axis=1))),
                }
            )
    return records


def write_jsonl(records: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_findings_doc(records: list[dict], findings_path: Path, records_path: Path) -> None:
    table = _markdown_table(records)
    findings_path.write_text(
        "\n".join(
            [
                "# PR-1b findings",
                "",
                "These toy-graph findings are generated from recorded runs in JSONL form.",
                "",
                table,
                "",
                "Run command:",
                "",
                "```text",
                "uv run python scripts/findings_pr1b.py --write-docs",
                "```",
                "",
                f"Records file: `{records_path.as_posix()}`",
                "",
                "Case (a), p=0.9 has at least three fixed points under canonical Φ:",
                "- Symmetric fixed point (captured by damped): V0≈[0.7466, 0.2534], V1≈[0.2534, 0.7466]",
                "- Asymmetric mirror pair (GS-id picks one by node order): V0≈[0.5228, 0.4772], V1≈[0.1011, 0.8989]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/runs/findings_pr1b.jsonl")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--write-docs", action="store_true")
    parser.add_argument("--findings-path", default="docs/FINDINGS.md")
    args = parser.parse_args()

    records = generate_records(args.max_iter)
    output = Path(args.output)
    write_jsonl(records, output)
    if args.write_docs:
        write_findings_doc(records, Path(args.findings_path), output)


if __name__ == "__main__":
    main()
