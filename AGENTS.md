# Repository Agent Instructions

This repository implements the EDSP evaluation specified in the Master Prompt. The Master Prompt is the governing specification; rules R1–R8 apply to every change.

The governing prompt is committed at `docs/MASTER_PROMPT.md`.

## Scope and commands

- Python >= 3.11; use `uv` when available.
- Unit/property tests: `pytest -q tests/unit`
- Toy examples and the 1M-edge benchmark are the only execution targets for PR-0/PR-1.
- Never download datasets or run full experiment matrices in an agent session; use GitHub Actions for heavy work (`dataset-stats.yml` for statistics, later `fetch-data.yml` / `run-matrix.yml`).
- Dataset statistics: `uv run python scripts/dataset_stats.py` (set `EDSP_DATA_DIR` for the raw-file cache). Sources and their verification status live in `data/REGISTRY.json`.

## Non-negotiable rules

- No fabricated numbers. Every result must be traceable to a logged run.
- Preserve test-label isolation. Test labels may only be accessed inside `edsp.eval.evaluate`; validation labels only inside `edsp.selection` / `edsp.eval`. Methods get labels exclusively through `GraphData.vault(split).train_labels()`; never read `GraphData._labels` or `LabelVault._labels` directly.
- Keep `EDSP-canonical` literal. Record ambiguities in `docs/SPEC_AMBIGUITIES.md`; implement alternatives only as named variants.
- Treat honest negative, non-convergent, and not-run outcomes as results.
- Use real data first; never substitute synthetic data without the audited exhaustion procedure.
- Keep tuning budgets equal across methods and validation-only.
- Mark unverified items `TO VERIFY` and record evidence.

## Running EDSP

- `from edsp.methods import run_edsp; run = run_edsp(graph, split, config)`; configurations and the zero-tuning default are documented in `edsp/regimes/inputs.py`.
- H-ORACLE is reachable only with `ORACLE=1` through `edsp.regimes.oracle`; never import it in method code.

## Definition of done for PR-1

The canonical NumPy/SciPy implementation, backend parity, T1–T11 tests, E1–E7 toy examples, and the 1M-edge benchmark are present and validated. T11 remains explicitly xfailed/skipped until LabelVault and the regimes are delivered by PR-2/PR-4.
