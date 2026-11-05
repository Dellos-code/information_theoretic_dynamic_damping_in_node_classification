# Progress

## Done

- Read the Master Prompt in full.
- Created `AGENTS.md` and `.github/copilot-instructions.md`.
- Added the Python package scaffold and literal NumPy EDSP-canonical implementation.
- Added PR-1 unit/property/toy/benchmark coverage, including explicit T11 xfail.
- Added the ambiguity register.
- Regenerated `uv.lock` with `uv lock` (no manual lockfile edits).
- PR-1b follow-up: torch test gating via `EDSP_REQUIRE_TORCH`, CI sets `EDSP_REQUIRE_TORCH=1`, GS sweep uses CSR offsets, canonical rejects `beta != 1`, regime diagnostics use H/H'/Hmax definitions, and GS diagnostics are now computed (not fabricated).
- Added §1.8 follow-up coverage: explicit Regime 1/3 tests, E7 specified setup (unlabelled 4-cycle with one non-uniform start), case (b) `p=0.3` convergence check, and P1/D1 hypothesis tests.
- Added `scripts/findings_pr1b.py`, JSONL findings records at `results/runs/findings_pr1b.jsonl`, and regenerated `docs/FINDINGS.md` with final states and `||Φ(V)-V||_1`.
- Validation:
  - `uv sync --frozen --group dev && uv run pytest -q tests/unit`
  - `24 passed, 1 xfailed in 5.43s`
  - `uv run --with ruff ruff check edsp tests`
  - `uv run --with mypy mypy edsp/core.py`

## Remaining

- PR-2: registry, loaders, LabelVault, dataset statistics, preregistration.
- PR-4: train-only regimes and a passing leakage guard.

## Blockers

- T11 cannot run until LabelVault (PR-2) and regimes (PR-4) exist; it is explicitly marked xfail rather than stubbed.
- No datasets were downloaded, per session scope.

## PR-2 handoff (no implementation started here)

- Implement dataset registry/loaders, LabelVault strict access isolation, dataset statistics/evidence records, and preregistration.
