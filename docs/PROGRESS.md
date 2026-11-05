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

- PR-2 (data layer): `edsp/data/` (graph format + canonicalisation, LabelVault with caller checks, registry, loaders, acquisition log, §3.2 statistics), `edsp/eval/evaluate.py` and `edsp/selection/` as the only val/test label consumers, `data/REGISTRY.json`, `data/PUBLISHED_STATS.json`, `scripts/dataset_stats.py`, workflow `dataset-stats.yml`, draft `docs/PREREGISTRATION.md`, new `tests/unit/test_data.py`.
  - Loaders: cora, citeseer, pubmed, amazon-computers, amazon-photo, coauthor-cs, coauthor-physics, wikics, roman-empire, amazon-ratings, minesweeper, tolokers, questions, chameleon-filtered, squirrel-filtered, sampson, ogbn-proteins (the last one untested on the real archive until its first Actions run).
  - DoD evidence: the `dataset-stats` workflow run on the PR (job summary + artifact). Numbers are cited only from that run.

## Remaining

- Owner decisions in docs/PREREGISTRATION.md (H7 replacement, non-convergence column, minesweeper), then freeze it.
- Verify the sources marked TO VERIFY in data/PUBLISHED_STATS.json (Planetoid/Amazon/Coauthor/WikiCS reference values).
- PR-3: fetch-data.yml and acquisition run for the remaining cells (LINKX, fraud, C4/C5 sources).
- PR-4: train-only regimes and a passing leakage guard (T11).

## Blockers

- T11 cannot run until the regimes (PR-4) exist; it is explicitly marked xfail rather than stubbed.
- snap.stanford.edu (ogbn-proteins) is unreachable from the development session; the loader is first exercised by `dataset-stats.yml` with `include_large=true`.
