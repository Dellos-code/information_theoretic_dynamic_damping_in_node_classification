# Progress

## Done

- Read the Master Prompt in full.
- Created `AGENTS.md` and `.github/copilot-instructions.md`.
- Added the Python package scaffold and literal NumPy EDSP-canonical implementation.
- Added PR-1 unit/property/toy/benchmark coverage, including explicit T11 xfail.
- Added the ambiguity register.
- Locked dependencies with `uv.lock`; CI and Copilot setup workflows run the unit suite.
- Validation: `12 passed, 1 skipped (optional torch), 1 xfailed (T11)`; 1M-edge one-iteration benchmark: `0.1M nodes, 999,987 edges, 1.155043 s`.

## Remaining

- Run and fix PR-1 tests and benchmark.
- Push the work-in-progress commit before the session limit (this checkpoint).
- PR-2: registry, loaders, LabelVault, dataset statistics, preregistration.
- PR-4: train-only regimes and a passing leakage guard.

## Blockers

- T11 cannot run until LabelVault (PR-2) and regimes (PR-4) exist; it is explicitly marked xfail rather than stubbed.
- No datasets were downloaded, per session scope.
