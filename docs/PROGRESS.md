# Progress

## Done

- Read the Master Prompt in full.
- Created `AGENTS.md` and `.github/copilot-instructions.md`.
- Added the Python package scaffold and literal NumPy EDSP-canonical implementation.
- Added PR-1 unit/property/toy/benchmark coverage, including explicit T11 xfail.
- Added the ambiguity register.
- Locked dependencies with `uv.lock`; CI and Copilot setup workflows run the unit suite.
- Validation: `19 passed, 1 skipped (torch unavailable on this host), 1 xfailed (T11)`; canonical oscillation regression and bounded diagnostics pass.
- PR-1b fix-up: canonical period-2 diagnostics, GS/damped variants, native torch iteration, CPU torch dependency declaration, expanded §1.8 coverage, CI/Docker/benchmark updates.

## Remaining

- PR-2: registry, loaders, LabelVault, dataset statistics, preregistration.
- PR-4: train-only regimes and a passing leakage guard.

## Blockers

- T11 cannot run until LabelVault (PR-2) and regimes (PR-4) exist; it is explicitly marked xfail rather than stubbed.
- No datasets were downloaded, per session scope.
- `uv lock` and torch parity are not run locally because the CPU wheel index is unavailable for this host architecture; CI remains the validation target.
