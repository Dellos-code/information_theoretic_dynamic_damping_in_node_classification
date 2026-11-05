# Specification ambiguities

This file records unresolved interpretations without changing the canonical implementation.

| Item | Literal interpretation used | Alternative / follow-up |
|---|---|---|
| Infinite trust | `exp(-inf)=0`, so calibration leaves the input unchanged | Add explicit finite/infinite trust property tests when LabelVault/regimes land |
| Duplicate undirected edges | Every input row is an edge and is expanded in both directions | Dataset loaders must document whether duplicate rows are pre-aggregated |
| Oscillation diagnostic | The implementation records period-2 detection as a diagnostic only; it does not alter `EDSP-canonical` | Named GS and damped variants are reported separately |
| Max-iteration prediction iterate | When a run reaches `max_iter` without convergence, predictions and calibration metrics use the last iterate; the last two iterates are also stored for diagnostics | Revisit only if a later evaluation protocol requires a different policy |
