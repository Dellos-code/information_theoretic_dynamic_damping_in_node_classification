# Specification ambiguities

This file records unresolved interpretations without changing the canonical implementation.

| Item | Literal interpretation used | Alternative / follow-up |
|---|---|---|
| Infinite trust | `exp(-inf)=0`, so calibration leaves the input unchanged | Add explicit finite/infinite trust property tests when LabelVault/regimes land |
| Duplicate undirected edges | Every input row is an edge and is expanded in both directions | Dataset loaders must document whether duplicate rows are pre-aggregated |
| Oscillation diagnostic | The implementation records the required residual/sign heuristic as a diagnostic only; it does not alter iteration | A labelled damping variant belongs in a later PR |
| E3 versus §1.5 | Literal §1.5 with one-hot `v`, deterministic opposite `S`, and `p=0.5` gives `alpha=1` and `V=S` | E3 states `alpha=1/3` and `V=[2/3,1/3]`; canonical tests use the literal equation until the specification is resolved |
