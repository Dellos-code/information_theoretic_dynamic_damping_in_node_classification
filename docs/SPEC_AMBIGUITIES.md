# Specification ambiguities

This file records unresolved interpretations without changing the canonical implementation.

| Item | Literal interpretation used | Alternative / follow-up |
|---|---|---|
| Infinite trust | `exp(-inf)=0`, so calibration leaves the input unchanged | Add explicit finite/infinite trust property tests when LabelVault/regimes land |
| Duplicate undirected edges | Every input row is an edge and is expanded in both directions | Dataset loaders must document whether duplicate rows are pre-aggregated |
| Oscillation diagnostic | The implementation records period-2 detection as a diagnostic only; it does not alter `EDSP-canonical` | Named GS and damped variants are reported separately |
| Max-iteration prediction iterate | When a run reaches `max_iter` without convergence, predictions and calibration metrics use the last iterate; the last two iterates are also stored for diagnostics | Revisit only if a later evaluation protocol requires a different policy |
| Graph canonicalisation (PR-2) | Every loader symmetrises edges, removes self-loops and merges duplicate/reciprocal rows (`edsp.data.graph.canonicalize_edges`); counts are kept in `meta['canonicalization']`. For edge attributes the first occurrence is kept and conflicting merges are counted; signs are summed (H-NATIVE rule §2.3a) | ogbn-proteins: if conflicting reciprocal attributes appear, decide between first/mean/max in PR-4 (P-NATIVE) |
| Planetoid preprocessing (PR-2) | Mirrors torch_geometric `read_planetoid_data`: test rows re-ordered, CiteSeer's 15 test-range gaps padded with zero features and label 0, padded nodes excluded from all splits | — |
| Registry format (PR-2) | `data/REGISTRY.json` instead of `REGISTRY.yaml`: no YAML dependency in the locked environment | Switch to YAML if a YAML parser is added to the lock |
| Sampson (PR-2) | Network `Sampson_T4` (18 novices, directed ±1 arcs) with partition `Sampson_factions_T4` (4 factions) from Pajek ESNA `Sampson.paj`; the Master Prompt's "25 nodes" refers to the unlabelled multi-time network | Owner may prefer another Sampson relation/time slice |
| Label access for dataset statistics (PR-2) | Full-label statistics read labels via `LabelVault.descriptive_labels()`, allowed only from `edsp.data.stats`; they describe datasets and are never inputs to a method | — |
| Constructed benchmark (PR-2) | `minesweeper` (Platonov et al.) is a constructed grid game, not a real network; kept as an established benchmark but flagged | Owner decision recorded in docs/PREREGISTRATION.md |
