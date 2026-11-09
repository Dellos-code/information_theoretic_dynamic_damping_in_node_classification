# E2 — native-polarity graphs (signed, relation-typed): Track-A EDSP vs Track-A baselines

Run records: 10 (git bdd1ad76f33a; runners actions-run-42626314495). Headline metric per dataset in brackets; mean ± std over splits (×100). Rows marked † are constructed benchmarks or case studies and are excluded from real-world aggregates.

| dataset [metric] | splits | EDSP-default | EDSP-tuned | EDSP-fullH-default | EDSP-fullH-tuned | majority | train_prior | harmonic | harmonic_cmn | lgc | ppr_lp | linbp | signed_vote | signed_lgc | snea |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wiki-editor [accuracy] | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — | **71.59 ± 0.61** |

## Pre-registered secondary column: EDSP-default on non-converged runs

| dataset | non-converged runs | headline, last iterate | headline, mean of last two iterates |
|---|---|---|---|
| wiki-editor | 0 / 0 | — | — |

## EDSP-tuned vs each baseline: win / tie / loss

Per dataset: two-sided Wilcoxon signed-rank over splits on the headline metric; Holm correction across datasets per baseline; win/loss if adjusted p < 0.05, else tie. Real-world datasets only.

| baseline | win | tie | loss | datasets (W / L) |
|---|---|---|---|---|
| harmonic | 0 | 0 | 0 | W: —; L: — |
| harmonic_cmn | 0 | 0 | 0 | W: —; L: — |
| lgc | 0 | 0 | 0 | W: —; L: — |
| ppr_lp | 0 | 0 | 0 | W: —; L: — |
| linbp | 0 | 0 | 0 | W: —; L: — |
| train_prior | 0 | 0 | 0 | W: —; L: — |
| signed_vote | 0 | 0 | 0 | W: —; L: — |
| signed_lgc | 0 | 0 | 0 | W: —; L: — |
| snea | 0 | 0 | 0 | W: —; L: — |

## Calibration (mean over splits; lower is better)

| dataset | EDSP-default NLL | EDSP-default ECE | EDSP-tuned NLL | EDSP-tuned ECE | best baseline by NLL (NLL / ECE) |
|---|---|---|---|---|---|
| wiki-editor | nan | nan | nan | nan | snea: 0.586 / 0.032 |

## H7 (descriptive): convergence and fixed-point non-uniqueness

| dataset | EDSP-default non-converged | oscillation flagged | mean fraction of nodes still moving | GS vs damped L1: mean | p95 | splits |
|---|---|---|---|---|---|---|
| wiki-editor | — | — | — | — | — | 0 |
