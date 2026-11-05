# E1 — Track-A EDSP vs B0/B1 Track-A baselines

Run records: 298 (git ee2fa3f35311; runners actions-run-42112122532). Headline metric per dataset in brackets; mean ± std over splits (×100). Rows marked † are constructed benchmarks or case studies and are excluded from real-world aggregates.

| dataset [metric] | splits | EDSP-default | EDSP-tuned | majority | train_prior | harmonic | harmonic_cmn | lgc | ppr_lp | linbp | signed_vote |
|---|---|---|---|---|---|---|---|---|---|---|---|
| coauthor-physics [accuracy] | 10 | 95.42 ± 0.29 | 95.52 ± 0.28 | 50.52 ± 0.00 | 50.52 ± 0.00 | 95.51 ± 0.26 | 95.68 ± 0.27 | **95.68 ± 0.30** | 95.21 ± 0.31 | 94.78 ± 0.27 | — |
| questions [roc_auc] | 10 | 63.05 ± 1.66 | 64.56 ± 1.42 | 50.00 ± 0.00 | 50.00 ± 0.00 | 62.52 ± 1.61 | 62.52 ± 1.61 | **68.69 ± 1.44** | 67.22 ± 1.39 | 40.92 ± 1.18 | — |
| wikics [accuracy] | 12 | 41.46 ± 6.32 | 67.19 ± 2.64 | 22.90 ± 0.00 | 22.90 ± 0.00 | 44.74 ± 6.34 | 65.78 ± 1.33 | **72.75 ± 0.71** | 69.68 ± 1.41 | 63.39 ± 1.28 | — |

## Pre-registered secondary column: EDSP-default on non-converged runs

| dataset | non-converged runs | headline, last iterate | headline, mean of last two iterates |
|---|---|---|---|
| coauthor-physics | 0 / 10 | — | — |
| questions | 0 / 10 | — | — |
| wikics | 0 / 12 | — | — |

## EDSP-tuned vs each baseline: win / tie / loss

Per dataset: two-sided Wilcoxon signed-rank over splits on the headline metric; Holm correction across datasets per baseline; win/loss if adjusted p < 0.05, else tie. Real-world datasets only.

| baseline | win | tie | loss | datasets (W / L) |
|---|---|---|---|---|
| harmonic | 2 | 1 | 0 | W: questions, wikics; L: — |
| harmonic_cmn | 1 | 1 | 1 | W: questions; L: coauthor-physics |
| lgc | 0 | 0 | 3 | W: —; L: coauthor-physics, questions, wikics |
| ppr_lp | 1 | 0 | 2 | W: coauthor-physics; L: questions, wikics |
| linbp | 3 | 0 | 0 | W: coauthor-physics, questions, wikics; L: — |
| train_prior | 3 | 0 | 0 | W: coauthor-physics, questions, wikics; L: — |

## Friedman / Nemenyi (real-world datasets, headline metric)

Datasets: 3; Friedman p = 0.0262; Nemenyi CD (α = 0.05) = 6.06.

| method | average rank |
|---|---|
| lgc | 1.00 |
| EDSP-tuned | 3.00 |
| ppr_lp | 3.33 |
| harmonic_cmn | 4.00 |
| harmonic | 5.00 |
| EDSP-default | 5.33 |
| linbp | 6.67 |
| train_prior | 7.67 |

## Calibration (mean over splits; lower is better)

| dataset | EDSP-default NLL | EDSP-default ECE | EDSP-tuned NLL | EDSP-tuned ECE | best baseline by NLL (NLL / ECE) |
|---|---|---|---|---|---|
| coauthor-physics | 0.200 | 0.077 | 0.195 | 0.072 | lgc: 0.176 / 0.055 |
| questions | 0.233 | 0.046 | 0.206 | 0.023 | train_prior: 0.134 / 0.000 |
| wikics | 1.589 | 0.178 | 1.793 | 0.468 | lgc: 1.005 / 0.197 |

## H7 (descriptive): convergence and fixed-point non-uniqueness

| dataset | EDSP-default non-converged | oscillation flagged | mean fraction of nodes still moving | GS vs damped L1: mean | p95 | splits |
|---|---|---|---|---|---|---|
| coauthor-physics | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3 |
| questions | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3 |
| wikics | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3 |
