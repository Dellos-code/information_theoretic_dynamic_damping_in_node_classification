# E1 — Track-A EDSP vs B0/B1 Track-A baselines

Run records: 1646 (git 694f4e582fc2; runners actions-run-42067613272). Headline metric per dataset in brackets; mean ± std over splits (×100). Rows marked † are constructed benchmarks or case studies and are excluded from real-world aggregates.

| dataset [metric] | splits | EDSP-default | EDSP-tuned | majority | train_prior | harmonic | harmonic_cmn | lgc | ppr_lp | linbp | signed_vote |
|---|---|---|---|---|---|---|---|---|---|---|---|
| amazon-computers [accuracy] | 10 | 87.71 ± 0.45 | 87.77 ± 0.46 | 37.49 ± 0.00 | 37.49 ± 0.00 | 87.84 ± 0.41 | 87.87 ± 0.42 | **89.04 ± 0.48** | 88.03 ± 0.40 | 85.42 ± 0.41 | — |
| amazon-photo [accuracy] | 10 | 90.71 ± 0.60 | 90.67 ± 0.64 | 25.38 ± 0.00 | 25.38 ± 0.00 | 90.80 ± 0.63 | 91.04 ± 0.52 | **91.31 ± 0.45** | 90.55 ± 0.57 | 89.41 ± 0.46 | — |
| amazon-ratings [accuracy] | 10 | 44.00 ± 0.40 | 44.07 ± 0.43 | 36.79 ± 0.00 | 36.79 ± 0.00 | 44.30 ± 0.42 | 43.89 ± 0.27 | **46.86 ± 0.47** | 46.02 ± 0.50 | 42.29 ± 0.60 | — |
| chameleon-filtered [accuracy] | 10 | 30.05 ± 5.60 | 29.95 ± 6.01 | 24.99 ± 3.52 | 24.99 ± 3.52 | 30.37 ± 5.61 | 34.60 ± 2.67 | **36.51 ± 3.57** | 33.08 ± 7.13 | 33.94 ± 2.36 | — |
| citeseer [accuracy] | 11 | 67.40 ± 5.34 | 67.88 ± 5.39 | 19.95 ± 3.88 | 19.95 ± 3.88 | 67.49 ± 5.33 | **68.61 ± 5.53** | 68.31 ± 5.72 | 68.34 ± 5.84 | 65.28 ± 13.05 | — |
| coauthor-cs [accuracy] | 10 | 91.03 ± 0.50 | 91.17 ± 0.49 | 22.56 ± 0.00 | 22.56 ± 0.00 | 91.11 ± 0.47 | **91.20 ± 0.51** | 91.10 ± 0.52 | 90.53 ± 0.45 | 89.64 ± 0.62 | — |
| coauthor-physics [accuracy] | 2 | 95.36 ± 0.42 | 95.50 ± 0.36 | 50.52 ± 0.00 | 50.52 ± 0.00 | 95.40 ± 0.37 | **95.57 ± 0.38** | 95.03 ± 0.00 | 94.74 ± 0.00 | 94.45 ± 0.00 | — |
| cora [accuracy] | 11 | 83.73 ± 4.04 | 83.75 ± 3.99 | 28.62 ± 4.94 | 28.62 ± 4.94 | 83.64 ± 3.94 | **84.64 ± 3.85** | 84.64 ± 4.24 | 83.63 ± 4.59 | 81.17 ± 7.62 | — |
| minesweeper† [roc_auc] | 10 | 50.62 ± 0.93 | 50.01 ± 1.16 | 50.00 ± 0.00 | 50.00 ± 0.00 | 50.60 ± 0.94 | 50.60 ± 0.94 | 51.00 ± 1.01 | **51.02 ± 1.01** | 50.30 ± 1.12 | — |
| pubmed [accuracy] | 11 | 77.05 ± 16.41 | 79.33 ± 10.02 | 37.95 ± 6.31 | 37.95 ± 6.31 | 76.82 ± 17.38 | 81.32 ± 2.99 | **82.23 ± 3.52** | 81.30 ± 3.83 | 75.94 ± 18.33 | — |
| questions [roc_auc] | 8 | 63.07 ± 1.85 | 64.55 ± 1.51 | 50.00 ± 0.00 | 50.00 ± 0.00 | 62.53 ± 1.80 | 62.53 ± 1.80 | **68.45 ± 1.20** | 67.16 ± 1.18 | 40.75 ± 1.10 | — |
| roman-empire [accuracy] | 10 | 5.25 ± 0.26 | 9.06 ± 0.28 | 13.96 ± 0.00 | 13.96 ± 0.00 | 5.05 ± 0.27 | 5.21 ± 0.26 | 10.60 ± 0.20 | 10.75 ± 0.29 | **27.54 ± 0.35** | — |
| sampson† [accuracy] | 18 | **66.67 ± 47.14** | **66.67 ± 47.14** | 38.89 ± 48.75 | 38.89 ± 48.75 | 44.44 ± 49.69 | 50.00 ± 50.00 | 44.44 ± 49.69 | 44.44 ± 49.69 | 38.89 ± 48.75 | **66.67 ± 47.14** |
| squirrel-filtered [accuracy] | 10 | 20.60 ± 1.82 | 21.43 ± 1.44 | **35.05 ± 1.22** | **35.05 ± 1.22** | 20.74 ± 2.06 | 27.12 ± 1.80 | 34.83 ± 1.53 | 22.82 ± 4.98 | 24.48 ± 2.30 | — |
| tolokers [roc_auc] | 10 | 71.07 ± 0.95 | 72.01 ± 1.34 | 50.00 ± 0.00 | 50.00 ± 0.00 | 71.13 ± 0.96 | 71.13 ± 0.96 | **72.65 ± 0.96** | 72.50 ± 0.91 | 58.78 ± 1.24 | — |
| wikics [accuracy] | 20 | 42.12 ± 6.78 | 67.21 ± 2.46 | 22.90 ± 0.00 | 22.90 ± 0.00 | 45.56 ± 6.42 | 65.96 ± 1.21 | **72.88 ± 0.59** | 69.86 ± 1.29 | 63.35 ± 1.72 | — |

## Pre-registered secondary column: EDSP-default on non-converged runs

| dataset | non-converged runs | headline, last iterate | headline, mean of last two iterates |
|---|---|---|---|
| amazon-computers | 0 / 10 | — | — |
| amazon-photo | 0 / 10 | — | — |
| amazon-ratings | 0 / 10 | — | — |
| chameleon-filtered | 0 / 10 | — | — |
| citeseer | 1 / 11 | 51.50 | 51.50 |
| coauthor-cs | 0 / 10 | — | — |
| coauthor-physics | 0 / 2 | — | — |
| cora | 1 / 11 | 71.40 | 71.40 |
| minesweeper | 0 / 10 | — | — |
| pubmed | 1 / 11 | 25.20 | 25.20 |
| questions | 0 / 8 | — | — |
| roman-empire | 0 / 10 | — | — |
| sampson | 0 / 18 | — | — |
| squirrel-filtered | 0 / 10 | — | — |
| tolokers | 0 / 10 | — | — |
| wikics | 0 / 20 | — | — |

## EDSP-tuned vs each baseline: win / tie / loss

Per dataset: two-sided Wilcoxon signed-rank over splits on the headline metric; Holm correction across datasets per baseline; win/loss if adjusted p < 0.05, else tie. Real-world datasets only.

| baseline | win | tie | loss | datasets (W / L) |
|---|---|---|---|---|
| harmonic | 2 | 11 | 1 | W: roman-empire, wikics; L: amazon-ratings |
| harmonic_cmn | 1 | 12 | 1 | W: roman-empire; L: squirrel-filtered |
| lgc | 0 | 3 | 10 | W: —; L: amazon-computers, amazon-photo, amazon-ratings, chameleon-filtered, cora, pubmed, questions, roman-empire, squirrel-filtered, wikics |
| ppr_lp | 1 | 9 | 3 | W: coauthor-cs; L: amazon-ratings, roman-empire, wikics |
| linbp | 8 | 4 | 1 | W: amazon-computers, amazon-photo, amazon-ratings, coauthor-cs, cora, pubmed, tolokers, wikics; L: roman-empire |
| train_prior | 10 | 2 | 2 | W: amazon-computers, amazon-photo, amazon-ratings, citeseer, coauthor-cs, cora, pubmed, questions, tolokers, wikics; L: roman-empire, squirrel-filtered |

## Friedman / Nemenyi (real-world datasets, headline metric)

Datasets: 14; Friedman p = 2.35e-07; Nemenyi CD (α = 0.05) = 2.81.

| method | average rank |
|---|---|
| lgc | 2.00 |
| harmonic_cmn | 3.07 |
| ppr_lp | 3.64 |
| EDSP-tuned | 4.00 |
| harmonic | 4.86 |
| EDSP-default | 5.43 |
| linbp | 6.00 |
| train_prior | 7.00 |

## Calibration (mean over splits; lower is better)

| dataset | EDSP-default NLL | EDSP-default ECE | EDSP-tuned NLL | EDSP-tuned ECE | best baseline by NLL (NLL / ECE) |
|---|---|---|---|---|---|
| amazon-computers | 0.512 | 0.153 | 0.509 | 0.151 | lgc: 0.408 / 0.061 |
| amazon-photo | 0.388 | 0.118 | 0.391 | 0.120 | lgc: 0.335 / 0.070 |
| amazon-ratings | 1.456 | 0.131 | 1.490 | 0.139 | ppr_lp: 1.325 / 0.054 |
| chameleon-filtered | 1.777 | 0.185 | 1.758 | 0.142 | lgc: 1.545 / 0.115 |
| citeseer | 2.169 | 0.131 | 2.221 | 0.128 | linbp: 1.589 / 0.419 |
| coauthor-cs | 0.438 | 0.149 | 0.426 | 0.136 | ppr_lp: 0.436 / 0.115 |
| coauthor-physics | 0.201 | 0.077 | 0.195 | 0.072 | lgc: 0.176 / 0.038 |
| cora | 0.583 | 0.078 | 0.588 | 0.077 | lgc: 0.565 / 0.117 |
| minesweeper | 0.571 | 0.089 | 0.636 | 0.137 | train_prior: 0.500 / 0.000 |
| pubmed | 0.616 | 0.080 | 0.607 | 0.073 | lgc: 0.489 / 0.050 |
| questions | 0.233 | 0.047 | 0.202 | 0.024 | train_prior: 0.134 / 0.000 |
| roman-empire | 5.005 | 0.472 | 2.895 | 0.033 | train_prior: 2.614 / 0.000 |
| sampson | 0.962 | 0.469 | 0.980 | 0.467 | linbp: 1.377 / 0.445 |
| squirrel-filtered | 1.875 | 0.178 | 1.660 | 0.042 | train_prior: 1.528 / 0.021 |
| tolokers | 0.549 | 0.076 | 0.527 | 0.069 | lgc: 0.508 / 0.074 |
| wikics | 1.588 | 0.180 | 1.800 | 0.470 | lgc: 1.018 / 0.212 |

## H7 (descriptive): convergence and fixed-point non-uniqueness

| dataset | EDSP-default non-converged | oscillation flagged | mean fraction of nodes still moving | GS vs damped L1: mean | p95 | splits |
|---|---|---|---|---|---|---|
| amazon-computers | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3 |
| amazon-photo | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3 |
| amazon-ratings | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3 |
| chameleon-filtered | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3 |
| citeseer | 0.091 | 0.000 | 0.003 | 0.000 | 0.001 | 3 |
| coauthor-cs | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3 |
| coauthor-physics | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 2 |
| cora | 0.091 | 0.000 | 0.001 | 0.000 | 0.000 | 3 |
| minesweeper | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3 |
| pubmed | 0.091 | 0.000 | 0.091 | 0.002 | 0.002 | 3 |
| questions | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3 |
| roman-empire | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3 |
| sampson | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3 |
| squirrel-filtered | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3 |
| tolokers | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3 |
| wikics | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3 |
