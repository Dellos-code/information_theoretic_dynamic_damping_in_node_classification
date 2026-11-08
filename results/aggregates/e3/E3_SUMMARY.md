# E3 — Track B (feature priors): EDSP-B vs logistic regression, Correct & Smooth and tuned GNNs, H4

Run records: 70 (git 872886226084; runners actions-run-42480358577). Headline metric per dataset in brackets; mean ± std over splits (×100). Rows marked † are constructed benchmarks or case studies and are excluded from real-world aggregates.

| dataset [metric] | splits | EDSP-default | EDSP-tuned | majority | train_prior | feature_lr | correct_and_smooth | mlp | gcn | sage | gat | gprgnn | bwgnn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coauthor-cs [accuracy] | 10 | **95.83 ± 0.27** | 95.78 ± 0.22 | 22.56 ± 0.00 | 22.56 ± 0.00 | 94.72 ± 0.23 | 95.83 ± 0.21 | 95.40 ± 0.22 | — | — | — | — | — |

## Pre-registered secondary column: EDSP-default on non-converged runs

| dataset | non-converged runs | headline, last iterate | headline, mean of last two iterates |
|---|---|---|---|
| coauthor-cs | 0 / 10 | — | — |

## EDSP-tuned vs each baseline: win / tie / loss

Per dataset: two-sided Wilcoxon signed-rank over splits on the headline metric; Holm correction across datasets per baseline; win/loss if adjusted p < 0.05, else tie. Real-world datasets only.

| baseline | win | tie | loss | datasets (W / L) |
|---|---|---|---|---|
| feature_lr | 1 | 0 | 0 | W: coauthor-cs; L: — |
| correct_and_smooth | 0 | 1 | 0 | W: —; L: — |
| mlp | 1 | 0 | 0 | W: coauthor-cs; L: — |
| gcn | 0 | 0 | 0 | W: —; L: — |
| sage | 0 | 0 | 0 | W: —; L: — |
| gat | 0 | 0 | 0 | W: —; L: — |
| gprgnn | 0 | 0 | 0 | W: —; L: — |
| bwgnn | 0 | 0 | 0 | W: —; L: — |

## Calibration (mean over splits; lower is better)

| dataset | EDSP-default NLL | EDSP-default ECE | EDSP-tuned NLL | EDSP-tuned ECE | best baseline by NLL (NLL / ECE) |
|---|---|---|---|---|---|
| coauthor-cs | 0.615 | 0.384 | 0.612 | 0.379 | mlp: 0.155 / 0.018 |

## H7 (descriptive): convergence and fixed-point non-uniqueness

| dataset | EDSP-default non-converged | oscillation flagged | mean fraction of nodes still moving | GS vs damped L1: mean | p95 | splits |
|---|---|---|---|---|---|---|
| coauthor-cs | 0.000 | 0.000 | 0.000 | — | — | 0 |

## Test accuracy by local homophily (descriptive, full labels; bin i = [i/10, (i+1)/10)), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| coauthor-cs | EDSP-default | 59.3 (783) | 52.2 (247) | 67.9 (607) | 78.7 (847) | 84.1 (787) | 90.3 (2196) | 94.6 (2763) | 97.0 (2811) | 98.8 (4718) | 99.6 (20891) |
| coauthor-cs | EDSP-tuned | 63.3 (783) | 53.8 (247) | 68.9 (607) | 79.1 (847) | 84.1 (787) | 90.2 (2196) | 93.9 (2763) | 96.8 (2811) | 98.7 (4718) | 99.4 (20891) |
| coauthor-cs | feature_lr | 75.6 (783) | 59.5 (247) | 72.8 (607) | 81.1 (847) | 84.6 (787) | 89.0 (2196) | 91.6 (2763) | 94.2 (2811) | 97.2 (4718) | 97.9 (20891) |
| coauthor-cs | correct_and_smooth | 66.4 (783) | 50.2 (247) | 68.0 (607) | 80.4 (847) | 85.0 (787) | 91.2 (2196) | 93.6 (2763) | 96.7 (2811) | 98.6 (4718) | 99.3 (20891) |

## Test accuracy by degree (bin 0: isolated; bin b: 2^(b-1) ≤ deg < 2^b; last bin open), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coauthor-cs | EDSP-default | — | 93.9 (2099) | 95.9 (7147) | 96.4 (12223) | 95.4 (10178) | 96.2 (3943) | 95.6 (955) | 98.1 (104) | 100.0 (1) | — | — |
| coauthor-cs | EDSP-tuned | — | 94.3 (2099) | 95.8 (7147) | 96.3 (12223) | 95.3 (10178) | 96.1 (3943) | 95.6 (955) | 98.1 (104) | 100.0 (1) | — | — |
| coauthor-cs | feature_lr | — | 93.6 (2099) | 94.3 (7147) | 94.8 (12223) | 94.6 (10178) | 95.7 (3943) | 95.8 (955) | 98.1 (104) | 100.0 (1) | — | — |
| coauthor-cs | correct_and_smooth | — | 95.0 (2099) | 96.0 (7147) | 96.1 (12223) | 95.5 (10178) | 96.1 (3943) | 95.2 (955) | 99.0 (104) | 100.0 (1) | — | — |

## Cell C3c (mixed by local structure: top-5 std of local homophily)

| dataset | std local homophily | min compat. distance | EDSP-default | EDSP-tuned | feature_lr | correct_and_smooth | mlp | gcn | sage | gat | gprgnn | bwgnn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coauthor-cs | 0.239 | 1.056 | **95.83** | 95.78 | 94.72 | 95.83 | 95.40 | — | — | — | — | — |

## Cell C3d (structured compatibility: top-5 min Frobenius distance to I and to (J-I)/(k-1))

| dataset | std local homophily | min compat. distance | EDSP-default | EDSP-tuned | feature_lr | correct_and_smooth | mlp | gcn | sage | gat | gprgnn | bwgnn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coauthor-cs | 0.239 | 1.056 | **95.83** | 95.78 | 94.72 | 95.83 | 95.40 | — | — | — | — | — |
