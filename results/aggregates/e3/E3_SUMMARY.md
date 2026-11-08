# E3 — Track B (feature priors): EDSP-B vs logistic regression, Correct & Smooth and tuned GNNs, H4

Run records: 9 (git 3b2ae0acb743; runners actions-run-42467757225). Headline metric per dataset in brackets; mean ± std over splits (×100). Rows marked † are constructed benchmarks or case studies and are excluded from real-world aggregates.

| dataset [metric] | splits | EDSP-default | EDSP-tuned | majority | train_prior | feature_lr | correct_and_smooth | mlp | gcn | sage | gat | gprgnn | bwgnn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tolokers [roc_auc] | 1 | 72.67 ± 0.00 | 74.87 ± 0.00 | 50.00 ± 0.00 | 50.00 ± 0.00 | 72.77 ± 0.00 | 75.81 ± 0.00 | 73.63 ± 0.00 | 82.83 ± 0.00 | **83.17 ± 0.00** | — | — | — |

## Pre-registered secondary column: EDSP-default on non-converged runs

| dataset | non-converged runs | headline, last iterate | headline, mean of last two iterates |
|---|---|---|---|
| tolokers | 0 / 1 | — | — |

## EDSP-tuned vs each baseline: win / tie / loss

Per dataset: two-sided Wilcoxon signed-rank over splits on the headline metric; Holm correction across datasets per baseline; win/loss if adjusted p < 0.05, else tie. Real-world datasets only.

| baseline | win | tie | loss | datasets (W / L) |
|---|---|---|---|---|
| feature_lr | 0 | 0 | 0 | W: —; L: — |
| correct_and_smooth | 0 | 0 | 0 | W: —; L: — |
| mlp | 0 | 0 | 0 | W: —; L: — |
| gcn | 0 | 0 | 0 | W: —; L: — |
| sage | 0 | 0 | 0 | W: —; L: — |
| gat | 0 | 0 | 0 | W: —; L: — |
| gprgnn | 0 | 0 | 0 | W: —; L: — |
| bwgnn | 0 | 0 | 0 | W: —; L: — |

## Calibration (mean over splits; lower is better)

| dataset | EDSP-default NLL | EDSP-default ECE | EDSP-tuned NLL | EDSP-tuned ECE | best baseline by NLL (NLL / ECE) |
|---|---|---|---|---|---|
| tolokers | 0.555 | 0.162 | 0.559 | 0.162 | sage: 0.389 / 0.008 |

## H7 (descriptive): convergence and fixed-point non-uniqueness

| dataset | EDSP-default non-converged | oscillation flagged | mean fraction of nodes still moving | GS vs damped L1: mean | p95 | splits |
|---|---|---|---|---|---|---|
| tolokers | 0.000 | 0.000 | 0.000 | — | — | 0 |

## Test accuracy by local homophily (descriptive, full labels; bin i = [i/10, (i+1)/10)), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tolokers | EDSP-default | 77.9 (95) | 22.7 (44) | 50.9 (114) | 46.9 (226) | 55.7 (350) | 78.2 (522) | 85.8 (344) | 87.5 (288) | 97.0 (470) | 91.2 (487) |
| tolokers | EDSP-tuned | 72.6 (95) | 20.5 (44) | 46.5 (114) | 49.1 (226) | 52.9 (350) | 74.9 (522) | 83.7 (344) | 87.8 (288) | 96.8 (470) | 91.4 (487) |
| tolokers | feature_lr | 74.7 (95) | 22.7 (44) | 51.8 (114) | 47.8 (226) | 56.3 (350) | 78.7 (522) | 86.0 (344) | 87.8 (288) | 96.6 (470) | 91.6 (487) |
| tolokers | correct_and_smooth | 54.7 (95) | 18.2 (44) | 43.9 (114) | 54.0 (226) | 49.7 (350) | 49.6 (522) | 59.3 (344) | 82.6 (288) | 94.7 (470) | 94.7 (487) |
| tolokers | gcn | 75.8 (95) | 18.2 (44) | 43.9 (114) | 51.3 (226) | 56.3 (350) | 81.0 (522) | 92.2 (344) | 95.5 (288) | 99.6 (470) | 93.0 (487) |
| tolokers | sage | 72.6 (95) | 20.5 (44) | 43.9 (114) | 54.0 (226) | 59.1 (350) | 80.7 (522) | 89.8 (344) | 94.8 (288) | 99.4 (470) | 92.8 (487) |

## Test accuracy by degree (bin 0: isolated; bin b: 2^(b-1) ≤ deg < 2^b; last bin open), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tolokers | EDSP-default | — | 73.0 (111) | 81.5 (233) | 87.5 (295) | 81.9 (430) | 76.6 (436) | 79.2 (504) | 78.4 (388) | 76.5 (277) | 73.5 (185) | 39.5 (81) |
| tolokers | EDSP-tuned | — | 71.2 (111) | 80.7 (233) | 84.4 (295) | 79.3 (430) | 78.4 (436) | 77.8 (504) | 74.7 (388) | 75.8 (277) | 74.1 (185) | 38.3 (81) |
| tolokers | feature_lr | — | 73.9 (111) | 82.0 (233) | 87.5 (295) | 82.1 (430) | 77.8 (436) | 79.0 (504) | 78.1 (388) | 76.5 (277) | 74.1 (185) | 39.5 (81) |
| tolokers | correct_and_smooth | — | 68.5 (111) | 76.4 (233) | 81.4 (295) | 74.4 (430) | 71.3 (436) | 64.3 (504) | 61.6 (388) | 62.1 (277) | 63.8 (185) | 43.2 (81) |
| tolokers | gcn | — | 74.8 (111) | 87.1 (233) | 90.5 (295) | 85.8 (430) | 80.7 (436) | 81.0 (504) | 80.7 (388) | 76.9 (277) | 73.5 (185) | 43.2 (81) |
| tolokers | sage | — | 73.9 (111) | 87.1 (233) | 89.5 (295) | 85.1 (430) | 80.7 (436) | 81.9 (504) | 78.9 (388) | 77.3 (277) | 73.0 (185) | 54.3 (81) |
