# E3 — Track B (feature priors): EDSP-B vs logistic regression, Correct & Smooth and tuned GNNs, H4

Run records: 82 (git 3b2ae0acb743, 5e6b302025fd; runners actions-run-42405479377, actions-run-42471157812). Headline metric per dataset in brackets; mean ± std over splits (×100). Rows marked † are constructed benchmarks or case studies and are excluded from real-world aggregates.

| dataset [metric] | splits | EDSP-default | EDSP-tuned | majority | train_prior | feature_lr | correct_and_smooth | mlp | gcn | sage | gat | gprgnn | bwgnn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| amazon-fraud [roc_auc] | 10 | 94.07 ± 1.05 | 96.62 ± 0.76 | 50.00 ± 0.00 | 50.00 ± 0.00 | 96.55 ± 0.91 | 95.65 ± 0.60 | **96.97 ± 0.34** | — | — | — | — | 96.96 ± 0.42 |
| questions [roc_auc] | 0 | — | — | — | — | — | — | — | — | — | — | **74.82 ± 0.54** | — |

## Pre-registered secondary column: EDSP-default on non-converged runs

| dataset | non-converged runs | headline, last iterate | headline, mean of last two iterates |
|---|---|---|---|
| amazon-fraud | 0 / 10 | — | — |
| questions | 0 / 0 | — | — |

## EDSP-tuned vs each baseline: win / tie / loss

Per dataset: two-sided Wilcoxon signed-rank over splits on the headline metric; Holm correction across datasets per baseline; win/loss if adjusted p < 0.05, else tie. Real-world datasets only.

| baseline | win | tie | loss | datasets (W / L) |
|---|---|---|---|---|
| feature_lr | 0 | 1 | 0 | W: —; L: — |
| correct_and_smooth | 1 | 0 | 0 | W: amazon-fraud; L: — |
| mlp | 0 | 1 | 0 | W: —; L: — |
| gcn | 0 | 0 | 0 | W: —; L: — |
| sage | 0 | 0 | 0 | W: —; L: — |
| gat | 0 | 0 | 0 | W: —; L: — |
| gprgnn | 0 | 0 | 0 | W: —; L: — |
| bwgnn | 0 | 1 | 0 | W: —; L: — |

## Calibration (mean over splits; lower is better)

| dataset | EDSP-default NLL | EDSP-default ECE | EDSP-tuned NLL | EDSP-tuned ECE | best baseline by NLL (NLL / ECE) |
|---|---|---|---|---|---|
| amazon-fraud | 0.309 | 0.184 | 0.284 | 0.200 | mlp: 0.106 / 0.012 |
| questions | nan | nan | nan | nan | gprgnn: 0.118 / 0.003 |

## H7 (descriptive): convergence and fixed-point non-uniqueness

| dataset | EDSP-default non-converged | oscillation flagged | mean fraction of nodes still moving | GS vs damped L1: mean | p95 | splits |
|---|---|---|---|---|---|---|
| amazon-fraud | 0.000 | 0.000 | 0.000 | — | — | 0 |
| questions | — | — | — | — | — | 0 |

## Test accuracy by local homophily (descriptive, full labels; bin i = [i/10, (i+1)/10)), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| amazon-fraud | EDSP-default | 29.5 (1632) | 69.0 (959) | 73.5 (388) | 69.2 (182) | 87.1 (85) | 90.2 (51) | 94.5 (73) | 97.2 (109) | 98.0 (1998) | 99.9 (29083) |
| amazon-fraud | EDSP-tuned | 58.9 (1632) | 88.5 (959) | 92.0 (388) | 97.8 (182) | 100.0 (85) | 100.0 (51) | 94.5 (73) | 95.4 (109) | 97.3 (1998) | 99.6 (29083) |
| amazon-fraud | feature_lr | 58.3 (1632) | 91.2 (959) | 94.3 (388) | 98.9 (182) | 100.0 (85) | 100.0 (51) | 94.5 (73) | 96.3 (109) | 97.3 (1998) | 99.8 (29083) |
| amazon-fraud | correct_and_smooth | 44.7 (1632) | 76.6 (959) | 85.1 (388) | 95.6 (182) | 96.5 (85) | 100.0 (51) | 94.5 (73) | 99.1 (109) | 98.1 (1998) | 99.9 (29083) |

## Test accuracy by degree (bin 0: isolated; bin b: 2^(b-1) ≤ deg < 2^b; last bin open), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| amazon-fraud | EDSP-default | — | — | 100.0 (6) | 83.3 (42) | 82.8 (134) | 84.7 (556) | 88.1 (1443) | 89.7 (3027) | 92.2 (5458) | 95.5 (7502) | 98.0 (16392) |
| amazon-fraud | EDSP-tuned | — | — | 100.0 (6) | 88.1 (42) | 91.8 (134) | 94.6 (556) | 96.3 (1443) | 96.4 (3027) | 95.7 (5458) | 96.6 (7502) | 98.2 (16392) |
| amazon-fraud | feature_lr | — | — | 100.0 (6) | 95.2 (42) | 93.3 (134) | 94.2 (556) | 97.5 (1443) | 96.7 (3027) | 95.8 (5458) | 97.0 (7502) | 98.3 (16392) |
| amazon-fraud | correct_and_smooth | — | — | 100.0 (6) | 88.1 (42) | 93.3 (134) | 93.9 (556) | 96.2 (1443) | 94.0 (3027) | 94.0 (5458) | 95.6 (7502) | 98.0 (16392) |
