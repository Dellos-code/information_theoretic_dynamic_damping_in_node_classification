# E3 — Track B (feature priors): EDSP-B vs logistic regression, Correct & Smooth and tuned GNNs, H4

Run records: 9 (git 3b2ae0acb743; runners actions-run-42471122471). Headline metric per dataset in brackets; mean ± std over splits (×100). Rows marked † are constructed benchmarks or case studies and are excluded from real-world aggregates.

| dataset [metric] | splits | EDSP-default | EDSP-tuned | majority | train_prior | feature_lr | correct_and_smooth | mlp | gcn | sage | gat | gprgnn | bwgnn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tolokers [roc_auc] | 1 | 71.57 ± 0.00 | 72.91 ± 0.00 | 50.00 ± 0.00 | 50.00 ± 0.00 | 71.45 ± 0.00 | 75.53 ± 0.00 | 72.10 ± 0.00 | **83.51 ± 0.00** | 83.36 ± 0.00 | — | — | — |

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
| tolokers | 0.552 | 0.157 | 0.558 | 0.151 | gcn: 0.392 / 0.033 |

## H7 (descriptive): convergence and fixed-point non-uniqueness

| dataset | EDSP-default non-converged | oscillation flagged | mean fraction of nodes still moving | GS vs damped L1: mean | p95 | splits |
|---|---|---|---|---|---|---|
| tolokers | 0.000 | 0.000 | 0.000 | — | — | 0 |

## Test accuracy by local homophily (descriptive, full labels; bin i = [i/10, (i+1)/10)), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tolokers | EDSP-default | 77.9 (77) | 21.9 (32) | 45.5 (99) | 48.2 (218) | 53.6 (362) | 77.5 (538) | 84.9 (385) | 85.7 (252) | 96.3 (487) | 93.3 (490) |
| tolokers | EDSP-tuned | 70.1 (77) | 21.9 (32) | 43.4 (99) | 48.6 (218) | 49.2 (362) | 74.2 (538) | 81.0 (385) | 86.5 (252) | 96.7 (487) | 92.2 (490) |
| tolokers | feature_lr | 74.0 (77) | 21.9 (32) | 47.5 (99) | 49.5 (218) | 53.6 (362) | 77.1 (538) | 84.9 (385) | 86.1 (252) | 96.1 (487) | 93.1 (490) |
| tolokers | correct_and_smooth | 58.4 (77) | 21.9 (32) | 36.4 (99) | 43.6 (218) | 49.4 (362) | 50.2 (538) | 59.2 (385) | 82.1 (252) | 96.3 (487) | 95.5 (490) |
| tolokers | gcn | 81.8 (77) | 15.6 (32) | 37.4 (99) | 48.6 (218) | 55.5 (362) | 80.7 (538) | 90.1 (385) | 94.8 (252) | 99.0 (487) | 93.9 (490) |
| tolokers | sage | 80.5 (77) | 15.6 (32) | 38.4 (99) | 50.0 (218) | 55.2 (362) | 79.9 (538) | 89.9 (385) | 95.6 (252) | 99.2 (487) | 94.9 (490) |

## Test accuracy by degree (bin 0: isolated; bin b: 2^(b-1) ≤ deg < 2^b; last bin open), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tolokers | EDSP-default | — | 72.6 (95) | 85.1 (202) | 85.2 (305) | 81.2 (394) | 79.5 (458) | 78.9 (497) | 77.2 (412) | 77.0 (296) | 73.2 (198) | 34.9 (83) |
| tolokers | EDSP-tuned | — | 64.2 (95) | 83.2 (202) | 82.3 (305) | 79.2 (394) | 78.8 (458) | 76.9 (497) | 75.0 (412) | 74.7 (296) | 73.2 (198) | 36.1 (83) |
| tolokers | feature_lr | — | 70.5 (95) | 84.7 (202) | 84.9 (305) | 82.2 (394) | 79.7 (458) | 78.3 (497) | 76.9 (412) | 77.0 (296) | 74.2 (198) | 34.9 (83) |
| tolokers | correct_and_smooth | — | 70.5 (95) | 76.2 (202) | 80.0 (305) | 72.3 (394) | 69.9 (458) | 65.0 (497) | 64.8 (412) | 60.1 (296) | 66.2 (198) | 42.2 (83) |
| tolokers | gcn | — | 73.7 (95) | 86.6 (202) | 87.9 (305) | 85.5 (394) | 82.5 (458) | 81.1 (497) | 80.3 (412) | 77.7 (296) | 75.3 (198) | 39.8 (83) |
| tolokers | sage | — | 81.1 (95) | 85.1 (202) | 89.2 (305) | 84.8 (394) | 82.3 (458) | 81.1 (497) | 78.9 (412) | 77.7 (296) | 74.2 (198) | 50.6 (83) |
