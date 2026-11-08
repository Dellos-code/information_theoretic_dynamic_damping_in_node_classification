# E3 — Track B (feature priors): EDSP-B vs logistic regression, Correct & Smooth and tuned GNNs, H4

Run records: 9 (git 3b2ae0acb743; runners actions-run-42467744211). Headline metric per dataset in brackets; mean ± std over splits (×100). Rows marked † are constructed benchmarks or case studies and are excluded from real-world aggregates.

| dataset [metric] | splits | EDSP-default | EDSP-tuned | majority | train_prior | feature_lr | correct_and_smooth | mlp | gcn | sage | gat | gprgnn | bwgnn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tolokers [roc_auc] | 1 | 71.85 ± 0.00 | 71.92 ± 0.00 | 50.00 ± 0.00 | 50.00 ± 0.00 | 70.98 ± 0.00 | 74.52 ± 0.00 | 72.11 ± 0.00 | **83.48 ± 0.00** | 82.20 ± 0.00 | — | — | — |

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
| tolokers | 0.550 | 0.155 | 0.550 | 0.155 | gcn: 0.394 / 0.027 |

## H7 (descriptive): convergence and fixed-point non-uniqueness

| dataset | EDSP-default non-converged | oscillation flagged | mean fraction of nodes still moving | GS vs damped L1: mean | p95 | splits |
|---|---|---|---|---|---|---|
| tolokers | 0.000 | 0.000 | 0.000 | — | — | 0 |

## Test accuracy by local homophily (descriptive, full labels; bin i = [i/10, (i+1)/10)), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tolokers | EDSP-default | 75.5 (94) | 18.2 (44) | 42.6 (115) | 49.5 (218) | 55.7 (334) | 76.6 (539) | 87.7 (399) | 89.3 (261) | 96.7 (480) | 91.2 (456) |
| tolokers | EDSP-tuned | 75.5 (94) | 18.2 (44) | 42.6 (115) | 49.5 (218) | 55.7 (334) | 76.6 (539) | 87.7 (399) | 89.3 (261) | 96.7 (480) | 91.2 (456) |
| tolokers | feature_lr | 72.3 (94) | 15.9 (44) | 46.1 (115) | 50.9 (218) | 55.1 (334) | 75.5 (539) | 87.7 (399) | 88.9 (261) | 96.7 (480) | 91.4 (456) |
| tolokers | correct_and_smooth | 42.6 (94) | 6.8 (44) | 24.3 (115) | 39.4 (218) | 48.2 (334) | 57.9 (539) | 78.2 (399) | 92.3 (261) | 98.1 (480) | 96.9 (456) |
| tolokers | gcn | 73.4 (94) | 20.5 (44) | 30.4 (115) | 49.5 (218) | 56.0 (334) | 80.1 (539) | 90.7 (399) | 96.9 (261) | 98.8 (480) | 93.9 (456) |
| tolokers | sage | 79.8 (94) | 15.9 (44) | 34.8 (115) | 47.7 (218) | 55.4 (334) | 77.9 (539) | 90.7 (399) | 94.3 (261) | 98.3 (480) | 93.4 (456) |

## Test accuracy by degree (bin 0: isolated; bin b: 2^(b-1) ≤ deg < 2^b; last bin open), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tolokers | EDSP-default | — | 67.3 (104) | 84.5 (233) | 86.2 (282) | 83.2 (404) | 75.1 (469) | 77.9 (488) | 78.7 (404) | 78.7 (287) | 73.8 (183) | 47.7 (86) |
| tolokers | EDSP-tuned | — | 67.3 (104) | 84.5 (233) | 86.2 (282) | 83.2 (404) | 75.1 (469) | 77.9 (488) | 78.7 (404) | 78.7 (287) | 73.8 (183) | 47.7 (86) |
| tolokers | feature_lr | — | 68.3 (104) | 83.3 (233) | 84.4 (282) | 83.9 (404) | 75.7 (469) | 77.9 (488) | 78.0 (404) | 78.7 (287) | 73.2 (183) | 47.7 (86) |
| tolokers | correct_and_smooth | — | 74.0 (104) | 73.4 (233) | 77.7 (282) | 75.0 (404) | 72.7 (469) | 69.9 (488) | 69.6 (404) | 70.4 (287) | 65.6 (183) | 47.7 (86) |
| tolokers | gcn | — | 74.0 (104) | 85.4 (233) | 87.9 (282) | 86.6 (404) | 77.8 (469) | 79.3 (488) | 80.0 (404) | 78.7 (287) | 74.9 (183) | 52.3 (86) |
| tolokers | sage | — | 76.9 (104) | 86.3 (233) | 86.9 (282) | 86.1 (404) | 75.7 (469) | 80.1 (488) | 79.0 (404) | 77.7 (287) | 73.8 (183) | 46.5 (86) |
