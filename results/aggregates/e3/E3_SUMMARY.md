# E3 — Track B (feature priors): EDSP-B vs logistic regression, Correct & Smooth and tuned GNNs, H4

Run records: 9 (git 3b2ae0acb743; runners actions-run-42471135631). Headline metric per dataset in brackets; mean ± std over splits (×100). Rows marked † are constructed benchmarks or case studies and are excluded from real-world aggregates.

| dataset [metric] | splits | EDSP-default | EDSP-tuned | majority | train_prior | feature_lr | correct_and_smooth | mlp | gcn | sage | gat | gprgnn | bwgnn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tolokers [roc_auc] | 1 | 73.75 ± 0.00 | 74.09 ± 0.00 | 50.00 ± 0.00 | 50.00 ± 0.00 | 71.99 ± 0.00 | 75.03 ± 0.00 | 74.07 ± 0.00 | **83.02 ± 0.00** | 81.93 ± 0.00 | — | — | — |

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
| tolokers | 0.563 | 0.170 | 0.569 | 0.167 | gcn: 0.392 / 0.019 |

## H7 (descriptive): convergence and fixed-point non-uniqueness

| dataset | EDSP-default non-converged | oscillation flagged | mean fraction of nodes still moving | GS vs damped L1: mean | p95 | splits |
|---|---|---|---|---|---|---|
| tolokers | 0.000 | 0.000 | 0.000 | — | — | 0 |

## Test accuracy by local homophily (descriptive, full labels; bin i = [i/10, (i+1)/10)), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tolokers | EDSP-default | 77.6 (85) | 30.6 (36) | 44.8 (116) | 44.4 (225) | 54.8 (352) | 78.8 (524) | 86.8 (387) | 84.5 (233) | 96.1 (484) | 93.4 (498) |
| tolokers | EDSP-tuned | 71.8 (85) | 27.8 (36) | 44.8 (116) | 41.3 (225) | 54.5 (352) | 74.2 (524) | 84.8 (387) | 85.8 (233) | 96.3 (484) | 93.0 (498) |
| tolokers | feature_lr | 77.6 (85) | 30.6 (36) | 44.8 (116) | 44.4 (225) | 54.8 (352) | 78.8 (524) | 86.8 (387) | 84.5 (233) | 96.1 (484) | 93.4 (498) |
| tolokers | correct_and_smooth | 49.4 (85) | 11.1 (36) | 28.4 (116) | 31.6 (225) | 47.2 (352) | 60.9 (524) | 73.6 (387) | 87.6 (233) | 97.7 (484) | 96.6 (498) |
| tolokers | gcn | 75.3 (85) | 22.2 (36) | 39.7 (116) | 46.7 (225) | 56.5 (352) | 82.1 (524) | 91.5 (387) | 94.4 (233) | 97.9 (484) | 94.2 (498) |
| tolokers | sage | 76.5 (85) | 33.3 (36) | 37.1 (116) | 45.8 (225) | 56.5 (352) | 80.3 (524) | 90.2 (387) | 94.0 (233) | 97.7 (484) | 95.0 (498) |

## Test accuracy by degree (bin 0: isolated; bin b: 2^(b-1) ≤ deg < 2^b; last bin open), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tolokers | EDSP-default | — | 73.3 (101) | 87.7 (204) | 86.6 (291) | 79.9 (407) | 78.3 (489) | 79.8 (526) | 79.8 (367) | 72.6 (292) | 71.0 (183) | 37.5 (80) |
| tolokers | EDSP-tuned | — | 70.3 (101) | 85.8 (204) | 83.5 (291) | 80.1 (407) | 77.5 (489) | 77.2 (526) | 78.7 (367) | 69.5 (292) | 72.1 (183) | 37.5 (80) |
| tolokers | feature_lr | — | 73.3 (101) | 87.7 (204) | 86.6 (291) | 79.9 (407) | 78.3 (489) | 79.8 (526) | 79.8 (367) | 72.6 (292) | 71.0 (183) | 37.5 (80) |
| tolokers | correct_and_smooth | — | 68.3 (101) | 78.4 (204) | 79.4 (291) | 75.9 (407) | 73.4 (489) | 68.4 (526) | 65.7 (367) | 67.1 (292) | 66.1 (183) | 40.0 (80) |
| tolokers | gcn | — | 73.3 (101) | 87.7 (204) | 88.0 (291) | 84.8 (407) | 81.4 (489) | 83.5 (526) | 81.5 (367) | 74.7 (292) | 68.9 (183) | 43.8 (80) |
| tolokers | sage | — | 75.2 (101) | 88.7 (204) | 88.3 (291) | 82.8 (407) | 81.0 (489) | 83.1 (526) | 80.9 (367) | 72.9 (292) | 71.0 (183) | 41.2 (80) |
