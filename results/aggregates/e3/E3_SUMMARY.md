# E3 — Track B (feature priors): EDSP-B vs logistic regression, Correct & Smooth and tuned GNNs, H4

Run records: 9 (git 3b2ae0acb743; runners actions-run-42467730714). Headline metric per dataset in brackets; mean ± std over splits (×100). Rows marked † are constructed benchmarks or case studies and are excluded from real-world aggregates.

| dataset [metric] | splits | EDSP-default | EDSP-tuned | majority | train_prior | feature_lr | correct_and_smooth | mlp | gcn | sage | gat | gprgnn | bwgnn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tolokers [roc_auc] | 1 | 73.63 ± 0.00 | 74.42 ± 0.00 | 50.00 ± 0.00 | 50.00 ± 0.00 | 72.10 ± 0.00 | 77.05 ± 0.00 | 72.20 ± 0.00 | **84.63 ± 0.00** | 83.49 ± 0.00 | — | — | — |

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
| tolokers | 0.548 | 0.155 | 0.554 | 0.159 | gcn: 0.384 / 0.038 |

## H7 (descriptive): convergence and fixed-point non-uniqueness

| dataset | EDSP-default non-converged | oscillation flagged | mean fraction of nodes still moving | GS vs damped L1: mean | p95 | splits |
|---|---|---|---|---|---|---|
| tolokers | 0.000 | 0.000 | 0.000 | — | — | 0 |

## Test accuracy by local homophily (descriptive, full labels; bin i = [i/10, (i+1)/10)), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tolokers | EDSP-default | 74.7 (91) | 13.9 (36) | 50.0 (106) | 46.9 (196) | 55.5 (348) | 75.1 (530) | 84.4 (358) | 87.2 (266) | 95.6 (521) | 93.6 (488) |
| tolokers | EDSP-tuned | 72.5 (91) | 13.9 (36) | 36.8 (106) | 46.9 (196) | 56.0 (348) | 74.0 (530) | 83.8 (358) | 86.1 (266) | 96.2 (521) | 91.6 (488) |
| tolokers | feature_lr | 72.5 (91) | 19.4 (36) | 51.9 (106) | 49.0 (196) | 56.0 (348) | 75.1 (530) | 84.6 (358) | 87.6 (266) | 95.4 (521) | 93.6 (488) |
| tolokers | correct_and_smooth | 52.7 (91) | 16.7 (36) | 35.8 (106) | 48.5 (196) | 54.0 (348) | 53.4 (530) | 64.8 (358) | 85.3 (266) | 97.7 (521) | 95.9 (488) |
| tolokers | gcn | 78.0 (91) | 13.9 (36) | 42.5 (106) | 48.0 (196) | 57.2 (348) | 79.8 (530) | 91.1 (358) | 95.9 (266) | 98.8 (521) | 94.5 (488) |
| tolokers | sage | 76.9 (91) | 11.1 (36) | 43.4 (106) | 53.1 (196) | 55.2 (348) | 76.8 (530) | 88.8 (358) | 93.2 (266) | 98.7 (521) | 94.3 (488) |

## Test accuracy by degree (bin 0: isolated; bin b: 2^(b-1) ≤ deg < 2^b; last bin open), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tolokers | EDSP-default | — | 73.6 (110) | 82.3 (192) | 84.9 (304) | 82.8 (418) | 75.6 (483) | 80.0 (471) | 80.5 (395) | 78.0 (286) | 68.9 (190) | 45.1 (91) |
| tolokers | EDSP-tuned | — | 65.5 (110) | 82.8 (192) | 80.3 (304) | 80.1 (418) | 76.6 (483) | 79.6 (471) | 78.5 (395) | 79.4 (286) | 67.9 (190) | 49.5 (91) |
| tolokers | feature_lr | — | 73.6 (110) | 82.8 (192) | 83.9 (304) | 83.0 (418) | 77.4 (483) | 80.3 (471) | 79.7 (395) | 79.0 (286) | 68.9 (190) | 45.1 (91) |
| tolokers | correct_and_smooth | — | 70.0 (110) | 79.2 (192) | 78.3 (304) | 76.8 (418) | 73.7 (483) | 68.2 (471) | 63.8 (395) | 68.5 (286) | 70.0 (190) | 52.7 (91) |
| tolokers | gcn | — | 78.2 (110) | 86.5 (192) | 86.8 (304) | 86.4 (418) | 81.4 (483) | 82.6 (471) | 82.5 (395) | 79.4 (286) | 72.1 (190) | 49.5 (91) |
| tolokers | sage | — | 77.3 (110) | 87.0 (192) | 85.2 (304) | 86.1 (418) | 80.3 (483) | 81.7 (471) | 80.0 (395) | 76.6 (286) | 68.4 (190) | 59.3 (91) |
