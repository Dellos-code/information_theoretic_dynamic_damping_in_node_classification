# E3 — Track B (feature priors): EDSP-B vs logistic regression, Correct & Smooth and tuned GNNs, H4

Run records: 9 (git d8543a63f80a; runners actions-run-42458905802). Headline metric per dataset in brackets; mean ± std over splits (×100). Rows marked † are constructed benchmarks or case studies and are excluded from real-world aggregates.

| dataset [metric] | splits | EDSP-default | EDSP-tuned | majority | train_prior | feature_lr | correct_and_smooth | mlp | gcn | sage | gat | gprgnn | bwgnn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| amazon-computers [accuracy] | 1 | 89.67 ± 0.00 | 89.16 ± 0.00 | 37.49 ± 0.00 | 37.49 ± 0.00 | 85.85 ± 0.00 | 91.42 ± 0.00 | 87.02 ± 0.00 | **92.91 ± 0.00** | 92.69 ± 0.00 | — | — | — |

## Pre-registered secondary column: EDSP-default on non-converged runs

| dataset | non-converged runs | headline, last iterate | headline, mean of last two iterates |
|---|---|---|---|
| amazon-computers | 0 / 1 | — | — |

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
| amazon-computers | 0.811 | 0.389 | 0.828 | 0.387 | gcn: 0.239 / 0.017 |

## H7 (descriptive): convergence and fixed-point non-uniqueness

| dataset | EDSP-default non-converged | oscillation flagged | mean fraction of nodes still moving | GS vs damped L1: mean | p95 | splits |
|---|---|---|---|---|---|---|
| amazon-computers | 0.000 | 0.000 | 0.000 | — | — | 0 |

## Test accuracy by local homophily (descriptive, full labels; bin i = [i/10, (i+1)/10)), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| amazon-computers | EDSP-default | 18.5 (65) | 27.6 (29) | 48.1 (54) | 66.7 (78) | 80.0 (95) | 76.0 (150) | 89.7 (195) | 92.6 (244) | 96.8 (496) | 98.2 (1288) |
| amazon-computers | EDSP-tuned | 23.1 (65) | 34.5 (29) | 50.0 (54) | 70.5 (78) | 75.8 (95) | 74.7 (150) | 88.2 (195) | 91.4 (244) | 95.4 (496) | 97.9 (1288) |
| amazon-computers | feature_lr | 32.3 (65) | 37.9 (29) | 55.6 (54) | 73.1 (78) | 77.9 (95) | 71.3 (150) | 84.6 (195) | 81.6 (244) | 91.5 (496) | 94.0 (1288) |
| amazon-computers | correct_and_smooth | 23.1 (65) | 31.0 (29) | 51.9 (54) | 76.9 (78) | 82.1 (95) | 80.0 (150) | 91.8 (195) | 94.3 (244) | 98.6 (496) | 98.9 (1288) |
| amazon-computers | gcn | 15.4 (65) | 20.7 (29) | 44.4 (54) | 71.8 (78) | 92.6 (95) | 88.0 (150) | 93.8 (195) | 98.8 (244) | 99.6 (496) | 99.6 (1288) |
| amazon-computers | sage | 10.8 (65) | 27.6 (29) | 46.3 (54) | 64.1 (78) | 87.4 (95) | 88.7 (150) | 94.9 (195) | 98.8 (244) | 99.6 (496) | 99.8 (1288) |

## Test accuracy by degree (bin 0: isolated; bin b: 2^(b-1) ≤ deg < 2^b; last bin open), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| amazon-computers | EDSP-default | 57.1 (56) | 67.1 (73) | 85.6 (132) | 90.5 (326) | 90.0 (488) | 90.6 (647) | 91.7 (651) | 95.0 (282) | 93.1 (72) | 88.9 (18) | 80.0 (5) |
| amazon-computers | EDSP-tuned | 57.1 (56) | 69.9 (73) | 86.4 (132) | 89.6 (326) | 89.5 (488) | 90.3 (647) | 90.6 (651) | 94.3 (282) | 93.1 (72) | 88.9 (18) | 60.0 (5) |
| amazon-computers | feature_lr | 57.1 (56) | 61.6 (73) | 78.0 (132) | 87.4 (326) | 86.1 (488) | 86.6 (647) | 87.4 (651) | 93.3 (282) | 93.1 (72) | 77.8 (18) | 60.0 (5) |
| amazon-computers | correct_and_smooth | 57.1 (56) | 69.9 (73) | 88.6 (132) | 92.0 (326) | 91.0 (488) | 93.8 (647) | 93.1 (651) | 95.4 (282) | 93.1 (72) | 94.4 (18) | 80.0 (5) |
| amazon-computers | gcn | 67.9 (56) | 71.2 (73) | 86.4 (132) | 92.9 (326) | 93.4 (488) | 95.2 (647) | 94.8 (651) | 95.7 (282) | 94.4 (72) | 94.4 (18) | 80.0 (5) |
| amazon-computers | sage | 66.1 (56) | 69.9 (73) | 86.4 (132) | 94.5 (326) | 93.9 (488) | 94.7 (647) | 94.0 (651) | 95.7 (282) | 90.3 (72) | 94.4 (18) | 80.0 (5) |
