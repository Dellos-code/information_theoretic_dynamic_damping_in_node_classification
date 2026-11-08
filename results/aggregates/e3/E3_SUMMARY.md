# E3 — Track B (feature priors): EDSP-B vs logistic regression, Correct & Smooth and tuned GNNs, H4

Run records: 9 (git 3b2ae0acb743; runners actions-run-42467704159). Headline metric per dataset in brackets; mean ± std over splits (×100). Rows marked † are constructed benchmarks or case studies and are excluded from real-world aggregates.

| dataset [metric] | splits | EDSP-default | EDSP-tuned | majority | train_prior | feature_lr | correct_and_smooth | mlp | gcn | sage | gat | gprgnn | bwgnn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tolokers [roc_auc] | 1 | 73.56 ± 0.00 | 75.12 ± 0.00 | 50.00 ± 0.00 | 50.00 ± 0.00 | 73.28 ± 0.00 | 76.55 ± 0.00 | 73.64 ± 0.00 | **84.09 ± 0.00** | 83.02 ± 0.00 | — | — | — |

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
| tolokers | 0.550 | 0.158 | 0.554 | 0.160 | gcn: 0.386 / 0.021 |

## H7 (descriptive): convergence and fixed-point non-uniqueness

| dataset | EDSP-default non-converged | oscillation flagged | mean fraction of nodes still moving | GS vs damped L1: mean | p95 | splits |
|---|---|---|---|---|---|---|
| tolokers | 0.000 | 0.000 | 0.000 | — | — | 0 |

## Test accuracy by local homophily (descriptive, full labels; bin i = [i/10, (i+1)/10)), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tolokers | EDSP-default | 79.3 (82) | 28.2 (39) | 46.7 (120) | 51.7 (201) | 54.8 (361) | 80.6 (520) | 82.6 (396) | 82.3 (277) | 95.9 (468) | 92.6 (476) |
| tolokers | EDSP-tuned | 72.0 (82) | 20.5 (39) | 43.3 (120) | 53.7 (201) | 55.4 (361) | 76.2 (520) | 79.3 (396) | 84.8 (277) | 96.4 (468) | 92.0 (476) |
| tolokers | feature_lr | 79.3 (82) | 28.2 (39) | 49.2 (120) | 51.7 (201) | 54.8 (361) | 81.0 (520) | 82.6 (396) | 82.3 (277) | 95.9 (468) | 93.1 (476) |
| tolokers | correct_and_smooth | 56.1 (82) | 20.5 (39) | 35.0 (120) | 50.7 (201) | 52.1 (361) | 51.3 (520) | 61.4 (396) | 87.7 (277) | 98.7 (468) | 96.8 (476) |
| tolokers | gcn | 78.0 (82) | 17.9 (39) | 42.5 (120) | 58.2 (201) | 58.2 (361) | 83.1 (520) | 88.6 (396) | 93.1 (277) | 99.6 (468) | 95.0 (476) |
| tolokers | sage | 82.9 (82) | 25.6 (39) | 45.0 (120) | 51.7 (201) | 54.8 (361) | 82.1 (520) | 83.6 (396) | 83.4 (277) | 96.8 (468) | 92.9 (476) |

## Test accuracy by degree (bin 0: isolated; bin b: 2^(b-1) ≤ deg < 2^b; last bin open), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tolokers | EDSP-default | — | 77.6 (85) | 84.6 (254) | 83.8 (321) | 80.0 (380) | 75.9 (453) | 76.3 (486) | 83.1 (402) | 76.8 (293) | 73.0 (174) | 46.7 (92) |
| tolokers | EDSP-tuned | — | 70.6 (85) | 82.7 (254) | 79.8 (321) | 79.2 (380) | 79.9 (453) | 75.1 (486) | 79.9 (402) | 73.7 (293) | 70.1 (174) | 52.2 (92) |
| tolokers | feature_lr | — | 78.8 (85) | 84.3 (254) | 83.8 (321) | 81.6 (380) | 76.6 (453) | 75.7 (486) | 83.3 (402) | 76.8 (293) | 73.0 (174) | 46.7 (92) |
| tolokers | correct_and_smooth | — | 72.9 (85) | 79.1 (254) | 78.5 (321) | 76.6 (380) | 73.7 (453) | 64.6 (486) | 60.9 (402) | 65.9 (293) | 68.4 (174) | 55.4 (92) |
| tolokers | gcn | — | 81.2 (85) | 87.4 (254) | 85.7 (321) | 87.6 (380) | 80.6 (453) | 80.0 (486) | 84.8 (402) | 79.9 (293) | 73.0 (174) | 57.6 (92) |
| tolokers | sage | — | 80.0 (85) | 85.8 (254) | 85.4 (321) | 81.3 (380) | 77.3 (453) | 75.9 (486) | 83.3 (402) | 77.1 (293) | 72.4 (174) | 46.7 (92) |
