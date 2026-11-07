# E3 — Track B (feature priors): EDSP-B vs logistic regression, Correct & Smooth and tuned GNNs, H4

Run records: 32 (git b5303b9ffdd1; runners actions-run-42385833158). Headline metric per dataset in brackets; mean ± std over splits (×100). Rows marked † are constructed benchmarks or case studies and are excluded from real-world aggregates.

| dataset [metric] | splits | EDSP-default | EDSP-tuned | majority | train_prior | feature_lr | correct_and_smooth | mlp | gcn | sage | gat | gprgnn | bwgnn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| amazon-fraud [roc_auc] | 4 | 93.71 ± 0.74 | 96.55 ± 0.93 | 50.00 ± 0.00 | 50.00 ± 0.00 | **96.58 ± 0.92** | 95.72 ± 0.46 | 92.66 ± 1.29 | 92.68 ± 1.45 | — | — | — | — |

## Pre-registered secondary column: EDSP-default on non-converged runs

| dataset | non-converged runs | headline, last iterate | headline, mean of last two iterates |
|---|---|---|---|
| amazon-fraud | 0 / 4 | — | — |

## EDSP-tuned vs each baseline: win / tie / loss

Per dataset: two-sided Wilcoxon signed-rank over splits on the headline metric; Holm correction across datasets per baseline; win/loss if adjusted p < 0.05, else tie. Real-world datasets only.

| baseline | win | tie | loss | datasets (W / L) |
|---|---|---|---|---|
| feature_lr | 0 | 1 | 0 | W: —; L: — |
| correct_and_smooth | 0 | 1 | 0 | W: —; L: — |
| mlp | 0 | 1 | 0 | W: —; L: — |
| gcn | 0 | 1 | 0 | W: —; L: — |
| sage | 0 | 0 | 0 | W: —; L: — |
| gat | 0 | 0 | 0 | W: —; L: — |
| gprgnn | 0 | 0 | 0 | W: —; L: — |
| bwgnn | 0 | 0 | 0 | W: —; L: — |

## Calibration (mean over splits; lower is better)

| dataset | EDSP-default NLL | EDSP-default ECE | EDSP-tuned NLL | EDSP-tuned ECE | best baseline by NLL (NLL / ECE) |
|---|---|---|---|---|---|
| amazon-fraud | 0.308 | 0.183 | 0.285 | 0.201 | feature_lr: 0.103 / 0.008 |

## H7 (descriptive): convergence and fixed-point non-uniqueness

| dataset | EDSP-default non-converged | oscillation flagged | mean fraction of nodes still moving | GS vs damped L1: mean | p95 | splits |
|---|---|---|---|---|---|---|
| amazon-fraud | 0.000 | 0.000 | 0.000 | — | — | 0 |

## Test accuracy by local homophily (descriptive, full labels; bin i = [i/10, (i+1)/10)), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| amazon-fraud | EDSP-default | 29.5 (671) | 70.5 (376) | 77.5 (142) | 70.7 (75) | 93.8 (32) | 95.7 (23) | 89.3 (28) | 96.1 (51) | 98.3 (785) | 99.9 (11641) |
| amazon-fraud | EDSP-tuned | 58.0 (671) | 91.0 (376) | 93.7 (142) | 97.3 (75) | 100.0 (32) | 100.0 (23) | 89.3 (28) | 94.1 (51) | 97.3 (785) | 99.5 (11641) |
| amazon-fraud | feature_lr | 57.1 (671) | 92.8 (376) | 95.8 (142) | 98.7 (75) | 100.0 (32) | 100.0 (23) | 89.3 (28) | 92.2 (51) | 97.6 (785) | 99.7 (11641) |
| amazon-fraud | correct_and_smooth | 47.1 (671) | 84.0 (376) | 93.7 (142) | 97.3 (75) | 100.0 (32) | 100.0 (23) | 89.3 (28) | 98.0 (51) | 98.1 (785) | 99.9 (11641) |
| amazon-fraud | gcn | 47.8 (671) | 82.4 (376) | 94.4 (142) | 96.0 (75) | 100.0 (32) | 91.3 (23) | 85.7 (28) | 80.4 (51) | 97.5 (785) | 99.9 (11641) |

## Test accuracy by degree (bin 0: isolated; bin b: 2^(b-1) ≤ deg < 2^b; last bin open), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| amazon-fraud | EDSP-default | — | — | 100.0 (3) | 77.8 (18) | 88.1 (59) | 85.3 (232) | 88.3 (579) | 90.0 (1212) | 92.3 (2155) | 95.4 (3028) | 98.0 (6538) |
| amazon-fraud | EDSP-tuned | — | — | 100.0 (3) | 83.3 (18) | 89.8 (59) | 95.3 (232) | 96.5 (579) | 96.1 (1212) | 95.7 (2155) | 96.6 (3028) | 98.0 (6538) |
| amazon-fraud | feature_lr | — | — | 100.0 (3) | 94.4 (18) | 91.5 (59) | 95.7 (232) | 97.6 (579) | 96.5 (1212) | 95.7 (2155) | 96.9 (3028) | 98.2 (6538) |
| amazon-fraud | correct_and_smooth | — | — | 100.0 (3) | 77.8 (18) | 91.5 (59) | 94.8 (232) | 97.2 (579) | 95.4 (1212) | 94.4 (2155) | 96.1 (3028) | 98.0 (6538) |
| amazon-fraud | gcn | — | — | 100.0 (3) | 77.8 (18) | 88.1 (59) | 94.0 (232) | 94.5 (579) | 94.4 (1212) | 94.3 (2155) | 96.3 (3028) | 98.2 (6538) |
