# E3 — Track B (feature priors): EDSP-B vs logistic regression, Correct & Smooth and tuned GNNs, H4

Run records: 30 (git 41d502c903c8; runners actions-run-42474871330). Headline metric per dataset in brackets; mean ± std over splits (×100). Rows marked † are constructed benchmarks or case studies and are excluded from real-world aggregates.

| dataset [metric] | splits | EDSP-default | EDSP-tuned | majority | train_prior | feature_lr | correct_and_smooth | mlp | gcn | sage | gat | gprgnn | bwgnn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| questions [roc_auc] | 5 | 69.62 ± 0.57 | 71.70 ± 0.96 | 50.00 ± 0.00 | 50.00 ± 0.00 | 70.04 ± 0.89 | **72.22 ± 0.92** | — | — | — | — | — | — |

## Pre-registered secondary column: EDSP-default on non-converged runs

| dataset | non-converged runs | headline, last iterate | headline, mean of last two iterates |
|---|---|---|---|
| questions | 0 / 5 | — | — |

## EDSP-tuned vs each baseline: win / tie / loss

Per dataset: two-sided Wilcoxon signed-rank over splits on the headline metric; Holm correction across datasets per baseline; win/loss if adjusted p < 0.05, else tie. Real-world datasets only.

| baseline | win | tie | loss | datasets (W / L) |
|---|---|---|---|---|
| feature_lr | 0 | 1 | 0 | W: —; L: — |
| correct_and_smooth | 0 | 1 | 0 | W: —; L: — |
| mlp | 0 | 0 | 0 | W: —; L: — |
| gcn | 0 | 0 | 0 | W: —; L: — |
| sage | 0 | 0 | 0 | W: —; L: — |
| gat | 0 | 0 | 0 | W: —; L: — |
| gprgnn | 0 | 0 | 0 | W: —; L: — |
| bwgnn | 0 | 0 | 0 | W: —; L: — |

## Calibration (mean over splits; lower is better)

| dataset | EDSP-default NLL | EDSP-default ECE | EDSP-tuned NLL | EDSP-tuned ECE | best baseline by NLL (NLL / ECE) |
|---|---|---|---|---|---|
| questions | 0.301 | 0.205 | 0.330 | 0.222 | feature_lr: 0.123 / 0.003 |

## H7 (descriptive): convergence and fixed-point non-uniqueness

| dataset | EDSP-default non-converged | oscillation flagged | mean fraction of nodes still moving | GS vs damped L1: mean | p95 | splits |
|---|---|---|---|---|---|---|
| questions | 0.000 | 0.000 | 0.000 | — | — | 0 |

## Test accuracy by local homophily (descriptive, full labels; bin i = [i/10, (i+1)/10)), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| questions | EDSP-default | 65.5 (3670) | 0.8 (261) | 7.5 (120) | 66.2 (148) | 84.4 (32) | 97.0 (1455) | 99.6 (1190) | 99.8 (1311) | 100.0 (3347) | 99.8 (49621) |
| questions | EDSP-tuned | 70.6 (3670) | 14.6 (261) | 21.7 (120) | 69.6 (148) | 84.4 (32) | 96.5 (1455) | 98.4 (1190) | 98.3 (1311) | 97.1 (3347) | 98.8 (49621) |
| questions | feature_lr | 68.6 (3670) | 3.4 (261) | 15.8 (120) | 68.2 (148) | 84.4 (32) | 97.0 (1455) | 99.3 (1190) | 99.8 (1311) | 99.7 (3347) | 99.7 (49621) |
| questions | correct_and_smooth | 67.2 (3670) | 0.4 (261) | 10.8 (120) | 66.2 (148) | 84.4 (32) | 97.2 (1455) | 99.6 (1190) | 99.8 (1311) | 99.9 (3347) | 99.8 (49621) |

## Test accuracy by degree (bin 0: isolated; bin b: 2^(b-1) ≤ deg < 2^b; last bin open), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| questions | EDSP-default | — | 98.0 (32453) | 97.2 (14083) | 96.7 (6762) | 94.7 (3705) | 92.6 (2181) | 90.4 (1083) | 87.3 (505) | 87.2 (274) | 84.1 (82) | 66.7 (27) |
| questions | EDSP-tuned | — | 97.8 (32453) | 96.9 (14083) | 96.2 (6762) | 92.5 (3705) | 88.7 (2181) | 87.6 (1083) | 82.2 (505) | 84.3 (274) | 79.3 (82) | 59.3 (27) |
| questions | feature_lr | — | 98.1 (32453) | 97.3 (14083) | 96.9 (6762) | 95.0 (3705) | 92.8 (2181) | 90.3 (1083) | 87.1 (505) | 87.2 (274) | 82.9 (82) | 66.7 (27) |
| questions | correct_and_smooth | — | 98.2 (32453) | 97.3 (14083) | 96.8 (6762) | 94.7 (3705) | 92.6 (2181) | 90.4 (1083) | 87.3 (505) | 87.2 (274) | 84.1 (82) | 66.7 (27) |

## Cell C3c (mixed by local structure: top-5 std of local homophily)

| dataset | std local homophily | min compat. distance | EDSP-default | EDSP-tuned | feature_lr | correct_and_smooth | mlp | gcn | sage | gat | gprgnn | bwgnn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| questions | 0.257 | 1.266 | 69.62 | 71.70 | 70.04 | **72.22** | — | — | — | — | — | — |

## Cell C3d (structured compatibility: top-5 min Frobenius distance to I and to (J-I)/(k-1))

| dataset | std local homophily | min compat. distance | EDSP-default | EDSP-tuned | feature_lr | correct_and_smooth | mlp | gcn | sage | gat | gprgnn | bwgnn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| questions | 0.257 | 1.266 | 69.62 | 71.70 | 70.04 | **72.22** | — | — | — | — | — | — |
