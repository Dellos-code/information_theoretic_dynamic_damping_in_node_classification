# E3 — Track B (feature priors): EDSP-B vs logistic regression, Correct & Smooth and tuned GNNs, H4

Run records: 69 (git 3b2ae0acb743; runners actions-run-42471148953). Headline metric per dataset in brackets; mean ± std over splits (×100). Rows marked † are constructed benchmarks or case studies and are excluded from real-world aggregates.

| dataset [metric] | splits | EDSP-default | EDSP-tuned | majority | train_prior | feature_lr | correct_and_smooth | mlp | gcn | sage | gat | gprgnn | bwgnn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| questions [roc_auc] | 9 | 69.99 ± 0.81 | 71.50 ± 0.76 | 50.00 ± 0.00 | 50.00 ± 0.00 | 70.27 ± 0.93 | 72.17 ± 0.98 | 70.35 ± 1.09 | 73.75 ± 1.21 | **75.57 ± 0.92** | — | — | — |

## Pre-registered secondary column: EDSP-default on non-converged runs

| dataset | non-converged runs | headline, last iterate | headline, mean of last two iterates |
|---|---|---|---|
| questions | 0 / 9 | — | — |

## EDSP-tuned vs each baseline: win / tie / loss

Per dataset: two-sided Wilcoxon signed-rank over splits on the headline metric; Holm correction across datasets per baseline; win/loss if adjusted p < 0.05, else tie. Real-world datasets only.

| baseline | win | tie | loss | datasets (W / L) |
|---|---|---|---|---|
| feature_lr | 1 | 0 | 0 | W: questions; L: — |
| correct_and_smooth | 0 | 1 | 0 | W: —; L: — |
| mlp | 0 | 1 | 0 | W: —; L: — |
| gcn | 0 | 1 | 0 | W: —; L: — |
| sage | 0 | 1 | 0 | W: —; L: — |
| gat | 0 | 0 | 0 | W: —; L: — |
| gprgnn | 0 | 0 | 0 | W: —; L: — |
| bwgnn | 0 | 0 | 0 | W: —; L: — |

## Calibration (mean over splits; lower is better)

| dataset | EDSP-default NLL | EDSP-default ECE | EDSP-tuned NLL | EDSP-tuned ECE | best baseline by NLL (NLL / ECE) |
|---|---|---|---|---|---|
| questions | 0.301 | 0.205 | 0.326 | 0.221 | sage: 0.119 / 0.015 |

## H7 (descriptive): convergence and fixed-point non-uniqueness

| dataset | EDSP-default non-converged | oscillation flagged | mean fraction of nodes still moving | GS vs damped L1: mean | p95 | splits |
|---|---|---|---|---|---|---|
| questions | 0.000 | 0.000 | 0.000 | — | — | 0 |

## Test accuracy by local homophily (descriptive, full labels; bin i = [i/10, (i+1)/10)), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| questions | EDSP-default | 65.8 (6599) | 0.6 (480) | 8.2 (231) | 65.9 (290) | 84.2 (57) | 97.3 (2591) | 99.5 (2147) | 99.9 (2375) | 100.0 (5959) | 99.8 (89350) |
| questions | EDSP-tuned | 70.6 (6599) | 11.5 (480) | 19.9 (231) | 69.3 (290) | 84.2 (57) | 96.8 (2591) | 98.8 (2147) | 98.7 (2375) | 97.6 (5959) | 99.0 (89350) |
| questions | feature_lr | 68.7 (6599) | 2.7 (480) | 14.3 (231) | 67.9 (290) | 84.2 (57) | 97.3 (2591) | 99.4 (2147) | 99.9 (2375) | 99.7 (5959) | 99.8 (89350) |
| questions | correct_and_smooth | 67.6 (6599) | 0.4 (480) | 10.0 (231) | 65.9 (290) | 84.2 (57) | 97.5 (2591) | 99.6 (2147) | 99.9 (2375) | 99.9 (5959) | 99.8 (89350) |
| questions | gcn | 68.0 (3670) | 0.0 (261) | 14.2 (120) | 66.9 (148) | 81.2 (32) | 96.8 (1455) | 99.6 (1190) | 99.8 (1311) | 99.9 (3347) | 99.8 (49621) |
| questions | sage | 69.5 (2940) | 5.2 (211) | 15.8 (101) | 67.5 (120) | 85.2 (27) | 97.4 (1151) | 99.7 (942) | 99.6 (1076) | 99.6 (2673) | 99.8 (39683) |

## Test accuracy by degree (bin 0: isolated; bin b: 2^(b-1) ≤ deg < 2^b; last bin open), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| questions | EDSP-default | — | 98.0 (58526) | 97.4 (25282) | 96.7 (12188) | 94.8 (6693) | 92.5 (3851) | 90.5 (1964) | 88.1 (890) | 86.0 (470) | 84.0 (169) | 69.6 (46) |
| questions | EDSP-tuned | — | 97.9 (58526) | 97.1 (25282) | 96.3 (12188) | 92.9 (6693) | 89.0 (3851) | 88.0 (1964) | 83.9 (890) | 82.8 (470) | 77.5 (169) | 60.9 (46) |
| questions | feature_lr | — | 98.1 (58526) | 97.4 (25282) | 96.8 (12188) | 94.9 (6693) | 92.8 (3851) | 90.5 (1964) | 87.9 (890) | 86.0 (470) | 82.8 (169) | 69.6 (46) |
| questions | correct_and_smooth | — | 98.1 (58526) | 97.4 (25282) | 96.7 (12188) | 94.7 (6693) | 92.5 (3851) | 90.5 (1964) | 88.1 (890) | 86.0 (470) | 84.0 (169) | 69.6 (46) |
| questions | gcn | — | 98.2 (32453) | 97.4 (14083) | 96.8 (6762) | 94.7 (3705) | 92.6 (2181) | 90.4 (1083) | 87.3 (505) | 87.2 (274) | 84.1 (82) | 66.7 (27) |
| questions | sage | — | 98.2 (25968) | 97.5 (11237) | 97.1 (5393) | 94.7 (2981) | 92.9 (1769) | 91.1 (869) | 88.0 (407) | 87.5 (216) | 82.0 (61) | 69.6 (23) |

## Cell C3c (mixed by local structure: top-5 std of local homophily)

| dataset | std local homophily | min compat. distance | EDSP-default | EDSP-tuned | feature_lr | correct_and_smooth | mlp | gcn | sage | gat | gprgnn | bwgnn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| questions | 0.257 | 1.266 | 69.99 | 71.50 | 70.27 | 72.17 | 70.35 | 73.75 | **75.57** | — | — | — |

## Cell C3d (structured compatibility: top-5 min Frobenius distance to I and to (J-I)/(k-1))

| dataset | std local homophily | min compat. distance | EDSP-default | EDSP-tuned | feature_lr | correct_and_smooth | mlp | gcn | sage | gat | gprgnn | bwgnn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| questions | 0.257 | 1.266 | 69.99 | 71.50 | 70.27 | 72.17 | 70.35 | 73.75 | **75.57** | — | — | — |

## H4 — EDSP-B-tuned vs C&S and vs the best tuned GNN (C2, PREREGISTRATION A7)

| dataset | best GNN | EDSP − C&S | p (Holm) | GNN − EDSP | p (Holm) |
|---|---|---|---|---|---|
| questions | sage | -0.67 | 0.164 | +4.19 | 0.125 |

EDSP-B ≈ C&S (no significant difference) on 1/1; best GNN significantly better on 0/1 → H4 **inconclusive**.
