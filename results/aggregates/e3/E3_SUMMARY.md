# E3 — Track B (feature priors): EDSP-B vs logistic regression, Correct & Smooth and tuned GNNs, H4

Run records: 9 (git 3b2ae0acb743; runners actions-run-42467691919). Headline metric per dataset in brackets; mean ± std over splits (×100). Rows marked † are constructed benchmarks or case studies and are excluded from real-world aggregates.

| dataset [metric] | splits | EDSP-default | EDSP-tuned | majority | train_prior | feature_lr | correct_and_smooth | mlp | gcn | sage | gat | gprgnn | bwgnn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tolokers [roc_auc] | 1 | 72.10 ± 0.00 | 73.08 ± 0.00 | 50.00 ± 0.00 | 50.00 ± 0.00 | 71.44 ± 0.00 | 75.41 ± 0.00 | 71.32 ± 0.00 | 82.41 ± 0.00 | **82.43 ± 0.00** | — | — | — |

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
| tolokers | 0.551 | 0.158 | 0.559 | 0.149 | sage: 0.395 / 0.026 |

## H7 (descriptive): convergence and fixed-point non-uniqueness

| dataset | EDSP-default non-converged | oscillation flagged | mean fraction of nodes still moving | GS vs damped L1: mean | p95 | splits |
|---|---|---|---|---|---|---|
| tolokers | 0.000 | 0.000 | 0.000 | — | — | 0 |

## Test accuracy by local homophily (descriptive, full labels; bin i = [i/10, (i+1)/10)), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tolokers | EDSP-default | 74.2 (97) | 27.1 (48) | 47.4 (95) | 53.2 (201) | 54.4 (353) | 80.2 (535) | 83.7 (406) | 83.0 (259) | 96.2 (477) | 90.8 (469) |
| tolokers | EDSP-tuned | 73.2 (97) | 18.8 (48) | 44.2 (95) | 48.3 (201) | 52.4 (353) | 74.6 (535) | 81.0 (406) | 85.3 (259) | 96.6 (477) | 88.9 (469) |
| tolokers | feature_lr | 76.3 (97) | 31.2 (48) | 48.4 (95) | 55.2 (201) | 55.0 (353) | 79.8 (535) | 84.7 (406) | 84.2 (259) | 96.4 (477) | 90.6 (469) |
| tolokers | correct_and_smooth | 57.7 (97) | 27.1 (48) | 44.2 (95) | 50.7 (201) | 49.6 (353) | 49.2 (535) | 61.6 (406) | 82.2 (259) | 95.0 (477) | 92.8 (469) |
| tolokers | gcn | 77.3 (97) | 25.0 (48) | 34.7 (95) | 55.2 (201) | 59.2 (353) | 81.5 (535) | 88.2 (406) | 93.4 (259) | 98.7 (477) | 93.0 (469) |
| tolokers | sage | 77.3 (97) | 20.8 (48) | 37.9 (95) | 59.7 (201) | 59.5 (353) | 79.3 (535) | 90.6 (406) | 93.4 (259) | 98.1 (477) | 93.0 (469) |

## Test accuracy by degree (bin 0: isolated; bin b: 2^(b-1) ≤ deg < 2^b; last bin open), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tolokers | EDSP-default | — | 73.1 (108) | 83.5 (231) | 87.1 (303) | 77.5 (396) | 77.8 (441) | 78.9 (493) | 78.9 (398) | 77.5 (316) | 73.8 (172) | 45.1 (82) |
| tolokers | EDSP-tuned | — | 66.7 (108) | 79.2 (231) | 82.8 (303) | 76.0 (396) | 78.7 (441) | 77.5 (493) | 76.1 (398) | 73.4 (316) | 70.3 (172) | 47.6 (82) |
| tolokers | feature_lr | — | 75.9 (108) | 82.3 (231) | 86.5 (303) | 78.8 (396) | 78.9 (441) | 79.7 (493) | 78.6 (398) | 78.8 (316) | 74.4 (172) | 45.1 (82) |
| tolokers | correct_and_smooth | — | 67.6 (108) | 71.4 (231) | 76.6 (303) | 74.2 (396) | 70.3 (441) | 64.7 (493) | 66.3 (398) | 59.5 (316) | 66.3 (172) | 52.4 (82) |
| tolokers | gcn | — | 78.7 (108) | 82.3 (231) | 88.4 (303) | 84.1 (396) | 82.8 (441) | 81.1 (493) | 80.9 (398) | 77.8 (316) | 76.2 (172) | 52.4 (82) |
| tolokers | sage | — | 75.0 (108) | 87.9 (231) | 88.8 (303) | 82.6 (396) | 82.5 (441) | 79.9 (493) | 80.4 (398) | 77.8 (316) | 77.9 (172) | 62.2 (82) |

## Cell C3c (mixed by local structure: top-5 std of local homophily)

| dataset | std local homophily | min compat. distance | EDSP-default | EDSP-tuned | feature_lr | correct_and_smooth | mlp | gcn | sage | gat | gprgnn | bwgnn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tolokers | 0.247 | 0.954 | 72.10 | 73.08 | 71.44 | 75.41 | 71.32 | 82.41 | **82.43** | — | — | — |

## Cell C3d (structured compatibility: top-5 min Frobenius distance to I and to (J-I)/(k-1))

| dataset | std local homophily | min compat. distance | EDSP-default | EDSP-tuned | feature_lr | correct_and_smooth | mlp | gcn | sage | gat | gprgnn | bwgnn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tolokers | 0.247 | 0.954 | 72.10 | 73.08 | 71.44 | 75.41 | 71.32 | 82.41 | **82.43** | — | — | — |

## H4 — EDSP-B-tuned vs C&S and vs the best tuned GNN (C2, PREREGISTRATION A7)

| dataset | best GNN | EDSP − C&S | p (Holm) | GNN − EDSP | p (Holm) |
|---|---|---|---|---|---|
| tolokers | sage | -2.32 | nan | +9.35 | nan |

EDSP-B ≈ C&S (no significant difference) on 0/1; best GNN significantly better on 0/1 → H4 **not supported**.
