# E3 — Track B (feature priors): EDSP-B vs logistic regression, Correct & Smooth and tuned GNNs, H4

Run records: 1098 (git c2e45ef6ba6f; runners actions-run-42707731268). Headline metric per dataset in brackets; mean ± std over splits (×100). Rows marked † are constructed benchmarks or case studies and are excluded from real-world aggregates.

| dataset [metric] | splits | EDSP-default | EDSP-tuned | majority | train_prior | feature_lr | correct_and_smooth | mlp | gcn | sage | gat | gprgnn | bwgnn | gat(h64) | cs_ref | cs_2hop | cs_2hop_ref | clp | clp_es | linbp_b |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| amazon-computers [accuracy] | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — | **90.40 ± 0.50** | 88.97 ± 0.71 | 89.05 ± 0.67 | 90.24 ± 0.52 | 90.21 ± 0.52 | 85.19 ± 0.61 |
| amazon-fraud [roc_auc] | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — | 96.00 ± 0.82 | 95.42 ± 0.85 | 95.65 ± 0.96 | 96.60 ± 1.03 | 96.60 ± 1.03 | **96.72 ± 1.11** |
| amazon-photo [accuracy] | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — | 94.35 ± 0.46 | 94.15 ± 0.51 | 94.08 ± 0.46 | **95.00 ± 0.54** | 94.98 ± 0.49 | 92.47 ± 0.51 |
| amazon-ratings [accuracy] | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — | 46.35 ± 0.47 | 48.11 ± 0.56 | **48.12 ± 0.57** | 44.47 ± 0.52 | 46.37 ± 0.41 | 41.21 ± 0.33 |
| chameleon-filtered [accuracy] | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — | 37.55 ± 2.45 | **38.90 ± 2.25** | 38.72 ± 2.19 | 37.63 ± 1.78 | 37.81 ± 2.94 | 35.64 ± 2.88 |
| citeseer [accuracy] | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — | **76.30 ± 2.27** | 74.52 ± 2.82 | 74.69 ± 2.57 | 75.90 ± 2.04 | 75.98 ± 2.04 | 73.44 ± 3.38 |
| coauthor-cs [accuracy] | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — | **95.89 ± 0.30** | 95.72 ± 0.25 | 95.78 ± 0.27 | 95.50 ± 0.27 | 95.50 ± 0.25 | 95.19 ± 0.18 |
| coauthor-physics [accuracy] | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — | **97.17 ± 0.19** | 97.07 ± 0.19 | 97.05 ± 0.20 | 97.08 ± 0.24 | 97.08 ± 0.25 | 96.46 ± 0.22 |
| cora [accuracy] | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — | **87.80 ± 4.73** | 86.09 ± 4.20 | 86.53 ± 4.13 | 85.71 ± 3.27 | 85.57 ± 3.35 | 78.63 ± 6.50 |
| minesweeper† [roc_auc] | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — | 50.49 ± 1.16 | 50.84 ± 0.88 | 50.89 ± 0.98 | 54.81 ± 1.00 | **62.98 ± 2.66** | 51.09 ± 0.97 |
| pubmed [accuracy] | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — | **88.97 ± 3.64** | 88.95 ± 3.25 | 88.84 ± 3.50 | 87.38 ± 3.39 | 87.58 ± 3.64 | 87.19 ± 4.22 |
| questions [roc_auc] | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — | 73.86 ± 1.13 | 74.09 ± 0.80 | **77.55 ± 1.07** | 70.90 ± 0.93 | 70.94 ± 0.91 | 66.82 ± 0.93 |
| roman-empire [accuracy] | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — | 50.99 ± 0.71 | 56.91 ± 0.57 | 56.75 ± 0.58 | 64.24 ± 0.52 | 64.24 ± 0.52 | **66.75 ± 0.56** |
| squirrel-filtered [accuracy] | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — | 24.10 ± 1.72 | 22.64 ± 2.05 | 22.46 ± 2.18 | 39.32 ± 2.06 | 39.29 ± 2.13 | **39.33 ± 2.08** |
| tolokers [roc_auc] | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — | 75.98 ± 0.78 | 75.43 ± 0.70 | 75.93 ± 0.86 | 76.29 ± 1.05 | **76.43 ± 1.12** | 71.69 ± 0.75 |
| wikics [accuracy] | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — | **79.04 ± 0.45** | 77.95 ± 0.51 | 77.93 ± 0.54 | 78.05 ± 0.40 | 78.00 ± 0.42 | 73.22 ± 0.63 |
| yelpchi [roc_auc] | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — | 72.07 ± 0.31 | 69.86 ± 0.46 | 71.17 ± 0.35 | **79.25 ± 0.30** | **79.25 ± 0.30** | 77.63 ± 0.34 |

## Pre-registered secondary column: EDSP-default on non-converged runs

| dataset | non-converged runs | headline, last iterate | headline, mean of last two iterates |
|---|---|---|---|
| amazon-computers | 0 / 0 | — | — |
| amazon-fraud | 0 / 0 | — | — |
| amazon-photo | 0 / 0 | — | — |
| amazon-ratings | 0 / 0 | — | — |
| chameleon-filtered | 0 / 0 | — | — |
| citeseer | 0 / 0 | — | — |
| coauthor-cs | 0 / 0 | — | — |
| coauthor-physics | 0 / 0 | — | — |
| cora | 0 / 0 | — | — |
| minesweeper | 0 / 0 | — | — |
| pubmed | 0 / 0 | — | — |
| questions | 0 / 0 | — | — |
| roman-empire | 0 / 0 | — | — |
| squirrel-filtered | 0 / 0 | — | — |
| tolokers | 0 / 0 | — | — |
| wikics | 0 / 0 | — | — |
| yelpchi | 0 / 0 | — | — |

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
| cs_ref | 0 | 0 | 0 | W: —; L: — |
| cs_2hop | 0 | 0 | 0 | W: —; L: — |
| cs_2hop_ref | 0 | 0 | 0 | W: —; L: — |
| clp | 0 | 0 | 0 | W: —; L: — |
| clp_es | 0 | 0 | 0 | W: —; L: — |
| linbp_b | 0 | 0 | 0 | W: —; L: — |

## Calibration (mean over splits; lower is better)

| dataset | EDSP-default NLL | EDSP-default ECE | EDSP-tuned NLL | EDSP-tuned ECE | best baseline by NLL (NLL / ECE) |
|---|---|---|---|---|---|
| amazon-computers | nan | nan | nan | nan | clp: 0.333 / 0.036 |
| amazon-fraud | nan | nan | nan | nan | linbp_b: 0.118 / 0.010 |
| amazon-photo | nan | nan | nan | nan | clp: 0.193 / 0.019 |
| amazon-ratings | nan | nan | nan | nan | cs_2hop: 1.509 / 0.141 |
| chameleon-filtered | nan | nan | nan | nan | linbp_b: 1.601 / 0.105 |
| citeseer | nan | nan | nan | nan | clp_es: 0.778 / 0.060 |
| coauthor-cs | nan | nan | nan | nan | clp: 0.177 / 0.021 |
| coauthor-physics | nan | nan | nan | nan | linbp_b: 0.141 / 0.009 |
| cora | nan | nan | nan | nan | clp_es: 0.518 / 0.089 |
| minesweeper | nan | nan | nan | nan | clp: 0.501 / 0.016 |
| pubmed | nan | nan | nan | nan | cs_2hop: 0.347 / 0.084 |
| questions | nan | nan | nan | nan | cs_2hop: 0.134 / 0.034 |
| roman-empire | nan | nan | nan | nan | clp: 1.155 / 0.040 |
| squirrel-filtered | nan | nan | nan | nan | clp: 1.427 / 0.056 |
| tolokers | nan | nan | nan | nan | linbp_b: 0.465 / 0.034 |
| wikics | nan | nan | nan | nan | clp: 0.743 / 0.086 |
| yelpchi | nan | nan | nan | nan | cs_2hop: 0.400 / 0.080 |

## H7 (descriptive): convergence and fixed-point non-uniqueness

| dataset | EDSP-default non-converged | oscillation flagged | mean fraction of nodes still moving | GS vs damped L1: mean | p95 | splits |
|---|---|---|---|---|---|---|
| amazon-computers | — | — | — | — | — | 0 |
| amazon-fraud | — | — | — | — | — | 0 |
| amazon-photo | — | — | — | — | — | 0 |
| amazon-ratings | — | — | — | — | — | 0 |
| chameleon-filtered | — | — | — | — | — | 0 |
| citeseer | — | — | — | — | — | 0 |
| coauthor-cs | — | — | — | — | — | 0 |
| coauthor-physics | — | — | — | — | — | 0 |
| cora | — | — | — | — | — | 0 |
| minesweeper | — | — | — | — | — | 0 |
| pubmed | — | — | — | — | — | 0 |
| questions | — | — | — | — | — | 0 |
| roman-empire | — | — | — | — | — | 0 |
| squirrel-filtered | — | — | — | — | — | 0 |
| tolokers | — | — | — | — | — | 0 |
| wikics | — | — | — | — | — | 0 |
| yelpchi | — | — | — | — | — | 0 |
