# E3 — Track B (feature priors): EDSP-B vs logistic regression, Correct & Smooth and tuned GNNs, H4

Run records: 40 (git bdd1ad76f33a; runners actions-run-42613047156). Headline metric per dataset in brackets; mean ± std over splits (×100). Rows marked † are constructed benchmarks or case studies and are excluded from real-world aggregates.

| dataset [metric] | splits | EDSP-default | EDSP-tuned | majority | train_prior | feature_lr | correct_and_smooth | mlp | gcn | sage | gat | gprgnn | bwgnn | gat(h64) | care_gnn | pc_gnn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| amazon-fraud [roc_auc] | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — | 95.18 ± 0.93 | **96.36 ± 0.54** |
| yelpchi [roc_auc] | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — | 79.16 ± 0.38 | **87.39 ± 0.40** |

## Pre-registered secondary column: EDSP-default on non-converged runs

| dataset | non-converged runs | headline, last iterate | headline, mean of last two iterates |
|---|---|---|---|
| amazon-fraud | 0 / 0 | — | — |
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
| care_gnn | 0 | 0 | 0 | W: —; L: — |
| pc_gnn | 0 | 0 | 0 | W: —; L: — |

## Calibration (mean over splits; lower is better)

| dataset | EDSP-default NLL | EDSP-default ECE | EDSP-tuned NLL | EDSP-tuned ECE | best baseline by NLL (NLL / ECE) |
|---|---|---|---|---|---|
| amazon-fraud | nan | nan | nan | nan | pc_gnn: 0.209 / 0.101 |
| yelpchi | nan | nan | nan | nan | pc_gnn: 0.465 / 0.135 |

## H7 (descriptive): convergence and fixed-point non-uniqueness

| dataset | EDSP-default non-converged | oscillation flagged | mean fraction of nodes still moving | GS vs damped L1: mean | p95 | splits |
|---|---|---|---|---|---|---|
| amazon-fraud | — | — | — | — | — | 0 |
| yelpchi | — | — | — | — | — | 0 |
