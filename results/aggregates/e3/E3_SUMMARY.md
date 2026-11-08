# E3 — Track B (feature priors): EDSP-B vs logistic regression, Correct & Smooth and tuned GNNs, H4

Run records: 9 (git 3b2ae0acb743; runners actions-run-42471109374). Headline metric per dataset in brackets; mean ± std over splits (×100). Rows marked † are constructed benchmarks or case studies and are excluded from real-world aggregates.

| dataset [metric] | splits | EDSP-default | EDSP-tuned | majority | train_prior | feature_lr | correct_and_smooth | mlp | gcn | sage | gat | gprgnn | bwgnn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tolokers [roc_auc] | 1 | 69.95 ± 0.00 | 72.08 ± 0.00 | 50.00 ± 0.00 | 50.00 ± 0.00 | 70.61 ± 0.00 | 74.60 ± 0.00 | 71.38 ± 0.00 | 81.06 ± 0.00 | **82.35 ± 0.00** | — | — | — |

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
| tolokers | 0.552 | 0.155 | 0.560 | 0.156 | sage: 0.404 / 0.035 |

## H7 (descriptive): convergence and fixed-point non-uniqueness

| dataset | EDSP-default non-converged | oscillation flagged | mean fraction of nodes still moving | GS vs damped L1: mean | p95 | splits |
|---|---|---|---|---|---|---|
| tolokers | 0.000 | 0.000 | 0.000 | — | — | 0 |

## Test accuracy by local homophily (descriptive, full labels; bin i = [i/10, (i+1)/10)), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tolokers | EDSP-default | 78.0 (100) | 20.9 (43) | 48.1 (108) | 49.5 (212) | 56.8 (368) | 78.0 (536) | 84.0 (362) | 86.4 (243) | 96.9 (508) | 91.5 (460) |
| tolokers | EDSP-tuned | 68.0 (100) | 18.6 (43) | 43.5 (108) | 51.9 (212) | 50.3 (368) | 75.0 (536) | 82.9 (362) | 88.1 (243) | 97.4 (508) | 91.3 (460) |
| tolokers | feature_lr | 75.0 (100) | 23.3 (43) | 50.0 (108) | 51.4 (212) | 57.3 (368) | 77.6 (536) | 85.1 (362) | 86.4 (243) | 97.0 (508) | 91.7 (460) |
| tolokers | correct_and_smooth | 56.0 (100) | 11.6 (43) | 36.1 (108) | 39.6 (212) | 44.3 (368) | 54.7 (536) | 76.2 (362) | 91.8 (243) | 97.6 (508) | 96.1 (460) |
| tolokers | gcn | 82.0 (100) | 18.6 (43) | 38.0 (108) | 46.2 (212) | 56.8 (368) | 79.5 (536) | 90.1 (362) | 93.4 (243) | 98.6 (508) | 92.6 (460) |
| tolokers | sage | 82.0 (100) | 18.6 (43) | 41.7 (108) | 49.5 (212) | 55.4 (368) | 79.7 (536) | 87.8 (362) | 92.6 (243) | 97.4 (508) | 93.5 (460) |

## Test accuracy by degree (bin 0: isolated; bin b: 2^(b-1) ≤ deg < 2^b; last bin open), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tolokers | EDSP-default | — | 75.0 (116) | 81.7 (246) | 84.2 (285) | 81.4 (409) | 77.9 (443) | 76.6 (488) | 82.3 (390) | 75.4 (301) | 70.4 (186) | 51.3 (76) |
| tolokers | EDSP-tuned | — | 63.8 (116) | 81.7 (246) | 81.8 (285) | 80.4 (409) | 79.0 (443) | 75.8 (488) | 78.5 (390) | 71.4 (301) | 69.9 (186) | 53.9 (76) |
| tolokers | feature_lr | — | 72.4 (116) | 82.1 (246) | 83.9 (285) | 82.6 (409) | 79.7 (443) | 77.0 (488) | 81.8 (390) | 76.1 (301) | 69.4 (186) | 51.3 (76) |
| tolokers | correct_and_smooth | — | 71.6 (116) | 76.8 (246) | 79.6 (285) | 76.8 (409) | 72.9 (443) | 68.2 (488) | 64.1 (390) | 66.1 (301) | 65.1 (186) | 50.0 (76) |
| tolokers | gcn | — | 80.2 (116) | 85.4 (246) | 82.5 (285) | 84.6 (409) | 79.7 (443) | 79.1 (488) | 82.6 (390) | 75.7 (301) | 71.0 (186) | 51.3 (76) |
| tolokers | sage | — | 83.6 (116) | 86.2 (246) | 83.9 (285) | 83.9 (409) | 78.8 (443) | 78.3 (488) | 82.3 (390) | 74.8 (301) | 71.0 (186) | 51.3 (76) |
