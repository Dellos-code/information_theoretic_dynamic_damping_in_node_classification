# E3 — Track B (feature priors): EDSP-B vs logistic regression, Correct & Smooth and tuned GNNs, H4

Run records: 80 (git bdd1ad76f33a; runners actions-run-42613056688). Headline metric per dataset in brackets; mean ± std over splits (×100). Rows marked † are constructed benchmarks or case studies and are excluded from real-world aggregates.

| dataset [metric] | splits | EDSP-default | EDSP-tuned | majority | train_prior | feature_lr | correct_and_smooth | mlp | gcn | sage | gat | gprgnn | bwgnn | gat(h64) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yelpchi [roc_auc] | 10 | 64.76 ± 5.85 | 76.85 ± 0.28 | 50.00 ± 0.00 | 50.00 ± 0.00 | 77.01 ± 0.34 | 70.97 ± 0.35 | 80.53 ± 0.40 | — | — | — | — | **82.28 ± 1.21** | — |

## Pre-registered secondary column: EDSP-default on non-converged runs

| dataset | non-converged runs | headline, last iterate | headline, mean of last two iterates |
|---|---|---|---|
| yelpchi | 0 / 10 | — | — |

## EDSP-tuned vs each baseline: win / tie / loss

Per dataset: two-sided Wilcoxon signed-rank over splits on the headline metric; Holm correction across datasets per baseline; win/loss if adjusted p < 0.05, else tie. Real-world datasets only.

| baseline | win | tie | loss | datasets (W / L) |
|---|---|---|---|---|
| feature_lr | 0 | 0 | 1 | W: —; L: yelpchi |
| correct_and_smooth | 1 | 0 | 0 | W: yelpchi; L: — |
| mlp | 0 | 0 | 1 | W: —; L: yelpchi |
| gcn | 0 | 0 | 0 | W: —; L: — |
| sage | 0 | 0 | 0 | W: —; L: — |
| gat | 0 | 0 | 0 | W: —; L: — |
| gprgnn | 0 | 0 | 0 | W: —; L: — |
| bwgnn | 0 | 0 | 1 | W: —; L: yelpchi |

## Calibration (mean over splits; lower is better)

| dataset | EDSP-default NLL | EDSP-default ECE | EDSP-tuned NLL | EDSP-tuned ECE | best baseline by NLL (NLL / ECE) |
|---|---|---|---|---|---|
| yelpchi | 0.477 | 0.165 | 0.476 | 0.188 | bwgnn: 0.340 / 0.052 |

## H7 (descriptive): convergence and fixed-point non-uniqueness

| dataset | EDSP-default non-converged | oscillation flagged | mean fraction of nodes still moving | GS vs damped L1: mean | p95 | splits |
|---|---|---|---|---|---|---|
| yelpchi | 0.000 | 0.000 | 0.000 | — | — | 0 |

## Test accuracy by local homophily (descriptive, full labels; bin i = [i/10, (i+1)/10)), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| yelpchi | EDSP-default | 0.5 (4768) | 1.1 (13645) | 3.0 (4955) | 27.5 (1866) | 29.5 (576) | 67.4 (737) | 73.1 (3705) | 99.4 (15543) | 99.9 (84971) | 99.5 (53003) |
| yelpchi | EDSP-tuned | 16.0 (4768) | 17.3 (13645) | 16.5 (4955) | 36.0 (1866) | 30.0 (576) | 68.0 (737) | 75.5 (3705) | 97.5 (15543) | 97.6 (84971) | 96.9 (53003) |
| yelpchi | feature_lr | 13.2 (4768) | 13.6 (13645) | 11.8 (4955) | 33.1 (1866) | 31.6 (576) | 68.4 (737) | 75.6 (3705) | 98.6 (15543) | 98.8 (84971) | 98.2 (53003) |
| yelpchi | correct_and_smooth | 0.3 (4768) | 5.2 (13645) | 11.4 (4955) | 36.0 (1866) | 39.2 (576) | 69.1 (737) | 80.5 (3705) | 98.6 (15543) | 99.6 (84971) | 99.8 (53003) |

## Test accuracy by degree (bin 0: isolated; bin b: 2^(b-1) ≤ deg < 2^b; last bin open), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yelpchi | EDSP-default | 74.5 (51) | 63.6 (143) | 73.5 (400) | 76.2 (1220) | 82.6 (2847) | 81.0 (7185) | 83.6 (18437) | 85.5 (39424) | 86.5 (79924) | 86.3 (34189) | — |
| yelpchi | EDSP-tuned | 74.5 (51) | 59.4 (143) | 69.0 (400) | 73.2 (1220) | 80.3 (2847) | 79.5 (7185) | 83.0 (18437) | 85.7 (39424) | 86.8 (79924) | 87.1 (34189) | — |
| yelpchi | feature_lr | 74.5 (51) | 65.7 (143) | 73.2 (400) | 77.0 (1220) | 81.5 (2847) | 80.2 (7185) | 83.6 (18437) | 86.0 (39424) | 87.2 (79924) | 87.7 (34189) | — |
| yelpchi | correct_and_smooth | 74.5 (51) | 73.4 (143) | 76.2 (400) | 78.4 (1220) | 83.6 (2847) | 81.9 (7185) | 84.4 (18437) | 86.0 (39424) | 87.0 (79924) | 87.4 (34189) | — |
