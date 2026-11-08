# E3 — Track B (feature priors): EDSP-B vs logistic regression, Correct & Smooth and tuned GNNs, H4

Run records: 12 (git 41d502c903c8; runners actions-run-42475964446). Headline metric per dataset in brackets; mean ± std over splits (×100). Rows marked † are constructed benchmarks or case studies and are excluded from real-world aggregates.

| dataset [metric] | splits | EDSP-default | EDSP-tuned | majority | train_prior | feature_lr | correct_and_smooth | mlp | gcn | sage | gat | gprgnn | bwgnn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chameleon-filtered [accuracy] | 1 | 22.36 ± 0.00 | 36.02 ± 0.00 | 26.09 ± 0.00 | 26.09 ± 0.00 | 34.78 ± 0.00 | 37.89 ± 0.00 | 40.99 ± 0.00 | 43.48 ± 0.00 | 42.86 ± 0.00 | 43.48 ± 0.00 | 41.61 ± 0.00 | **44.10 ± 0.00** |

## Pre-registered secondary column: EDSP-default on non-converged runs

| dataset | non-converged runs | headline, last iterate | headline, mean of last two iterates |
|---|---|---|---|
| chameleon-filtered | 0 / 1 | — | — |

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
| chameleon-filtered | 1.594 | 0.027 | 1.583 | 0.152 | gcn: 1.349 / 0.152 |

## H7 (descriptive): convergence and fixed-point non-uniqueness

| dataset | EDSP-default non-converged | oscillation flagged | mean fraction of nodes still moving | GS vs damped L1: mean | p95 | splits |
|---|---|---|---|---|---|---|
| chameleon-filtered | 0.000 | 0.000 | 0.000 | — | — | 0 |

## Test accuracy by local homophily (descriptive, full labels; bin i = [i/10, (i+1)/10)), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chameleon-filtered | EDSP-default | 11.3 (53) | 0.0 (14) | 18.2 (44) | 26.3 (19) | 41.7 (12) | 50.0 (8) | 66.7 (6) | 100.0 (1) | 100.0 (1) | 66.7 (3) |
| chameleon-filtered | EDSP-tuned | 34.0 (53) | 64.3 (14) | 34.1 (44) | 36.8 (19) | 16.7 (12) | 37.5 (8) | 50.0 (6) | 0.0 (1) | 100.0 (1) | 0.0 (3) |
| chameleon-filtered | feature_lr | 34.0 (53) | 71.4 (14) | 29.5 (44) | 31.6 (19) | 25.0 (12) | 37.5 (8) | 16.7 (6) | 0.0 (1) | 100.0 (1) | 33.3 (3) |
| chameleon-filtered | correct_and_smooth | 32.1 (53) | 64.3 (14) | 27.3 (44) | 36.8 (19) | 33.3 (12) | 50.0 (8) | 66.7 (6) | 100.0 (1) | 100.0 (1) | 66.7 (3) |
| chameleon-filtered | gcn | 45.3 (53) | 71.4 (14) | 36.4 (44) | 47.4 (19) | 16.7 (12) | 25.0 (8) | 66.7 (6) | 0.0 (1) | 100.0 (1) | 66.7 (3) |
| chameleon-filtered | sage | 39.6 (53) | 71.4 (14) | 43.2 (44) | 47.4 (19) | 16.7 (12) | 25.0 (8) | 66.7 (6) | 0.0 (1) | 100.0 (1) | 33.3 (3) |
| chameleon-filtered | gat | 41.5 (53) | 71.4 (14) | 34.1 (44) | 42.1 (19) | 16.7 (12) | 50.0 (8) | 83.3 (6) | 0.0 (1) | 100.0 (1) | 100.0 (3) |
| chameleon-filtered | gprgnn | 30.2 (53) | 78.6 (14) | 38.6 (44) | 36.8 (19) | 33.3 (12) | 50.0 (8) | 83.3 (6) | 0.0 (1) | 100.0 (1) | 66.7 (3) |

## Test accuracy by degree (bin 0: isolated; bin b: 2^(b-1) ≤ deg < 2^b; last bin open), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chameleon-filtered | EDSP-default | — | 44.4 (9) | 17.9 (28) | 20.0 (50) | 15.4 (39) | 33.3 (12) | 16.7 (6) | 35.3 (17) | — | — | — |
| chameleon-filtered | EDSP-tuned | — | 44.4 (9) | 21.4 (28) | 48.0 (50) | 38.5 (39) | 16.7 (12) | 33.3 (6) | 29.4 (17) | — | — | — |
| chameleon-filtered | feature_lr | — | 55.6 (9) | 21.4 (28) | 46.0 (50) | 38.5 (39) | 8.3 (12) | 16.7 (6) | 29.4 (17) | — | — | — |
| chameleon-filtered | correct_and_smooth | — | 44.4 (9) | 25.0 (28) | 50.0 (50) | 48.7 (39) | 8.3 (12) | 0.0 (6) | 29.4 (17) | — | — | — |
| chameleon-filtered | gcn | — | 55.6 (9) | 39.3 (28) | 48.0 (50) | 46.2 (39) | 33.3 (12) | 16.7 (6) | 41.2 (17) | — | — | — |
| chameleon-filtered | sage | — | 44.4 (9) | 32.1 (28) | 48.0 (50) | 46.2 (39) | 33.3 (12) | 33.3 (6) | 47.1 (17) | — | — | — |
| chameleon-filtered | gat | — | 55.6 (9) | 42.9 (28) | 48.0 (50) | 56.4 (39) | 0.0 (12) | 16.7 (6) | 35.3 (17) | — | — | — |
| chameleon-filtered | gprgnn | — | 55.6 (9) | 21.4 (28) | 54.0 (50) | 56.4 (39) | 16.7 (12) | 16.7 (6) | 23.5 (17) | — | — | — |
