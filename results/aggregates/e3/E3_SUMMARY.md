# E3 — Track B (feature priors): EDSP-B vs logistic regression, Correct & Smooth and tuned GNNs, H4

Run records: 12 (git 41d502c903c8; runners actions-run-42475969615). Headline metric per dataset in brackets; mean ± std over splits (×100). Rows marked † are constructed benchmarks or case studies and are excluded from real-world aggregates.

| dataset [metric] | splits | EDSP-default | EDSP-tuned | majority | train_prior | feature_lr | correct_and_smooth | mlp | gcn | sage | gat | gprgnn | bwgnn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chameleon-filtered [accuracy] | 1 | 30.19 ± 0.00 | 33.33 ± 0.00 | 27.04 ± 0.00 | 27.04 ± 0.00 | 35.22 ± 0.00 | 35.22 ± 0.00 | 40.88 ± 0.00 | 45.91 ± 0.00 | 47.80 ± 0.00 | **49.06 ± 0.00** | 37.74 ± 0.00 | 46.54 ± 0.00 |

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
| chameleon-filtered | 1.587 | 0.083 | 1.590 | 0.137 | gat: 1.282 / 0.090 |

## H7 (descriptive): convergence and fixed-point non-uniqueness

| dataset | EDSP-default non-converged | oscillation flagged | mean fraction of nodes still moving | GS vs damped L1: mean | p95 | splits |
|---|---|---|---|---|---|---|
| chameleon-filtered | 0.000 | 0.000 | 0.000 | — | — | 0 |

## Test accuracy by local homophily (descriptive, full labels; bin i = [i/10, (i+1)/10)), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chameleon-filtered | EDSP-default | 20.8 (48) | 37.5 (16) | 25.5 (47) | 30.0 (20) | 25.0 (4) | 62.5 (8) | 33.3 (6) | 50.0 (2) | 100.0 (2) | 50.0 (6) |
| chameleon-filtered | EDSP-tuned | 37.5 (48) | 37.5 (16) | 25.5 (47) | 30.0 (20) | 0.0 (4) | 50.0 (8) | 50.0 (6) | 50.0 (2) | 100.0 (2) | 16.7 (6) |
| chameleon-filtered | feature_lr | 41.7 (48) | 37.5 (16) | 34.0 (47) | 25.0 (20) | 0.0 (4) | 50.0 (8) | 50.0 (6) | 50.0 (2) | 50.0 (2) | 0.0 (6) |
| chameleon-filtered | correct_and_smooth | 27.1 (48) | 31.2 (16) | 29.8 (47) | 30.0 (20) | 25.0 (4) | 62.5 (8) | 100.0 (6) | 50.0 (2) | 100.0 (2) | 50.0 (6) |
| chameleon-filtered | gcn | 35.4 (48) | 31.2 (16) | 42.6 (47) | 50.0 (20) | 25.0 (4) | 75.0 (8) | 100.0 (6) | 100.0 (2) | 100.0 (2) | 66.7 (6) |
| chameleon-filtered | sage | 39.6 (48) | 43.8 (16) | 57.4 (47) | 40.0 (20) | 25.0 (4) | 75.0 (8) | 66.7 (6) | 0.0 (2) | 100.0 (2) | 33.3 (6) |
| chameleon-filtered | gat | 39.6 (48) | 43.8 (16) | 55.3 (47) | 50.0 (20) | 0.0 (4) | 50.0 (8) | 100.0 (6) | 50.0 (2) | 100.0 (2) | 50.0 (6) |
| chameleon-filtered | gprgnn | 25.0 (48) | 37.5 (16) | 38.3 (47) | 35.0 (20) | 0.0 (4) | 62.5 (8) | 100.0 (6) | 0.0 (2) | 100.0 (2) | 66.7 (6) |

## Test accuracy by degree (bin 0: isolated; bin b: 2^(b-1) ≤ deg < 2^b; last bin open), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chameleon-filtered | EDSP-default | — | 16.7 (6) | 28.1 (32) | 45.9 (37) | 42.1 (38) | 16.7 (12) | 7.7 (13) | 10.0 (20) | 0.0 (1) | — | — |
| chameleon-filtered | EDSP-tuned | — | 33.3 (6) | 40.6 (32) | 45.9 (37) | 34.2 (38) | 16.7 (12) | 23.1 (13) | 15.0 (20) | 0.0 (1) | — | — |
| chameleon-filtered | feature_lr | — | 33.3 (6) | 40.6 (32) | 43.2 (37) | 36.8 (38) | 25.0 (12) | 23.1 (13) | 25.0 (20) | 0.0 (1) | — | — |
| chameleon-filtered | correct_and_smooth | — | 16.7 (6) | 37.5 (32) | 59.5 (37) | 42.1 (38) | 16.7 (12) | 7.7 (13) | 10.0 (20) | 0.0 (1) | — | — |
| chameleon-filtered | gcn | — | 33.3 (6) | 56.2 (32) | 59.5 (37) | 39.5 (38) | 41.7 (12) | 38.5 (13) | 30.0 (20) | 0.0 (1) | — | — |
| chameleon-filtered | sage | — | 33.3 (6) | 46.9 (32) | 51.4 (37) | 52.6 (38) | 33.3 (12) | 23.1 (13) | 65.0 (20) | 0.0 (1) | — | — |
| chameleon-filtered | gat | — | 33.3 (6) | 46.9 (32) | 56.8 (37) | 52.6 (38) | 41.7 (12) | 38.5 (13) | 50.0 (20) | 0.0 (1) | — | — |
| chameleon-filtered | gprgnn | — | 16.7 (6) | 37.5 (32) | 62.2 (37) | 36.8 (38) | 41.7 (12) | 15.4 (13) | 15.0 (20) | 0.0 (1) | — | — |
