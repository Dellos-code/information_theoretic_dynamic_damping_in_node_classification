# E3 — Track B (feature priors): EDSP-B vs logistic regression, Correct & Smooth and tuned GNNs, H4

Run records: 12 (git 41d502c903c8; runners actions-run-42475974592). Headline metric per dataset in brackets; mean ± std over splits (×100). Rows marked † are constructed benchmarks or case studies and are excluded from real-world aggregates.

| dataset [metric] | splits | EDSP-default | EDSP-tuned | majority | train_prior | feature_lr | correct_and_smooth | mlp | gcn | sage | gat | gprgnn | bwgnn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chameleon-filtered [accuracy] | 1 | 34.88 ± 0.00 | 37.21 ± 0.00 | 25.58 ± 0.00 | 25.58 ± 0.00 | 35.47 ± 0.00 | 35.47 ± 0.00 | **44.19 ± 0.00** | 39.53 ± 0.00 | 40.12 ± 0.00 | 43.60 ± 0.00 | 40.70 ± 0.00 | 43.02 ± 0.00 |

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
| chameleon-filtered | 1.578 | 0.120 | 1.581 | 0.168 | mlp: 1.361 / 0.104 |

## H7 (descriptive): convergence and fixed-point non-uniqueness

| dataset | EDSP-default non-converged | oscillation flagged | mean fraction of nodes still moving | GS vs damped L1: mean | p95 | splits |
|---|---|---|---|---|---|---|
| chameleon-filtered | 0.000 | 0.000 | 0.000 | — | — | 0 |

## Test accuracy by local homophily (descriptive, full labels; bin i = [i/10, (i+1)/10)), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chameleon-filtered | EDSP-default | 16.7 (48) | 25.0 (16) | 38.5 (39) | 32.4 (34) | 70.0 (10) | 45.5 (11) | 33.3 (6) | 100.0 (1) | 100.0 (1) | 100.0 (6) |
| chameleon-filtered | EDSP-tuned | 35.4 (48) | 37.5 (16) | 41.0 (39) | 35.3 (34) | 50.0 (10) | 45.5 (11) | 16.7 (6) | 0.0 (1) | 100.0 (1) | 16.7 (6) |
| chameleon-filtered | feature_lr | 31.2 (48) | 37.5 (16) | 43.6 (39) | 32.4 (34) | 50.0 (10) | 36.4 (11) | 16.7 (6) | 0.0 (1) | 100.0 (1) | 16.7 (6) |
| chameleon-filtered | correct_and_smooth | 14.6 (48) | 56.2 (16) | 15.4 (39) | 44.1 (34) | 70.0 (10) | 54.5 (11) | 66.7 (6) | 0.0 (1) | 100.0 (1) | 100.0 (6) |
| chameleon-filtered | gcn | 18.8 (48) | 50.0 (16) | 25.6 (39) | 55.9 (34) | 40.0 (10) | 63.6 (11) | 66.7 (6) | 0.0 (1) | 100.0 (1) | 100.0 (6) |
| chameleon-filtered | sage | 37.5 (48) | 50.0 (16) | 33.3 (39) | 41.2 (34) | 40.0 (10) | 45.5 (11) | 33.3 (6) | 0.0 (1) | 100.0 (1) | 66.7 (6) |
| chameleon-filtered | gat | 22.9 (48) | 56.2 (16) | 43.6 (39) | 41.2 (34) | 60.0 (10) | 63.6 (11) | 66.7 (6) | 100.0 (1) | 100.0 (1) | 83.3 (6) |
| chameleon-filtered | gprgnn | 29.2 (48) | 62.5 (16) | 33.3 (39) | 38.2 (34) | 60.0 (10) | 45.5 (11) | 50.0 (6) | 0.0 (1) | 100.0 (1) | 83.3 (6) |

## Test accuracy by degree (bin 0: isolated; bin b: 2^(b-1) ≤ deg < 2^b; last bin open), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chameleon-filtered | EDSP-default | — | 37.5 (8) | 32.0 (25) | 37.0 (46) | 39.1 (46) | 33.3 (15) | 40.0 (15) | 17.6 (17) | — | — | — |
| chameleon-filtered | EDSP-tuned | — | 25.0 (8) | 28.0 (25) | 43.5 (46) | 45.7 (46) | 20.0 (15) | 53.3 (15) | 17.6 (17) | — | — | — |
| chameleon-filtered | feature_lr | — | 25.0 (8) | 20.0 (25) | 41.3 (46) | 43.5 (46) | 20.0 (15) | 53.3 (15) | 23.5 (17) | — | — | — |
| chameleon-filtered | correct_and_smooth | — | 25.0 (8) | 32.0 (25) | 43.5 (46) | 41.3 (46) | 13.3 (15) | 13.3 (15) | 47.1 (17) | — | — | — |
| chameleon-filtered | gcn | — | 50.0 (8) | 32.0 (25) | 45.7 (46) | 47.8 (46) | 13.3 (15) | 26.7 (15) | 41.2 (17) | — | — | — |
| chameleon-filtered | sage | — | 50.0 (8) | 24.0 (25) | 39.1 (46) | 45.7 (46) | 40.0 (15) | 46.7 (15) | 41.2 (17) | — | — | — |
| chameleon-filtered | gat | — | 62.5 (8) | 28.0 (25) | 39.1 (46) | 52.2 (46) | 40.0 (15) | 60.0 (15) | 35.3 (17) | — | — | — |
| chameleon-filtered | gprgnn | — | 25.0 (8) | 28.0 (25) | 45.7 (46) | 47.8 (46) | 33.3 (15) | 53.3 (15) | 29.4 (17) | — | — | — |
