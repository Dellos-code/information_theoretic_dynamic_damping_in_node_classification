# E3 — Track B (feature priors): EDSP-B vs logistic regression, Correct & Smooth and tuned GNNs, H4

Run records: 12 (git 41d502c903c8; runners actions-run-42475979970). Headline metric per dataset in brackets; mean ± std over splits (×100). Rows marked † are constructed benchmarks or case studies and are excluded from real-world aggregates.

| dataset [metric] | splits | EDSP-default | EDSP-tuned | majority | train_prior | feature_lr | correct_and_smooth | mlp | gcn | sage | gat | gprgnn | bwgnn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| squirrel-filtered [accuracy] | 1 | 34.95 ± 0.00 | 39.78 ± 0.00 | 33.85 ± 0.00 | 33.85 ± 0.00 | 40.00 ± 0.00 | 27.25 ± 0.00 | 39.78 ± 0.00 | 39.34 ± 0.00 | 39.56 ± 0.00 | 38.02 ± 0.00 | **40.88 ± 0.00** | 40.66 ± 0.00 |

## Pre-registered secondary column: EDSP-default on non-converged runs

| dataset | non-converged runs | headline, last iterate | headline, mean of last two iterates |
|---|---|---|---|
| squirrel-filtered | 0 / 1 | — | — |

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
| squirrel-filtered | 1.578 | 0.133 | 1.570 | 0.179 | gat: 1.384 / 0.088 |

## H7 (descriptive): convergence and fixed-point non-uniqueness

| dataset | EDSP-default non-converged | oscillation flagged | mean fraction of nodes still moving | GS vs damped L1: mean | p95 | splits |
|---|---|---|---|---|---|---|
| squirrel-filtered | 0.000 | 0.000 | 0.000 | — | — | 0 |

## Test accuracy by local homophily (descriptive, full labels; bin i = [i/10, (i+1)/10)), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| squirrel-filtered | EDSP-default | 42.1 (133) | 31.7 (123) | 15.8 (114) | 44.4 (45) | 61.9 (21) | 45.5 (11) | 100.0 (7) | 100.0 (1) | — | — |
| squirrel-filtered | EDSP-tuned | 51.9 (133) | 35.0 (123) | 22.8 (114) | 44.4 (45) | 66.7 (21) | 36.4 (11) | 57.1 (7) | 100.0 (1) | — | — |
| squirrel-filtered | feature_lr | 54.1 (133) | 35.0 (123) | 23.7 (114) | 40.0 (45) | 61.9 (21) | 36.4 (11) | 57.1 (7) | 100.0 (1) | — | — |
| squirrel-filtered | correct_and_smooth | 12.8 (133) | 20.3 (123) | 26.3 (114) | 48.9 (45) | 76.2 (21) | 72.7 (11) | 71.4 (7) | 100.0 (1) | — | — |
| squirrel-filtered | gcn | 45.9 (133) | 37.4 (123) | 22.8 (114) | 53.3 (45) | 61.9 (21) | 36.4 (11) | 57.1 (7) | 100.0 (1) | — | — |
| squirrel-filtered | sage | 54.9 (133) | 37.4 (123) | 20.2 (114) | 35.6 (45) | 66.7 (21) | 27.3 (11) | 57.1 (7) | 100.0 (1) | — | — |
| squirrel-filtered | gat | 46.6 (133) | 31.7 (123) | 26.3 (114) | 44.4 (45) | 61.9 (21) | 27.3 (11) | 71.4 (7) | 100.0 (1) | — | — |
| squirrel-filtered | gprgnn | 44.4 (133) | 39.8 (123) | 28.9 (114) | 51.1 (45) | 61.9 (21) | 27.3 (11) | 71.4 (7) | 100.0 (1) | — | — |

## Test accuracy by degree (bin 0: isolated; bin b: 2^(b-1) ≤ deg < 2^b; last bin open), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| squirrel-filtered | EDSP-default | — | 23.1 (13) | 36.1 (61) | 48.1 (106) | 39.2 (97) | 26.2 (65) | 25.0 (32) | 36.8 (19) | 25.5 (51) | 0.0 (10) | 0.0 (1) |
| squirrel-filtered | EDSP-tuned | — | 23.1 (13) | 50.8 (61) | 51.9 (106) | 39.2 (97) | 27.7 (65) | 34.4 (32) | 47.4 (19) | 31.4 (51) | 0.0 (10) | 0.0 (1) |
| squirrel-filtered | feature_lr | — | 38.5 (13) | 52.5 (61) | 51.9 (106) | 41.2 (97) | 26.2 (65) | 25.0 (32) | 47.4 (19) | 31.4 (51) | 0.0 (10) | 0.0 (1) |
| squirrel-filtered | correct_and_smooth | — | 7.7 (13) | 26.2 (61) | 30.2 (106) | 33.0 (97) | 23.1 (65) | 18.8 (32) | 36.8 (19) | 29.4 (51) | 0.0 (10) | 0.0 (1) |
| squirrel-filtered | gcn | — | 38.5 (13) | 37.7 (61) | 50.9 (106) | 44.3 (97) | 29.2 (65) | 40.6 (32) | 36.8 (19) | 29.4 (51) | 0.0 (10) | 0.0 (1) |
| squirrel-filtered | sage | — | 53.8 (13) | 45.9 (61) | 50.0 (106) | 40.2 (97) | 30.8 (65) | 28.1 (32) | 47.4 (19) | 29.4 (51) | 0.0 (10) | 0.0 (1) |
| squirrel-filtered | gat | — | 46.2 (13) | 36.1 (61) | 52.8 (106) | 38.1 (97) | 38.5 (65) | 25.0 (32) | 36.8 (19) | 21.6 (51) | 10.0 (10) | 0.0 (1) |
| squirrel-filtered | gprgnn | — | 53.8 (13) | 41.0 (61) | 46.2 (106) | 41.2 (97) | 26.2 (65) | 37.5 (32) | 52.6 (19) | 37.3 (51) | 60.0 (10) | 100.0 (1) |
