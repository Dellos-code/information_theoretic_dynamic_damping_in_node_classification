# E3 — Track B (feature priors): EDSP-B vs logistic regression, Correct & Smooth and tuned GNNs, H4

Run records: 9 (git d8543a63f80a; runners actions-run-42458915368). Headline metric per dataset in brackets; mean ± std over splits (×100). Rows marked † are constructed benchmarks or case studies and are excluded from real-world aggregates.

| dataset [metric] | splits | EDSP-default | EDSP-tuned | majority | train_prior | feature_lr | correct_and_smooth | mlp | gcn | sage | gat | gprgnn | bwgnn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| amazon-computers [accuracy] | 1 | 87.60 ± 0.00 | 87.60 ± 0.00 | 37.49 ± 0.00 | 37.49 ± 0.00 | 84.80 ± 0.00 | 89.85 ± 0.00 | 85.78 ± 0.00 | 91.20 ± 0.00 | **91.56 ± 0.00** | — | — | — |

## Pre-registered secondary column: EDSP-default on non-converged runs

| dataset | non-converged runs | headline, last iterate | headline, mean of last two iterates |
|---|---|---|---|
| amazon-computers | 0 / 1 | — | — |

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
| amazon-computers | 0.826 | 0.368 | 0.826 | 0.368 | sage: 0.277 / 0.025 |

## H7 (descriptive): convergence and fixed-point non-uniqueness

| dataset | EDSP-default non-converged | oscillation flagged | mean fraction of nodes still moving | GS vs damped L1: mean | p95 | splits |
|---|---|---|---|---|---|---|
| amazon-computers | 0.000 | 0.000 | 0.000 | — | — | 0 |

## Test accuracy by local homophily (descriptive, full labels; bin i = [i/10, (i+1)/10)), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| amazon-computers | EDSP-default | 21.9 (64) | 34.9 (43) | 46.9 (64) | 61.2 (67) | 63.0 (92) | 75.9 (162) | 85.0 (180) | 91.2 (250) | 97.3 (484) | 96.7 (1285) |
| amazon-computers | EDSP-tuned | 21.9 (64) | 34.9 (43) | 46.9 (64) | 61.2 (67) | 63.0 (92) | 75.9 (162) | 85.0 (180) | 91.2 (250) | 97.3 (484) | 96.7 (1285) |
| amazon-computers | feature_lr | 34.4 (64) | 53.5 (43) | 59.4 (64) | 65.7 (67) | 65.2 (92) | 69.8 (162) | 80.0 (180) | 84.8 (250) | 92.6 (484) | 93.0 (1285) |
| amazon-computers | correct_and_smooth | 26.6 (64) | 25.6 (43) | 53.1 (64) | 64.2 (67) | 69.6 (92) | 81.5 (162) | 90.0 (180) | 93.6 (250) | 98.8 (484) | 98.3 (1285) |
| amazon-computers | gcn | 18.8 (64) | 14.0 (43) | 39.1 (64) | 62.7 (67) | 78.3 (92) | 89.5 (162) | 93.3 (180) | 97.6 (250) | 100.0 (484) | 99.5 (1285) |
| amazon-computers | sage | 17.2 (64) | 18.6 (43) | 32.8 (64) | 65.7 (67) | 72.8 (92) | 89.5 (162) | 97.8 (180) | 99.2 (250) | 100.0 (484) | 99.5 (1285) |

## Test accuracy by degree (bin 0: isolated; bin b: 2^(b-1) ≤ deg < 2^b; last bin open), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| amazon-computers | EDSP-default | 55.9 (59) | 62.3 (69) | 80.7 (119) | 88.8 (331) | 86.8 (493) | 88.2 (684) | 90.4 (627) | 94.2 (278) | 92.3 (65) | 93.8 (16) | 88.9 (9) |
| amazon-computers | EDSP-tuned | 55.9 (59) | 62.3 (69) | 80.7 (119) | 88.8 (331) | 86.8 (493) | 88.2 (684) | 90.4 (627) | 94.2 (278) | 92.3 (65) | 93.8 (16) | 88.9 (9) |
| amazon-computers | feature_lr | 55.9 (59) | 62.3 (69) | 77.3 (119) | 84.3 (331) | 82.6 (493) | 84.4 (684) | 88.4 (627) | 94.2 (278) | 92.3 (65) | 100.0 (16) | 100.0 (9) |
| amazon-computers | correct_and_smooth | 55.9 (59) | 69.6 (69) | 84.0 (119) | 90.6 (331) | 88.4 (493) | 91.5 (684) | 92.5 (627) | 95.7 (278) | 90.8 (65) | 93.8 (16) | 88.9 (9) |
| amazon-computers | gcn | 52.5 (59) | 75.4 (69) | 88.2 (119) | 91.5 (331) | 93.1 (493) | 92.5 (684) | 92.7 (627) | 94.6 (278) | 89.2 (65) | 93.8 (16) | 88.9 (9) |
| amazon-computers | sage | 59.3 (59) | 69.6 (69) | 88.2 (119) | 93.1 (331) | 93.1 (493) | 92.4 (684) | 93.0 (627) | 95.7 (278) | 90.8 (65) | 93.8 (16) | 88.9 (9) |
