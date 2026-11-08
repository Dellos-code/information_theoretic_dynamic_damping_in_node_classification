# E3 — Track B (feature priors): EDSP-B vs logistic regression, Correct & Smooth and tuned GNNs, H4

Run records: 9 (git 3b2ae0acb743; runners actions-run-42467716806). Headline metric per dataset in brackets; mean ± std over splits (×100). Rows marked † are constructed benchmarks or case studies and are excluded from real-world aggregates.

| dataset [metric] | splits | EDSP-default | EDSP-tuned | majority | train_prior | feature_lr | correct_and_smooth | mlp | gcn | sage | gat | gprgnn | bwgnn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tolokers [roc_auc] | 1 | 73.52 ± 0.00 | 74.06 ± 0.00 | 50.00 ± 0.00 | 50.00 ± 0.00 | 71.84 ± 0.00 | 75.74 ± 0.00 | 72.56 ± 0.00 | 81.29 ± 0.00 | **82.94 ± 0.00** | — | — | — |

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
| tolokers | 0.550 | 0.159 | 0.555 | 0.150 | sage: 0.400 / 0.032 |

## H7 (descriptive): convergence and fixed-point non-uniqueness

| dataset | EDSP-default non-converged | oscillation flagged | mean fraction of nodes still moving | GS vs damped L1: mean | p95 | splits |
|---|---|---|---|---|---|---|
| tolokers | 0.000 | 0.000 | 0.000 | — | — | 0 |

## Test accuracy by local homophily (descriptive, full labels; bin i = [i/10, (i+1)/10)), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tolokers | EDSP-default | 74.1 (85) | 28.6 (35) | 41.8 (110) | 50.5 (216) | 55.3 (333) | 79.4 (539) | 81.4 (355) | 87.3 (267) | 95.8 (523) | 91.2 (477) |
| tolokers | EDSP-tuned | 68.2 (85) | 22.9 (35) | 34.5 (110) | 48.6 (216) | 52.6 (333) | 74.6 (539) | 78.9 (355) | 87.3 (267) | 96.6 (523) | 91.0 (477) |
| tolokers | feature_lr | 74.1 (85) | 28.6 (35) | 41.8 (110) | 50.5 (216) | 55.3 (333) | 79.4 (539) | 81.4 (355) | 87.3 (267) | 95.8 (523) | 91.2 (477) |
| tolokers | correct_and_smooth | 52.9 (85) | 8.6 (35) | 38.2 (110) | 44.4 (216) | 52.9 (333) | 50.6 (539) | 66.5 (355) | 86.5 (267) | 97.7 (523) | 96.2 (477) |
| tolokers | gcn | 74.1 (85) | 25.7 (35) | 40.0 (110) | 48.1 (216) | 55.0 (333) | 80.7 (539) | 84.8 (355) | 95.1 (267) | 97.9 (523) | 92.9 (477) |
| tolokers | sage | 75.3 (85) | 25.7 (35) | 40.0 (110) | 50.5 (216) | 55.0 (333) | 80.5 (539) | 84.2 (355) | 93.3 (267) | 97.1 (523) | 92.0 (477) |

## Test accuracy by degree (bin 0: isolated; bin b: 2^(b-1) ≤ deg < 2^b; last bin open), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tolokers | EDSP-default | — | 68.0 (100) | 80.7 (218) | 84.7 (301) | 80.7 (399) | 80.0 (446) | 79.0 (480) | 78.8 (406) | 76.9 (316) | 73.3 (191) | 45.8 (83) |
| tolokers | EDSP-tuned | — | 64.0 (100) | 78.9 (218) | 80.4 (301) | 79.4 (399) | 78.7 (446) | 79.2 (480) | 75.9 (406) | 71.8 (316) | 72.8 (191) | 45.8 (83) |
| tolokers | feature_lr | — | 68.0 (100) | 80.7 (218) | 84.7 (301) | 80.7 (399) | 80.0 (446) | 79.0 (480) | 78.8 (406) | 76.9 (316) | 73.3 (191) | 45.8 (83) |
| tolokers | correct_and_smooth | — | 69.0 (100) | 77.1 (218) | 81.1 (301) | 78.9 (399) | 71.7 (446) | 67.9 (480) | 63.3 (406) | 67.4 (316) | 64.9 (191) | 43.4 (83) |
| tolokers | gcn | — | 71.0 (100) | 85.8 (218) | 87.0 (301) | 81.7 (399) | 82.7 (446) | 81.5 (480) | 79.3 (406) | 76.6 (316) | 73.3 (191) | 45.8 (83) |
| tolokers | sage | — | 71.0 (100) | 84.9 (218) | 86.7 (301) | 81.7 (399) | 82.7 (446) | 80.2 (480) | 78.6 (406) | 76.9 (316) | 74.3 (191) | 44.6 (83) |
