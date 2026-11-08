# E1 — Track-A EDSP vs B0/B1 Track-A baselines

Run records: 11 (git c87e8e837725; runners actions-run-42490815688). Headline metric per dataset in brackets; mean ± std over splits (×100). Rows marked † are constructed benchmarks or case studies and are excluded from real-world aggregates.

| dataset [metric] | splits | EDSP-default | EDSP-tuned | EDSP-fullH-default | EDSP-fullH-tuned | majority | train_prior | harmonic | harmonic_cmn | lgc | ppr_lp | linbp | signed_vote | signed_lgc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| genius [roc_auc] | 1 | 51.35 ± 0.00 | 60.09 ± 0.00 | — | — | 50.00 ± 0.00 | 50.00 ± 0.00 | 51.17 ± 0.00 | 51.17 ± 0.00 | **70.61 ± 0.00** | 60.93 ± 0.00 | 58.26 ± 0.00 | — | — |

## Pre-registered secondary column: EDSP-default on non-converged runs

| dataset | non-converged runs | headline, last iterate | headline, mean of last two iterates |
|---|---|---|---|
| genius | 0 / 1 | — | — |

## EDSP-tuned vs each baseline: win / tie / loss

Per dataset: two-sided Wilcoxon signed-rank over splits on the headline metric; Holm correction across datasets per baseline; win/loss if adjusted p < 0.05, else tie. Real-world datasets only.

| baseline | win | tie | loss | datasets (W / L) |
|---|---|---|---|---|
| harmonic | 0 | 0 | 0 | W: —; L: — |
| harmonic_cmn | 0 | 0 | 0 | W: —; L: — |
| lgc | 0 | 0 | 0 | W: —; L: — |
| ppr_lp | 0 | 0 | 0 | W: —; L: — |
| linbp | 0 | 0 | 0 | W: —; L: — |
| train_prior | 0 | 0 | 0 | W: —; L: — |
| signed_vote | 0 | 0 | 0 | W: —; L: — |
| signed_lgc | 0 | 0 | 0 | W: —; L: — |

## Calibration (mean over splits; lower is better)

| dataset | EDSP-default NLL | EDSP-default ECE | EDSP-tuned NLL | EDSP-tuned ECE | best baseline by NLL (NLL / ECE) |
|---|---|---|---|---|---|
| genius | 1.383 | 0.229 | 0.958 | 0.153 | lgc: 0.460 / 0.035 |

## H7 (descriptive): convergence and fixed-point non-uniqueness

| dataset | EDSP-default non-converged | oscillation flagged | mean fraction of nodes still moving | GS vs damped L1: mean | p95 | splits |
|---|---|---|---|---|---|---|
| genius | 0.000 | 0.000 | 0.000 | — | — | 0 |

## Test accuracy by local homophily (descriptive, full labels; bin i = [i/10, (i+1)/10)), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| genius | EDSP-default | 27.9 (34166) | 8.8 (114) | 26.5 (820) | 54.4 (2029) | 61.2 (459) | 60.5 (8360) | 95.7 (1680) | 99.6 (955) | 100.0 (1028) | 93.3 (34762) |
| genius | EDSP-tuned | 68.7 (34166) | 90.4 (114) | 93.3 (820) | 96.7 (2029) | 90.0 (459) | 71.3 (8360) | 79.5 (1680) | 81.6 (955) | 91.2 (1028) | 70.4 (34762) |
| genius | harmonic_cmn | 33.4 (34166) | 31.6 (114) | 60.9 (820) | 84.6 (2029) | 78.4 (459) | 79.3 (8360) | 94.9 (1680) | 94.7 (955) | 99.2 (1028) | 93.5 (34762) |
| genius | lgc | 73.6 (34166) | 45.6 (114) | 76.6 (820) | 91.6 (2029) | 79.5 (459) | 80.7 (8360) | 94.9 (1680) | 94.3 (955) | 98.7 (1028) | 84.7 (34762) |
| genius | ppr_lp | 73.9 (34166) | 45.6 (114) | 76.7 (820) | 91.7 (2029) | 79.3 (459) | 79.5 (8360) | 94.9 (1680) | 94.1 (955) | 98.7 (1028) | 83.9 (34762) |
| genius | linbp | 86.2 (34166) | 93.9 (114) | 96.7 (820) | 91.6 (2029) | 85.2 (459) | 72.6 (8360) | 70.4 (1680) | 64.2 (955) | 76.3 (1028) | 49.5 (34762) |

## Test accuracy by degree (bin 0: isolated; bin b: 2^(b-1) ≤ deg < 2^b; last bin open), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| genius | EDSP-default | 100.0 (19) | 62.2 (55757) | 54.6 (18798) | 80.3 (4559) | 90.2 (1899) | 64.3 (1604) | 43.0 (1198) | 66.3 (294) | 72.2 (126) | 52.4 (105) | 84.8 (33) |
| genius | EDSP-tuned | 100.0 (19) | 66.9 (55757) | 76.1 (18798) | 88.2 (4559) | 90.9 (1899) | 83.7 (1604) | 86.3 (1198) | 85.0 (294) | 80.2 (126) | 60.0 (105) | 81.8 (33) |
| genius | harmonic_cmn | 100.0 (19) | 62.2 (55757) | 74.7 (18798) | 92.4 (4559) | 91.5 (1899) | 68.9 (1604) | 41.7 (1198) | 68.4 (294) | 76.2 (126) | 53.3 (105) | 84.8 (33) |
| genius | lgc | 100.0 (19) | 77.6 (55757) | 86.8 (18798) | 96.9 (4559) | 91.8 (1899) | 69.6 (1604) | 41.8 (1198) | 67.7 (294) | 76.2 (126) | 54.3 (105) | 84.8 (33) |
| genius | ppr_lp | 100.0 (19) | 77.3 (55757) | 86.1 (18798) | 96.9 (4559) | 91.8 (1899) | 69.6 (1604) | 41.7 (1198) | 68.4 (294) | 76.2 (126) | 52.4 (105) | 84.8 (33) |
| genius | linbp | 100.0 (19) | 65.9 (55757) | 75.8 (18798) | 75.9 (4559) | 76.3 (1899) | 75.3 (1604) | 80.6 (1198) | 75.9 (294) | 57.1 (126) | 55.2 (105) | 15.2 (33) |
