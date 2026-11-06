# E2 — native-polarity graphs (signed, relation-typed): Track-A EDSP vs Track-A baselines

Run records: 96 (git a2bc65936cab; runners actions-run-42257945936). Headline metric per dataset in brackets; mean ± std over splits (×100). Rows marked † are constructed benchmarks or case studies and are excluded from real-world aggregates.

| dataset [metric] | splits | EDSP-default | EDSP-tuned | majority | train_prior | harmonic | harmonic_cmn | lgc | ppr_lp | linbp | signed_vote | signed_lgc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yelpchi [roc_auc] | 10 | 61.75 ± 1.13 | 62.60 ± 0.41 | 50.00 ± 0.00 | 50.00 ± 0.00 | 62.98 ± 0.41 | 62.98 ± 0.41 | **63.54 ± 0.33** | 63.50 ± 0.39 | 58.18 ± 0.39 | — | — |

## Pre-registered secondary column: EDSP-default on non-converged runs

| dataset | non-converged runs | headline, last iterate | headline, mean of last two iterates |
|---|---|---|---|
| yelpchi | 0 / 10 | — | — |

## EDSP-tuned vs each baseline: win / tie / loss

Per dataset: two-sided Wilcoxon signed-rank over splits on the headline metric; Holm correction across datasets per baseline; win/loss if adjusted p < 0.05, else tie. Real-world datasets only.

| baseline | win | tie | loss | datasets (W / L) |
|---|---|---|---|---|
| harmonic | 0 | 0 | 1 | W: —; L: yelpchi |
| harmonic_cmn | 0 | 0 | 1 | W: —; L: yelpchi |
| lgc | 0 | 0 | 1 | W: —; L: yelpchi |
| ppr_lp | 0 | 0 | 1 | W: —; L: yelpchi |
| linbp | 1 | 0 | 0 | W: yelpchi; L: — |
| train_prior | 1 | 0 | 0 | W: yelpchi; L: — |
| signed_vote | 0 | 0 | 0 | W: —; L: — |
| signed_lgc | 0 | 0 | 0 | W: —; L: — |

## Calibration (mean over splits; lower is better)

| dataset | EDSP-default NLL | EDSP-default ECE | EDSP-tuned NLL | EDSP-tuned ECE | best baseline by NLL (NLL / ECE) |
|---|---|---|---|---|---|
| yelpchi | 0.434 | 0.048 | 0.401 | 0.010 | harmonic_cmn: 0.405 / 0.012 |

## H7 (descriptive): convergence and fixed-point non-uniqueness

| dataset | EDSP-default non-converged | oscillation flagged | mean fraction of nodes still moving | GS vs damped L1: mean | p95 | splits |
|---|---|---|---|---|---|---|
| yelpchi | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3 |

## Test accuracy by local homophily (descriptive, full labels; bin i = [i/10, (i+1)/10)), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| yelpchi | EDSP-default | 2.3 (4768) | 3.8 (13645) | 11.8 (4955) | 15.2 (1866) | 37.7 (576) | 60.4 (737) | 86.9 (3705) | 91.3 (15543) | 97.5 (84971) | 98.6 (53003) |
| yelpchi | EDSP-tuned | 0.4 (4768) | 0.2 (13645) | 0.9 (4955) | 2.4 (1866) | 18.8 (576) | 66.8 (737) | 96.5 (3705) | 99.1 (15543) | 99.9 (84971) | 99.5 (53003) |
| yelpchi | harmonic_cmn | 0.0 (4768) | 0.0 (13645) | 0.2 (4955) | 1.9 (1866) | 17.2 (576) | 65.3 (737) | 97.2 (3705) | 99.8 (15543) | 100.0 (84971) | 99.9 (53003) |
| yelpchi | lgc | 0.1 (4768) | 0.1 (13645) | 0.4 (4955) | 5.7 (1866) | 33.7 (576) | 66.8 (737) | 94.7 (3705) | 99.5 (15543) | 99.9 (84971) | 99.9 (53003) |
| yelpchi | ppr_lp | 0.0 (4768) | 0.0 (13645) | 0.2 (4955) | 3.4 (1866) | 23.8 (576) | 69.2 (737) | 96.1 (3705) | 99.7 (15543) | 100.0 (84971) | 99.9 (53003) |
| yelpchi | linbp | 0.0 (4768) | 0.0 (13645) | 0.2 (4955) | 3.3 (1866) | 25.2 (576) | 68.9 (737) | 96.1 (3705) | 99.7 (15543) | 100.0 (84971) | 99.9 (53003) |

## Test accuracy by degree (bin 0: isolated; bin b: 2^(b-1) ≤ deg < 2^b; last bin open), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yelpchi | EDSP-default | 76.5 (51) | 76.2 (143) | 73.5 (400) | 71.6 (1220) | 77.8 (2847) | 78.3 (7185) | 78.2 (18437) | 81.6 (39424) | 86.3 (79924) | 87.5 (34189) | — |
| yelpchi | EDSP-tuned | 76.5 (51) | 64.3 (143) | 71.2 (400) | 73.8 (1220) | 78.4 (2847) | 80.5 (7185) | 83.2 (18437) | 85.4 (39424) | 86.4 (79924) | 87.5 (34189) | — |
| yelpchi | harmonic_cmn | 23.5 (51) | 79.0 (143) | 73.8 (400) | 77.0 (1220) | 83.9 (2847) | 82.2 (7185) | 83.2 (18437) | 85.4 (39424) | 86.4 (79924) | 87.5 (34189) | — |
| yelpchi | lgc | 76.5 (51) | 77.6 (143) | 77.0 (400) | 73.3 (1220) | 81.6 (2847) | 82.1 (7185) | 83.7 (18437) | 85.4 (39424) | 86.4 (79924) | 87.5 (34189) | — |
| yelpchi | ppr_lp | 76.5 (51) | 77.6 (143) | 78.8 (400) | 75.3 (1220) | 83.0 (2847) | 82.1 (7185) | 83.6 (18437) | 85.4 (39424) | 86.4 (79924) | 87.5 (34189) | — |
| yelpchi | linbp | 76.5 (51) | 77.6 (143) | 79.0 (400) | 75.6 (1220) | 83.1 (2847) | 82.1 (7185) | 83.6 (18437) | 85.4 (39424) | 86.4 (79924) | 87.5 (34189) | — |
