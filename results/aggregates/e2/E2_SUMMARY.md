# E2 — native-polarity graphs (signed, relation-typed): Track-A EDSP vs Track-A baselines

Run records: 52 (git e04a805e982a; runners actions-run-42230951809). Headline metric per dataset in brackets; mean ± std over splits (×100). Rows marked † are constructed benchmarks or case studies and are excluded from real-world aggregates.

| dataset [metric] | splits | EDSP-default | EDSP-tuned | majority | train_prior | harmonic | harmonic_cmn | lgc | ppr_lp | linbp | signed_vote | signed_lgc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yelpchi [roc_auc] | 6 | 61.13 ± 1.01 | 62.61 ± 0.32 | 50.00 ± 0.00 | 50.00 ± 0.00 | 62.96 ± 0.41 | 62.96 ± 0.41 | **63.48 ± 0.28** | 63.43 ± 0.35 | 58.06 ± 0.37 | — | — |

## Pre-registered secondary column: EDSP-default on non-converged runs

| dataset | non-converged runs | headline, last iterate | headline, mean of last two iterates |
|---|---|---|---|
| yelpchi | 0 / 6 | — | — |

## EDSP-tuned vs each baseline: win / tie / loss

Per dataset: two-sided Wilcoxon signed-rank over splits on the headline metric; Holm correction across datasets per baseline; win/loss if adjusted p < 0.05, else tie. Real-world datasets only.

| baseline | win | tie | loss | datasets (W / L) |
|---|---|---|---|---|
| harmonic | 0 | 1 | 0 | W: —; L: — |
| harmonic_cmn | 0 | 1 | 0 | W: —; L: — |
| lgc | 0 | 1 | 0 | W: —; L: — |
| ppr_lp | 0 | 1 | 0 | W: —; L: — |
| linbp | 0 | 1 | 0 | W: —; L: — |
| train_prior | 0 | 1 | 0 | W: —; L: — |
| signed_vote | 0 | 0 | 0 | W: —; L: — |
| signed_lgc | 0 | 0 | 0 | W: —; L: — |

## Calibration (mean over splits; lower is better)

| dataset | EDSP-default NLL | EDSP-default ECE | EDSP-tuned NLL | EDSP-tuned ECE | best baseline by NLL (NLL / ECE) |
|---|---|---|---|---|---|
| yelpchi | 0.457 | 0.073 | 0.401 | 0.009 | harmonic_cmn: 0.405 / 0.011 |

## H7 (descriptive): convergence and fixed-point non-uniqueness

| dataset | EDSP-default non-converged | oscillation flagged | mean fraction of nodes still moving | GS vs damped L1: mean | p95 | splits |
|---|---|---|---|---|---|---|
| yelpchi | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3 |

## Test accuracy by local homophily (descriptive, full labels; bin i = [i/10, (i+1)/10)), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| yelpchi | EDSP-default | 3.7 (2857) | 6.4 (8174) | 19.4 (3002) | 25.4 (1096) | 52.6 (344) | 55.4 (435) | 80.5 (2249) | 85.7 (9368) | 95.8 (50970) | 97.6 (31763) |
| yelpchi | EDSP-tuned | 0.6 (2389) | 0.2 (6809) | 1.2 (2497) | 3.3 (909) | 17.4 (281) | 66.6 (359) | 96.6 (1879) | 99.0 (7849) | 99.8 (42413) | 99.4 (26499) |
| yelpchi | harmonic_cmn | 0.0 (2389) | 0.0 (6809) | 0.4 (2497) | 3.2 (909) | 17.4 (281) | 63.0 (359) | 97.7 (1879) | 99.7 (7849) | 100.0 (42413) | 99.9 (26499) |
| yelpchi | lgc | 0.1 (2389) | 0.1 (6809) | 0.4 (2497) | 5.7 (909) | 37.4 (281) | 66.0 (359) | 95.4 (1879) | 99.5 (7849) | 99.9 (42413) | 99.9 (26499) |
| yelpchi | ppr_lp | 0.0 (2389) | 0.0 (6809) | 0.3 (2497) | 3.9 (909) | 25.3 (281) | 69.6 (359) | 96.7 (1879) | 99.6 (7849) | 100.0 (42413) | 99.9 (26499) |
| yelpchi | linbp | 0.0 (2389) | 0.0 (6809) | 0.4 (2497) | 3.7 (909) | 27.4 (281) | 69.1 (359) | 96.8 (1879) | 99.6 (7849) | 100.0 (42413) | 99.9 (26499) |

## Test accuracy by degree (bin 0: isolated; bin b: 2^(b-1) ≤ deg < 2^b; last bin open), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yelpchi | EDSP-default | 76.5 (34) | 72.0 (82) | 68.6 (242) | 66.9 (749) | 73.3 (1720) | 75.6 (4262) | 75.1 (11113) | 79.2 (23676) | 86.2 (47859) | 87.6 (20555) | — |
| yelpchi | EDSP-tuned | 80.8 (26) | 62.0 (71) | 67.7 (201) | 73.4 (638) | 76.5 (1437) | 80.2 (3547) | 83.4 (9215) | 85.6 (19786) | 86.4 (39840) | 87.6 (17149) | — |
| yelpchi | harmonic_cmn | 19.2 (26) | 78.9 (71) | 72.6 (201) | 75.5 (638) | 83.5 (1437) | 82.3 (3547) | 83.5 (9215) | 85.6 (19786) | 86.4 (39840) | 87.6 (17149) | — |
| yelpchi | lgc | 80.8 (26) | 77.5 (71) | 76.1 (201) | 71.5 (638) | 80.6 (1437) | 82.3 (3547) | 84.0 (9215) | 85.7 (19786) | 86.4 (39840) | 87.6 (17149) | — |
| yelpchi | ppr_lp | 80.8 (26) | 78.9 (71) | 78.1 (201) | 73.7 (638) | 82.2 (1437) | 82.3 (3547) | 83.9 (9215) | 85.6 (19786) | 86.4 (39840) | 87.6 (17149) | — |
| yelpchi | linbp | 80.8 (26) | 78.9 (71) | 78.6 (201) | 74.0 (638) | 82.3 (1437) | 82.3 (3547) | 83.9 (9215) | 85.6 (19786) | 86.4 (39840) | 87.6 (17149) | — |
