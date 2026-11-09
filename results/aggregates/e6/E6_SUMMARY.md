# E6

Run records: 14 (git d8fbacfe6c6c; runners actions-run-42603215359). Headline metric per dataset in brackets; mean ± std over splits (×100). Rows marked † are constructed benchmarks or case studies and are excluded from real-world aggregates.

| dataset [metric] | splits | EDSP-default | EDSP-tuned | EDSP-fullH-default | EDSP-fullH-tuned | majority | train_prior | harmonic | harmonic_cmn | lgc | ppr_lp | linbp | signed_vote | signed_lgc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cora~ln-uniform-0.3 [accuracy] | 1 | 67.10 ± 0.00 | 67.40 ± 0.00 | — | — | 31.90 ± 0.00 | 31.90 ± 0.00 | 67.10 ± 0.00 | 67.30 ± 0.00 | **70.20 ± 0.00** | 61.90 ± 0.00 | 39.90 ± 0.00 | — | — |

## Pre-registered secondary column: EDSP-default on non-converged runs

| dataset | non-converged runs | headline, last iterate | headline, mean of last two iterates |
|---|---|---|---|
| cora~ln-uniform-0.3 | 1 / 1 | 67.10 | 67.10 |

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
| cora~ln-uniform-0.3 | 1.240 | 0.283 | 1.229 | 0.271 | ppr_lp: 1.143 / 0.141 |

## H7 (descriptive): convergence and fixed-point non-uniqueness

| dataset | EDSP-default non-converged | oscillation flagged | mean fraction of nodes still moving | GS vs damped L1: mean | p95 | splits |
|---|---|---|---|---|---|---|
| cora~ln-uniform-0.3 | 1.000 | 0.000 | 0.012 | — | — | 0 |

## Test accuracy by local homophily (descriptive, full labels; bin i = [i/10, (i+1)/10)), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cora~ln-uniform-0.3 | EDSP-default | 18.3 (60) | 20.0 (5) | 35.0 (20) | 33.3 (27) | 33.3 (6) | 45.5 (77) | 63.9 (72) | 66.7 (39) | 64.6 (48) | 77.9 (646) |
| cora~ln-uniform-0.3 | EDSP-tuned | 18.3 (60) | 20.0 (5) | 35.0 (20) | 33.3 (27) | 33.3 (6) | 48.1 (77) | 63.9 (72) | 66.7 (39) | 64.6 (48) | 78.0 (646) |
| cora~ln-uniform-0.3 | harmonic_cmn | 23.3 (60) | 20.0 (5) | 40.0 (20) | 48.1 (27) | 16.7 (6) | 45.5 (77) | 62.5 (72) | 76.9 (39) | 68.8 (48) | 76.3 (646) |
| cora~ln-uniform-0.3 | lgc | 18.3 (60) | 20.0 (5) | 50.0 (20) | 37.0 (27) | 33.3 (6) | 48.1 (77) | 68.1 (72) | 74.4 (39) | 75.0 (48) | 80.0 (646) |
| cora~ln-uniform-0.3 | ppr_lp | 13.3 (60) | 20.0 (5) | 45.0 (20) | 33.3 (27) | 33.3 (6) | 44.2 (77) | 58.3 (72) | 59.0 (39) | 54.2 (48) | 72.0 (646) |
| cora~ln-uniform-0.3 | linbp | 20.0 (60) | 0.0 (5) | 30.0 (20) | 29.6 (27) | 50.0 (6) | 23.4 (77) | 47.2 (72) | 30.8 (39) | 37.5 (48) | 44.6 (646) |

## Test accuracy by degree (bin 0: isolated; bin b: 2^(b-1) ≤ deg < 2^b; last bin open), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cora~ln-uniform-0.3 | EDSP-default | — | 54.5 (178) | 67.9 (436) | 71.0 (317) | 81.0 (58) | 42.9 (7) | 66.7 (3) | 100.0 (1) | — | — | — |
| cora~ln-uniform-0.3 | EDSP-tuned | — | 55.6 (178) | 68.1 (436) | 71.0 (317) | 81.0 (58) | 42.9 (7) | 66.7 (3) | 100.0 (1) | — | — | — |
| cora~ln-uniform-0.3 | harmonic_cmn | — | 56.2 (178) | 66.7 (436) | 72.9 (317) | 77.6 (58) | 42.9 (7) | 66.7 (3) | 100.0 (1) | — | — | — |
| cora~ln-uniform-0.3 | lgc | — | 56.7 (178) | 69.0 (436) | 76.3 (317) | 87.9 (58) | 57.1 (7) | 66.7 (3) | 100.0 (1) | — | — | — |
| cora~ln-uniform-0.3 | ppr_lp | — | 50.6 (178) | 63.1 (436) | 65.0 (317) | 72.4 (58) | 42.9 (7) | 66.7 (3) | 100.0 (1) | — | — | — |
| cora~ln-uniform-0.3 | linbp | — | 30.3 (178) | 40.4 (436) | 43.2 (317) | 48.3 (58) | 57.1 (7) | 0.0 (3) | 0.0 (1) | — | — | — |

## Cell C3c (mixed by local structure: top-5 std of local homophily)

| dataset | std local homophily | min compat. distance | EDSP-default | EDSP-tuned | train_prior | harmonic | harmonic_cmn | lgc | ppr_lp | linbp |
|---|---|---|---|---|---|---|---|---|---|---|
| cora~ln-uniform-0.3 | 0.293 | 0.623 | 67.10 | 67.40 | 31.90 | 67.10 | 67.30 | **70.20** | 61.90 | 39.90 |

## Cell C3d (structured compatibility: top-5 min Frobenius distance to I and to (J-I)/(k-1))

| dataset | std local homophily | min compat. distance | EDSP-default | EDSP-tuned | train_prior | harmonic | harmonic_cmn | lgc | ppr_lp | linbp |
|---|---|---|---|---|---|---|---|---|---|---|
| cora~ln-uniform-0.3 | 0.293 | 0.623 | 67.10 | 67.40 | 31.90 | 67.10 | 67.30 | **70.20** | 61.90 | 39.90 |
