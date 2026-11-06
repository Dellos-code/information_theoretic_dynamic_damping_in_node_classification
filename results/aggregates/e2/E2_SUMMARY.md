# E2 — native-polarity graphs (signed, relation-typed): Track-A EDSP vs Track-A baselines

Run records: 441 (git 933c69c344dc; runners actions-run-42210967089). Headline metric per dataset in brackets; mean ± std over splits (×100). Rows marked † are constructed benchmarks or case studies and are excluded from real-world aggregates.

| dataset [metric] | splits | EDSP-default | EDSP-tuned | majority | train_prior | harmonic | harmonic_cmn | lgc | ppr_lp | linbp | signed_vote | signed_lgc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| amazon-fraud [roc_auc] | 1 | **79.33 ± 0.00** | — | — | — | — | — | — | — | — | — | — |
| sampson† [accuracy] | 18 | **66.67 ± 47.14** | **66.67 ± 47.14** | 38.89 ± 48.75 | 38.89 ± 48.75 | 44.44 ± 49.69 | 50.00 ± 50.00 | 44.44 ± 49.69 | 44.44 ± 49.69 | 38.89 ± 48.75 | **66.67 ± 47.14** | 61.11 ± 48.75 |
| wiki-editor [accuracy] | 1 | **86.83 ± 0.00** | — | — | — | — | — | — | — | — | — | — |
| wiki-elec† [accuracy] | 10 | 80.54 ± 1.45 | 80.58 ± 1.42 | 52.69 ± 0.00 | 52.69 ± 0.00 | 55.85 ± 0.92 | 60.58 ± 2.55 | 59.20 ± 0.76 | 56.75 ± 1.08 | 56.75 ± 1.05 | 78.09 ± 1.43 | **81.66 ± 0.78** |
| wiki-rfa† [accuracy] | 10 | 72.22 ± 1.22 | 72.23 ± 1.18 | 51.87 ± 0.00 | 51.87 ± 0.00 | 60.81 ± 2.15 | 64.12 ± 2.23 | 63.99 ± 1.29 | 61.27 ± 1.47 | 61.22 ± 1.59 | 71.73 ± 1.57 | **76.59 ± 0.81** |

## Pre-registered secondary column: EDSP-default on non-converged runs

| dataset | non-converged runs | headline, last iterate | headline, mean of last two iterates |
|---|---|---|---|
| amazon-fraud | 0 / 1 | — | — |
| sampson | 0 / 18 | — | — |
| wiki-editor | 0 / 1 | — | — |
| wiki-elec | 0 / 10 | — | — |
| wiki-rfa | 0 / 10 | — | — |

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
| amazon-fraud | 0.290 | 0.062 | nan | nan | harmonic: nan / nan |
| sampson | 0.962 | 0.469 | 0.980 | 0.467 | signed_lgc: 0.909 / 0.446 |
| wiki-editor | 0.395 | 0.081 | nan | nan | harmonic: nan / nan |
| wiki-elec | 0.476 | 0.127 | 0.475 | 0.127 | linbp: 0.687 / 0.062 |
| wiki-rfa | 0.589 | 0.098 | 0.587 | 0.098 | lgc: 0.652 / 0.050 |

## H7 (descriptive): convergence and fixed-point non-uniqueness

| dataset | EDSP-default non-converged | oscillation flagged | mean fraction of nodes still moving | GS vs damped L1: mean | p95 | splits |
|---|---|---|---|---|---|---|
| amazon-fraud | 0.000 | 0.000 | 0.000 | — | — | 0 |
| sampson | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3 |
| wiki-editor | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1 |
| wiki-elec | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3 |
| wiki-rfa | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3 |

## Test accuracy by local-homophily bin (pooled over splits, ×100)

Bins of each node's local homophily over all its labelled neighbours: [0, .2), [.2, .4), [.4, .6), [.6, .8), [.8, 1]. Descriptive (full labels define the bins, never the predictions). Cell: accuracy (test nodes).

| dataset | method | bin 0 | bin 1 | bin 2 | bin 3 | bin 4 |
|---|---|---|---|---|---|---|
| sampson | EDSP-default | 0.0 (6) | 100.0 (3) | 100.0 (5) | 100.0 (1) | 100.0 (3) |
| sampson | EDSP-tuned | 0.0 (6) | 100.0 (3) | 100.0 (5) | 100.0 (1) | 100.0 (3) |
| sampson | harmonic_cmn | 0.0 (6) | 33.3 (3) | 80.0 (5) | 100.0 (1) | 100.0 (3) |
| sampson | lgc | 0.0 (6) | 0.0 (3) | 80.0 (5) | 100.0 (1) | 100.0 (3) |
| sampson | ppr_lp | 0.0 (6) | 0.0 (3) | 80.0 (5) | 100.0 (1) | 100.0 (3) |
| sampson | linbp | 0.0 (6) | 0.0 (3) | 80.0 (5) | 100.0 (1) | 66.7 (3) |
| sampson | signed_vote | 0.0 (6) | 100.0 (3) | 100.0 (5) | 100.0 (1) | 100.0 (3) |
| wiki-elec | EDSP-default | 70.2 (597) | 57.2 (996) | 66.8 (618) | 96.3 (1467) | 95.8 (972) |
| wiki-elec | EDSP-tuned | 70.2 (597) | 57.4 (996) | 66.8 (618) | 96.3 (1467) | 95.8 (972) |
| wiki-elec | harmonic_cmn | 6.2 (597) | 51.2 (996) | 65.0 (618) | 61.7 (1467) | 99.1 (972) |
| wiki-elec | lgc | 2.8 (597) | 11.0 (996) | 48.5 (618) | 92.8 (1467) | 99.3 (972) |
| wiki-elec | ppr_lp | 0.0 (597) | 1.5 (996) | 40.9 (618) | 96.2 (1467) | 98.8 (972) |
| wiki-elec | linbp | 0.0 (597) | 1.5 (996) | 40.9 (618) | 96.2 (1467) | 98.8 (972) |
| wiki-elec | signed_vote | 63.5 (597) | 58.7 (996) | 61.0 (618) | 93.7 (1467) | 94.1 (972) |
| wiki-rfa | EDSP-default | 79.7 (281) | 53.0 (1105) | 63.6 (2552) | 87.6 (2332) | 80.0 (670) |
| wiki-rfa | EDSP-tuned | 79.7 (281) | 53.0 (1105) | 63.6 (2552) | 87.6 (2332) | 80.0 (670) |
| wiki-rfa | harmonic_cmn | 11.0 (281) | 35.1 (1105) | 59.3 (2552) | 79.2 (2332) | 99.9 (670) |
| wiki-rfa | lgc | 8.5 (281) | 9.9 (1105) | 55.7 (2552) | 95.5 (2332) | 98.2 (670) |
| wiki-rfa | ppr_lp | 2.8 (281) | 4.1 (1105) | 51.8 (2552) | 95.5 (2332) | 97.3 (670) |
| wiki-rfa | linbp | 3.2 (281) | 4.3 (1105) | 51.5 (2552) | 95.5 (2332) | 97.2 (670) |
| wiki-rfa | signed_vote | 83.3 (281) | 55.3 (1105) | 62.1 (2552) | 86.4 (2332) | 79.6 (670) |
