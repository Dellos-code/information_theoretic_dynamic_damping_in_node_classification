# E2U

Run records: 1170 (git 1afaf38a8669; runners actions-run-42377245757). Headline metric per dataset in brackets; mean ± std over splits (×100). Rows marked † are constructed benchmarks or case studies and are excluded from real-world aggregates.

| dataset [metric] | splits | EDSP-default | EDSP-tuned | EDSP-fullH-default | EDSP-fullH-tuned | majority | train_prior | harmonic | harmonic_cmn | lgc | ppr_lp | linbp | signed_vote | signed_lgc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| amazon-fraud~unsigned [roc_auc] | 10 | 81.97 ± 1.04 | **82.79 ± 1.16** | — | — | 50.00 ± 0.00 | 50.00 ± 0.00 | 82.13 ± 1.06 | 82.13 ± 1.06 | 82.68 ± 1.02 | 81.69 ± 1.10 | 81.01 ± 1.04 | — | — |
| congress-house-100~unsigned [accuracy] | 10 | 47.08 ± 3.74 | **87.19 ± 2.98** | — | — | 57.30 ± 0.00 | 57.30 ± 0.00 | 47.64 ± 3.38 | 46.63 ± 2.76 | 56.74 ± 0.75 | 56.97 ± 0.72 | 73.71 ± 2.98 | — | — |
| congress-house-107~unsigned [accuracy] | 10 | 19.66 ± 3.77 | **95.84 ± 1.33** | — | — | 51.69 ± 0.00 | 51.69 ± 0.00 | 20.45 ± 3.95 | 13.03 ± 2.94 | 47.98 ± 1.01 | 48.09 ± 0.45 | 91.01 ± 2.97 | — | — |
| congress-house-114~unsigned [accuracy] | 10 | 14.27 ± 3.86 | **96.18 ± 1.83** | — | — | 56.18 ± 0.00 | 56.18 ± 0.00 | 14.49 ± 3.91 | 37.87 ± 3.26 | 56.18 ± 0.00 | 56.18 ± 0.00 | 91.80 ± 1.82 | — | — |
| congress-house-93~unsigned [accuracy] | 10 | 60.00 ± 2.26 | **80.22 ± 3.49** | — | — | 56.18 ± 0.00 | 56.18 ± 0.00 | 59.55 ± 2.46 | 65.62 ± 2.94 | 67.08 ± 2.01 | 61.80 ± 2.25 | 55.73 ± 9.54 | — | — |
| congress-senate-100~unsigned [accuracy] | 10 | 49.00 ± 9.43 | **76.50 ± 12.85** | — | — | 55.00 ± 0.00 | 55.00 ± 0.00 | 48.50 ± 8.67 | 50.00 ± 9.22 | 53.50 ± 7.09 | 51.00 ± 8.31 | 62.50 ± 5.59 | — | — |
| congress-senate-107~unsigned [accuracy] | 10 | 25.79 ± 6.84 | **93.16 ± 4.74** | — | — | 52.63 ± 0.00 | 52.63 ± 0.00 | 27.37 ± 7.74 | 25.26 ± 7.74 | 52.11 ± 1.58 | 51.05 ± 2.41 | 79.47 ± 9.83 | — | — |
| congress-senate-114~unsigned [accuracy] | 10 | 29.50 ± 7.57 | **98.00 ± 2.45** | — | — | 55.00 ± 0.00 | 55.00 ± 0.00 | 30.50 ± 7.89 | 36.00 ± 7.00 | 54.00 ± 2.00 | 53.00 ± 4.58 | 83.00 ± 6.78 | — | — |
| congress-senate-93~unsigned [accuracy] | 10 | 53.33 ± 7.32 | **70.48 ± 12.38** | — | — | 57.14 ± 0.00 | 57.14 ± 0.00 | 54.29 ± 6.80 | 54.29 ± 7.74 | 52.86 ± 6.55 | 52.86 ± 5.41 | 62.86 ± 6.67 | — | — |
| wiki-editor~unsigned [accuracy] | 10 | 73.91 ± 0.49 | 74.01 ± 0.39 | — | — | 63.09 ± 0.00 | 63.09 ± 0.00 | 73.63 ± 0.64 | 76.08 ± 0.47 | **76.54 ± 0.48** | 73.41 ± 0.48 | 72.78 ± 0.60 | — | — |
| wiki-elec~unsigned [accuracy] | 10 | 56.19 ± 0.93 | **72.77 ± 2.10** | — | — | 52.69 ± 0.00 | 52.69 ± 0.00 | 55.85 ± 0.92 | 60.58 ± 2.55 | 59.20 ± 0.76 | 56.75 ± 1.08 | 56.75 ± 1.05 | — | — |
| wiki-rfa~unsigned [accuracy] | 10 | 61.22 ± 2.03 | 63.95 ± 1.74 | — | — | 51.87 ± 0.00 | 51.87 ± 0.00 | 60.81 ± 2.15 | **64.12 ± 2.23** | 63.99 ± 1.29 | 61.27 ± 1.47 | 61.22 ± 1.59 | — | — |
| yelpchi~unsigned [roc_auc] | 10 | 62.90 ± 0.43 | 62.86 ± 0.48 | — | — | 50.00 ± 0.00 | 50.00 ± 0.00 | 62.98 ± 0.41 | 62.98 ± 0.41 | **63.54 ± 0.33** | 63.50 ± 0.39 | 58.18 ± 0.39 | — | — |

## Pre-registered secondary column: EDSP-default on non-converged runs

| dataset | non-converged runs | headline, last iterate | headline, mean of last two iterates |
|---|---|---|---|
| amazon-fraud~unsigned | 0 / 10 | — | — |
| congress-house-100~unsigned | 0 / 10 | — | — |
| congress-house-107~unsigned | 0 / 10 | — | — |
| congress-house-114~unsigned | 0 / 10 | — | — |
| congress-house-93~unsigned | 0 / 10 | — | — |
| congress-senate-100~unsigned | 0 / 10 | — | — |
| congress-senate-107~unsigned | 0 / 10 | — | — |
| congress-senate-114~unsigned | 0 / 10 | — | — |
| congress-senate-93~unsigned | 0 / 10 | — | — |
| wiki-editor~unsigned | 0 / 10 | — | — |
| wiki-elec~unsigned | 0 / 10 | — | — |
| wiki-rfa~unsigned | 0 / 10 | — | — |
| yelpchi~unsigned | 0 / 10 | — | — |

## EDSP-tuned vs each baseline: win / tie / loss

Per dataset: two-sided Wilcoxon signed-rank over splits on the headline metric; Holm correction across datasets per baseline; win/loss if adjusted p < 0.05, else tie. Real-world datasets only.

| baseline | win | tie | loss | datasets (W / L) |
|---|---|---|---|---|
| harmonic | 8 | 4 | 1 | W: congress-house-100~unsigned, congress-house-107~unsigned, congress-house-114~unsigned, congress-house-93~unsigned, congress-senate-100~unsigned, congress-senate-107~unsigned, congress-senate-114~unsigned, wiki-elec~unsigned; L: yelpchi~unsigned |
| harmonic_cmn | 8 | 3 | 2 | W: congress-house-100~unsigned, congress-house-107~unsigned, congress-house-114~unsigned, congress-house-93~unsigned, congress-senate-100~unsigned, congress-senate-107~unsigned, congress-senate-114~unsigned, wiki-elec~unsigned; L: wiki-editor~unsigned, yelpchi~unsigned |
| lgc | 8 | 3 | 2 | W: congress-house-100~unsigned, congress-house-107~unsigned, congress-house-114~unsigned, congress-house-93~unsigned, congress-senate-100~unsigned, congress-senate-107~unsigned, congress-senate-114~unsigned, wiki-elec~unsigned; L: wiki-editor~unsigned, yelpchi~unsigned |
| ppr_lp | 12 | 0 | 1 | W: amazon-fraud~unsigned, congress-house-100~unsigned, congress-house-107~unsigned, congress-house-114~unsigned, congress-house-93~unsigned, congress-senate-100~unsigned, congress-senate-107~unsigned, congress-senate-114~unsigned, congress-senate-93~unsigned, wiki-editor~unsigned, wiki-elec~unsigned, wiki-rfa~unsigned; L: yelpchi~unsigned |
| linbp | 11 | 2 | 0 | W: amazon-fraud~unsigned, congress-house-100~unsigned, congress-house-107~unsigned, congress-house-114~unsigned, congress-house-93~unsigned, congress-senate-100~unsigned, congress-senate-107~unsigned, congress-senate-114~unsigned, wiki-editor~unsigned, wiki-elec~unsigned, yelpchi~unsigned; L: — |
| train_prior | 13 | 0 | 0 | W: amazon-fraud~unsigned, congress-house-100~unsigned, congress-house-107~unsigned, congress-house-114~unsigned, congress-house-93~unsigned, congress-senate-100~unsigned, congress-senate-107~unsigned, congress-senate-114~unsigned, congress-senate-93~unsigned, wiki-editor~unsigned, wiki-elec~unsigned, wiki-rfa~unsigned, yelpchi~unsigned; L: — |
| signed_vote | 0 | 0 | 0 | W: —; L: — |
| signed_lgc | 0 | 0 | 0 | W: —; L: — |

## Friedman / Nemenyi (real-world datasets, headline metric)

Datasets: 13; Friedman p = 3.14e-05; Nemenyi CD (α = 0.05) = 2.91.

| method | average rank |
|---|---|
| EDSP-tuned | 1.69 |
| lgc | 3.38 |
| linbp | 4.08 |
| harmonic_cmn | 4.69 |
| ppr_lp | 4.73 |
| train_prior | 5.31 |
| harmonic | 5.92 |
| EDSP-default | 6.19 |

## Calibration (mean over splits; lower is better)

| dataset | EDSP-default NLL | EDSP-default ECE | EDSP-tuned NLL | EDSP-tuned ECE | best baseline by NLL (NLL / ECE) |
|---|---|---|---|---|---|
| amazon-fraud~unsigned | 0.295 | 0.045 | 0.293 | 0.047 | harmonic_cmn: 0.268 / 0.045 |
| congress-house-100~unsigned | 0.767 | 0.202 | 0.395 | 0.150 | linbp: 0.605 / 0.181 |
| congress-house-107~unsigned | 0.826 | 0.371 | 0.153 | 0.084 | linbp: 0.589 / 0.357 |
| congress-house-114~unsigned | 0.787 | 0.408 | 0.136 | 0.074 | linbp: 0.618 / 0.375 |
| congress-house-93~unsigned | 0.664 | 0.097 | 0.527 | 0.184 | lgc: 0.634 / 0.104 |
| congress-senate-100~unsigned | 0.714 | 0.191 | 0.566 | 0.221 | linbp: 0.662 / 0.138 |
| congress-senate-107~unsigned | 0.800 | 0.337 | 0.291 | 0.182 | linbp: 0.624 / 0.261 |
| congress-senate-114~unsigned | 0.797 | 0.300 | 0.123 | 0.108 | linbp: 0.650 / 0.305 |
| congress-senate-93~unsigned | 0.721 | 0.166 | 0.607 | 0.204 | linbp: 0.668 / 0.122 |
| wiki-editor~unsigned | 0.602 | 0.071 | 0.615 | 0.071 | lgc: 0.522 / 0.084 |
| wiki-elec~unsigned | 0.784 | 0.162 | 0.563 | 0.046 | linbp: 0.687 / 0.062 |
| wiki-rfa~unsigned | 0.674 | 0.048 | 0.671 | 0.051 | lgc: 0.652 / 0.050 |
| yelpchi~unsigned | 0.402 | 0.028 | 0.402 | 0.026 | harmonic_cmn: 0.405 / 0.012 |

## H7 (descriptive): convergence and fixed-point non-uniqueness

| dataset | EDSP-default non-converged | oscillation flagged | mean fraction of nodes still moving | GS vs damped L1: mean | p95 | splits |
|---|---|---|---|---|---|---|
| amazon-fraud~unsigned | 0.000 | 0.000 | 0.000 | — | — | 0 |
| congress-house-100~unsigned | 0.000 | 0.000 | 0.000 | — | — | 0 |
| congress-house-107~unsigned | 0.000 | 0.000 | 0.000 | — | — | 0 |
| congress-house-114~unsigned | 0.000 | 0.000 | 0.000 | — | — | 0 |
| congress-house-93~unsigned | 0.000 | 0.000 | 0.000 | — | — | 0 |
| congress-senate-100~unsigned | 0.000 | 0.000 | 0.000 | — | — | 0 |
| congress-senate-107~unsigned | 0.000 | 0.000 | 0.000 | — | — | 0 |
| congress-senate-114~unsigned | 0.000 | 0.000 | 0.000 | — | — | 0 |
| congress-senate-93~unsigned | 0.000 | 0.000 | 0.000 | — | — | 0 |
| wiki-editor~unsigned | 0.000 | 0.000 | 0.000 | — | — | 0 |
| wiki-elec~unsigned | 0.000 | 0.000 | 0.000 | — | — | 0 |
| wiki-rfa~unsigned | 0.000 | 0.000 | 0.000 | — | — | 0 |
| yelpchi~unsigned | 0.000 | 0.000 | 0.000 | — | — | 0 |

## Test accuracy by local homophily (descriptive, full labels; bin i = [i/10, (i+1)/10)), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| amazon-fraud~unsigned | EDSP-default | 0.0 (1632) | 0.0 (959) | 0.0 (388) | 1.6 (182) | 0.0 (85) | 33.3 (51) | 86.3 (73) | 97.2 (109) | 100.0 (1998) | 100.0 (29083) |
| amazon-fraud~unsigned | EDSP-tuned | 1.2 (1632) | 1.3 (959) | 3.9 (388) | 8.8 (182) | 3.5 (85) | 41.2 (51) | 78.1 (73) | 93.6 (109) | 98.5 (1998) | 99.9 (29083) |
| amazon-fraud~unsigned | harmonic_cmn | 0.0 (1632) | 0.0 (959) | 0.0 (388) | 2.2 (182) | 0.0 (85) | 33.3 (51) | 86.3 (73) | 97.2 (109) | 100.0 (1998) | 100.0 (29083) |
| amazon-fraud~unsigned | lgc | 0.0 (1632) | 0.0 (959) | 0.0 (388) | 1.6 (182) | 3.5 (85) | 35.3 (51) | 89.0 (73) | 97.2 (109) | 100.0 (1998) | 100.0 (29083) |
| amazon-fraud~unsigned | ppr_lp | 0.0 (1632) | 0.0 (959) | 0.0 (388) | 1.6 (182) | 0.0 (85) | 33.3 (51) | 86.3 (73) | 97.2 (109) | 100.0 (1998) | 100.0 (29083) |
| amazon-fraud~unsigned | linbp | 0.0 (1632) | 0.0 (959) | 1.0 (388) | 6.6 (182) | 11.8 (85) | 52.9 (51) | 87.7 (73) | 98.2 (109) | 100.0 (1998) | 100.0 (29083) |
| congress-house-100~unsigned | EDSP-default | 0.0 (2) | 0.0 (47) | 0.0 (143) | 3.2 (156) | 55.9 (247) | 89.2 (176) | 100.0 (102) | 100.0 (17) | — | — |
| congress-house-100~unsigned | EDSP-tuned | 100.0 (2) | 100.0 (47) | 100.0 (143) | 92.3 (156) | 85.8 (247) | 86.4 (176) | 72.5 (102) | 11.8 (17) | — | — |
| congress-house-100~unsigned | harmonic_cmn | 0.0 (2) | 0.0 (47) | 0.0 (143) | 1.9 (156) | 55.5 (247) | 88.6 (176) | 100.0 (102) | 100.0 (17) | — | — |
| congress-house-100~unsigned | lgc | 0.0 (2) | 4.3 (47) | 8.4 (143) | 34.0 (156) | 66.0 (247) | 88.6 (176) | 100.0 (102) | 100.0 (17) | — | — |
| congress-house-100~unsigned | ppr_lp | 0.0 (2) | 2.1 (47) | 10.5 (143) | 34.0 (156) | 66.0 (247) | 88.6 (176) | 100.0 (102) | 100.0 (17) | — | — |
| congress-house-100~unsigned | linbp | 100.0 (2) | 100.0 (47) | 100.0 (143) | 100.0 (156) | 90.3 (247) | 48.3 (176) | 0.0 (102) | 0.0 (17) | — | — |
| congress-house-107~unsigned | EDSP-default | 0.0 (2) | 0.0 (66) | 0.0 (147) | 0.0 (266) | 27.1 (310) | 90.6 (85) | 100.0 (13) | 100.0 (1) | — | — |
| congress-house-107~unsigned | EDSP-tuned | 100.0 (2) | 100.0 (66) | 100.0 (147) | 100.0 (266) | 99.4 (310) | 72.9 (85) | 15.4 (13) | 0.0 (1) | — | — |
| congress-house-107~unsigned | harmonic_cmn | 0.0 (2) | 0.0 (66) | 0.0 (147) | 0.4 (266) | 19.0 (310) | 49.4 (85) | 100.0 (13) | 100.0 (1) | — | — |
| congress-house-107~unsigned | lgc | 100.0 (2) | 18.2 (66) | 15.0 (147) | 35.0 (266) | 74.2 (310) | 74.1 (85) | 38.5 (13) | 0.0 (1) | — | — |
| congress-house-107~unsigned | ppr_lp | 50.0 (2) | 18.2 (66) | 16.3 (147) | 35.0 (266) | 74.2 (310) | 74.1 (85) | 38.5 (13) | 0.0 (1) | — | — |
| congress-house-107~unsigned | linbp | 100.0 (2) | 100.0 (66) | 100.0 (147) | 100.0 (266) | 99.0 (310) | 25.9 (85) | 0.0 (13) | 0.0 (1) | — | — |
| congress-house-114~unsigned | EDSP-default | 0.0 (3) | 0.0 (19) | 0.0 (74) | 0.0 (215) | 9.6 (498) | 96.9 (64) | 100.0 (17) | — | — | — |
| congress-house-114~unsigned | EDSP-tuned | 100.0 (3) | 100.0 (19) | 100.0 (74) | 100.0 (215) | 99.2 (498) | 78.1 (64) | 5.9 (17) | — | — | — |
| congress-house-114~unsigned | harmonic_cmn | 0.0 (3) | 0.0 (19) | 0.0 (74) | 10.2 (215) | 47.8 (498) | 93.8 (64) | 100.0 (17) | — | — | — |
| congress-house-114~unsigned | lgc | 0.0 (3) | 42.1 (19) | 75.7 (74) | 56.3 (215) | 47.8 (498) | 93.8 (64) | 100.0 (17) | — | — | — |
| congress-house-114~unsigned | ppr_lp | 0.0 (3) | 42.1 (19) | 75.7 (74) | 56.3 (215) | 47.8 (498) | 93.8 (64) | 100.0 (17) | — | — | — |
| congress-house-114~unsigned | linbp | 100.0 (3) | 100.0 (19) | 100.0 (74) | 100.0 (215) | 99.6 (498) | 15.6 (64) | 0.0 (17) | — | — | — |
| congress-house-93~unsigned | EDSP-default | 0.0 (29) | 0.0 (51) | 0.0 (73) | 0.0 (77) | 20.0 (115) | 72.1 (122) | 100.0 (82) | 100.0 (140) | 100.0 (117) | 100.0 (84) |
| congress-house-93~unsigned | EDSP-tuned | 96.6 (29) | 92.2 (51) | 84.9 (73) | 62.3 (77) | 76.5 (115) | 94.3 (122) | 89.0 (82) | 80.7 (140) | 69.2 (117) | 70.2 (84) |
| congress-house-93~unsigned | harmonic_cmn | 0.0 (29) | 0.0 (51) | 0.0 (73) | 22.1 (77) | 73.9 (115) | 56.6 (122) | 87.8 (82) | 100.0 (140) | 100.0 (117) | 100.0 (84) |
| congress-house-93~unsigned | lgc | 0.0 (29) | 0.0 (51) | 0.0 (73) | 2.6 (77) | 49.6 (115) | 94.3 (122) | 100.0 (82) | 100.0 (140) | 100.0 (117) | 100.0 (84) |
| congress-house-93~unsigned | ppr_lp | 0.0 (29) | 0.0 (51) | 0.0 (73) | 0.0 (77) | 24.3 (115) | 81.1 (122) | 100.0 (82) | 100.0 (140) | 100.0 (117) | 100.0 (84) |
| congress-house-93~unsigned | linbp | 31.0 (29) | 23.5 (51) | 24.7 (73) | 31.2 (77) | 43.5 (115) | 76.2 (122) | 68.3 (82) | 69.3 (140) | 67.5 (117) | 69.0 (84) |
| congress-senate-100~unsigned | EDSP-default | — | — | 0.0 (17) | 2.4 (42) | 38.2 (55) | 82.8 (58) | 100.0 (25) | — | 100.0 (3) | — |
| congress-senate-100~unsigned | EDSP-tuned | — | — | 100.0 (17) | 92.9 (42) | 90.9 (55) | 58.6 (58) | 48.0 (25) | — | 33.3 (3) | — |
| congress-senate-100~unsigned | harmonic_cmn | — | — | 0.0 (17) | 2.4 (42) | 43.6 (55) | 81.0 (58) | 100.0 (25) | — | 100.0 (3) | — |
| congress-senate-100~unsigned | lgc | — | — | 0.0 (17) | 2.4 (42) | 69.1 (55) | 75.9 (58) | 84.0 (25) | — | 100.0 (3) | — |
| congress-senate-100~unsigned | ppr_lp | — | — | 0.0 (17) | 0.0 (42) | 54.5 (55) | 84.5 (58) | 80.0 (25) | — | 100.0 (3) | — |
| congress-senate-100~unsigned | linbp | — | — | 100.0 (17) | 100.0 (42) | 90.9 (55) | 25.9 (58) | 4.0 (25) | — | 0.0 (3) | — |
| congress-senate-107~unsigned | EDSP-default | 0.0 (3) | 0.0 (26) | 0.0 (28) | 0.0 (61) | 32.3 (31) | 94.1 (17) | 91.7 (12) | 100.0 (5) | 100.0 (7) | — |
| congress-senate-107~unsigned | EDSP-tuned | 100.0 (3) | 100.0 (26) | 100.0 (28) | 100.0 (61) | 100.0 (31) | 82.4 (17) | 83.3 (12) | 60.0 (5) | 14.3 (7) | — |
| congress-senate-107~unsigned | harmonic_cmn | 0.0 (3) | 0.0 (26) | 0.0 (28) | 3.3 (61) | 25.8 (31) | 88.2 (17) | 91.7 (12) | 100.0 (5) | 100.0 (7) | — |
| congress-senate-107~unsigned | lgc | 100.0 (3) | 19.2 (26) | 28.6 (28) | 50.8 (61) | 67.7 (31) | 52.9 (17) | 100.0 (12) | 100.0 (5) | 71.4 (7) | — |
| congress-senate-107~unsigned | ppr_lp | 33.3 (3) | 19.2 (26) | 28.6 (28) | 50.8 (61) | 67.7 (31) | 52.9 (17) | 100.0 (12) | 100.0 (5) | 71.4 (7) | — |
| congress-senate-107~unsigned | linbp | 100.0 (3) | 100.0 (26) | 100.0 (28) | 100.0 (61) | 93.5 (31) | 11.8 (17) | 16.7 (12) | 0.0 (5) | 0.0 (7) | — |
| congress-senate-114~unsigned | EDSP-default | — | 0.0 (18) | 0.0 (19) | 1.7 (59) | 36.2 (69) | 93.5 (31) | 100.0 (4) | — | — | — |
| congress-senate-114~unsigned | EDSP-tuned | — | 100.0 (18) | 100.0 (19) | 100.0 (59) | 100.0 (69) | 93.5 (31) | 50.0 (4) | — | — | — |
| congress-senate-114~unsigned | harmonic_cmn | — | 0.0 (18) | 0.0 (19) | 1.7 (59) | 53.6 (69) | 96.8 (31) | 100.0 (4) | — | — | — |
| congress-senate-114~unsigned | lgc | — | 16.7 (18) | 68.4 (19) | 27.1 (59) | 63.8 (69) | 90.3 (31) | 100.0 (4) | — | — | — |
| congress-senate-114~unsigned | ppr_lp | — | 11.1 (18) | 63.2 (19) | 27.1 (59) | 63.8 (69) | 90.3 (31) | 100.0 (4) | — | — | — |
| congress-senate-114~unsigned | linbp | — | 100.0 (18) | 100.0 (19) | 100.0 (59) | 92.8 (69) | 19.4 (31) | 0.0 (4) | — | — | — |
| congress-senate-93~unsigned | EDSP-default | — | 0.0 (5) | 0.0 (12) | 6.2 (48) | 45.3 (53) | 90.2 (61) | 96.8 (31) | — | — | — |
| congress-senate-93~unsigned | EDSP-tuned | — | 80.0 (5) | 91.7 (12) | 87.5 (48) | 67.9 (53) | 70.5 (61) | 38.7 (31) | — | — | — |
| congress-senate-93~unsigned | harmonic_cmn | — | 0.0 (5) | 0.0 (12) | 6.2 (48) | 50.9 (53) | 88.5 (61) | 96.8 (31) | — | — | — |
| congress-senate-93~unsigned | lgc | — | 0.0 (5) | 8.3 (12) | 4.2 (48) | 50.9 (53) | 86.9 (61) | 90.3 (31) | — | — | — |
| congress-senate-93~unsigned | ppr_lp | — | 0.0 (5) | 8.3 (12) | 10.4 (48) | 49.1 (53) | 85.2 (61) | 87.1 (31) | — | — | — |
| congress-senate-93~unsigned | linbp | — | 100.0 (5) | 100.0 (12) | 100.0 (48) | 81.1 (53) | 37.7 (61) | 3.2 (31) | — | — | — |
| wiki-editor~unsigned | EDSP-default | 6.8 (3284) | 0.0 (416) | 4.1 (1147) | 11.7 (1708) | 19.7 (1987) | 64.9 (5013) | 87.5 (4508) | 97.2 (4191) | 99.3 (4809) | 97.1 (13337) |
| wiki-editor~unsigned | EDSP-tuned | 11.7 (3284) | 0.7 (416) | 5.8 (1147) | 14.2 (1708) | 20.9 (1987) | 66.2 (5013) | 87.6 (4508) | 96.6 (4191) | 98.9 (4809) | 95.4 (13337) |
| wiki-editor~unsigned | harmonic_cmn | 11.6 (3284) | 1.2 (416) | 12.5 (1147) | 31.5 (1708) | 58.8 (1987) | 59.9 (5013) | 85.2 (4508) | 96.0 (4191) | 99.1 (4809) | 96.5 (13337) |
| wiki-editor~unsigned | lgc | 24.5 (3284) | 8.2 (416) | 31.0 (1147) | 44.3 (1708) | 43.3 (1987) | 68.3 (5013) | 80.7 (4508) | 91.5 (4191) | 97.2 (4809) | 94.0 (13337) |
| wiki-editor~unsigned | ppr_lp | 7.0 (3284) | 0.2 (416) | 6.2 (1147) | 13.9 (1708) | 22.3 (1987) | 62.0 (5013) | 84.8 (4508) | 96.1 (4191) | 99.2 (4809) | 97.1 (13337) |
| wiki-editor~unsigned | linbp | 7.0 (3284) | 0.7 (416) | 5.7 (1147) | 14.3 (1708) | 22.3 (1987) | 60.4 (5013) | 82.6 (4508) | 94.8 (4191) | 98.8 (4809) | 97.0 (13337) |
| wiki-elec~unsigned | EDSP-default | 0.0 (307) | 0.0 (290) | 0.0 (525) | 1.5 (471) | 9.1 (265) | 60.1 (353) | 90.9 (538) | 99.6 (929) | 99.7 (697) | 94.9 (275) |
| wiki-elec~unsigned | EDSP-tuned | 86.3 (307) | 22.8 (290) | 37.5 (525) | 39.3 (471) | 42.6 (265) | 69.4 (353) | 90.9 (538) | 99.0 (929) | 95.6 (697) | 86.5 (275) |
| wiki-elec~unsigned | harmonic_cmn | 8.8 (307) | 3.4 (290) | 34.5 (525) | 69.9 (471) | 81.5 (265) | 52.7 (353) | 29.6 (538) | 80.3 (929) | 99.1 (697) | 98.9 (275) |
| wiki-elec~unsigned | lgc | 4.6 (307) | 1.0 (290) | 5.9 (525) | 16.8 (471) | 42.6 (265) | 53.0 (353) | 83.5 (538) | 98.2 (929) | 99.7 (697) | 98.2 (275) |
| wiki-elec~unsigned | ppr_lp | 0.0 (307) | 0.0 (290) | 0.6 (525) | 2.5 (471) | 17.0 (265) | 58.9 (353) | 90.7 (538) | 99.4 (929) | 99.9 (697) | 96.0 (275) |
| wiki-elec~unsigned | linbp | 0.0 (307) | 0.0 (290) | 0.6 (525) | 2.5 (471) | 17.4 (265) | 58.6 (353) | 90.7 (538) | 99.4 (929) | 99.9 (697) | 96.0 (275) |
| wiki-rfa~unsigned | EDSP-default | 3.6 (167) | 0.0 (114) | 1.6 (376) | 4.0 (729) | 18.4 (936) | 72.1 (1616) | 93.5 (1487) | 98.1 (845) | 99.2 (376) | 94.6 (294) |
| wiki-rfa~unsigned | EDSP-tuned | 59.3 (167) | 12.3 (114) | 19.7 (376) | 17.3 (729) | 23.3 (936) | 72.8 (1616) | 91.3 (1487) | 94.8 (845) | 95.7 (376) | 72.1 (294) |
| wiki-rfa~unsigned | harmonic_cmn | 17.4 (167) | 1.8 (114) | 14.6 (376) | 45.7 (729) | 75.4 (936) | 50.0 (1616) | 68.8 (1487) | 97.6 (845) | 100.0 (376) | 99.7 (294) |
| wiki-rfa~unsigned | lgc | 14.4 (167) | 0.0 (114) | 5.6 (376) | 12.1 (729) | 30.6 (936) | 70.3 (1616) | 93.7 (1487) | 98.8 (845) | 98.7 (376) | 97.6 (294) |
| wiki-rfa~unsigned | ppr_lp | 4.8 (167) | 0.0 (114) | 1.9 (376) | 5.2 (729) | 18.6 (936) | 71.0 (1616) | 93.9 (1487) | 98.2 (845) | 98.9 (376) | 95.2 (294) |
| wiki-rfa~unsigned | linbp | 5.4 (167) | 0.0 (114) | 2.1 (376) | 5.5 (729) | 18.4 (936) | 70.7 (1616) | 93.8 (1487) | 98.3 (845) | 98.7 (376) | 95.2 (294) |
| yelpchi~unsigned | EDSP-default | 0.0 (4768) | 0.0 (13645) | 0.2 (4955) | 2.1 (1866) | 17.2 (576) | 66.5 (737) | 97.2 (3705) | 99.8 (15543) | 100.0 (84971) | 99.9 (53003) |
| yelpchi~unsigned | EDSP-tuned | 0.0 (4768) | 0.0 (13645) | 0.3 (4955) | 2.3 (1866) | 17.2 (576) | 66.4 (737) | 97.1 (3705) | 99.7 (15543) | 100.0 (84971) | 99.9 (53003) |
| yelpchi~unsigned | harmonic_cmn | 0.0 (4768) | 0.0 (13645) | 0.2 (4955) | 1.9 (1866) | 17.2 (576) | 65.3 (737) | 97.2 (3705) | 99.8 (15543) | 100.0 (84971) | 99.9 (53003) |
| yelpchi~unsigned | lgc | 0.1 (4768) | 0.1 (13645) | 0.4 (4955) | 5.7 (1866) | 33.7 (576) | 66.8 (737) | 94.7 (3705) | 99.5 (15543) | 99.9 (84971) | 99.9 (53003) |
| yelpchi~unsigned | ppr_lp | 0.0 (4768) | 0.0 (13645) | 0.2 (4955) | 3.4 (1866) | 23.8 (576) | 69.2 (737) | 96.1 (3705) | 99.7 (15543) | 100.0 (84971) | 99.9 (53003) |
| yelpchi~unsigned | linbp | 0.0 (4768) | 0.0 (13645) | 0.2 (4955) | 3.3 (1866) | 25.2 (576) | 68.9 (737) | 96.1 (3705) | 99.7 (15543) | 100.0 (84971) | 99.9 (53003) |

## Test accuracy by degree (bin 0: isolated; bin b: 2^(b-1) ≤ deg < 2^b; last bin open), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| amazon-fraud~unsigned | EDSP-default | — | — | 100.0 (6) | 69.0 (42) | 49.3 (134) | 62.4 (556) | 66.9 (1443) | 77.7 (3027) | 86.1 (5458) | 90.8 (7502) | 97.6 (16392) |
| amazon-fraud~unsigned | EDSP-tuned | — | — | 83.3 (6) | 69.0 (42) | 59.0 (134) | 61.3 (556) | 67.1 (1443) | 77.7 (3027) | 86.1 (5458) | 90.8 (7502) | 97.6 (16392) |
| amazon-fraud~unsigned | harmonic_cmn | — | — | 100.0 (6) | 69.0 (42) | 50.7 (134) | 62.2 (556) | 66.9 (1443) | 77.7 (3027) | 86.1 (5458) | 90.8 (7502) | 97.6 (16392) |
| amazon-fraud~unsigned | lgc | — | — | 100.0 (6) | 69.0 (42) | 50.7 (134) | 62.4 (556) | 67.1 (1443) | 77.7 (3027) | 86.1 (5458) | 90.8 (7502) | 97.6 (16392) |
| amazon-fraud~unsigned | ppr_lp | — | — | 100.0 (6) | 69.0 (42) | 49.3 (134) | 62.4 (556) | 66.9 (1443) | 77.7 (3027) | 86.1 (5458) | 90.8 (7502) | 97.6 (16392) |
| amazon-fraud~unsigned | linbp | — | — | 100.0 (6) | 69.0 (42) | 53.7 (134) | 63.3 (556) | 67.6 (1443) | 78.0 (3027) | 86.1 (5458) | 90.8 (7502) | 97.6 (16392) |
| congress-house-100~unsigned | EDSP-default | — | — | — | — | — | — | — | 64.2 (53) | 43.3 (580) | 52.1 (257) | — |
| congress-house-100~unsigned | EDSP-tuned | — | — | — | — | — | — | — | 88.7 (53) | 84.8 (580) | 92.2 (257) | — |
| congress-house-100~unsigned | harmonic_cmn | — | — | — | — | — | — | — | 62.3 (53) | 42.6 (580) | 52.5 (257) | — |
| congress-house-100~unsigned | lgc | — | — | — | — | — | — | — | 75.5 (53) | 52.2 (580) | 63.0 (257) | — |
| congress-house-100~unsigned | ppr_lp | — | — | — | — | — | — | — | 79.2 (53) | 52.2 (580) | 63.0 (257) | — |
| congress-house-100~unsigned | linbp | — | — | — | — | — | — | — | 64.2 (53) | 71.4 (580) | 80.9 (257) | — |
| congress-house-107~unsigned | EDSP-default | — | — | — | — | — | — | — | 77.8 (9) | 23.1 (273) | 17.3 (608) | — |
| congress-house-107~unsigned | EDSP-tuned | — | — | — | — | — | — | — | 66.7 (9) | 91.2 (273) | 98.4 (608) | — |
| congress-house-107~unsigned | harmonic_cmn | — | — | — | — | — | — | — | 66.7 (9) | 21.2 (273) | 8.6 (608) | — |
| congress-house-107~unsigned | lgc | — | — | — | — | — | — | — | 33.3 (9) | 33.3 (273) | 54.8 (608) | — |
| congress-house-107~unsigned | ppr_lp | — | — | — | — | — | — | — | 33.3 (9) | 33.3 (273) | 54.9 (608) | — |
| congress-house-107~unsigned | linbp | — | — | — | — | — | — | — | 33.3 (9) | 83.9 (273) | 95.1 (608) | — |
| congress-house-114~unsigned | EDSP-default | — | — | — | — | — | — | — | — | 30.1 (143) | 11.2 (747) | — |
| congress-house-114~unsigned | EDSP-tuned | — | — | — | — | — | — | — | — | 88.1 (143) | 97.7 (747) | — |
| congress-house-114~unsigned | harmonic_cmn | — | — | — | — | — | — | — | — | 43.4 (143) | 36.8 (747) | — |
| congress-house-114~unsigned | lgc | — | — | — | — | — | — | — | — | 91.6 (143) | 49.4 (747) | — |
| congress-house-114~unsigned | ppr_lp | — | — | — | — | — | — | — | — | 91.6 (143) | 49.4 (747) | — |
| congress-house-114~unsigned | linbp | — | — | — | — | — | — | — | — | 76.2 (143) | 94.8 (747) | — |
| congress-house-93~unsigned | EDSP-default | — | — | — | — | — | 89.2 (74) | 54.7 (393) | 48.8 (328) | 97.2 (72) | 100.0 (23) | — |
| congress-house-93~unsigned | EDSP-tuned | — | — | — | — | — | 89.2 (74) | 76.6 (393) | 77.1 (328) | 98.6 (72) | 100.0 (23) | — |
| congress-house-93~unsigned | harmonic_cmn | — | — | — | — | — | 89.2 (74) | 67.4 (393) | 68.0 (328) | 41.7 (72) | 0.0 (23) | — |
| congress-house-93~unsigned | lgc | — | — | — | — | — | 89.2 (74) | 64.4 (393) | 60.1 (328) | 88.9 (72) | 73.9 (23) | — |
| congress-house-93~unsigned | ppr_lp | — | — | — | — | — | 89.2 (74) | 57.8 (393) | 50.3 (328) | 97.2 (72) | 95.7 (23) | — |
| congress-house-93~unsigned | linbp | — | — | — | — | — | 58.1 (74) | 53.7 (393) | 49.1 (328) | 83.3 (72) | 91.3 (23) | — |
| congress-senate-100~unsigned | EDSP-default | — | — | — | — | 0.0 (6) | 66.7 (48) | 47.1 (121) | 36.0 (25) | — | — | — |
| congress-senate-100~unsigned | EDSP-tuned | — | — | — | — | 66.7 (6) | 75.0 (48) | 77.7 (121) | 76.0 (25) | — | — | — |
| congress-senate-100~unsigned | harmonic_cmn | — | — | — | — | 0.0 (6) | 68.8 (48) | 44.6 (121) | 52.0 (25) | — | — | — |
| congress-senate-100~unsigned | lgc | — | — | — | — | 0.0 (6) | 72.9 (48) | 47.9 (121) | 56.0 (25) | — | — | — |
| congress-senate-100~unsigned | ppr_lp | — | — | — | — | 0.0 (6) | 75.0 (48) | 44.6 (121) | 48.0 (25) | — | — | — |
| congress-senate-100~unsigned | linbp | — | — | — | — | 100.0 (6) | 52.1 (48) | 62.8 (121) | 72.0 (25) | — | — | — |
| congress-senate-107~unsigned | EDSP-default | — | — | — | — | — | 43.5 (23) | 27.1 (129) | 10.5 (38) | — | — | — |
| congress-senate-107~unsigned | EDSP-tuned | — | — | — | — | — | 82.6 (23) | 93.0 (129) | 100.0 (38) | — | — | — |
| congress-senate-107~unsigned | harmonic_cmn | — | — | — | — | — | 47.8 (23) | 26.4 (129) | 7.9 (38) | — | — | — |
| congress-senate-107~unsigned | lgc | — | — | — | — | — | 69.6 (23) | 41.1 (129) | 78.9 (38) | — | — | — |
| congress-senate-107~unsigned | ppr_lp | — | — | — | — | — | 60.9 (23) | 41.1 (129) | 78.9 (38) | — | — | — |
| congress-senate-107~unsigned | linbp | — | — | — | — | — | 56.5 (23) | 78.3 (129) | 97.4 (38) | — | — | — |
| congress-senate-114~unsigned | EDSP-default | — | — | — | — | — | — | 21.4 (56) | 32.6 (144) | — | — | — |
| congress-senate-114~unsigned | EDSP-tuned | — | — | — | — | — | — | 96.4 (56) | 98.6 (144) | — | — | — |
| congress-senate-114~unsigned | harmonic_cmn | — | — | — | — | — | — | 23.2 (56) | 41.0 (144) | — | — | — |
| congress-senate-114~unsigned | lgc | — | — | — | — | — | — | 57.1 (56) | 52.8 (144) | — | — | — |
| congress-senate-114~unsigned | ppr_lp | — | — | — | — | — | — | 53.6 (56) | 52.8 (144) | — | — | — |
| congress-senate-114~unsigned | linbp | — | — | — | — | — | — | 87.5 (56) | 81.2 (144) | — | — | — |
| congress-senate-93~unsigned | EDSP-default | — | — | — | — | 50.0 (2) | 69.0 (42) | 52.5 (122) | 40.9 (44) | — | — | — |
| congress-senate-93~unsigned | EDSP-tuned | — | — | — | — | 0.0 (2) | 66.7 (42) | 66.4 (122) | 88.6 (44) | — | — | — |
| congress-senate-93~unsigned | harmonic_cmn | — | — | — | — | 0.0 (2) | 69.0 (42) | 54.9 (122) | 40.9 (44) | — | — | — |
| congress-senate-93~unsigned | lgc | — | — | — | — | 0.0 (2) | 64.3 (42) | 54.1 (122) | 40.9 (44) | — | — | — |
| congress-senate-93~unsigned | ppr_lp | — | — | — | — | 0.0 (2) | 64.3 (42) | 54.1 (122) | 40.9 (44) | — | — | — |
| congress-senate-93~unsigned | linbp | — | — | — | — | 50.0 (2) | 54.8 (42) | 56.6 (122) | 88.6 (44) | — | — | — |
| wiki-editor~unsigned | EDSP-default | — | 63.5 (5561) | 67.0 (7895) | 73.0 (8718) | 80.4 (7449) | 83.3 (4204) | 79.7 (2057) | 79.2 (1440) | 85.4 (1534) | 68.3 (1377) | 91.5 (165) |
| wiki-editor~unsigned | EDSP-tuned | — | 63.2 (5561) | 67.1 (7895) | 73.1 (8718) | 80.9 (7449) | 83.4 (4204) | 79.6 (2057) | 79.2 (1440) | 85.3 (1534) | 68.3 (1377) | 90.9 (165) |
| wiki-editor~unsigned | harmonic_cmn | — | 64.4 (5561) | 70.2 (7895) | 78.2 (8718) | 83.9 (7449) | 83.4 (4204) | 75.8 (2057) | 78.1 (1440) | 87.0 (1534) | 65.9 (1377) | 66.7 (165) |
| wiki-editor~unsigned | lgc | — | 65.2 (5561) | 72.4 (7895) | 80.1 (8718) | 84.3 (7449) | 82.0 (4204) | 71.2 (2057) | 71.5 (1440) | 86.3 (1534) | 66.5 (1377) | 78.2 (165) |
| wiki-editor~unsigned | ppr_lp | — | 63.6 (5561) | 65.4 (7895) | 72.5 (8718) | 79.6 (7449) | 83.0 (4204) | 79.5 (2057) | 79.7 (1440) | 87.1 (1534) | 68.3 (1377) | 91.5 (165) |
| wiki-editor~unsigned | linbp | — | 63.6 (5561) | 65.1 (7895) | 71.1 (8718) | 78.3 (7449) | 82.5 (4204) | 79.9 (2057) | 79.7 (1440) | 87.6 (1534) | 68.3 (1377) | 91.5 (165) |
| wiki-elec~unsigned | EDSP-default | — | 11.4 (132) | 25.9 (247) | 20.0 (475) | 32.3 (647) | 59.0 (1070) | 75.2 (1040) | 80.0 (661) | 73.6 (307) | 90.9 (66) | 40.0 (5) |
| wiki-elec~unsigned | EDSP-tuned | — | 84.1 (132) | 78.1 (247) | 81.1 (475) | 66.5 (647) | 61.0 (1070) | 75.4 (1040) | 80.6 (661) | 75.9 (307) | 90.9 (66) | 40.0 (5) |
| wiki-elec~unsigned | harmonic_cmn | — | 31.1 (132) | 44.9 (247) | 54.9 (475) | 53.5 (647) | 65.6 (1070) | 65.9 (1040) | 63.1 (661) | 68.4 (307) | 60.6 (66) | 80.0 (5) |
| wiki-elec~unsigned | lgc | — | 22.7 (132) | 32.4 (247) | 33.1 (475) | 39.4 (647) | 62.0 (1070) | 74.7 (1040) | 77.0 (661) | 74.3 (307) | 78.8 (66) | 40.0 (5) |
| wiki-elec~unsigned | ppr_lp | — | 12.1 (132) | 21.5 (247) | 22.1 (475) | 33.5 (647) | 59.9 (1070) | 75.8 (1040) | 80.0 (661) | 74.3 (307) | 90.9 (66) | 40.0 (5) |
| wiki-elec~unsigned | linbp | — | 12.1 (132) | 21.9 (247) | 21.9 (475) | 33.5 (647) | 59.9 (1070) | 75.8 (1040) | 80.0 (661) | 74.3 (307) | 90.9 (66) | 40.0 (5) |
| wiki-rfa~unsigned | EDSP-default | — | 54.3 (127) | 47.6 (368) | 46.9 (684) | 40.4 (832) | 59.5 (1021) | 73.2 (1668) | 72.6 (1490) | 58.8 (565) | 55.4 (168) | 76.5 (17) |
| wiki-rfa~unsigned | EDSP-tuned | — | 66.9 (127) | 58.7 (368) | 61.3 (684) | 45.7 (832) | 59.9 (1021) | 73.1 (1668) | 71.9 (1490) | 57.5 (565) | 57.1 (168) | 76.5 (17) |
| wiki-rfa~unsigned | harmonic_cmn | — | 72.4 (127) | 63.3 (368) | 72.4 (684) | 67.9 (832) | 71.4 (1021) | 62.9 (1668) | 58.1 (1490) | 54.0 (565) | 61.3 (168) | 70.6 (17) |
| wiki-rfa~unsigned | lgc | — | 66.9 (127) | 55.2 (368) | 53.9 (684) | 48.4 (832) | 64.7 (1021) | 72.5 (1668) | 70.8 (1490) | 61.2 (565) | 58.3 (168) | 70.6 (17) |
| wiki-rfa~unsigned | ppr_lp | — | 57.5 (127) | 46.5 (368) | 45.6 (684) | 41.1 (832) | 60.1 (1021) | 73.1 (1668) | 71.8 (1490) | 59.5 (565) | 60.1 (168) | 76.5 (17) |
| wiki-rfa~unsigned | linbp | — | 58.3 (127) | 45.1 (368) | 45.5 (684) | 41.5 (832) | 60.3 (1021) | 73.1 (1668) | 71.9 (1490) | 59.1 (565) | 58.9 (168) | 76.5 (17) |
| yelpchi~unsigned | EDSP-default | 76.5 (51) | 77.6 (143) | 78.0 (400) | 77.0 (1220) | 83.8 (2847) | 82.2 (7185) | 83.3 (18437) | 85.4 (39424) | 86.4 (79924) | 87.5 (34189) | — |
| yelpchi~unsigned | EDSP-tuned | 76.5 (51) | 74.1 (143) | 77.8 (400) | 76.6 (1220) | 83.0 (2847) | 81.9 (7185) | 83.3 (18437) | 85.4 (39424) | 86.4 (79924) | 87.5 (34189) | — |
| yelpchi~unsigned | harmonic_cmn | 23.5 (51) | 79.0 (143) | 73.8 (400) | 77.0 (1220) | 83.9 (2847) | 82.2 (7185) | 83.2 (18437) | 85.4 (39424) | 86.4 (79924) | 87.5 (34189) | — |
| yelpchi~unsigned | lgc | 76.5 (51) | 77.6 (143) | 77.0 (400) | 73.3 (1220) | 81.6 (2847) | 82.1 (7185) | 83.7 (18437) | 85.4 (39424) | 86.4 (79924) | 87.5 (34189) | — |
| yelpchi~unsigned | ppr_lp | 76.5 (51) | 77.6 (143) | 78.8 (400) | 75.3 (1220) | 83.0 (2847) | 82.1 (7185) | 83.6 (18437) | 85.4 (39424) | 86.4 (79924) | 87.5 (34189) | — |
| yelpchi~unsigned | linbp | 76.5 (51) | 77.6 (143) | 79.0 (400) | 75.6 (1220) | 83.1 (2847) | 82.1 (7185) | 83.6 (18437) | 85.4 (39424) | 86.4 (79924) | 87.5 (34189) | — |
