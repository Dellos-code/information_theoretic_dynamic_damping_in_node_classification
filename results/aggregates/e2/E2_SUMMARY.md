# E2 — native-polarity graphs (signed, relation-typed): Track-A EDSP vs Track-A baselines

Run records: 928 (git 08499faaf308; runners actions-run-42366240266). Headline metric per dataset in brackets; mean ± std over splits (×100). Rows marked † are constructed benchmarks or case studies and are excluded from real-world aggregates.

| dataset [metric] | splits | EDSP-default | EDSP-tuned | EDSP-fullH-default | EDSP-fullH-tuned | majority | train_prior | harmonic | harmonic_cmn | lgc | ppr_lp | linbp | signed_vote | signed_lgc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| congress-house-100 [accuracy] | 10 | 88.09 ± 2.90 | 88.65 ± 2.53 | — | — | 57.30 ± 0.00 | 57.30 ± 0.00 | 47.64 ± 3.38 | 46.63 ± 2.76 | 56.74 ± 0.75 | 56.97 ± 0.72 | 73.71 ± 2.98 | **88.76 ± 2.56** | 88.65 ± 2.38 |
| congress-house-107 [accuracy] | 10 | 96.97 ± 1.95 | 96.97 ± 1.95 | — | — | 51.69 ± 0.00 | 51.69 ± 0.00 | 20.45 ± 3.95 | 13.03 ± 2.94 | 47.98 ± 1.01 | 48.09 ± 0.45 | 91.01 ± 2.97 | **97.08 ± 1.96** | **97.08 ± 2.08** |
| congress-house-114 [accuracy] | 10 | **97.08 ± 1.76** | **97.08 ± 1.76** | — | — | 56.18 ± 0.00 | 56.18 ± 0.00 | 14.49 ± 3.91 | 37.87 ± 3.26 | 56.18 ± 0.00 | 56.18 ± 0.00 | 91.80 ± 1.82 | 96.97 ± 2.01 | 96.97 ± 2.01 |
| congress-house-93 [accuracy] | 10 | 73.15 ± 3.20 | **80.45 ± 3.23** | — | — | 56.18 ± 0.00 | 56.18 ± 0.00 | 59.55 ± 2.46 | 65.62 ± 2.94 | 67.08 ± 2.01 | 61.80 ± 2.25 | 55.73 ± 9.54 | 75.06 ± 3.47 | 78.88 ± 3.13 |
| congress-senate-100 [accuracy] | 10 | 88.00 ± 5.10 | **88.50 ± 5.02** | — | — | 55.00 ± 0.00 | 55.00 ± 0.00 | 48.50 ± 8.67 | 50.00 ± 9.22 | 53.50 ± 7.09 | 51.00 ± 8.31 | 62.50 ± 5.59 | 88.00 ± 6.00 | 88.00 ± 6.00 |
| congress-senate-107 [accuracy] | 10 | 96.84 ± 2.58 | 96.84 ± 2.58 | — | — | 52.63 ± 0.00 | 52.63 ± 0.00 | 27.37 ± 7.74 | 25.26 ± 7.74 | 52.11 ± 1.58 | 51.05 ± 2.41 | 79.47 ± 9.83 | **97.37 ± 2.63** | **97.37 ± 2.63** |
| congress-senate-114 [accuracy] | 10 | 96.50 ± 4.50 | 96.00 ± 4.36 | — | — | 55.00 ± 0.00 | 55.00 ± 0.00 | 30.50 ± 7.89 | 36.00 ± 7.00 | 54.00 ± 2.00 | 53.00 ± 4.58 | 83.00 ± 6.78 | **97.50 ± 3.35** | **97.50 ± 3.35** |
| congress-senate-93 [accuracy] | 10 | 80.48 ± 8.37 | 80.48 ± 8.37 | — | — | 57.14 ± 0.00 | 57.14 ± 0.00 | 54.29 ± 6.80 | 54.29 ± 7.74 | 52.86 ± 6.55 | 52.86 ± 5.41 | 62.86 ± 6.67 | **81.90 ± 7.62** | **81.90 ± 7.62** |

## Pre-registered secondary column: EDSP-default on non-converged runs

| dataset | non-converged runs | headline, last iterate | headline, mean of last two iterates |
|---|---|---|---|
| congress-house-100 | 0 / 10 | — | — |
| congress-house-107 | 0 / 10 | — | — |
| congress-house-114 | 0 / 10 | — | — |
| congress-house-93 | 0 / 10 | — | — |
| congress-senate-100 | 0 / 10 | — | — |
| congress-senate-107 | 0 / 10 | — | — |
| congress-senate-114 | 0 / 10 | — | — |
| congress-senate-93 | 0 / 10 | — | — |

## EDSP-tuned vs each baseline: win / tie / loss

Per dataset: two-sided Wilcoxon signed-rank over splits on the headline metric; Holm correction across datasets per baseline; win/loss if adjusted p < 0.05, else tie. Real-world datasets only.

| baseline | win | tie | loss | datasets (W / L) |
|---|---|---|---|---|
| harmonic | 8 | 0 | 0 | W: congress-house-100, congress-house-107, congress-house-114, congress-house-93, congress-senate-100, congress-senate-107, congress-senate-114, congress-senate-93; L: — |
| harmonic_cmn | 8 | 0 | 0 | W: congress-house-100, congress-house-107, congress-house-114, congress-house-93, congress-senate-100, congress-senate-107, congress-senate-114, congress-senate-93; L: — |
| lgc | 8 | 0 | 0 | W: congress-house-100, congress-house-107, congress-house-114, congress-house-93, congress-senate-100, congress-senate-107, congress-senate-114, congress-senate-93; L: — |
| ppr_lp | 8 | 0 | 0 | W: congress-house-100, congress-house-107, congress-house-114, congress-house-93, congress-senate-100, congress-senate-107, congress-senate-114, congress-senate-93; L: — |
| linbp | 8 | 0 | 0 | W: congress-house-100, congress-house-107, congress-house-114, congress-house-93, congress-senate-100, congress-senate-107, congress-senate-114, congress-senate-93; L: — |
| train_prior | 8 | 0 | 0 | W: congress-house-100, congress-house-107, congress-house-114, congress-house-93, congress-senate-100, congress-senate-107, congress-senate-114, congress-senate-93; L: — |
| signed_vote | 0 | 8 | 0 | W: —; L: — |
| signed_lgc | 0 | 8 | 0 | W: —; L: — |

## Friedman / Nemenyi (real-world datasets, headline metric)

Datasets: 8; Friedman p = 1.81e-07; Nemenyi CD (α = 0.05) = 3.71.

| method | average rank |
|---|---|
| EDSP-tuned | 1.38 |
| EDSP-default | 1.62 |
| linbp | 3.62 |
| train_prior | 4.50 |
| lgc | 5.25 |
| ppr_lp | 5.75 |
| harmonic_cmn | 6.81 |
| harmonic | 7.06 |

## Calibration (mean over splits; lower is better)

| dataset | EDSP-default NLL | EDSP-default ECE | EDSP-tuned NLL | EDSP-tuned ECE | best baseline by NLL (NLL / ECE) |
|---|---|---|---|---|---|
| congress-house-100 | 0.407 | 0.199 | 0.401 | 0.178 | linbp: 0.605 / 0.181 |
| congress-house-107 | 0.181 | 0.135 | 0.181 | 0.135 | linbp: 0.589 / 0.357 |
| congress-house-114 | 0.147 | 0.116 | 0.147 | 0.116 | linbp: 0.618 / 0.375 |
| congress-house-93 | 0.562 | 0.082 | 0.529 | 0.176 | lgc: 0.634 / 0.104 |
| congress-senate-100 | 0.469 | 0.261 | 0.476 | 0.264 | linbp: 0.662 / 0.138 |
| congress-senate-107 | 0.273 | 0.194 | 0.273 | 0.194 | signed_lgc: 0.545 / 0.026 |
| congress-senate-114 | 0.075 | 0.066 | 0.102 | 0.089 | signed_vote: 0.431 / 0.022 |
| congress-senate-93 | 0.506 | 0.227 | 0.506 | 0.227 | linbp: 0.668 / 0.122 |

## H7 (descriptive): convergence and fixed-point non-uniqueness

| dataset | EDSP-default non-converged | oscillation flagged | mean fraction of nodes still moving | GS vs damped L1: mean | p95 | splits |
|---|---|---|---|---|---|---|
| congress-house-100 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3 |
| congress-house-107 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3 |
| congress-house-114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3 |
| congress-house-93 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3 |
| congress-senate-100 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3 |
| congress-senate-107 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3 |
| congress-senate-114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3 |
| congress-senate-93 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3 |

## Test accuracy by local homophily (descriptive, full labels; bin i = [i/10, (i+1)/10)), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| congress-house-100 | EDSP-default | 100.0 (2) | 100.0 (47) | 100.0 (143) | 96.2 (156) | 89.9 (247) | 78.4 (176) | 78.4 (102) | 11.8 (17) | — | — |
| congress-house-100 | EDSP-tuned | 100.0 (2) | 100.0 (47) | 100.0 (143) | 94.9 (156) | 89.1 (247) | 82.4 (176) | 80.4 (102) | 11.8 (17) | — | — |
| congress-house-100 | harmonic_cmn | 0.0 (2) | 0.0 (47) | 0.0 (143) | 1.9 (156) | 55.5 (247) | 88.6 (176) | 100.0 (102) | 100.0 (17) | — | — |
| congress-house-100 | lgc | 0.0 (2) | 4.3 (47) | 8.4 (143) | 34.0 (156) | 66.0 (247) | 88.6 (176) | 100.0 (102) | 100.0 (17) | — | — |
| congress-house-100 | ppr_lp | 0.0 (2) | 2.1 (47) | 10.5 (143) | 34.0 (156) | 66.0 (247) | 88.6 (176) | 100.0 (102) | 100.0 (17) | — | — |
| congress-house-100 | linbp | 100.0 (2) | 100.0 (47) | 100.0 (143) | 100.0 (156) | 90.3 (247) | 48.3 (176) | 0.0 (102) | 0.0 (17) | — | — |
| congress-house-100 | signed_vote | 100.0 (2) | 100.0 (47) | 100.0 (143) | 95.5 (156) | 92.7 (247) | 78.4 (176) | 78.4 (102) | 11.8 (17) | — | — |
| congress-house-100 | signed_lgc | 100.0 (2) | 100.0 (47) | 100.0 (143) | 94.2 (156) | 91.5 (247) | 80.1 (176) | 79.4 (102) | 11.8 (17) | — | — |
| congress-house-107 | EDSP-default | 100.0 (2) | 100.0 (66) | 100.0 (147) | 100.0 (266) | 99.4 (310) | 82.4 (85) | 30.8 (13) | 0.0 (1) | — | — |
| congress-house-107 | EDSP-tuned | 100.0 (2) | 100.0 (66) | 100.0 (147) | 100.0 (266) | 99.4 (310) | 82.4 (85) | 30.8 (13) | 0.0 (1) | — | — |
| congress-house-107 | harmonic_cmn | 0.0 (2) | 0.0 (66) | 0.0 (147) | 0.4 (266) | 19.0 (310) | 49.4 (85) | 100.0 (13) | 100.0 (1) | — | — |
| congress-house-107 | lgc | 100.0 (2) | 18.2 (66) | 15.0 (147) | 35.0 (266) | 74.2 (310) | 74.1 (85) | 38.5 (13) | 0.0 (1) | — | — |
| congress-house-107 | ppr_lp | 50.0 (2) | 18.2 (66) | 16.3 (147) | 35.0 (266) | 74.2 (310) | 74.1 (85) | 38.5 (13) | 0.0 (1) | — | — |
| congress-house-107 | linbp | 100.0 (2) | 100.0 (66) | 100.0 (147) | 100.0 (266) | 99.0 (310) | 25.9 (85) | 0.0 (13) | 0.0 (1) | — | — |
| congress-house-107 | signed_vote | 100.0 (2) | 100.0 (66) | 100.0 (147) | 100.0 (266) | 99.4 (310) | 83.5 (85) | 30.8 (13) | 0.0 (1) | — | — |
| congress-house-107 | signed_lgc | 100.0 (2) | 100.0 (66) | 100.0 (147) | 100.0 (266) | 99.0 (310) | 84.7 (85) | 30.8 (13) | 0.0 (1) | — | — |
| congress-house-114 | EDSP-default | 100.0 (3) | 100.0 (19) | 100.0 (74) | 100.0 (215) | 99.4 (498) | 78.1 (64) | 47.1 (17) | — | — | — |
| congress-house-114 | EDSP-tuned | 100.0 (3) | 100.0 (19) | 100.0 (74) | 100.0 (215) | 99.4 (498) | 78.1 (64) | 47.1 (17) | — | — | — |
| congress-house-114 | harmonic_cmn | 0.0 (3) | 0.0 (19) | 0.0 (74) | 10.2 (215) | 47.8 (498) | 93.8 (64) | 100.0 (17) | — | — | — |
| congress-house-114 | lgc | 0.0 (3) | 42.1 (19) | 75.7 (74) | 56.3 (215) | 47.8 (498) | 93.8 (64) | 100.0 (17) | — | — | — |
| congress-house-114 | ppr_lp | 0.0 (3) | 42.1 (19) | 75.7 (74) | 56.3 (215) | 47.8 (498) | 93.8 (64) | 100.0 (17) | — | — | — |
| congress-house-114 | linbp | 100.0 (3) | 100.0 (19) | 100.0 (74) | 100.0 (215) | 99.6 (498) | 15.6 (64) | 0.0 (17) | — | — | — |
| congress-house-114 | signed_vote | 100.0 (3) | 100.0 (19) | 100.0 (74) | 100.0 (215) | 99.4 (498) | 76.6 (64) | 47.1 (17) | — | — | — |
| congress-house-114 | signed_lgc | 100.0 (3) | 100.0 (19) | 100.0 (74) | 100.0 (215) | 99.4 (498) | 76.6 (64) | 47.1 (17) | — | — | — |
| congress-house-93 | EDSP-default | 100.0 (29) | 100.0 (51) | 100.0 (73) | 98.7 (77) | 93.9 (115) | 98.4 (122) | 70.7 (82) | 55.0 (140) | 44.4 (117) | 8.3 (84) |
| congress-house-93 | EDSP-tuned | 96.6 (29) | 92.2 (51) | 84.9 (73) | 68.8 (77) | 80.0 (115) | 96.7 (122) | 87.8 (82) | 80.7 (140) | 65.8 (117) | 64.3 (84) |
| congress-house-93 | harmonic_cmn | 0.0 (29) | 0.0 (51) | 0.0 (73) | 22.1 (77) | 73.9 (115) | 56.6 (122) | 87.8 (82) | 100.0 (140) | 100.0 (117) | 100.0 (84) |
| congress-house-93 | lgc | 0.0 (29) | 0.0 (51) | 0.0 (73) | 2.6 (77) | 49.6 (115) | 94.3 (122) | 100.0 (82) | 100.0 (140) | 100.0 (117) | 100.0 (84) |
| congress-house-93 | ppr_lp | 0.0 (29) | 0.0 (51) | 0.0 (73) | 0.0 (77) | 24.3 (115) | 81.1 (122) | 100.0 (82) | 100.0 (140) | 100.0 (117) | 100.0 (84) |
| congress-house-93 | linbp | 31.0 (29) | 23.5 (51) | 24.7 (73) | 31.2 (77) | 43.5 (115) | 76.2 (122) | 68.3 (82) | 69.3 (140) | 67.5 (117) | 69.0 (84) |
| congress-house-93 | signed_vote | 100.0 (29) | 100.0 (51) | 100.0 (73) | 98.7 (77) | 93.9 (115) | 96.7 (122) | 76.8 (82) | 60.7 (140) | 49.6 (117) | 8.3 (84) |
| congress-house-93 | signed_lgc | 100.0 (29) | 100.0 (51) | 100.0 (73) | 98.7 (77) | 93.0 (115) | 97.5 (122) | 81.7 (82) | 67.9 (140) | 59.0 (117) | 19.0 (84) |
| congress-senate-100 | EDSP-default | — | — | 100.0 (17) | 95.2 (42) | 92.7 (55) | 82.8 (58) | 68.0 (25) | — | 100.0 (3) | — |
| congress-senate-100 | EDSP-tuned | — | — | 100.0 (17) | 95.2 (42) | 92.7 (55) | 84.5 (58) | 68.0 (25) | — | 100.0 (3) | — |
| congress-senate-100 | harmonic_cmn | — | — | 0.0 (17) | 2.4 (42) | 43.6 (55) | 81.0 (58) | 100.0 (25) | — | 100.0 (3) | — |
| congress-senate-100 | lgc | — | — | 0.0 (17) | 2.4 (42) | 69.1 (55) | 75.9 (58) | 84.0 (25) | — | 100.0 (3) | — |
| congress-senate-100 | ppr_lp | — | — | 0.0 (17) | 0.0 (42) | 54.5 (55) | 84.5 (58) | 80.0 (25) | — | 100.0 (3) | — |
| congress-senate-100 | linbp | — | — | 100.0 (17) | 100.0 (42) | 90.9 (55) | 25.9 (58) | 4.0 (25) | — | 0.0 (3) | — |
| congress-senate-100 | signed_vote | — | — | 100.0 (17) | 92.9 (42) | 92.7 (55) | 84.5 (58) | 68.0 (25) | — | 100.0 (3) | — |
| congress-senate-100 | signed_lgc | — | — | 100.0 (17) | 92.9 (42) | 90.9 (55) | 86.2 (58) | 68.0 (25) | — | 100.0 (3) | — |
| congress-senate-107 | EDSP-default | 100.0 (3) | 100.0 (26) | 100.0 (28) | 100.0 (61) | 100.0 (31) | 88.2 (17) | 83.3 (12) | 100.0 (5) | 71.4 (7) | — |
| congress-senate-107 | EDSP-tuned | 100.0 (3) | 100.0 (26) | 100.0 (28) | 100.0 (61) | 100.0 (31) | 88.2 (17) | 83.3 (12) | 100.0 (5) | 71.4 (7) | — |
| congress-senate-107 | harmonic_cmn | 0.0 (3) | 0.0 (26) | 0.0 (28) | 3.3 (61) | 25.8 (31) | 88.2 (17) | 91.7 (12) | 100.0 (5) | 100.0 (7) | — |
| congress-senate-107 | lgc | 100.0 (3) | 19.2 (26) | 28.6 (28) | 50.8 (61) | 67.7 (31) | 52.9 (17) | 100.0 (12) | 100.0 (5) | 71.4 (7) | — |
| congress-senate-107 | ppr_lp | 33.3 (3) | 19.2 (26) | 28.6 (28) | 50.8 (61) | 67.7 (31) | 52.9 (17) | 100.0 (12) | 100.0 (5) | 71.4 (7) | — |
| congress-senate-107 | linbp | 100.0 (3) | 100.0 (26) | 100.0 (28) | 100.0 (61) | 93.5 (31) | 11.8 (17) | 16.7 (12) | 0.0 (5) | 0.0 (7) | — |
| congress-senate-107 | signed_vote | 100.0 (3) | 100.0 (26) | 100.0 (28) | 100.0 (61) | 100.0 (31) | 88.2 (17) | 91.7 (12) | 100.0 (5) | 71.4 (7) | — |
| congress-senate-107 | signed_lgc | 100.0 (3) | 100.0 (26) | 100.0 (28) | 100.0 (61) | 100.0 (31) | 88.2 (17) | 91.7 (12) | 100.0 (5) | 71.4 (7) | — |
| congress-senate-114 | EDSP-default | — | 100.0 (18) | 100.0 (19) | 100.0 (59) | 100.0 (69) | 90.3 (31) | 0.0 (4) | — | — | — |
| congress-senate-114 | EDSP-tuned | — | 100.0 (18) | 100.0 (19) | 100.0 (59) | 100.0 (69) | 87.1 (31) | 0.0 (4) | — | — | — |
| congress-senate-114 | harmonic_cmn | — | 0.0 (18) | 0.0 (19) | 1.7 (59) | 53.6 (69) | 96.8 (31) | 100.0 (4) | — | — | — |
| congress-senate-114 | lgc | — | 16.7 (18) | 68.4 (19) | 27.1 (59) | 63.8 (69) | 90.3 (31) | 100.0 (4) | — | — | — |
| congress-senate-114 | ppr_lp | — | 11.1 (18) | 63.2 (19) | 27.1 (59) | 63.8 (69) | 90.3 (31) | 100.0 (4) | — | — | — |
| congress-senate-114 | linbp | — | 100.0 (18) | 100.0 (19) | 100.0 (59) | 92.8 (69) | 19.4 (31) | 0.0 (4) | — | — | — |
| congress-senate-114 | signed_vote | — | 100.0 (18) | 100.0 (19) | 100.0 (59) | 100.0 (69) | 93.5 (31) | 25.0 (4) | — | — | — |
| congress-senate-114 | signed_lgc | — | 100.0 (18) | 100.0 (19) | 100.0 (59) | 100.0 (69) | 93.5 (31) | 25.0 (4) | — | — | — |
| congress-senate-93 | EDSP-default | — | 100.0 (5) | 100.0 (12) | 100.0 (48) | 81.1 (53) | 73.8 (61) | 51.6 (31) | — | — | — |
| congress-senate-93 | EDSP-tuned | — | 100.0 (5) | 100.0 (12) | 100.0 (48) | 81.1 (53) | 73.8 (61) | 51.6 (31) | — | — | — |
| congress-senate-93 | harmonic_cmn | — | 0.0 (5) | 0.0 (12) | 6.2 (48) | 50.9 (53) | 88.5 (61) | 96.8 (31) | — | — | — |
| congress-senate-93 | lgc | — | 0.0 (5) | 8.3 (12) | 4.2 (48) | 50.9 (53) | 86.9 (61) | 90.3 (31) | — | — | — |
| congress-senate-93 | ppr_lp | — | 0.0 (5) | 8.3 (12) | 10.4 (48) | 49.1 (53) | 85.2 (61) | 87.1 (31) | — | — | — |
| congress-senate-93 | linbp | — | 100.0 (5) | 100.0 (12) | 100.0 (48) | 81.1 (53) | 37.7 (61) | 3.2 (31) | — | — | — |
| congress-senate-93 | signed_vote | — | 100.0 (5) | 100.0 (12) | 100.0 (48) | 81.1 (53) | 77.0 (61) | 54.8 (31) | — | — | — |
| congress-senate-93 | signed_lgc | — | 100.0 (5) | 100.0 (12) | 100.0 (48) | 81.1 (53) | 77.0 (61) | 54.8 (31) | — | — | — |

## Test accuracy by degree (bin 0: isolated; bin b: 2^(b-1) ≤ deg < 2^b; last bin open), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| congress-house-100 | EDSP-default | — | — | — | — | — | — | — | 83.0 (53) | 88.8 (580) | 87.5 (257) | — |
| congress-house-100 | EDSP-tuned | — | — | — | — | — | — | — | 88.7 (53) | 88.4 (580) | 89.1 (257) | — |
| congress-house-100 | harmonic_cmn | — | — | — | — | — | — | — | 62.3 (53) | 42.6 (580) | 52.5 (257) | — |
| congress-house-100 | lgc | — | — | — | — | — | — | — | 75.5 (53) | 52.2 (580) | 63.0 (257) | — |
| congress-house-100 | ppr_lp | — | — | — | — | — | — | — | 79.2 (53) | 52.2 (580) | 63.0 (257) | — |
| congress-house-100 | linbp | — | — | — | — | — | — | — | 64.2 (53) | 71.4 (580) | 80.9 (257) | — |
| congress-house-100 | signed_vote | — | — | — | — | — | — | — | 83.0 (53) | 88.8 (580) | 89.9 (257) | — |
| congress-house-100 | signed_lgc | — | — | — | — | — | — | — | 84.9 (53) | 88.6 (580) | 89.5 (257) | — |
| congress-house-107 | EDSP-default | — | — | — | — | — | — | — | 55.6 (9) | 92.7 (273) | 99.5 (608) | — |
| congress-house-107 | EDSP-tuned | — | — | — | — | — | — | — | 55.6 (9) | 92.7 (273) | 99.5 (608) | — |
| congress-house-107 | harmonic_cmn | — | — | — | — | — | — | — | 66.7 (9) | 21.2 (273) | 8.6 (608) | — |
| congress-house-107 | lgc | — | — | — | — | — | — | — | 33.3 (9) | 33.3 (273) | 54.8 (608) | — |
| congress-house-107 | ppr_lp | — | — | — | — | — | — | — | 33.3 (9) | 33.3 (273) | 54.9 (608) | — |
| congress-house-107 | linbp | — | — | — | — | — | — | — | 33.3 (9) | 83.9 (273) | 95.1 (608) | — |
| congress-house-107 | signed_vote | — | — | — | — | — | — | — | 55.6 (9) | 93.4 (273) | 99.3 (608) | — |
| congress-house-107 | signed_lgc | — | — | — | — | — | — | — | 55.6 (9) | 93.8 (273) | 99.2 (608) | — |
| congress-house-114 | EDSP-default | — | — | — | — | — | — | — | — | 90.9 (143) | 98.3 (747) | — |
| congress-house-114 | EDSP-tuned | — | — | — | — | — | — | — | — | 90.9 (143) | 98.3 (747) | — |
| congress-house-114 | harmonic_cmn | — | — | — | — | — | — | — | — | 43.4 (143) | 36.8 (747) | — |
| congress-house-114 | lgc | — | — | — | — | — | — | — | — | 91.6 (143) | 49.4 (747) | — |
| congress-house-114 | ppr_lp | — | — | — | — | — | — | — | — | 91.6 (143) | 49.4 (747) | — |
| congress-house-114 | linbp | — | — | — | — | — | — | — | — | 76.2 (143) | 94.8 (747) | — |
| congress-house-114 | signed_vote | — | — | — | — | — | — | — | — | 90.9 (143) | 98.1 (747) | — |
| congress-house-114 | signed_lgc | — | — | — | — | — | — | — | — | 90.9 (143) | 98.1 (747) | — |
| congress-house-93 | EDSP-default | — | — | — | — | — | 21.6 (74) | 69.2 (393) | 81.7 (328) | 100.0 (72) | 100.0 (23) | — |
| congress-house-93 | EDSP-tuned | — | — | — | — | — | 83.8 (74) | 76.3 (393) | 79.3 (328) | 98.6 (72) | 100.0 (23) | — |
| congress-house-93 | harmonic_cmn | — | — | — | — | — | 89.2 (74) | 67.4 (393) | 68.0 (328) | 41.7 (72) | 0.0 (23) | — |
| congress-house-93 | lgc | — | — | — | — | — | 89.2 (74) | 64.4 (393) | 60.1 (328) | 88.9 (72) | 73.9 (23) | — |
| congress-house-93 | ppr_lp | — | — | — | — | — | 89.2 (74) | 57.8 (393) | 50.3 (328) | 97.2 (72) | 95.7 (23) | — |
| congress-house-93 | linbp | — | — | — | — | — | 58.1 (74) | 53.7 (393) | 49.1 (328) | 83.3 (72) | 91.3 (23) | — |
| congress-house-93 | signed_vote | — | — | — | — | — | 25.7 (74) | 71.0 (393) | 84.5 (328) | 98.6 (72) | 95.7 (23) | — |
| congress-house-93 | signed_lgc | — | — | — | — | — | 31.1 (74) | 76.6 (393) | 86.6 (328) | 100.0 (72) | 95.7 (23) | — |
| congress-senate-100 | EDSP-default | — | — | — | — | 66.7 (6) | 83.3 (48) | 93.4 (121) | 76.0 (25) | — | — | — |
| congress-senate-100 | EDSP-tuned | — | — | — | — | 66.7 (6) | 83.3 (48) | 94.2 (121) | 76.0 (25) | — | — | — |
| congress-senate-100 | harmonic_cmn | — | — | — | — | 0.0 (6) | 68.8 (48) | 44.6 (121) | 52.0 (25) | — | — | — |
| congress-senate-100 | lgc | — | — | — | — | 0.0 (6) | 72.9 (48) | 47.9 (121) | 56.0 (25) | — | — | — |
| congress-senate-100 | ppr_lp | — | — | — | — | 0.0 (6) | 75.0 (48) | 44.6 (121) | 48.0 (25) | — | — | — |
| congress-senate-100 | linbp | — | — | — | — | 100.0 (6) | 52.1 (48) | 62.8 (121) | 72.0 (25) | — | — | — |
| congress-senate-100 | signed_vote | — | — | — | — | 66.7 (6) | 81.2 (48) | 94.2 (121) | 76.0 (25) | — | — | — |
| congress-senate-100 | signed_lgc | — | — | — | — | 66.7 (6) | 83.3 (48) | 93.4 (121) | 76.0 (25) | — | — | — |
| congress-senate-107 | EDSP-default | — | — | — | — | — | 91.3 (23) | 96.9 (129) | 100.0 (38) | — | — | — |
| congress-senate-107 | EDSP-tuned | — | — | — | — | — | 91.3 (23) | 96.9 (129) | 100.0 (38) | — | — | — |
| congress-senate-107 | harmonic_cmn | — | — | — | — | — | 47.8 (23) | 26.4 (129) | 7.9 (38) | — | — | — |
| congress-senate-107 | lgc | — | — | — | — | — | 69.6 (23) | 41.1 (129) | 78.9 (38) | — | — | — |
| congress-senate-107 | ppr_lp | — | — | — | — | — | 60.9 (23) | 41.1 (129) | 78.9 (38) | — | — | — |
| congress-senate-107 | linbp | — | — | — | — | — | 56.5 (23) | 78.3 (129) | 97.4 (38) | — | — | — |
| congress-senate-107 | signed_vote | — | — | — | — | — | 95.7 (23) | 96.9 (129) | 100.0 (38) | — | — | — |
| congress-senate-107 | signed_lgc | — | — | — | — | — | 95.7 (23) | 96.9 (129) | 100.0 (38) | — | — | — |
| congress-senate-114 | EDSP-default | — | — | — | — | — | — | 87.5 (56) | 100.0 (144) | — | — | — |
| congress-senate-114 | EDSP-tuned | — | — | — | — | — | — | 87.5 (56) | 99.3 (144) | — | — | — |
| congress-senate-114 | harmonic_cmn | — | — | — | — | — | — | 23.2 (56) | 41.0 (144) | — | — | — |
| congress-senate-114 | lgc | — | — | — | — | — | — | 57.1 (56) | 52.8 (144) | — | — | — |
| congress-senate-114 | ppr_lp | — | — | — | — | — | — | 53.6 (56) | 52.8 (144) | — | — | — |
| congress-senate-114 | linbp | — | — | — | — | — | — | 87.5 (56) | 81.2 (144) | — | — | — |
| congress-senate-114 | signed_vote | — | — | — | — | — | — | 91.1 (56) | 100.0 (144) | — | — | — |
| congress-senate-114 | signed_lgc | — | — | — | — | — | — | 91.1 (56) | 100.0 (144) | — | — | — |
| congress-senate-93 | EDSP-default | — | — | — | — | 100.0 (2) | 81.0 (42) | 77.0 (122) | 88.6 (44) | — | — | — |
| congress-senate-93 | EDSP-tuned | — | — | — | — | 100.0 (2) | 81.0 (42) | 77.0 (122) | 88.6 (44) | — | — | — |
| congress-senate-93 | harmonic_cmn | — | — | — | — | 0.0 (2) | 69.0 (42) | 54.9 (122) | 40.9 (44) | — | — | — |
| congress-senate-93 | lgc | — | — | — | — | 0.0 (2) | 64.3 (42) | 54.1 (122) | 40.9 (44) | — | — | — |
| congress-senate-93 | ppr_lp | — | — | — | — | 0.0 (2) | 64.3 (42) | 54.1 (122) | 40.9 (44) | — | — | — |
| congress-senate-93 | linbp | — | — | — | — | 50.0 (2) | 54.8 (42) | 56.6 (122) | 88.6 (44) | — | — | — |
| congress-senate-93 | signed_vote | — | — | — | — | 100.0 (2) | 85.7 (42) | 77.9 (122) | 88.6 (44) | — | — | — |
| congress-senate-93 | signed_lgc | — | — | — | — | 100.0 (2) | 85.7 (42) | 77.9 (122) | 88.6 (44) | — | — | — |
