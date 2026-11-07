# E3 — Track B (feature priors): EDSP-B vs logistic regression, Correct & Smooth and tuned GNNs, H4

Run records: 172 (git e9dd5ae507fe; runners actions-run-42368756508). Headline metric per dataset in brackets; mean ± std over splits (×100). Rows marked † are constructed benchmarks or case studies and are excluded from real-world aggregates.

| dataset [metric] | splits | EDSP-default | EDSP-tuned | majority | train_prior | feature_lr | correct_and_smooth | mlp | gcn | sage | gat | gprgnn | bwgnn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| amazon-computers [accuracy] | 5 | **88.87 ± 0.59** | — | — | — | — | — | — | — | — | — | — | — |
| amazon-photo [accuracy] | 4 | 94.60 ± 0.37 | **94.87 ± 0.16** | 25.38 ± 0.00 | 25.38 ± 0.00 | — | — | — | — | — | — | — | — |
| citeseer [accuracy] | 8 | 75.89 ± 0.77 | 75.87 ± 1.09 | 21.18 ± 0.00 | 21.18 ± 0.00 | 72.89 ± 0.98 | **77.65 ± 0.60** | — | — | — | — | — | — |
| cora [accuracy] | 7 | 84.14 ± 5.82 | 84.48 ± 6.19 | 27.73 ± 6.01 | 27.73 ± 6.01 | 73.35 ± 7.11 | **87.72 ± 4.95** | 67.75 ± 10.95 | — | — | — | — | — |
| pubmed [accuracy] | 7 | 88.91 ± 0.43 | 89.31 ± 0.33 | 39.94 ± 0.00 | 39.94 ± 0.00 | 87.54 ± 0.34 | **90.04 ± 0.46** | — | — | — | — | — | — |
| wikics [accuracy] | 6 | 76.78 ± 0.60 | 76.89 ± 0.63 | 22.90 ± 0.00 | 22.90 ± 0.00 | 72.37 ± 0.90 | **79.04 ± 0.41** | — | — | — | — | — | — |

## Pre-registered secondary column: EDSP-default on non-converged runs

| dataset | non-converged runs | headline, last iterate | headline, mean of last two iterates |
|---|---|---|---|
| amazon-computers | 0 / 5 | — | — |
| amazon-photo | 0 / 4 | — | — |
| citeseer | 0 / 8 | — | — |
| cora | 1 / 7 | 70.10 | 70.10 |
| pubmed | 0 / 7 | — | — |
| wikics | 0 / 6 | — | — |

## EDSP-tuned vs each baseline: win / tie / loss

Per dataset: two-sided Wilcoxon signed-rank over splits on the headline metric; Holm correction across datasets per baseline; win/loss if adjusted p < 0.05, else tie. Real-world datasets only.

| baseline | win | tie | loss | datasets (W / L) |
|---|---|---|---|---|
| feature_lr | 0 | 4 | 0 | W: —; L: — |
| correct_and_smooth | 0 | 4 | 0 | W: —; L: — |
| mlp | 0 | 1 | 0 | W: —; L: — |
| gcn | 0 | 0 | 0 | W: —; L: — |
| sage | 0 | 0 | 0 | W: —; L: — |
| gat | 0 | 0 | 0 | W: —; L: — |
| gprgnn | 0 | 0 | 0 | W: —; L: — |
| bwgnn | 0 | 0 | 0 | W: —; L: — |

## Calibration (mean over splits; lower is better)

| dataset | EDSP-default NLL | EDSP-default ECE | EDSP-tuned NLL | EDSP-tuned ECE | best baseline by NLL (NLL / ECE) |
|---|---|---|---|---|---|
| amazon-computers | 0.814 | 0.380 | nan | nan | feature_lr: nan / nan |
| amazon-photo | 0.615 | 0.367 | 0.607 | 0.366 | majority: 20.619 / 0.746 |
| citeseer | 1.018 | 0.288 | 1.030 | 0.295 | feature_lr: 0.847 / 0.059 |
| cora | 1.014 | 0.402 | 1.011 | 0.404 | correct_and_smooth: 0.538 / 0.193 |
| pubmed | 0.554 | 0.262 | 0.554 | 0.268 | correct_and_smooth: 0.304 / 0.056 |
| wikics | 1.164 | 0.361 | 1.168 | 0.363 | feature_lr: 0.834 / 0.025 |

## H7 (descriptive): convergence and fixed-point non-uniqueness

| dataset | EDSP-default non-converged | oscillation flagged | mean fraction of nodes still moving | GS vs damped L1: mean | p95 | splits |
|---|---|---|---|---|---|---|
| amazon-computers | 0.000 | 0.000 | 0.000 | — | — | 0 |
| amazon-photo | 0.000 | 0.000 | 0.000 | — | — | 0 |
| citeseer | 0.000 | 0.000 | 0.000 | — | — | 0 |
| cora | 0.143 | 0.000 | 0.000 | — | — | 0 |
| pubmed | 0.000 | 0.000 | 0.000 | — | — | 0 |
| wikics | 0.000 | 0.000 | 0.000 | — | — | 0 |

## Test accuracy by local homophily (descriptive, full labels; bin i = [i/10, (i+1)/10)), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| amazon-photo | EDSP-default | 34.4 (96) | 62.7 (67) | 53.8 (117) | 65.8 (111) | 85.1 (121) | 86.9 (274) | 94.0 (381) | 96.9 (447) | 98.9 (741) | 99.3 (3669) |
| amazon-photo | EDSP-tuned | 46.8 (47) | 60.0 (35) | 60.3 (68) | 64.7 (51) | 91.8 (73) | 87.2 (117) | 93.6 (203) | 96.4 (223) | 98.7 (379) | 99.1 (1828) |
| citeseer | EDSP-default | 21.0 (917) | 4.8 (21) | 25.5 (51) | 44.8 (116) | 69.0 (29) | 67.1 (431) | 86.3 (277) | 94.7 (131) | 96.4 (138) | 94.0 (3091) |
| citeseer | EDSP-tuned | 24.7 (695) | 7.1 (14) | 32.6 (43) | 53.7 (82) | 72.0 (25) | 66.9 (329) | 81.6 (196) | 93.8 (97) | 94.6 (111) | 93.3 (2301) |
| citeseer | feature_lr | 34.2 (584) | 7.7 (13) | 51.4 (35) | 62.5 (72) | 77.3 (22) | 65.8 (260) | 78.4 (162) | 83.8 (80) | 85.4 (89) | 86.0 (1922) |
| citeseer | correct_and_smooth | 20.0 (469) | 0.0 (9) | 30.8 (26) | 66.1 (56) | 89.5 (19) | 68.4 (212) | 86.5 (126) | 93.8 (64) | 98.5 (66) | 96.3 (1547) |
| cora | EDSP-default | 31.4 (236) | 18.8 (16) | 28.2 (71) | 47.8 (115) | 46.3 (41) | 62.7 (295) | 77.9 (280) | 87.0 (185) | 88.0 (209) | 92.6 (2792) |
| cora | EDSP-tuned | 30.5 (236) | 18.8 (16) | 31.0 (71) | 48.7 (115) | 46.3 (41) | 64.4 (295) | 78.2 (280) | 87.0 (185) | 88.5 (209) | 92.6 (2792) |
| cora | feature_lr | 49.6 (236) | 43.8 (16) | 32.4 (71) | 45.2 (115) | 61.0 (41) | 58.6 (295) | 60.0 (280) | 71.4 (185) | 65.6 (209) | 78.7 (2792) |
| cora | correct_and_smooth | 23.3 (236) | 12.5 (16) | 35.2 (71) | 45.2 (115) | 58.5 (41) | 68.1 (295) | 89.3 (280) | 93.0 (185) | 96.7 (209) | 96.0 (2792) |
| pubmed | EDSP-default | 52.7 (3544) | 23.8 (122) | 41.0 (234) | 56.3 (375) | 69.0 (213) | 82.1 (1487) | 88.9 (943) | 95.9 (789) | 98.3 (1291) | 97.3 (18603) |
| pubmed | EDSP-tuned | 59.1 (3544) | 25.4 (122) | 45.3 (234) | 58.4 (375) | 71.4 (213) | 83.0 (1487) | 89.3 (943) | 94.7 (789) | 98.1 (1291) | 96.5 (18603) |
| pubmed | feature_lr | 69.9 (3544) | 40.2 (122) | 54.7 (234) | 65.3 (375) | 74.6 (213) | 80.7 (1487) | 85.5 (943) | 88.7 (789) | 92.8 (1291) | 92.5 (18603) |
| pubmed | correct_and_smooth | 57.4 (3544) | 13.1 (122) | 41.0 (234) | 59.5 (375) | 73.2 (213) | 84.3 (1487) | 91.4 (943) | 96.7 (789) | 99.5 (1291) | 97.6 (18603) |
| wikics | EDSP-default | 17.9 (1704) | 20.3 (1272) | 28.6 (1704) | 50.3 (1854) | 65.7 (1884) | 73.2 (3486) | 86.2 (3810) | 91.4 (3762) | 94.9 (4308) | 94.7 (10248) |
| wikics | EDSP-tuned | 17.8 (1704) | 21.1 (1272) | 29.6 (1704) | 51.0 (1854) | 65.9 (1884) | 73.5 (3486) | 86.2 (3810) | 91.2 (3762) | 94.8 (4308) | 94.6 (10248) |
| wikics | feature_lr | 28.9 (1704) | 29.5 (1272) | 38.1 (1704) | 54.1 (1854) | 60.3 (1884) | 67.3 (3486) | 77.4 (3810) | 83.1 (3762) | 86.7 (4308) | 87.1 (10248) |
| wikics | correct_and_smooth | 18.3 (1704) | 16.4 (1272) | 30.9 (1704) | 54.2 (1854) | 70.0 (1884) | 77.2 (3486) | 89.8 (3810) | 94.9 (3762) | 97.5 (4308) | 95.9 (10248) |

## Test accuracy by degree (bin 0: isolated; bin b: 2^(b-1) ≤ deg < 2^b; last bin open), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| amazon-photo | EDSP-default | 71.7 (92) | 84.7 (131) | 85.5 (345) | 92.1 (674) | 94.7 (1104) | 96.5 (1664) | 98.1 (1559) | 95.7 (414) | 92.2 (103) | 72.7 (22) | 62.5 (8) |
| amazon-photo | EDSP-tuned | 73.5 (34) | 81.4 (70) | 84.1 (164) | 91.3 (310) | 94.5 (547) | 97.3 (888) | 98.4 (762) | 96.7 (212) | 91.8 (61) | 66.7 (6) | 50.0 (4) |
| citeseer | EDSP-default | 52.3 (86) | 71.1 (2076) | 77.3 (1977) | 83.8 (874) | 83.3 (227) | 89.1 (46) | 100.0 (1) | 100.0 (1) | — | — | — |
| citeseer | EDSP-tuned | 52.1 (73) | 71.0 (1556) | 77.7 (1468) | 83.1 (670) | 82.7 (162) | 91.4 (35) | 100.0 (1) | 100.0 (1) | — | — | — |
| citeseer | feature_lr | 51.5 (66) | 67.2 (1309) | 75.7 (1211) | 80.0 (551) | 80.4 (138) | 85.7 (28) | 100.0 (1) | 100.0 (1) | — | — | — |
| citeseer | correct_and_smooth | 56.0 (50) | 73.4 (1075) | 78.9 (952) | 85.5 (435) | 84.0 (106) | 91.7 (24) | 100.0 (1) | 100.0 (1) | — | — | — |
| cora | EDSP-default | — | 81.2 (729) | 81.7 (1784) | 84.6 (1418) | 80.1 (241) | 85.7 (49) | 91.7 (12) | 100.0 (6) | 100.0 (1) | — | — |
| cora | EDSP-tuned | — | 81.1 (729) | 82.2 (1784) | 84.8 (1418) | 80.5 (241) | 85.7 (49) | 91.7 (12) | 100.0 (6) | 100.0 (1) | — | — |
| cora | feature_lr | — | 72.3 (729) | 70.7 (1784) | 72.8 (1418) | 65.1 (241) | 71.4 (49) | 91.7 (12) | 83.3 (6) | 100.0 (1) | — | — |
| cora | correct_and_smooth | — | 83.4 (729) | 85.4 (1784) | 88.4 (1418) | 90.5 (241) | 83.7 (49) | 100.0 (12) | 100.0 (6) | 100.0 (1) | — | — |
| pubmed | EDSP-default | — | 88.4 (12705) | 89.3 (6974) | 89.8 (3434) | 89.1 (2669) | 88.4 (1419) | 91.9 (347) | 79.5 (44) | 100.0 (9) | — | — |
| pubmed | EDSP-tuned | — | 89.0 (12705) | 89.8 (6974) | 89.7 (3434) | 88.9 (2669) | 88.7 (1419) | 91.9 (347) | 79.5 (44) | 100.0 (9) | — | — |
| pubmed | feature_lr | — | 87.6 (12705) | 88.3 (6974) | 87.3 (3434) | 85.6 (2669) | 87.2 (1419) | 89.0 (347) | 81.8 (44) | 100.0 (9) | — | — |
| pubmed | correct_and_smooth | — | 89.6 (12705) | 90.8 (6974) | 90.4 (3434) | 89.5 (2669) | 90.1 (1419) | 92.2 (347) | 81.8 (44) | 100.0 (9) | — | — |
| wikics | EDSP-default | 62.0 (1050) | 76.3 (1860) | 73.7 (4494) | 77.0 (5790) | 74.4 (6090) | 74.4 (4338) | 80.9 (4638) | 81.4 (4542) | 83.4 (1884) | 79.5 (336) | 90.0 (60) |
| wikics | EDSP-tuned | 62.0 (1050) | 76.6 (1860) | 73.8 (4494) | 77.3 (5790) | 74.6 (6090) | 74.6 (4338) | 80.8 (4638) | 81.3 (4542) | 83.2 (1884) | 79.8 (336) | 90.0 (60) |
| wikics | feature_lr | 62.0 (1050) | 69.5 (1860) | 69.8 (4494) | 72.5 (5790) | 70.8 (6090) | 70.9 (4338) | 74.8 (4638) | 76.2 (4542) | 78.2 (1884) | 79.8 (336) | 90.0 (60) |
| wikics | correct_and_smooth | 62.0 (1050) | 78.2 (1860) | 76.6 (4494) | 80.0 (5790) | 77.1 (6090) | 77.4 (4338) | 82.0 (4638) | 83.1 (4542) | 85.9 (1884) | 73.5 (336) | 90.0 (60) |

## Cell C3c (mixed by local structure: top-5 std of local homophily)

| dataset | std local homophily | min compat. distance | EDSP-default | EDSP-tuned | feature_lr | correct_and_smooth | mlp | gcn | sage | gat | gprgnn | bwgnn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cora | 0.293 | 0.623 | 84.14 | 84.48 | 73.35 | **87.72** | 67.75 | — | — | — | — | — |
| wikics | 0.291 | 1.426 | 76.78 | 76.89 | 72.37 | **79.04** | — | — | — | — | — | — |
| amazon-photo | 0.228 | 0.897 | 94.60 | **94.87** | — | — | — | — | — | — | — | — |

## Cell C3d (structured compatibility: top-5 min Frobenius distance to I and to (J-I)/(k-1))

| dataset | std local homophily | min compat. distance | EDSP-default | EDSP-tuned | feature_lr | correct_and_smooth | mlp | gcn | sage | gat | gprgnn | bwgnn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wikics | 0.291 | 1.426 | 76.78 | 76.89 | 72.37 | **79.04** | — | — | — | — | — | — |
| amazon-photo | 0.228 | 0.897 | 94.60 | **94.87** | — | — | — | — | — | — | — | — |
| cora | 0.293 | 0.623 | 84.14 | 84.48 | 73.35 | **87.72** | 67.75 | — | — | — | — | — |
