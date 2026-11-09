# E3 — Track B (feature priors): EDSP-B vs logistic regression, Correct & Smooth and tuned GNNs, H4

Run records: 220 (git 092aee8e1a7e, 2aff2f69014f, 48685088d25d, 6eece72134df, 7cb3edd54b81, 872886226084; runners actions-run-42477345347, actions-run-42480358577, actions-run-42480364987, actions-run-42490821424, actions-run-42498139858, actions-run-42498145946, actions-run-42498152359, actions-run-42506763207, actions-run-42506769170, actions-run-42506775076, actions-run-42514179439, actions-run-42520540110, actions-run-42520547071). Headline metric per dataset in brackets; mean ± std over splits (×100). Rows marked † are constructed benchmarks or case studies and are excluded from real-world aggregates.

| dataset [metric] | splits | EDSP-default | EDSP-tuned | majority | train_prior | feature_lr | correct_and_smooth | mlp | gcn | sage | gat | gprgnn | bwgnn | gat(h64) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coauthor-cs [accuracy] | 10 | 95.83 ± 0.27 | 95.78 ± 0.22 | 22.56 ± 0.00 | 22.56 ± 0.00 | 94.72 ± 0.23 | 95.83 ± 0.21 | 95.40 ± 0.22 | 95.52 ± 0.20 | 95.38 ± 0.38 | 95.93 ± 0.30 | **95.97 ± 0.20** | 95.88 ± 0.29 | — |
| coauthor-physics [accuracy] | 10 | 97.02 ± 0.22 | 97.03 ± 0.22 | 50.52 ± 0.00 | 50.52 ± 0.00 | 96.25 ± 0.19 | 97.21 ± 0.23 | 96.98 ± 0.23 | 97.12 ± 0.17 | — | — | **97.37 ± 0.21** | 97.31 ± 0.24 | — |

## Pre-registered secondary column: EDSP-default on non-converged runs

| dataset | non-converged runs | headline, last iterate | headline, mean of last two iterates |
|---|---|---|---|
| coauthor-cs | 0 / 10 | — | — |
| coauthor-physics | 0 / 10 | — | — |

## EDSP-tuned vs each baseline: win / tie / loss

Per dataset: two-sided Wilcoxon signed-rank over splits on the headline metric; Holm correction across datasets per baseline; win/loss if adjusted p < 0.05, else tie. Real-world datasets only.

| baseline | win | tie | loss | datasets (W / L) |
|---|---|---|---|---|
| feature_lr | 2 | 0 | 0 | W: coauthor-cs, coauthor-physics; L: — |
| correct_and_smooth | 0 | 1 | 1 | W: —; L: coauthor-physics |
| mlp | 1 | 1 | 0 | W: coauthor-cs; L: — |
| gcn | 1 | 1 | 0 | W: coauthor-cs; L: — |
| sage | 1 | 0 | 0 | W: coauthor-cs; L: — |
| gat | 0 | 1 | 0 | W: —; L: — |
| gprgnn | 0 | 0 | 2 | W: —; L: coauthor-cs, coauthor-physics |
| bwgnn | 0 | 1 | 1 | W: —; L: coauthor-physics |

## Calibration (mean over splits; lower is better)

| dataset | EDSP-default NLL | EDSP-default ECE | EDSP-tuned NLL | EDSP-tuned ECE | best baseline by NLL (NLL / ECE) |
|---|---|---|---|---|---|
| coauthor-cs | 0.615 | 0.384 | 0.612 | 0.379 | gat: 0.131 / 0.011 |
| coauthor-physics | 0.459 | 0.317 | 0.459 | 0.316 | gcn: 0.087 / 0.008 |

## H7 (descriptive): convergence and fixed-point non-uniqueness

| dataset | EDSP-default non-converged | oscillation flagged | mean fraction of nodes still moving | GS vs damped L1: mean | p95 | splits |
|---|---|---|---|---|---|---|
| coauthor-cs | 0.000 | 0.000 | 0.000 | — | — | 0 |
| coauthor-physics | 0.000 | 0.000 | 0.000 | — | — | 0 |

## Test accuracy by local homophily (descriptive, full labels; bin i = [i/10, (i+1)/10)), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| coauthor-cs | EDSP-default | 59.3 (783) | 52.2 (247) | 67.9 (607) | 78.7 (847) | 84.1 (787) | 90.3 (2196) | 94.6 (2763) | 97.0 (2811) | 98.8 (4718) | 99.6 (20891) |
| coauthor-cs | EDSP-tuned | 63.3 (783) | 53.8 (247) | 68.9 (607) | 79.1 (847) | 84.1 (787) | 90.2 (2196) | 93.9 (2763) | 96.8 (2811) | 98.7 (4718) | 99.4 (20891) |
| coauthor-cs | feature_lr | 75.6 (783) | 59.5 (247) | 72.8 (607) | 81.1 (847) | 84.6 (787) | 89.0 (2196) | 91.6 (2763) | 94.2 (2811) | 97.2 (4718) | 97.9 (20891) |
| coauthor-cs | correct_and_smooth | 66.4 (783) | 50.2 (247) | 68.0 (607) | 80.4 (847) | 85.0 (787) | 91.2 (2196) | 93.6 (2763) | 96.7 (2811) | 98.6 (4718) | 99.3 (20891) |
| coauthor-cs | gcn | 48.3 (783) | 39.7 (247) | 57.8 (607) | 71.8 (847) | 81.6 (787) | 90.7 (2196) | 95.3 (2763) | 98.3 (2811) | 99.4 (4718) | 99.8 (20891) |
| coauthor-cs | sage | 45.2 (783) | 45.3 (247) | 62.1 (607) | 73.0 (847) | 80.6 (787) | 89.5 (2196) | 94.5 (2763) | 97.9 (2811) | 99.1 (4718) | 99.8 (20891) |
| coauthor-cs | gat | 52.0 (783) | 44.9 (247) | 61.9 (607) | 75.6 (847) | 83.2 (787) | 92.1 (2196) | 96.1 (2763) | 98.3 (2811) | 99.5 (4718) | 99.7 (20891) |
| coauthor-cs | gprgnn | 58.4 (783) | 44.1 (247) | 63.3 (607) | 75.9 (847) | 83.7 (787) | 91.6 (2196) | 95.1 (2763) | 98.0 (2811) | 99.4 (4718) | 99.7 (20891) |
| coauthor-physics | EDSP-default | 41.8 (517) | 29.3 (191) | 55.8 (464) | 54.6 (692) | 69.9 (748) | 80.8 (1854) | 92.7 (2473) | 96.0 (2997) | 98.6 (6116) | 99.7 (52928) |
| coauthor-physics | EDSP-tuned | 48.2 (517) | 29.8 (191) | 58.8 (464) | 57.4 (692) | 71.3 (748) | 81.8 (1854) | 92.2 (2473) | 95.6 (2997) | 98.4 (6116) | 99.6 (52928) |
| coauthor-physics | feature_lr | 60.5 (517) | 38.7 (191) | 68.1 (464) | 68.4 (692) | 75.4 (748) | 82.3 (1854) | 89.0 (2473) | 93.1 (2997) | 96.1 (6116) | 98.7 (52928) |
| coauthor-physics | correct_and_smooth | 51.5 (517) | 28.8 (191) | 60.8 (464) | 62.3 (692) | 75.4 (748) | 83.9 (1854) | 92.1 (2473) | 95.8 (2997) | 98.4 (6116) | 99.6 (52928) |
| coauthor-physics | gcn | 27.9 (517) | 15.7 (191) | 39.9 (464) | 52.7 (692) | 68.4 (748) | 82.0 (1854) | 94.3 (2473) | 97.9 (2997) | 99.6 (6116) | 99.9 (52928) |
| coauthor-physics | gprgnn | 36.0 (517) | 26.2 (191) | 48.5 (464) | 61.4 (692) | 72.5 (748) | 85.2 (1854) | 94.0 (2473) | 97.3 (2997) | 99.1 (6116) | 99.9 (52928) |

## Test accuracy by degree (bin 0: isolated; bin b: 2^(b-1) ≤ deg < 2^b; last bin open), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coauthor-cs | EDSP-default | — | 93.9 (2099) | 95.9 (7147) | 96.4 (12223) | 95.4 (10178) | 96.2 (3943) | 95.6 (955) | 98.1 (104) | 100.0 (1) | — | — |
| coauthor-cs | EDSP-tuned | — | 94.3 (2099) | 95.8 (7147) | 96.3 (12223) | 95.3 (10178) | 96.1 (3943) | 95.6 (955) | 98.1 (104) | 100.0 (1) | — | — |
| coauthor-cs | feature_lr | — | 93.6 (2099) | 94.3 (7147) | 94.8 (12223) | 94.6 (10178) | 95.7 (3943) | 95.8 (955) | 98.1 (104) | 100.0 (1) | — | — |
| coauthor-cs | correct_and_smooth | — | 95.0 (2099) | 96.0 (7147) | 96.1 (12223) | 95.5 (10178) | 96.1 (3943) | 95.2 (955) | 99.0 (104) | 100.0 (1) | — | — |
| coauthor-cs | gcn | — | 93.1 (2099) | 95.8 (7147) | 95.8 (12223) | 95.1 (10178) | 96.5 (3943) | 96.3 (955) | 97.1 (104) | 100.0 (1) | — | — |
| coauthor-cs | sage | — | 91.7 (2099) | 95.7 (7147) | 95.7 (12223) | 95.0 (10178) | 96.3 (3943) | 96.5 (955) | 98.1 (104) | 100.0 (1) | — | — |
| coauthor-cs | gat | — | 93.3 (2099) | 96.3 (7147) | 96.3 (12223) | 95.5 (10178) | 96.6 (3943) | 96.2 (955) | 98.1 (104) | 100.0 (1) | — | — |
| coauthor-cs | gprgnn | — | 94.3 (2099) | 96.3 (7147) | 96.3 (12223) | 95.3 (10178) | 96.6 (3943) | 96.8 (955) | 96.2 (104) | 100.0 (1) | — | — |
| coauthor-physics | EDSP-default | — | 93.1 (1713) | 95.0 (7048) | 96.4 (17135) | 97.2 (22165) | 98.0 (14953) | 99.2 (4844) | 99.2 (1003) | 97.4 (114) | 100.0 (5) | — |
| coauthor-physics | EDSP-tuned | — | 93.2 (1713) | 95.1 (7048) | 96.4 (17135) | 97.1 (22165) | 98.0 (14953) | 99.2 (4844) | 99.2 (1003) | 97.4 (114) | 100.0 (5) | — |
| coauthor-physics | feature_lr | — | 91.9 (1713) | 93.9 (7048) | 95.1 (17135) | 96.4 (22165) | 97.9 (14953) | 99.1 (4844) | 99.2 (1003) | 97.4 (114) | 100.0 (5) | — |
| coauthor-physics | correct_and_smooth | — | 93.9 (1713) | 95.7 (7048) | 96.6 (17135) | 97.3 (22165) | 98.1 (14953) | 99.3 (4844) | 99.2 (1003) | 97.4 (114) | 100.0 (5) | — |
| coauthor-physics | gcn | — | 93.1 (1713) | 95.2 (7048) | 96.4 (17135) | 97.2 (22165) | 98.3 (14953) | 99.3 (4844) | 99.3 (1003) | 97.4 (114) | 100.0 (5) | — |
| coauthor-physics | gprgnn | — | 93.5 (1713) | 96.0 (7048) | 96.9 (17135) | 97.4 (22165) | 98.3 (14953) | 99.2 (4844) | 99.5 (1003) | 99.1 (114) | 100.0 (5) | — |

## Cell C3c (mixed by local structure: top-5 std of local homophily)

| dataset | std local homophily | min compat. distance | EDSP-default | EDSP-tuned | feature_lr | correct_and_smooth | mlp | gcn | sage | gat | gprgnn | bwgnn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coauthor-cs | 0.239 | 1.056 | 95.83 | 95.78 | 94.72 | 95.83 | 95.40 | 95.52 | 95.38 | 95.93 | **95.97** | 95.88 |
| coauthor-physics | 0.174 | 0.370 | 97.02 | 97.03 | 96.25 | 97.21 | 96.98 | 97.12 | — | — | **97.37** | 97.31 |

## Cell C3d (structured compatibility: top-5 min Frobenius distance to I and to (J-I)/(k-1))

| dataset | std local homophily | min compat. distance | EDSP-default | EDSP-tuned | feature_lr | correct_and_smooth | mlp | gcn | sage | gat | gprgnn | bwgnn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coauthor-cs | 0.239 | 1.056 | 95.83 | 95.78 | 94.72 | 95.83 | 95.40 | 95.52 | 95.38 | 95.93 | **95.97** | 95.88 |
| coauthor-physics | 0.174 | 0.370 | 97.02 | 97.03 | 96.25 | 97.21 | 96.98 | 97.12 | — | — | **97.37** | 97.31 |
