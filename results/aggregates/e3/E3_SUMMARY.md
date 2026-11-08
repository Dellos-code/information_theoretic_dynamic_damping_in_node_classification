# E3 — Track B (feature priors): EDSP-B vs logistic regression, Correct & Smooth and tuned GNNs, H4

Run records: 70 (git 48685088d25d; runners actions-run-42498139858). Headline metric per dataset in brackets; mean ± std over splits (×100). Rows marked † are constructed benchmarks or case studies and are excluded from real-world aggregates.

| dataset [metric] | splits | EDSP-default | EDSP-tuned | majority | train_prior | feature_lr | correct_and_smooth | mlp | gcn | sage | gat | gprgnn | bwgnn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coauthor-physics [accuracy] | 10 | 97.02 ± 0.22 | 97.03 ± 0.22 | 50.52 ± 0.00 | 50.52 ± 0.00 | 96.25 ± 0.19 | **97.21 ± 0.23** | 96.98 ± 0.23 | — | — | — | — | — |

## Pre-registered secondary column: EDSP-default on non-converged runs

| dataset | non-converged runs | headline, last iterate | headline, mean of last two iterates |
|---|---|---|---|
| coauthor-physics | 0 / 10 | — | — |

## EDSP-tuned vs each baseline: win / tie / loss

Per dataset: two-sided Wilcoxon signed-rank over splits on the headline metric; Holm correction across datasets per baseline; win/loss if adjusted p < 0.05, else tie. Real-world datasets only.

| baseline | win | tie | loss | datasets (W / L) |
|---|---|---|---|---|
| feature_lr | 1 | 0 | 0 | W: coauthor-physics; L: — |
| correct_and_smooth | 0 | 0 | 1 | W: —; L: coauthor-physics |
| mlp | 0 | 1 | 0 | W: —; L: — |
| gcn | 0 | 0 | 0 | W: —; L: — |
| sage | 0 | 0 | 0 | W: —; L: — |
| gat | 0 | 0 | 0 | W: —; L: — |
| gprgnn | 0 | 0 | 0 | W: —; L: — |
| bwgnn | 0 | 0 | 0 | W: —; L: — |

## Calibration (mean over splits; lower is better)

| dataset | EDSP-default NLL | EDSP-default ECE | EDSP-tuned NLL | EDSP-tuned ECE | best baseline by NLL (NLL / ECE) |
|---|---|---|---|---|---|
| coauthor-physics | 0.459 | 0.317 | 0.459 | 0.316 | mlp: 0.096 / 0.012 |

## H7 (descriptive): convergence and fixed-point non-uniqueness

| dataset | EDSP-default non-converged | oscillation flagged | mean fraction of nodes still moving | GS vs damped L1: mean | p95 | splits |
|---|---|---|---|---|---|---|
| coauthor-physics | 0.000 | 0.000 | 0.000 | — | — | 0 |

## Test accuracy by local homophily (descriptive, full labels; bin i = [i/10, (i+1)/10)), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| coauthor-physics | EDSP-default | 41.8 (517) | 29.3 (191) | 55.8 (464) | 54.6 (692) | 69.9 (748) | 80.8 (1854) | 92.7 (2473) | 96.0 (2997) | 98.6 (6116) | 99.7 (52928) |
| coauthor-physics | EDSP-tuned | 48.2 (517) | 29.8 (191) | 58.8 (464) | 57.4 (692) | 71.3 (748) | 81.8 (1854) | 92.2 (2473) | 95.6 (2997) | 98.4 (6116) | 99.6 (52928) |
| coauthor-physics | feature_lr | 60.5 (517) | 38.7 (191) | 68.1 (464) | 68.4 (692) | 75.4 (748) | 82.3 (1854) | 89.0 (2473) | 93.1 (2997) | 96.1 (6116) | 98.7 (52928) |
| coauthor-physics | correct_and_smooth | 51.5 (517) | 28.8 (191) | 60.8 (464) | 62.3 (692) | 75.4 (748) | 83.9 (1854) | 92.1 (2473) | 95.8 (2997) | 98.4 (6116) | 99.6 (52928) |

## Test accuracy by degree (bin 0: isolated; bin b: 2^(b-1) ≤ deg < 2^b; last bin open), pooled over splits (×100)

Cell: accuracy (number of test nodes). Bins are never method inputs.

| dataset | method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coauthor-physics | EDSP-default | — | 93.1 (1713) | 95.0 (7048) | 96.4 (17135) | 97.2 (22165) | 98.0 (14953) | 99.2 (4844) | 99.2 (1003) | 97.4 (114) | 100.0 (5) | — |
| coauthor-physics | EDSP-tuned | — | 93.2 (1713) | 95.1 (7048) | 96.4 (17135) | 97.1 (22165) | 98.0 (14953) | 99.2 (4844) | 99.2 (1003) | 97.4 (114) | 100.0 (5) | — |
| coauthor-physics | feature_lr | — | 91.9 (1713) | 93.9 (7048) | 95.1 (17135) | 96.4 (22165) | 97.9 (14953) | 99.1 (4844) | 99.2 (1003) | 97.4 (114) | 100.0 (5) | — |
| coauthor-physics | correct_and_smooth | — | 93.9 (1713) | 95.7 (7048) | 96.6 (17135) | 97.3 (22165) | 98.1 (14953) | 99.3 (4844) | 99.2 (1003) | 97.4 (114) | 100.0 (5) | — |

## Cell C3c (mixed by local structure: top-5 std of local homophily)

| dataset | std local homophily | min compat. distance | EDSP-default | EDSP-tuned | feature_lr | correct_and_smooth | mlp | gcn | sage | gat | gprgnn | bwgnn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coauthor-physics | 0.174 | 0.370 | 97.02 | 97.03 | 96.25 | **97.21** | 96.98 | — | — | — | — | — |

## Cell C3d (structured compatibility: top-5 min Frobenius distance to I and to (J-I)/(k-1))

| dataset | std local homophily | min compat. distance | EDSP-default | EDSP-tuned | feature_lr | correct_and_smooth | mlp | gcn | sage | gat | gprgnn | bwgnn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coauthor-physics | 0.174 | 0.370 | 97.02 | 97.03 | 96.25 | **97.21** | 96.98 | — | — | — | — | — |
