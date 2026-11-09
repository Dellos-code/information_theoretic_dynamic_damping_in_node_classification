# C4 — P-NATIVE vs P-CONST on native probabilistic graphs (H5), Track A

Run records: 1792 (git d501eec4ce55, d55872bbb885). Mean ± std over tasks (×100) of the headline metric; 95% bootstrap CI over tasks in brackets. Descriptive only: the H5 decision rule is the one registered in docs/PREREGISTRATION.md.

## ogbn-proteins — split species [roc_auc]

| method | tasks | mean ± std | 95% CI |
|---|---|---|---|
| EDSP-const-default | 112 | 77.73 ± 7.95 | [76.25, 79.19] |
| EDSP-native-default | 112 | 78.56 ± 8.10 | [77.04, 80.02] |
| EDSP-const-tuned | 112 | 77.79 ± 7.92 | [76.31, 79.23] |
| EDSP-native-tuned | 112 | 78.81 ± 8.05 | [77.29, 80.27] |
| majority | 112 | 50.00 ± 0.00 | [50.00, 50.00] |
| train_prior | 112 | 50.00 ± 0.00 | [50.00, 50.00] |
| harmonic | 112 | 77.78 ± 7.94 | [76.30, 79.23] |
| harmonic_cmn | 112 | 77.78 ± 7.94 | [76.30, 79.23] |
| lgc | 112 | 83.20 ± 8.24 | [81.66, 84.67] |
| ppr_lp | 112 | 83.32 ± 8.09 | [81.79, 84.76] |
| linbp | 112 | 74.73 ± 9.66 | [72.90, 76.53] |
| harmonic+w | 112 | 78.63 ± 8.02 | [77.14, 80.09] |
| harmonic_cmn+w | 112 | 78.63 ± 8.02 | [77.14, 80.09] |
| lgc+w | 112 | 83.34 ± 8.23 | [81.79, 84.80] |
| ppr_lp+w | 112 | 83.47 ± 8.09 | [81.95, 84.91] |
| linbp+w | 112 | 75.15 ± 9.58 | [73.33, 76.92] |

Paired per-task comparisons (A − B, ×100; Wilcoxon signed-rank over tasks, two-sided, uncorrected):

| A | B | tasks | mean Δ | p | A > B | tie | A < B |
|---|---|---|---|---|---|---|---|
| EDSP-native-default | EDSP-const-default | 112 | +0.82 | 1.44e-19 | 105 | 0 | 7 |
| EDSP-native-tuned | EDSP-const-tuned | 112 | +1.01 | 1.88e-19 | 108 | 0 | 4 |
| harmonic+w | harmonic | 112 | +0.85 | 5.36e-20 | 109 | 0 | 3 |
| harmonic_cmn+w | harmonic_cmn | 112 | +0.85 | 5.36e-20 | 109 | 0 | 3 |
| lgc+w | lgc | 112 | +0.14 | 2.9e-18 | 103 | 0 | 9 |
| ppr_lp+w | ppr_lp | 112 | +0.15 | 2.39e-19 | 106 | 0 | 6 |
| linbp+w | linbp | 112 | +0.42 | 1.18e-18 | 109 | 0 | 3 |
| EDSP-native-tuned | harmonic+w | 112 | +0.17 | 0.00504 | 69 | 0 | 43 |
| EDSP-native-tuned | harmonic_cmn+w | 112 | +0.17 | 0.00504 | 69 | 0 | 43 |
| EDSP-native-tuned | lgc+w | 112 | -4.53 | 1.69e-19 | 4 | 0 | 108 |
| EDSP-native-tuned | ppr_lp+w | 112 | -4.67 | 6.3e-20 | 3 | 0 | 109 |
| EDSP-native-tuned | linbp+w | 112 | +3.65 | 2.77e-10 | 82 | 0 | 30 |

| EDSP method | non-converged tasks | oscillation flagged | median iterations | chosen configurations (count) |
|---|---|---|---|---|
| EDSP-const-default | 0 / 112 | 0 | 156 | — |
| EDSP-native-default | 0 / 112 | 0 | 140 | — |
| EDSP-const-tuned | 0 / 112 | 0 | 155 | {"h": {"fill": "b1", "name": "trainonly"}, "p": {"name": "const", "p0": 0.1}} ×87; {"h": {"name": "learned"}, "p": {"name": "const", "p0": 0.1}} ×23; {"h": {"name": "learned"}, "p": {"name": "const", "p0": 0.9}} ×1; {"h": {"fill": "b1", "name": "trainonly"}, "p": {"name": "const", "p0": 0.5}} ×1 |
| EDSP-native-tuned | 17 / 112 | 6 | 128 | {"h": {"fill": "b1", "name": "trainonly"}, "p": {"mode": "max", "name": "native"}} ×57; {"h": {"fill": "b1", "name": "trainonly"}, "p": {"mode": "mean", "name": "native"}} ×23; {"h": {"name": "learned"}, "p": {"mode": "max", "name": "native"}} ×20; {"h": {"name": "learned"}, "p": {"mode": "mean", "name": "native"}} ×7; {"h": {"fill": "b1", "name": "trainonly"}, "p": {"mode": "noisy-or", "name": "native"}} ×5 |
| ↳ secondary column (non-converged tasks) | last iterate 77.81 | mean of last two 77.81 | | |

