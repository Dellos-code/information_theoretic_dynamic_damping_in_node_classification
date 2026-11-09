# C4 — P-NATIVE vs P-CONST on native probabilistic graphs (H5), Track A

Run records: 896 (git d501eec4ce55). Mean ± std over tasks (×100) of the headline metric; 95% bootstrap CI over tasks in brackets. Descriptive only: the H5 decision rule is the one registered in docs/PREREGISTRATION.md.

## ogbn-proteins — split species [roc_auc]

| method | tasks | mean ± std | 95% CI |
|---|---|---|---|
| EDSP-const-default | 56 | 80.21 ± 6.91 | [78.41, 82.02] |
| EDSP-native-default | 56 | 81.01 ± 7.02 | [79.16, 82.85] |
| EDSP-const-tuned | 56 | 80.23 ± 6.88 | [78.43, 82.03] |
| EDSP-native-tuned | 56 | 81.24 ± 7.05 | [79.39, 83.10] |
| majority | 56 | 50.00 ± 0.00 | [50.00, 50.00] |
| train_prior | 56 | 50.00 ± 0.00 | [50.00, 50.00] |
| harmonic | 56 | 80.30 ± 6.82 | [78.51, 82.08] |
| harmonic_cmn | 56 | 80.30 ± 6.82 | [78.51, 82.08] |
| lgc | 56 | 85.80 ± 7.05 | [83.87, 87.60] |
| ppr_lp | 56 | 86.00 ± 6.73 | [84.17, 87.72] |
| linbp | 56 | 74.71 ± 11.42 | [71.60, 77.66] |
| harmonic+w | 56 | 81.18 ± 6.84 | [79.38, 82.98] |
| harmonic_cmn+w | 56 | 81.18 ± 6.84 | [79.38, 82.98] |
| lgc+w | 56 | 85.93 ± 7.02 | [84.02, 87.73] |
| ppr_lp+w | 56 | 86.13 ± 6.71 | [84.30, 87.85] |
| linbp+w | 56 | 75.10 ± 11.36 | [72.02, 78.04] |

Paired per-task comparisons (A − B, ×100; Wilcoxon signed-rank over tasks, two-sided, uncorrected):

| A | B | tasks | mean Δ | p | A > B | tie | A < B |
|---|---|---|---|---|---|---|---|
| EDSP-native-default | EDSP-const-default | 56 | +0.80 | 1.23e-10 | 53 | 0 | 3 |
| EDSP-native-tuned | EDSP-const-tuned | 56 | +1.01 | 7.55e-11 | 56 | 0 | 0 |
| harmonic+w | harmonic | 56 | +0.88 | 7.55e-11 | 56 | 0 | 0 |
| harmonic_cmn+w | harmonic_cmn | 56 | +0.88 | 7.55e-11 | 56 | 0 | 0 |
| lgc+w | lgc | 56 | +0.14 | 1.74e-09 | 51 | 0 | 5 |
| ppr_lp+w | ppr_lp | 56 | +0.13 | 9.49e-10 | 50 | 0 | 6 |
| linbp+w | linbp | 56 | +0.40 | 7.55e-11 | 56 | 0 | 0 |
| EDSP-native-tuned | harmonic+w | 56 | +0.06 | 0.324 | 32 | 0 | 24 |
| EDSP-native-tuned | harmonic_cmn+w | 56 | +0.06 | 0.324 | 32 | 0 | 24 |
| EDSP-native-tuned | lgc+w | 56 | -4.69 | 4.15e-10 | 4 | 0 | 52 |
| EDSP-native-tuned | ppr_lp+w | 56 | -4.89 | 1.52e-10 | 3 | 0 | 53 |
| EDSP-native-tuned | linbp+w | 56 | +6.14 | 1.37e-08 | 47 | 0 | 9 |

| EDSP method | non-converged tasks | oscillation flagged | median iterations | chosen configurations (count) |
|---|---|---|---|---|
| EDSP-const-default | 0 / 56 | 0 | 158 | — |
| EDSP-native-default | 0 / 56 | 0 | 142 | — |
| EDSP-const-tuned | 0 / 56 | 0 | 158 | {"h": {"fill": "b1", "name": "trainonly"}, "p": {"name": "const", "p0": 0.1}} ×52; {"h": {"name": "learned"}, "p": {"name": "const", "p0": 0.1}} ×3; {"h": {"fill": "b1", "name": "trainonly"}, "p": {"name": "const", "p0": 0.5}} ×1 |
| EDSP-native-tuned | 4 / 56 | 2 | 132 | {"h": {"fill": "b1", "name": "trainonly"}, "p": {"mode": "max", "name": "native"}} ×29; {"h": {"fill": "b1", "name": "trainonly"}, "p": {"mode": "mean", "name": "native"}} ×17; {"h": {"name": "learned"}, "p": {"mode": "max", "name": "native"}} ×4; {"h": {"fill": "b1", "name": "trainonly"}, "p": {"mode": "noisy-or", "name": "native"}} ×4; {"h": {"name": "learned"}, "p": {"mode": "mean", "name": "native"}} ×2 |
| ↳ secondary column (non-converged tasks) | last iterate 82.91 | mean of last two 82.91 | | |

