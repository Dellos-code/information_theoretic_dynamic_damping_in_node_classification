# C4 — P-NATIVE vs P-CONST on native probabilistic graphs (H5), Track A

Run records: 1042 (git d55872bbb885). Mean ± std over tasks (×100) of the headline metric; 95% bootstrap CI over tasks in brackets. Descriptive only: the H5 decision rule is the one registered in docs/PREREGISTRATION.md.

## ogbn-proteins — split species [roc_auc]

| method | tasks | mean ± std | 95% CI |
|---|---|---|---|
| EDSP-const-default | 74 | 76.31 ± 8.07 | [74.46, 78.08] |
| EDSP-native-default | 65 | 76.65 ± 8.30 | [74.57, 78.55] |
| EDSP-const-tuned | 74 | 76.39 ± 8.05 | [74.53, 78.16] |
| EDSP-native-tuned | 65 | 76.92 ± 8.21 | [74.86, 78.81] |
| majority | 65 | 50.00 ± 0.00 | [50.00, 50.00] |
| train_prior | 65 | 50.00 ± 0.00 | [50.00, 50.00] |
| harmonic | 65 | 75.83 ± 8.13 | [73.78, 77.70] |
| harmonic_cmn | 65 | 75.83 ± 8.13 | [73.78, 77.70] |
| lgc | 63 | 81.19 ± 8.35 | [79.02, 83.10] |
| ppr_lp | 63 | 81.24 ± 8.29 | [79.09, 83.14] |
| linbp | 63 | 75.00 ± 7.54 | [73.09, 76.79] |
| harmonic+w | 63 | 76.60 ± 8.24 | [74.49, 78.55] |
| harmonic_cmn+w | 63 | 76.60 ± 8.24 | [74.49, 78.55] |
| lgc+w | 63 | 81.33 ± 8.37 | [79.15, 83.24] |
| ppr_lp+w | 63 | 81.42 ± 8.30 | [79.26, 83.31] |
| linbp+w | 63 | 75.43 ± 7.43 | [73.54, 77.20] |

Paired per-task comparisons (A − B, ×100; Wilcoxon signed-rank over tasks, two-sided, uncorrected):

| A | B | tasks | mean Δ | p | A > B | tie | A < B |
|---|---|---|---|---|---|---|---|
| EDSP-native-default | EDSP-const-default | 65 | +0.84 | 5.8e-12 | 60 | 0 | 5 |
| EDSP-native-tuned | EDSP-const-tuned | 65 | +1.02 | 8.75e-12 | 61 | 0 | 4 |
| harmonic+w | harmonic | 63 | +0.83 | 7.23e-12 | 60 | 0 | 3 |
| harmonic_cmn+w | harmonic_cmn | 63 | +0.83 | 7.23e-12 | 60 | 0 | 3 |
| lgc+w | lgc | 63 | +0.14 | 2.71e-11 | 58 | 0 | 5 |
| ppr_lp+w | ppr_lp | 63 | +0.17 | 5.17e-12 | 63 | 0 | 0 |
| linbp+w | linbp | 63 | +0.43 | 1.84e-10 | 60 | 0 | 3 |
| EDSP-native-tuned | harmonic+w | 63 | +0.27 | 0.000835 | 42 | 0 | 21 |
| EDSP-native-tuned | harmonic_cmn+w | 63 | +0.27 | 0.000835 | 42 | 0 | 21 |
| EDSP-native-tuned | lgc+w | 63 | -4.46 | 5.17e-12 | 0 | 0 | 63 |
| EDSP-native-tuned | ppr_lp+w | 63 | -4.54 | 5.17e-12 | 0 | 0 | 63 |
| EDSP-native-tuned | linbp+w | 63 | +1.44 | 0.000468 | 41 | 0 | 22 |

| EDSP method | non-converged tasks | oscillation flagged | median iterations | chosen configurations (count) |
|---|---|---|---|---|
| EDSP-const-default | 0 / 74 | 0 | 154 | — |
| EDSP-native-default | 0 / 65 | 0 | 138 | — |
| EDSP-const-tuned | 0 / 74 | 0 | 152 | {"h": {"fill": "b1", "name": "trainonly"}, "p": {"name": "const", "p0": 0.1}} ×51; {"h": {"name": "learned"}, "p": {"name": "const", "p0": 0.1}} ×22; {"h": {"name": "learned"}, "p": {"name": "const", "p0": 0.9}} ×1 |
| EDSP-native-tuned | 16 / 65 | 6 | 127 | {"h": {"fill": "b1", "name": "trainonly"}, "p": {"mode": "max", "name": "native"}} ×30; {"h": {"name": "learned"}, "p": {"mode": "max", "name": "native"}} ×19; {"h": {"fill": "b1", "name": "trainonly"}, "p": {"mode": "mean", "name": "native"}} ×9; {"h": {"name": "learned"}, "p": {"mode": "mean", "name": "native"}} ×6; {"h": {"fill": "b1", "name": "trainonly"}, "p": {"mode": "noisy-or", "name": "native"}} ×1 |
| ↳ secondary column (non-converged tasks) | last iterate 77.25 | mean of last two 77.25 | | |

