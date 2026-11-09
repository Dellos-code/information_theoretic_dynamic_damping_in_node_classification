# C4 — P-NATIVE vs P-CONST on native probabilistic graphs (H5), Track A

Run records: 32 (git d55872bbb885). Mean ± std over tasks (×100) of the headline metric; 95% bootstrap CI over tasks in brackets. Descriptive only: the H5 decision rule is the one registered in docs/PREREGISTRATION.md.

## ogbn-proteins — split species [roc_auc]

| method | tasks | mean ± std | 95% CI |
|---|---|---|---|
| EDSP-const-default | 2 | 52.24 ± 1.11 | [51.12, 53.35] |
| EDSP-native-default | 2 | 52.27 ± 1.45 | [50.83, 53.72] |
| EDSP-const-tuned | 2 | 52.67 ± 3.34 | [49.33, 56.01] |
| EDSP-native-tuned | 2 | 52.71 ± 3.90 | [48.81, 56.61] |
| majority | 2 | 50.00 ± 0.00 | [50.00, 50.00] |
| train_prior | 2 | 50.00 ± 0.00 | [50.00, 50.00] |
| harmonic | 2 | 51.99 ± 1.36 | [50.63, 53.35] |
| harmonic_cmn | 2 | 51.99 ± 1.36 | [50.63, 53.35] |
| lgc | 2 | 55.44 ± 1.87 | [53.57, 57.31] |
| ppr_lp | 2 | 55.34 ± 2.09 | [53.24, 57.43] |
| linbp | 2 | 56.58 ± 0.52 | [56.06, 57.10] |
| harmonic+w | 2 | 51.98 ± 1.59 | [50.39, 53.56] |
| harmonic_cmn+w | 2 | 51.98 ± 1.59 | [50.39, 53.56] |
| lgc+w | 2 | 55.54 ± 1.89 | [53.65, 57.43] |
| ppr_lp+w | 2 | 55.47 ± 2.16 | [53.31, 57.63] |
| linbp+w | 2 | 56.42 ± 0.83 | [55.59, 57.25] |

Paired per-task comparisons (A − B, ×100; Wilcoxon signed-rank over tasks, two-sided, uncorrected):

| A | B | tasks | mean Δ | p | A > B | tie | A < B |
|---|---|---|---|---|---|---|---|
| EDSP-native-default | EDSP-const-default | 2 | +0.03 | 1 | 1 | 0 | 1 |
| EDSP-native-tuned | EDSP-const-tuned | 2 | +0.04 | 1 | 1 | 0 | 1 |
| harmonic+w | harmonic | 2 | -0.02 | 1 | 1 | 0 | 1 |
| harmonic_cmn+w | harmonic_cmn | 2 | -0.02 | 1 | 1 | 0 | 1 |
| lgc+w | lgc | 2 | +0.10 | 0.5 | 2 | 0 | 0 |
| ppr_lp+w | ppr_lp | 2 | +0.13 | 0.5 | 2 | 0 | 0 |
| linbp+w | linbp | 2 | -0.16 | 1 | 1 | 0 | 1 |
| EDSP-native-tuned | harmonic+w | 2 | +0.74 | 1 | 1 | 0 | 1 |
| EDSP-native-tuned | harmonic_cmn+w | 2 | +0.74 | 1 | 1 | 0 | 1 |
| EDSP-native-tuned | lgc+w | 2 | -2.83 | 0.5 | 0 | 0 | 2 |
| EDSP-native-tuned | ppr_lp+w | 2 | -2.76 | 0.5 | 0 | 0 | 2 |
| EDSP-native-tuned | linbp+w | 2 | -3.71 | 0.5 | 0 | 0 | 2 |

| EDSP method | non-converged tasks | oscillation flagged | median iterations | chosen configurations (count) |
|---|---|---|---|---|
| EDSP-const-default | 0 / 2 | 0 | 148 | — |
| EDSP-native-default | 0 / 2 | 0 | 133 | — |
| EDSP-const-tuned | 0 / 2 | 0 | 128 | {"h": {"name": "learned"}, "p": {"name": "const", "p0": 0.1}} ×1; {"h": {"name": "learned"}, "p": {"name": "const", "p0": 0.9}} ×1 |
| EDSP-native-tuned | 1 / 2 | 1 | 556 | {"h": {"name": "learned"}, "p": {"mode": "mean", "name": "native"}} ×1; {"h": {"name": "learned"}, "p": {"mode": "max", "name": "native"}} ×1 |
| ↳ secondary column (non-converged tasks) | last iterate 56.61 | mean of last two 56.61 | | |

