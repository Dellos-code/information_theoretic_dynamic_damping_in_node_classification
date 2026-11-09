# H5 — P-NATIVE vs P-CONST on ogbn-proteins (PREREGISTRATION A15)

**Verdict: SUPPORTED** (supported iff the two-sided Wilcoxon over tasks gives p < 0.05 and the mean Δ > 0).

1792 run records; 112 of 112 tasks have both primary records (0 left out); 0 NOT-RUN records. ROC-AUC ×100; Δ = A − B per task; 95% bootstrap CI of the mean Δ over tasks; win / tie / loss = tasks where A > B / = / <.

| comparison (A vs B) | tasks | mean Δ [95% CI] | median Δ | p | win / tie / loss | p < 0.05 |
|---|---|---|---|---|---|---|
| **primary:** EDSP-native-tuned vs EDSP-const-tuned | 112 | +1.011 [+0.846, +1.185] | +0.751 | 1.88e-19 | 108 / 0 / 4 | yes |
| EDSP-native-default vs EDSP-const-default | 112 | +0.823 [+0.700, +0.958] | +0.666 | 1.44e-19 | 105 / 0 / 7 | yes |
| EDSP-native-tuned vs best weighted (ppr_lp+w) | 112 | -4.667 [-5.153, -4.198] | -4.069 | 6.3e-20 | 3 / 0 / 109 | yes |
| EDSP-native-tuned vs best unweighted (ppr_lp) | 112 | -4.513 [-4.997, -4.052] | -3.845 | 6.47e-20 | 3 / 0 / 109 | yes |

Mean ROC-AUC over tasks (×100):

| method | mean |
|---|---|
| ppr_lp+w | 83.47 |
| lgc+w | 83.34 |
| ppr_lp | 83.32 |
| lgc | 83.20 |
| EDSP-native-tuned | 78.81 |
| harmonic_cmn+w | 78.63 |
| harmonic+w | 78.63 |
| EDSP-native-default | 78.56 |
| EDSP-const-tuned | 77.79 |
| harmonic | 77.78 |
| harmonic_cmn | 77.78 |
| EDSP-const-default | 77.73 |
| linbp+w | 75.15 |
| linbp | 74.73 |
| majority | 50.00 |
| train_prior | 50.00 |

