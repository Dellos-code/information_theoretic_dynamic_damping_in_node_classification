# E2 — native-polarity graphs (signed, relation-typed): Track-A EDSP vs Track-A baselines

Run records: 472 (git bdd1ad76f33a; runners actions-run-42626291721). Headline metric per dataset in brackets; mean ± std over splits (×100). Rows marked † are constructed benchmarks or case studies and are excluded from real-world aggregates.

| dataset [metric] | splits | EDSP-default | EDSP-tuned | EDSP-fullH-default | EDSP-fullH-tuned | majority | train_prior | harmonic | harmonic_cmn | lgc | ppr_lp | linbp | signed_vote | signed_lgc | sgcn | snea | sdgnn | sigat |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| congress-house-100 [accuracy] | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — | 92.47 ± 1.88 | 87.98 ± 3.52 | 90.11 ± 1.80 | **93.26 ± 1.51** |
| congress-house-107 [accuracy] | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — | **98.43 ± 1.25** | 96.97 ± 1.51 | 97.42 ± 1.51 | 97.98 ± 1.57 |
| congress-house-114 [accuracy] | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — | **98.65 ± 0.98** | 98.43 ± 1.25 | **98.65 ± 1.10** | 98.54 ± 0.88 |
| congress-house-93 [accuracy] | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — | **89.21 ± 2.85** | 76.29 ± 4.53 | 81.12 ± 4.32 | 85.96 ± 2.47 |
| congress-senate-100 [accuracy] | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — | **86.50 ± 8.38** | 82.00 ± 10.54 | 79.00 ± 9.17 | 82.50 ± 6.02 |
| congress-senate-107 [accuracy] | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — | **92.63 ± 5.37** | 86.32 ± 5.37 | 92.11 ± 5.39 | 92.11 ± 3.53 |
| congress-senate-114 [accuracy] | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — | 97.50 ± 3.35 | **99.00 ± 2.00** | 97.50 ± 3.35 | 97.50 ± 3.35 |
| congress-senate-93 [accuracy] | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — | 82.38 ± 9.05 | **84.29 ± 6.04** | 79.52 ± 9.29 | 76.67 ± 6.19 |
| sampson† [accuracy] | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — | 50.00 ± 50.00 | 44.44 ± 49.69 | **61.11 ± 48.75** | **61.11 ± 48.75** |
| wiki-elec† [accuracy] | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — | 89.14 ± 1.38 | **91.83 ± 1.38** | 91.27 ± 1.43 | 89.05 ± 1.33 |
| wiki-rfa† [accuracy] | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — | 86.76 ± 1.03 | 88.03 ± 1.38 | **89.61 ± 1.25** | 89.52 ± 1.55 |

## Pre-registered secondary column: EDSP-default on non-converged runs

| dataset | non-converged runs | headline, last iterate | headline, mean of last two iterates |
|---|---|---|---|
| congress-house-100 | 0 / 0 | — | — |
| congress-house-107 | 0 / 0 | — | — |
| congress-house-114 | 0 / 0 | — | — |
| congress-house-93 | 0 / 0 | — | — |
| congress-senate-100 | 0 / 0 | — | — |
| congress-senate-107 | 0 / 0 | — | — |
| congress-senate-114 | 0 / 0 | — | — |
| congress-senate-93 | 0 / 0 | — | — |
| sampson | 0 / 0 | — | — |
| wiki-elec | 0 / 0 | — | — |
| wiki-rfa | 0 / 0 | — | — |

## EDSP-tuned vs each baseline: win / tie / loss

Per dataset: two-sided Wilcoxon signed-rank over splits on the headline metric; Holm correction across datasets per baseline; win/loss if adjusted p < 0.05, else tie. Real-world datasets only.

| baseline | win | tie | loss | datasets (W / L) |
|---|---|---|---|---|
| harmonic | 0 | 0 | 0 | W: —; L: — |
| harmonic_cmn | 0 | 0 | 0 | W: —; L: — |
| lgc | 0 | 0 | 0 | W: —; L: — |
| ppr_lp | 0 | 0 | 0 | W: —; L: — |
| linbp | 0 | 0 | 0 | W: —; L: — |
| train_prior | 0 | 0 | 0 | W: —; L: — |
| signed_vote | 0 | 0 | 0 | W: —; L: — |
| signed_lgc | 0 | 0 | 0 | W: —; L: — |
| sgcn | 0 | 0 | 0 | W: —; L: — |
| snea | 0 | 0 | 0 | W: —; L: — |
| sdgnn | 0 | 0 | 0 | W: —; L: — |
| sigat | 0 | 0 | 0 | W: —; L: — |

## Calibration (mean over splits; lower is better)

| dataset | EDSP-default NLL | EDSP-default ECE | EDSP-tuned NLL | EDSP-tuned ECE | best baseline by NLL (NLL / ECE) |
|---|---|---|---|---|---|
| congress-house-100 | nan | nan | nan | nan | sigat: 0.223 / 0.051 |
| congress-house-107 | nan | nan | nan | nan | sigat: 0.056 / 0.028 |
| congress-house-114 | nan | nan | nan | nan | sgcn: 0.048 / 0.018 |
| congress-house-93 | nan | nan | nan | nan | sgcn: 0.271 / 0.076 |
| congress-senate-100 | nan | nan | nan | nan | sigat: 0.497 / 0.156 |
| congress-senate-107 | nan | nan | nan | nan | sgcn: 0.294 / 0.085 |
| congress-senate-114 | nan | nan | nan | nan | snea: 0.035 / 0.024 |
| congress-senate-93 | nan | nan | nan | nan | snea: 0.460 / 0.163 |
| sampson | nan | nan | nan | nan | sigat: 1.979 / 0.400 |
| wiki-elec | nan | nan | nan | nan | sdgnn: 0.216 / 0.030 |
| wiki-rfa | nan | nan | nan | nan | sdgnn: 0.243 / 0.027 |

## H7 (descriptive): convergence and fixed-point non-uniqueness

| dataset | EDSP-default non-converged | oscillation flagged | mean fraction of nodes still moving | GS vs damped L1: mean | p95 | splits |
|---|---|---|---|---|---|---|
| congress-house-100 | — | — | — | — | — | 0 |
| congress-house-107 | — | — | — | — | — | 0 |
| congress-house-114 | — | — | — | — | — | 0 |
| congress-house-93 | — | — | — | — | — | 0 |
| congress-senate-100 | — | — | — | — | — | 0 |
| congress-senate-107 | — | — | — | — | — | 0 |
| congress-senate-114 | — | — | — | — | — | 0 |
| congress-senate-93 | — | — | — | — | — | 0 |
| sampson | — | — | — | — | — | 0 |
| wiki-elec | — | — | — | — | — | 0 |
| wiki-rfa | — | — | — | — | — | 0 |
