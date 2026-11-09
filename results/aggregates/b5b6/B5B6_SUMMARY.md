# B5 / B6 — EDSP against signed GNNs and fraud GNNs (PREREGISTRATION A17)

New-method records from git bdd1ad76f33a. Mean ± std over splits (×100); bold = best mean among the columns shown. Δ = mean paired difference EDSP − method (×100); p = two-sided Wilcoxon over splits; Holm across the ranked graphs of the family, separately per method; win / loss if Holm-adjusted p < 0.05. 'EDSP rank' counts only the columns shown.

## B5 — signed graphs (Track A: structure + train labels)

| graph [metric] | EDSP-tuned | SGCN | SNEA | SDGNN | SiGAT | EDSP-default | signed_vote | signed_lgc | LGC | harmonic+CMN | LinBP | EDSP rank |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| congress-house-93 [accuracy] | 80.45 ± 3.23 | **89.21 ± 2.85** | 76.29 ± 4.53 | 81.12 ± 4.32 | 85.96 ± 2.47 | 73.15 ± 3.20 | 75.06 ± 3.47 | 78.88 ± 3.13 | 67.08 ± 2.01 | 65.62 ± 2.94 | 55.73 ± 9.54 | 4 / 11 |
| congress-house-100 [accuracy] | 88.65 ± 2.53 | 92.47 ± 1.88 | 87.98 ± 3.52 | 90.11 ± 1.80 | **93.26 ± 1.51** | 88.09 ± 2.90 | 88.76 ± 2.56 | 88.65 ± 2.38 | 56.74 ± 0.75 | 46.63 ± 2.76 | 73.71 ± 2.98 | 6 / 11 |
| congress-house-107 [accuracy] | 96.97 ± 1.95 | **98.43 ± 1.25** | 96.97 ± 1.51 | 97.42 ± 1.51 | 97.98 ± 1.57 | 96.97 ± 1.95 | 97.08 ± 1.96 | 97.08 ± 2.08 | 47.98 ± 1.01 | 13.03 ± 2.94 | 91.01 ± 2.97 | 6 / 11 |
| congress-house-114 [accuracy] | 97.08 ± 1.76 | **98.65 ± 0.98** | 98.43 ± 1.25 | 98.65 ± 1.10 | 98.54 ± 0.88 | 97.08 ± 1.76 | 96.97 ± 2.01 | 96.97 ± 2.01 | 56.18 ± 0.00 | 37.87 ± 3.26 | 91.80 ± 1.82 | 5 / 11 |
| congress-senate-93 [accuracy] | 80.48 ± 8.37 | 82.38 ± 9.05 | **84.29 ± 6.04** | 79.52 ± 9.29 | 76.67 ± 6.19 | 80.48 ± 8.37 | 81.90 ± 7.62 | 81.90 ± 7.62 | 52.86 ± 6.55 | 54.29 ± 7.74 | 62.86 ± 6.67 | 5 / 11 |
| congress-senate-100 [accuracy] | **88.50 ± 5.02** | 86.50 ± 8.38 | 82.00 ± 10.54 | 79.00 ± 9.17 | 82.50 ± 6.02 | 88.00 ± 5.10 | 88.00 ± 6.00 | 88.00 ± 6.00 | 53.50 ± 7.09 | 50.00 ± 9.22 | 62.50 ± 5.59 | 1 / 11 |
| congress-senate-107 [accuracy] | 96.84 ± 2.58 | 92.63 ± 5.37 | 86.32 ± 5.37 | 92.11 ± 5.39 | 92.11 ± 3.53 | 96.84 ± 2.58 | **97.37 ± 2.63** | 97.37 ± 2.63 | 52.11 ± 1.58 | 25.26 ± 7.74 | 79.47 ± 9.83 | 3 / 11 |
| congress-senate-114 [accuracy] | 96.00 ± 4.36 | 97.50 ± 3.35 | **99.00 ± 2.00** | 97.50 ± 3.35 | 97.50 ± 3.35 | 96.50 ± 4.50 | 97.50 ± 3.35 | 97.50 ± 3.35 | 54.00 ± 2.00 | 36.00 ± 7.00 | 83.00 ± 6.78 | 8 / 11 |
| wiki-editor [accuracy] | 86.12 ± 0.67 | 81.42 ± 0.61 | 71.59 ± 0.61 | 76.82 ± 2.03 | 82.13 ± 0.76 | 86.15 ± 0.65 | 84.33 ± 0.63 | **86.43 ± 0.42** | 76.54 ± 0.48 | 76.08 ± 0.47 | 72.78 ± 0.60 | 3 / 11 |
| wiki-rfa (case study) [accuracy] | 72.23 ± 1.18 | 86.76 ± 1.03 | 88.03 ± 1.38 | **89.61 ± 1.25** | 89.52 ± 1.55 | 72.22 ± 1.22 | 71.73 ± 1.57 | 76.59 ± 0.81 | 63.99 ± 1.29 | 64.12 ± 2.23 | 61.22 ± 1.59 | 6 / 11 |
| wiki-elec (case study) [accuracy] | 80.58 ± 1.42 | 89.14 ± 1.38 | **91.83 ± 1.38** | 91.27 ± 1.43 | 89.05 ± 1.33 | 80.54 ± 1.45 | 78.09 ± 1.43 | 81.66 ± 0.78 | 59.20 ± 0.76 | 60.58 ± 2.55 | 56.75 ± 1.05 | 6 / 11 |
| sampson (case study) [accuracy] | **66.67 ± 47.14** | 50.00 ± 50.00 | 44.44 ± 49.69 | 61.11 ± 48.75 | 61.11 ± 48.75 | 66.67 ± 47.14 | 66.67 ± 47.14 | 61.11 ± 48.75 | 44.44 ± 49.69 | 50.00 ± 50.00 | 38.89 ± 48.75 | 1 / 11 |

| graph | Δ vs SGCN (p; p Holm; verdict) | Δ vs SNEA (p; p Holm; verdict) | Δ vs SDGNN (p; p Holm; verdict) | Δ vs SiGAT (p; p Holm; verdict) |
|---|---|---|---|---|
| congress-house-93 | -8.76 (0.00195; 0.0176; loss) | +4.16 (0.129; 0.645; tie) | -0.67 (0.576; 1; tie) | -5.51 (0.00391; 0.0312; loss) |
| congress-house-100 | -3.82 (0.00195; 0.0176; loss) | +0.67 (0.578; 1; tie) | -1.46 (0.238; 1; tie) | -4.61 (0.00391; 0.0312; loss) |
| congress-house-107 | -1.46 (0.125; 0.625; tie) | -0.00 (0.906; 1; tie) | -0.45 (0.43; 1; tie) | -1.01 (0.148; 0.445; tie) |
| congress-house-114 | -1.57 (0.00781; 0.0469; loss) | -1.35 (0.0625; 0.375; tie) | -1.57 (0.0156; 0.109; tie) | -1.46 (0.0469; 0.188; tie) |
| congress-senate-93 | -1.90 (0.562; 0.75; tie) | -3.81 (0.422; 1; tie) | +0.95 (0.875; 1; tie) | +3.81 (0.281; 0.562; tie) |
| congress-senate-100 | +2.00 (0.266; 0.75; tie) | +6.50 (0.0312; 0.219; tie) | +9.50 (0.00781; 0.0625; tie) | +6.00 (0.00781; 0.0469; win) |
| congress-senate-107 | +4.21 (0.125; 0.625; tie) | +10.53 (0.00391; 0.0312; win) | +4.74 (0.0156; 0.109; tie) | +4.74 (0.00781; 0.0469; win) |
| congress-senate-114 | -1.50 (0.25; 0.75; tie) | -3.00 (0.156; 0.645; tie) | -1.50 (0.25; 1; tie) | -1.50 (0.531; 0.562; tie) |
| wiki-editor | +4.70 (0.00195; 0.0176; win) | +14.53 (0.00195; 0.0176; win) | +9.30 (0.00195; 0.0176; win) | +3.99 (0.00195; 0.0176; win) |
| wiki-rfa | -14.52 (0.00195; —; case study) | -15.79 (0.00195; —; case study) | -17.38 (0.00195; —; case study) | -17.29 (0.00195; —; case study) |
| wiki-elec | -8.56 (0.00195; —; case study) | -11.25 (0.00195; —; case study) | -10.69 (0.00195; —; case study) | -8.47 (0.00195; —; case study) |
| sampson | +16.67 (0.18; —; case study) | +22.22 (0.0455; —; case study) | +5.56 (0.564; —; case study) | +5.56 (0.705; —; case study) |

| EDSP-tuned vs | win | tie | loss | win graphs | loss graphs |
|---|---|---|---|---|---|
| SGCN | 1 | 5 | 3 | wiki-editor | congress-house-93, congress-house-100, congress-house-114 |
| SNEA | 2 | 7 | 0 | congress-senate-107, wiki-editor | — |
| SDGNN | 1 | 8 | 0 | wiki-editor | — |
| SiGAT | 3 | 4 | 2 | congress-senate-100, congress-senate-107, wiki-editor | congress-house-93, congress-house-100 |

### Exploratory: average ranks on the ranked signed graphs (not preregistered)

9 graphs; Friedman p = 0.275; Nemenyi CD (α = 0.05) = 3.00. SGCN 2.56, SiGAT 3.56, signed_lgc 3.83, signed_vote 4.17, EDSP-tuned 4.39, SDGNN 4.44, SNEA 5.06.

## B6 — fraud graphs (Track B: node features)

| graph [metric] | EDSP-B-tuned | CARE-GNN | PC-GNN | EDSP-default | LR | C&S | MLP | BWGNN | EDSP rank |
|---|---|---|---|---|---|---|---|---|---|
| amazon-fraud [roc_auc] | 96.62 ± 0.76 | 95.18 ± 0.93 | 96.36 ± 0.54 | 94.07 ± 1.05 | 96.55 ± 0.91 | 95.65 ± 0.60 | **96.97 ± 0.34** | 96.96 ± 0.42 | 3 / 8 |
| yelpchi [roc_auc] | 76.85 ± 0.28 | 79.16 ± 0.38 | **87.39 ± 0.40** | 64.76 ± 5.85 | 77.01 ± 0.34 | 70.97 ± 0.35 | 80.53 ± 0.40 | 82.28 ± 1.21 | 6 / 8 |

| graph | Δ vs CARE-GNN (p; p Holm; verdict) | Δ vs PC-GNN (p; p Holm; verdict) |
|---|---|---|
| amazon-fraud | +1.44 (0.00781; 0.00781; win) | +0.25 (0.375; 0.375; tie) |
| yelpchi | -2.31 (0.00195; 0.00391; loss) | -10.53 (0.00195; 0.00391; loss) |

| EDSP-B-tuned vs | win | tie | loss | win graphs | loss graphs |
|---|---|---|---|---|---|
| CARE-GNN | 1 | 0 | 1 | amazon-fraud | yelpchi |
| PC-GNN | 0 | 1 | 1 | — | yelpchi |

NOT-RUN records: B5 0, B6 0.
