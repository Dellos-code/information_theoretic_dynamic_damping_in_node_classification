# A13 — native mixing (real graphs; E1 Track A, `--learned-sign-baselines`, `--h7-splits 0`)

**(a) transfer verdict: INCONCLUSIVE** — 4 wins, 0 ties, 9 losses over 13 tested datasets; win families: FB100.

Means ×100 over splits (headline metric per dataset). (a): EDSP-tuned vs the best sign-blind baseline (by mean), Wilcoxon over splits, Holm across the tested datasets. (b) only for (a) wins: EDSP-tuned vs the better of the two +HL baselines, Holm over those datasets.

| dataset | metric | splits | EDSP-def | EDSP-tuned | majority | harmonic | harmonic_cmn | lgc | ppr_lp | linbp | signed_vote+HL | signed_lgc+HL | best sign-blind | Δ (a) | p Holm (a) | (a) | Δ (b) | p Holm (b) | (b) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fb100-penn94 | accuracy | 10 | 59.12 | 71.94 | 52.65 | 59.09 | 59.19 | 63.51 | 60.58 | 59.96 | 70.74 | 72.59 | lgc | +8.43 | 0.0254 | win | -0.64 | 0.00781 | loss |
| fb100-amherst41 | accuracy | 10 | 63.14 | 71.65 | 50.12 | 63.22 | 63.54 | 63.76 | 62.41 | 62.33 | 71.13 | 71.06 | lgc | +7.89 | 0.0254 | win | +0.52 | 0.0625 | tie |
| fb100-cornell5 | accuracy | 10 | 65.55 | 71.53 | 51.65 | 65.47 | 65.74 | 67.92 | 65.46 | 64.57 | 70.79 | 72.29 | lgc | +3.61 | 0.0254 | win | -0.76 | 0.00781 | loss |
| fb100-johnshopkins55 | accuracy | 10 | 65.18 | 70.03 | 54.83 | 65.26 | 65.46 | 67.26 | 65.51 | 64.50 | 69.02 | 71.08 | lgc | +2.77 | 0.0254 | win | -1.05 | 0.0625 | tie |
| fb100-reed98 | accuracy | 10 | 60.23 | 59.42 | 58.38 | 60.81 | 59.94 | 59.31 | 59.94 | 59.02 | 56.88 | 55.95 | harmonic | -1.39 | 0.0293 | loss | — | — | — |
| genius | roc_auc | 10 | 50.85 | 59.16 | 50.00 | 50.48 | 50.48 | 70.37 | 61.00 | 58.74 | 58.06 | 54.79 | lgc | -11.21 | 0.0254 | loss | — | — | — |
| deezer-europe | accuracy | 10 | 54.92 | 54.89 | 55.67 | 55.04 | 54.50 | 56.40 | 56.23 | 55.67 | 54.72 | 56.27 | lgc | -1.51 | 0.0254 | loss | — | — | — |
| twitch-de | roc_auc | 10 | 67.72 | 67.72 | 50.00 | 67.72 | 67.72 | 68.77 | 69.51 | 70.40 | 68.86 | 58.54 | linbp | -2.68 | 0.0254 | loss | — | — | — |
| twitch-en | roc_auc | 10 | 57.60 | 57.57 | 50.00 | 57.54 | 57.54 | 60.02 | 59.13 | 61.29 | 59.83 | 56.27 | linbp | -3.72 | 0.0254 | loss | — | — | — |
| twitch-es | roc_auc | 10 | 66.27 | 66.41 | 50.00 | 66.27 | 66.27 | 67.33 | 67.74 | 59.63 | 59.74 | 57.06 | ppr_lp | -1.32 | 0.0254 | loss | — | — | — |
| twitch-fr | roc_auc | 10 | 64.81 | 64.76 | 50.00 | 64.83 | 64.83 | 66.43 | 65.68 | 60.62 | 59.96 | 58.50 | lgc | -1.68 | 0.0254 | loss | — | — | — |
| twitch-pt | roc_auc | 10 | 65.40 | 65.12 | 50.00 | 65.41 | 65.41 | 66.39 | 66.25 | 61.48 | 59.97 | 55.44 | lgc | -1.27 | 0.0293 | loss | — | — | — |
| twitch-ru | roc_auc | 10 | 57.30 | 56.94 | 50.00 | 57.27 | 57.27 | 58.36 | 57.61 | 51.57 | 52.19 | 51.69 | lgc | -1.42 | 0.0293 | loss | — | — | — |
| actor (not tested) | accuracy | 10 | 23.30 | 23.33 | 25.24 | 23.41 | 23.58 | 25.27 | 25.02 | 20.92 | 15.66 | 20.86 | lgc | -1.94 | 0.00391 (raw) | — | — | — | — |
