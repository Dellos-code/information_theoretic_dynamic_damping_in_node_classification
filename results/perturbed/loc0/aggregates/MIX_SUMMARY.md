# MIX — PERTURBED-REAL mixing sweep (not real-world results)

Run records: 2800 (git 1afaf38a8669). Real nodes, labels, features and splits; edges re-wired by `edsp/data/perturb.py` to the mixing condition in the first column. Achieved statistics are means over re-wiring seeds (full labels, descriptive). Method cells: headline metric ×100, mean ± std over (seed, split) runs; best in bold. Δ = EDSP-tuned − best baseline; p = Wilcoxon over paired runs, Holm over the conditions of the base graph.

## amazon-ratings [accuracy]

| condition | h_edge | h_adj | LI | std local h | EDSP-default | EDSP-tuned | EDSP-fullH-default | EDSP-fullH-tuned | train_prior | harmonic | harmonic_cmn | lgc | ppr_lp | linbp | Δ vs best baseline | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| beta:m=0.2,c=0.5,loc=0 | 0.20 | -0.09 | 0.03 | 0.23 | 22.3 ± 0.5 | 37.2 ± 0.6 | — | — | 36.8 ± 0.0 | 22.7 ± 0.6 | 23.1 ± 0.5 | 36.8 ± 0.0 | 36.8 ± 0.0 | **48.1 ± 0.8** | -10.9 (linbp) | 3.45e-06 |
| beta:m=0.5,c=0.5,loc=0 | 0.50 | +0.32 | 0.12 | 0.28 | 60.5 ± 0.6 | 60.7 ± 0.6 | — | — | 36.8 ± 0.0 | 61.0 ± 0.5 | **61.0 ± 0.6** | 60.0 ± 0.5 | 60.6 ± 0.4 | 59.3 ± 0.8 | -0.4 (harmonic_cmn) | 3.45e-06 |

## wiki-editor [accuracy]

| condition | h_edge | h_adj | LI | std local h | EDSP-default | EDSP-tuned | EDSP-fullH-default | EDSP-fullH-tuned | train_prior | harmonic | harmonic_cmn | lgc | ppr_lp | linbp | Δ vs best baseline | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| beta:m=0.2,c=0.5,loc=0 | 0.20 | -0.60 | 0.28 | 0.23 | 18.9 ± 0.8 | **89.6 ± 0.5** | — | — | 63.1 ± 0.0 | 18.8 ± 0.7 | 36.8 ± 1.2 | 63.1 ± 0.0 | 36.1 ± 0.2 | 89.5 ± 0.5 | +0.1 (linbp) | 0.113 |
| beta:m=0.5,c=0.5,loc=0 | 0.50 | -0.02 | 0.00 | 0.29 | 49.1 ± 1.3 | **71.0 ± 2.4** | — | — | 63.1 ± 0.0 | 48.9 ± 1.3 | 60.9 ± 0.9 | 65.7 ± 0.7 | 49.5 ± 1.3 | 60.1 ± 4.7 | +5.3 (lgc) | 1.21e-05 |
| beta:m=0.5,c=5,loc=0 | 0.50 | -0.02 | 0.00 | 0.26 | 45.1 ± 1.1 | 61.4 ± 0.8 | — | — | **63.1 ± 0.0** | 44.8 ± 1.1 | 57.2 ± 0.8 | 63.1 ± 0.1 | 44.6 ± 1.2 | 56.9 ± 0.9 | -1.7 (lgc) | 1.21e-05 |
| beta:m=0.8,c=0.5,loc=0 | 0.79 | +0.57 | 0.25 | 0.24 | 85.3 ± 0.6 | 87.7 ± 0.7 | — | — | 63.1 ± 0.0 | 85.3 ± 0.8 | 87.7 ± 0.7 | **89.0 ± 0.6** | 85.3 ± 0.6 | 84.3 ± 0.6 | -1.2 (lgc) | 1.21e-05 |
| uniform:loc=0 | 0.50 | -0.02 | 0.00 | 0.27 | 46.5 ± 1.4 | **64.1 ± 0.9** | — | — | 63.1 ± 0.0 | 46.3 ± 1.5 | 58.6 ± 0.9 | 63.0 ± 0.2 | 46.4 ± 1.5 | 58.3 ± 4.0 | +1.1 (lgc) | 1.21e-05 |
| logistic:loc=0 | 0.50 | -0.02 | 0.00 | 0.28 | 47.8 ± 1.1 | **66.3 ± 0.9** | — | — | 63.1 ± 0.0 | 47.5 ± 1.1 | 59.9 ± 0.7 | 63.4 ± 0.4 | 48.1 ± 1.2 | 59.2 ± 4.9 | +2.9 (lgc) | 1.21e-05 |
| regional:s=20,hi=0.9,lo=0.1,loc=0 | 0.51 | -0.00 | 0.19 | 0.25 | 50.4 ± 26.9 | **82.3 ± 7.9** | — | — | 63.1 ± 0.0 | 50.2 ± 26.9 | 62.2 ± 20.5 | 72.7 ± 11.1 | 58.0 ± 19.7 | 77.4 ± 12.5 | +4.9 (linbp) | 1.49e-08 |
| community:hi=0.9,lo=0.1,loc=0 | 0.53 | -0.17 | 0.03 | 0.32 | 43.9 ± 3.0 | **70.4 ± 1.6** | — | — | 63.1 ± 0.0 | 44.0 ± 3.4 | 46.1 ± 3.1 | 48.3 ± 9.7 | 48.2 ± 9.7 | 61.9 ± 4.6 | +8.5 (linbp) | 1.21e-05 |
