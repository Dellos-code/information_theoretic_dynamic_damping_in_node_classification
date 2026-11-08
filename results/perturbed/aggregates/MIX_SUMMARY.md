# MIX — PERTURBED-REAL mixing sweep (not real-world results)

Run records: 1020 (git 4e594052580c). Real nodes, labels, features and splits; edges re-wired by `edsp/data/perturb.py` to the mixing condition in the first column. Achieved statistics are means over re-wiring seeds (full labels, descriptive). Method cells: headline metric ×100, mean ± std over (seed, split) runs; best in bold. Δ = EDSP-tuned − best baseline; p = Wilcoxon over paired runs, Holm over the conditions of the base graph.

## amazon-ratings [accuracy]

| condition | h_edge | h_adj | LI | std local h | EDSP-default | EDSP-tuned | EDSP-fullH-default | EDSP-fullH-tuned | train_prior | harmonic | harmonic_cmn | lgc | ppr_lp | linbp | Δ vs best baseline | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global:h=0 | 0.00 | -0.36 | 0.22 | 0.00 | 4.3 ± 0.3 | **76.1 ± 0.5** | — | — | 36.8 ± 0.0 | 4.4 ± 0.3 | 5.7 ± 0.3 | 36.7 ± 0.0 | 36.7 ± 0.0 | 74.8 ± 0.4 | +1.3 (linbp) | 5.19e-06 |
| beta:m=0.2,c=0.5 | 0.20 | -0.09 | 0.03 | 0.32 | 24.9 ± 0.5 | **60.8 ± 0.7** | — | — | 36.8 ± 0.0 | 25.1 ± 0.5 | 25.6 ± 0.5 | 42.0 ± 0.4 | 42.1 ± 0.3 | 53.0 ± 1.3 | +7.8 (linbp) | 5.19e-06 |
| beta:m=0.5,c=0.5 | 0.50 | +0.31 | 0.11 | 0.40 | 56.5 ± 0.6 | **63.0 ± 1.5** | — | — | 36.8 ± 0.0 | 56.7 ± 0.6 | 56.8 ± 0.6 | 61.4 ± 0.6 | 61.2 ± 0.6 | 55.2 ± 0.8 | +1.6 (lgc) | 0.000241 |
