# MIX — PERTURBED-REAL mixing sweep (not real-world results)

Run records: 680 (git 4e594052580c). Real nodes, labels, features and splits; edges re-wired by `edsp/data/perturb.py` to the mixing condition in the first column. Achieved statistics are means over re-wiring seeds (full labels, descriptive). Method cells: headline metric ×100, mean ± std over (seed, split) runs; best in bold. Δ = EDSP-tuned − best baseline; p = Wilcoxon over paired runs, Holm over the conditions of the base graph.

## amazon-ratings [accuracy]

| condition | h_edge | h_adj | LI | std local h | EDSP-default | EDSP-tuned | EDSP-fullH-default | EDSP-fullH-tuned | train_prior | harmonic | harmonic_cmn | lgc | ppr_lp | linbp | Δ vs best baseline | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| beta:m=0.2,c=0.5,loc=0 | 0.20 | -0.09 | 0.03 | 0.23 | 22.3 ± 0.5 | 37.2 ± 0.6 | — | — | 36.8 ± 0.0 | 22.7 ± 0.6 | 23.1 ± 0.5 | 36.8 ± 0.0 | 36.8 ± 0.0 | **48.1 ± 0.8** | -10.9 (linbp) | 3.45e-06 |
| beta:m=0.5,c=0.5,loc=0 | 0.50 | +0.32 | 0.12 | 0.28 | 60.5 ± 0.6 | 60.7 ± 0.6 | — | — | 36.8 ± 0.0 | 61.0 ± 0.5 | **61.0 ± 0.6** | 60.0 ± 0.5 | 60.6 ± 0.4 | 59.3 ± 0.8 | -0.4 (harmonic_cmn) | 3.45e-06 |
