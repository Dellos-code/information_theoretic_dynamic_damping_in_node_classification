# A18 post-hoc smoothers — Phase 1 (exploratory, E3 graphs)

Base = `feature_lr`; d = method − base per split (headline metric ×100). NI margin δ = 1.0; one-sided α = 0.025; Nadeau–Bengio corrected t (default n_test/n_train = 0.333); heterophilous iff adjusted homophily < 0.2. Left out: amazon-fraud:rand8.

Mean d (sd; one-sided NI p). NI holds where p < α.

| graph [metric] | h_adj | EDSP-tuned | correct_and_smooth | cs_ref | cs_2hop | cs_2hop_ref | clp | clp_es | linbp_b |
|---|---|---|---|---|---|---|---|---|---|
| amazon-computers [accuracy] | 0.68 | +3.87 (0.61; p 3.7e-07) | +5.25 (0.27; p 3.1e-11) | +5.55 (0.40; p 6.2e-10) | +4.12 (0.37; p 2.6e-09) | +4.20 (0.33; p 8.8e-10) | +5.39 (0.44; p 1.8e-09) | +5.35 (0.42; p 1.3e-09) | +0.33 (0.27; p 1.7e-05) |
| amazon-fraud [roc_auc] | 0.06 | +0.08 (0.29; p 0.0039) | -0.71 (0.83; p 0.37) | -0.63 (0.52; p 0.26) | -1.16 (1.04; p 0.56) | -0.98 (0.25; p 0.47) | -0.04 (0.12; p 3.8e-05) | -0.04 (0.12; p 3.9e-05) | +0.10 (0.34; p 0.0073) |
| amazon-photo [accuracy] | 0.79 | +3.06 (0.28; p 1.9e-09) | +2.68 (0.61; p 3.8e-06) | +2.64 (0.51; p 8.7e-07) | +2.44 (0.48; p 9e-07) | +2.37 (0.54; p 2.8e-06) | +3.29 (0.45; p 8.1e-08) | +3.27 (0.45; p 7.6e-08) | +0.76 (0.26; p 1.5e-06) |
| amazon-ratings [accuracy] | 0.14 | +1.34 (0.46; p 4.8e-05) | +7.98 (0.58; p 4.4e-09) | +7.92 (0.50; p 1.2e-09) | +9.69 (0.43; p 6.9e-11) | +9.69 (0.43; p 7.5e-11) | +6.05 (0.72; p 2.6e-07) | +7.95 (0.47; p 6.8e-10) | +2.79 (0.26; p 7.9e-09) |
| chameleon-filtered [accuracy] | 0.03 | +0.21 (2.32; p 0.26) | +2.67 (3.09; p 0.08) | +2.61 (2.32; p 0.038) | +3.97 (1.90; p 0.0041) | +3.78 (1.83; p 0.0041) | +2.70 (1.48; p 0.0051) | +2.88 (1.79; p 0.01) | +0.71 (1.21; p 0.051) |
| citeseer [accuracy] | 0.67 | +3.87 (1.71; p 0.00068) | +4.64 (0.87; p 8.1e-07) | +4.68 (1.37; p 4.1e-05) | +2.90 (1.28; p 0.00043) | +3.07 (1.29; p 0.00034) | +4.29 (1.47; p 0.00012) | +4.36 (1.42; p 8.5e-05) | +1.82 (0.90; p 0.00035) |
| coauthor-cs [accuracy] | 0.78 | +1.06 (0.16; p 6.2e-09) | +1.11 (0.23; p 1e-07) | +1.17 (0.22; p 6.5e-08) | +1.01 (0.22; p 1e-07) | +1.06 (0.21; p 6.7e-08) | +0.78 (0.19; p 7.3e-08) | +0.79 (0.18; p 5.9e-08) | +0.47 (0.16; p 8.7e-08) |
| coauthor-physics [accuracy] | 0.87 | +0.77 (0.14; p 4.7e-09) | +0.95 (0.09; p 3.8e-11) | +0.91 (0.09; p 7.3e-11) | +0.82 (0.09; p 6.7e-11) | +0.80 (0.16; p 2.1e-08) | +0.83 (0.19; p 7.5e-08) | +0.82 (0.18; p 4.3e-08) | +0.21 (0.06; p 1.2e-10) |
| cora [accuracy] | 0.77 | +10.84 (1.25; p 2.3e-08) | +13.84 (2.32; p 9.3e-07) | +13.63 (1.99; p 2.6e-07) | +11.92 (2.68; p 1.1e-05) | +12.36 (2.56; p 5.8e-06) | +11.55 (2.87; p 2.6e-05) | +11.40 (2.88; p 3e-05) | +4.46 (0.91; p 1.8e-06) |
| minesweeper [roc_auc] | 0.01 | +10.65 (2.68; p 0.00017) | -0.32 (1.60; p 0.3) | -0.62 (1.57; p 0.38) | -0.28 (1.45; p 0.27) | -0.23 (1.51; p 0.26) | +3.69 (1.15; p 0.00027) | +11.87 (3.01; p 0.00019) | -0.03 (1.22; p 0.16) |
| pubmed [accuracy] | 0.69 | +1.84 (0.47; p 1.7e-06) | +2.65 (0.47; p 1.7e-07) | +2.69 (0.50; p 2.4e-07) | +2.67 (0.83; p 2.3e-05) | +2.56 (0.61; p 2.2e-06) | +1.10 (0.66; p 0.00032) | +1.30 (0.54; p 3.2e-05) | +0.92 (0.35; p 3.6e-06) |
| questions [roc_auc] | 0.02 | +1.29 (0.91; p 0.0052) | +1.96 (1.32; p 0.0089) | +3.51 (1.30; p 0.00078) | +3.75 (0.23; p 3.2e-10) | +7.21 (0.81; p 1.8e-07) | +0.55 (0.53; p 0.0023) | +0.60 (0.48; p 0.0011) | -3.53 (0.65; p 1) |
| roman-empire [accuracy] | -0.05 | -0.02 (0.06; p 1.6e-09) | -13.04 (0.73; p 1) | -13.34 (0.76; p 1) | -7.41 (0.59; p 1) | -7.58 (0.57; p 1) | -0.08 (0.26; p 0.00067) | -0.08 (0.24; p 0.00043) | +2.42 (0.33; p 1.6e-07) |
| squirrel-filtered [accuracy] | 0.01 | -0.23 (0.79; p 0.12) | -14.75 (2.77; p 1) | -15.29 (2.86; p 1) | -16.76 (2.44; p 1) | -16.94 (2.33; p 1) | -0.08 (0.49; p 0.02) | -0.11 (0.43; p 0.012) | -0.07 (0.21; p 0.00013) |
| tolokers [roc_auc] | 0.09 | +1.77 (0.43; p 8.7e-06) | +3.74 (0.58; p 1.1e-06) | +4.15 (0.61; p 8.8e-07) | +3.60 (0.62; p 2.6e-06) | +4.09 (0.72; p 3.8e-06) | +4.45 (0.54; p 2e-07) | +4.60 (0.70; p 1.4e-06) | -0.15 (0.06; p 1.2e-08) |
| wikics [accuracy] | 0.58 | +4.48 (0.67; p 2.3e-11) | +6.57 (0.56; p 3.7e-15) | +6.56 (0.61; p 1.7e-14) | +5.47 (0.42; p 2.9e-16) | +5.45 (0.44; p 7.4e-16) | +5.56 (0.62; p 2.6e-13) | +5.52 (0.62; p 3.3e-13) | +0.74 (0.26; p 7.4e-10) |
| yelpchi [roc_auc] | 0.05 | -0.16 (0.20; p 0.0015) | -6.05 (0.52; p 1) | -4.95 (0.46; p 1) | -7.16 (0.64; p 1) | -5.84 (0.49; p 1) | +2.24 (0.16; p 5e-09) | +2.24 (0.16; p 5e-09) | +0.62 (0.17; p 4.2e-06) |

| method | graphs | NI holds | NI fails | significant harm (> δ below base, Holm) | EDSP-B-tuned vs it: win / tie / loss |
|---|---|---|---|---|---|
| EDSP-tuned | 17 | 15 | chameleon-filtered, squirrel-filtered | — | — |
| correct_and_smooth | 17 | 11 | amazon-fraud, chameleon-filtered, minesweeper, roman-empire, squirrel-filtered, yelpchi | roman-empire, squirrel-filtered, yelpchi | 4 / 11 / 2 |
| cs_ref | 17 | 11 | amazon-fraud, chameleon-filtered, minesweeper, roman-empire, squirrel-filtered, yelpchi | roman-empire, squirrel-filtered, yelpchi | 4 / 8 / 5 |
| cs_2hop | 17 | 12 | amazon-fraud, minesweeper, roman-empire, squirrel-filtered, yelpchi | roman-empire, squirrel-filtered, yelpchi | 4 / 12 / 1 |
| cs_2hop_ref | 17 | 12 | amazon-fraud, minesweeper, roman-empire, squirrel-filtered, yelpchi | roman-empire, squirrel-filtered, yelpchi | 4 / 11 / 2 |
| clp | 17 | 17 | — | — | 1 / 11 / 5 |
| clp_es | 17 | 17 | — | — | 0 / 12 / 5 |
| linbp_b | 17 | 14 | chameleon-filtered, minesweeper, questions | questions | 9 / 6 / 2 |
