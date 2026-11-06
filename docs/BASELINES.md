# Baselines (Master Prompt §4)

Status of every baseline. "Parity" compares our run with a REPORTED number
(`data/PUBLISHED_BASELINES.json`, tolerance 1 point); REPORTED numbers are never
mixed into REPRODUCED columns. Evidence for B0/B1 is the `baseline-parity`
workflow run on the PR that introduced them.

| id | baseline | group | track | implementation | hyper-parameters (selected by the validation value of the dataset's headline metric) | parity |
|---|---|---|---|---|---|---|
| majority | majority class | B0 | A | `edsp/baselines/propagation.py` | — | n/a |
| train_prior | train class prior | B0 | A | same | — | n/a |
| signed_vote | Σ sign·1[y=c] over train neighbours | B0 | A | same | — | n/a (signed graphs only) |
| signed_lgc | Label spreading with the signed normalised adjacency on centred one-hot labels (Kunegis et al. 2010; signed-Laplacian LP family of Mercado et al. 2019) | B1 | A | 9-point α grid (as LGC) | — | n/a (signed graphs only) |
| feature_lr | logistic regression on node features (= Track-B base predictor, temperature-scaled) | B0 | B | `edsp/regimes/priors.py` | l2 × standardise | n/a |
| harmonic | Zhu et al. 2003, exact solve, uncovered components uniform | B1 | A | `propagation.py` | — | Planetoid public vs "LP" of Yang et al. 2016 — see workflow; reference TO VERIFY |
| harmonic_cmn | harmonic + class mass normalisation | B1 | A | same | — | same reference |
| lgc | Zhou et al. 2004, exact solve | B1 | A | same | α ∈ {0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999} (9 points, rule R6) | no reference on these splits (documented) |
| ppr_lp | PPR label propagation / training-free APPNP, exact solve | B1 | A | same | α ∈ same 9 points | none (documented) |
| linbp | LinBP* with myopic compatibility estimation (MCE) | B1 | A | same | scale ∈ {0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95} | none (documented) |
| correct_and_smooth | Huang et al. 2021, autoscale, same base predictor as Track-B EDSP | B1 | B | same | α_correct, α_smooth ∈ {0.5, 0.8, 0.95} | C&S paper uses other splits; none (documented) |
| possible_worlds_harmonic | harmonic LP averaged over sampled worlds | B1 | A | same | samples ∈ {10, 50} (fixed) | n/a (probabilistic graphs, E2) |

Not yet implemented (later PRs): LinBP with DCE (Gatterbauer & Kumar 2020; needed when train–train edges are scarce), ZooBP (code TO VERIFY), signed-Laplacian LP and Mercado et al. GL, uBayes/uBayes+ (C++), HAPPY/H-PPR, fixed-aggregation features, REV2 — and all B2–B6 GNN/transformer/signed/fraud baselines (PR-6).
