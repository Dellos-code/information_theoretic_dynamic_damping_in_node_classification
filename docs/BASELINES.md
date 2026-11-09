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
| signed_lgc | Label spreading with the signed normalised adjacency on centred one-hot labels (Kunegis et al. 2010; signed-Laplacian LP family of Mercado et al. 2019) | B1 | A | same | α, 9-point grid (as LGC) | n/a (signed graphs only) |
| feature_lr | logistic regression on node features (= Track-B base predictor, temperature-scaled) | B0 | B | `edsp/regimes/priors.py` | l2 × standardise | n/a |
| harmonic | Zhu et al. 2003, exact solve, uncovered components uniform | B1 | A | `propagation.py` | — | Planetoid public vs "LP" of Yang et al. 2016 — see workflow; reference TO VERIFY |
| harmonic_cmn | harmonic + class mass normalisation | B1 | A | same | — | same reference |
| lgc | Zhou et al. 2004, exact solve | B1 | A | same | α ∈ {0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999} (9 points, rule R6) | no reference on these splits (documented) |
| ppr_lp | PPR label propagation / training-free APPNP, exact solve | B1 | A | same | α ∈ same 9 points | none (documented) |
| linbp | LinBP* with myopic compatibility estimation (MCE) | B1 | A | same | scale ∈ {0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95} | none (documented) |
| correct_and_smooth | Huang et al. 2021, autoscale, same base predictor as Track-B EDSP | B1 | B | same | α_correct, α_smooth ∈ {0.5, 0.8, 0.95} | C&S paper uses other splits; none (documented) |
| cs_2hop | C&S 2-hop (Lim et al., NeurIPS 2021): C&S whose propagation applies S = D^-1/2 A D^-1/2 twice per step (their `num_hops=2`; S² never formed), same base predictor as Track-B EDSP | B1 | B | `edsp/baselines/posthoc.py` → `propagation.correct_and_smooth(hops=2)`; behaviour of CUAI/Non-Homophily-Large-Scale@82f8f05 | α_correct, α_smooth ∈ {0.5, 0.8, 0.95} (as C&S) | not run (REPORTED values are for the LINKX graphs with an MLP base); SPEC_AMBIGUITIES "C&S 2-hop" |
| clp | Compatible Label Propagation (Zhong, Ivanov & Pang, TMLR 2022): Sinkhorn-normalised compatibility from train labels + base predictions, per-class compatibility edge weights, 50 layers, same base predictor | B1 | B | `edsp/baselines/posthoc.py`, port of zhiqiangzhongddu/TMLR-CLP@2f7901d (no code copied) | α ∈ {0.1, 0.5, 0.9} × start matrix ∈ {base, base+train, train} | not run; the official notebook selects on test accuracy (SPEC_AMBIGUITIES "CLP") |
| linbp_b | LinBP* (MCE) whose explicit beliefs are the base predictor's centred outputs (train nodes: centred one-hot) | B1 | B | `edsp/baselines/posthoc.py` → `propagation.linbp(priors=...)` | scale, the 9 LinBP points | n/a (our construction) |
| possible_worlds_harmonic | harmonic LP averaged over sampled worlds | B1 | A | same | samples ∈ {10, 50} (fixed) | n/a (probabilistic graphs, E2) |
| sgcn | SGCN (Derr et al., ICDM 2018) node embeddings + logistic regression | B5 | A | `edsp/baselines/signed_gnn.py`: PyGSD 1.2.0 `nn.signed.SGCN` (commit 2968b55), optional install `requirements/signed-gnn.txt` | embedding size {20, 32, 64} × LR l2 {1e-3, 1e-2, 1e-1}; early stopping on validation (A7) | not run (signed graphs only; see SPEC_AMBIGUITIES "Signed-GNN protocol") |
| snea | SNEA (Li et al., AAAI 2020), same protocol | B5 | A | same, `SNEA` | same | not run |
| sdgnn | SDGNN (Huang et al., AAAI 2021), same protocol; sees the symmetrised graph | B5 | A | same, `SDGNN` | same | not run |
| sigat | SiGAT (Huang et al., ICANN 2019), same protocol; sees the symmetrised graph | B5 | A | same, `SiGAT` | same | not run |
| care_gnn | CARE-GNN (Dou et al., CIKM 2020): label-aware similarity, RL-adjusted per-relation neighbour filtering, threshold-weighted inter-relation aggregation | B6 | B | `edsp/baselines/fraud_gnn.py`: plain-PyTorch re-implementation of YingtongDou/CARE-GNN@a64ff75 | lr {0.01, 0.005, 0.001} × λ_1 {1, 2, 4}; embedding 64; early stopping on validation ROC-AUC | not run (relation-typed graphs only); REPORTED values to add |
| pc_gnn | PC-GNN (Liu et al., WWW 2021): label-balanced pick, neighbour choose with minority over-sampling | B6 | B | same file, re-implementation of PonderLY/PC-GNN@9d7d7fa | lr {0.01, 0.005, 0.001} × ρ {0.2, 0.5, 0.8}; embedding 64; early stopping on validation ROC-AUC | not run; REPORTED values to add |

Not yet implemented (later PRs): LinBP with DCE (Gatterbauer & Kumar 2020; needed when train–train edges are scarce), ZooBP (code TO VERIFY), signed-Laplacian LP and Mercado et al. GL, uBayes/uBayes+ (C++), HAPPY/H-PPR, fixed-aggregation features, REV2 — B3 graph transformers, GAGA and H²-FDetector (B6). The B2 / B4 GNNs and BWGNN (B6) are in `edsp/baselines/gnn.py` (PR-12; protocol in docs/SPEC_AMBIGUITIES.md "GNN baselines (PR-12)"); B5 and CARE-GNN / PC-GNN (B6) are listed above (PR-29). The Track-B post-hoc comparators `cs_2hop`, `clp`, `linbp_b` (PR-39) run only with `scripts/run_e1.py --track B --posthoc cs2hop,clp,linbp_b`.
