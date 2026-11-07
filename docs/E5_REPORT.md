# E5 report — EDSP-fullH: estimated compatibility instead of the complement (H3)

**What was tested.** The named variant EDSP-fullH (Master Prompt §2.5): on edges with
h_ij = 1 the message is V Ĥ instead of the canonical complement C(V) = (1 − V)/(k − 1),
with Ĥ the row-normalised, Laplace-smoothed (+1) class counts on the train–train
edges that the active polarity regime marks heterophilic (`edsp/regimes/compat.py`).
With Ĥ = (J − I)/(k − 1) it reproduces canonical EDSP exactly, and with Ĥ = I and all
edges heterophilic it reproduces all-homophilic EDSP exactly (both checked to 0.0).
EDSP-fullH-default uses the EDSP-default inputs, EDSP-fullH-tuned the same 9-point
grid as EDSP-tuned (validation-selected). Decision rule: PREREGISTRATION A5.

**Provenance (R1).** 2,123 run records in `results/runs/e5/`, Actions run
[42364282669](https://github.com/Dellos-code/information_theoretic_dynamic_damping_in_node_classification/actions/runs/42364282669)
(commit b7d3513); all E1 datasets, all splits, no NOT-RUN. Tables:
`results/aggregates/e5/E5_SUMMARY.md`. PERTURBED-REAL part: 8,500 records (incl. 250 DATASET-STATS) in
`results/perturbed/fullh/runs/` from run
[42364290816](https://github.com/Dellos-code/information_theoretic_dynamic_damping_in_node_classification/actions/runs/42364290816)
(seeds 10–19 × 3 splits = 30 paired runs per condition), tables in
`results/perturbed/fullh/aggregates/MIX_SUMMARY.md`.

## 1. H3 on real data — not supported

EDSP-fullH-tuned vs EDSP-tuned (headline metric ×100; Wilcoxon over splits; Holm across
the five C3d datasets):

| dataset | k | C3d | EDSP-tuned | EDSP-fullH-tuned | Δ | p (Holm) | verdict | LinBP |
|---|---|---|---|---|---|---|---|---|
| roman-empire | 18 | yes | 9.06 | 25.65 | +16.58 | 0.0078 | **win** | 27.54 |
| coauthor-cs | 15 | yes | 91.17 | 91.18 | +0.01 | 0.75 | tie | 89.64 |
| questions [AUC] | 2 | yes | 64.56 | 64.71 | +0.14 | 0.21 | tie | 40.92 |
| amazon-computers | 10 | yes | 87.77 | 87.57 | −0.20 | 0.023 | loss | 85.42 |
| wikics | 10 | yes | 67.21 | 43.72 | −23.48 | 9.5e-6 | **loss** | 63.35 |

- **Part 1 (C3d): 1 win / 2 ties / 2 losses → not supported** (rule: ≥ 3 wins and no loss).
- **Part 2 ("the gap grows with k"):** Spearman ρ between k and the gap over all 14
  real datasets = −0.36, one-sided p = 0.89 → **not supported**.

Reading:
- **roman-empire (18 classes, structured compatibility):** the one place where the
  complement assumption is clearly wrong on real data. fullH lifts EDSP from 9.1 % (below
  the 14.0 % majority class) to 25.7 %, close to LinBP (27.5 %) — but every label-only
  method stays far below what feature-based GNNs reach on this dataset.
- **wikics:** fullH collapses (67.2 → 43.7). With ≈ 5 % training labels there are only
  ≈ 160–180 train–train edges that H-LEARNED marks heterophilic, for a 10 × 10 matrix with
  Laplace +1 per cell; Ĥ is then dominated by smoothing and the heterophilic messages
  carry noise instead of the complement's "not this class" signal (validation accuracy of
  the H-LEARNED configurations: 0.65 with the complement, 0.23 with Ĥ). The variant needs
  enough labelled heterophilic edges to estimate Ĥ.
- Elsewhere the differences are ≤ 0.6 points: with H-TRAINONLY few edges are marked
  heterophilic, so the operator rarely acts.

## 2. Structured mixing on re-wired real graphs (PERTURBED-REAL, exploratory)

Secondary, A5. Mean headline accuracy ×100 over 30 paired runs; p values are
unadjusted two-sided Wilcoxon (exploratory). Conditions: global h = 0 / 0.2,
permutation h = 0.1 / 0.3 (heterophilic edges go to one fixed other class), community
(half the classes homophilic, half heterophilic).

| base (k) | condition | EDSP-tuned | EDSP-fullH-tuned | LinBP |
|---|---|---|---|---|
| amazon-ratings (5) | permutation 0.1 / 0.3 | 58.9 / 38.9 | 80.9 / 68.5 | **83.6 / 75.1** |
| amazon-ratings (5) | community | 68.2 | **87.6** | 85.9 |
| amazon-ratings (5) | global 0 | **76.1** | 59.7 | 74.8 |
| cora (7) | permutation 0.1 / 0.3 | 45.1 / 30.6 | 51.7 / 45.4 | **57.8 / 48.0** |
| cora (7) | community | 50.3 | **59.8** | 55.7 |
| citeseer (6) | permutation 0.1 / 0.3 | 43.4 / 30.4 | 50.0 / 42.0 | **55.0 / 45.2** |
| citeseer (6) | community | 45.7 | **56.5** | 53.9 (p = 0.12 vs fullH) |
| roman-empire (18) | permutation 0.1 / 0.3 | 48.9 / 35.8 | 65.3 / 54.9 | **65.7 / 55.7** |
| roman-empire (18) | community | 50.0 | **57.1** | 56.4 |
| wiki-editor (2) | community | **70.9** | 51.6 | 63.0 |

- With k > 2 and structured heterophily, fullH lifts EDSP by 6–30 points in every
  permutation and community condition and closes most of the gap to LinBP; it is
  ahead of LinBP in the community conditions (amazon-ratings, cora, roman-empire;
  citeseer not significant), still behind it in the permutation conditions.
- It hurts where the complement is already right: k = 2 (wiki-editor community
  −19.3) and uniform heterophily (amazon-ratings global h = 0, −16.4).

## 3. Bottom line

The complement operator is the right default when heterophily is uniform or k = 2,
and the wrong one when heterophily is structured with many classes. Replacing it by an
estimated Ĥ fixes the latter (roman-empire, re-wired structured conditions) but hurts
the former and fails when few heterophilic training edges exist (wikics). H3 as
pre-registered is **not supported** on real data. A natural follow-up — choosing between
C and Ĥ on validation, i.e. adding the operator to EDSP's tuning grid — was not
pre-registered and would enlarge EDSP's tuning budget; it is listed, not run.
