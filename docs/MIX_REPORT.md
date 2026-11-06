# MIX report — PERTURBED-REAL mixing sweep (exploratory)

**Not real-world results (rule R4).** Real nodes, labels and splits of five real
graphs; only the edges are re-drawn by `edsp/data/perturb.py` to a controlled
mixing pattern. These numbers never enter the E1/E2 tables or ranks.

**Design (PREREGISTRATION A2).** Base graphs: cora (k = 7), citeseer (6),
amazon-ratings (5), roman-empire (18), wiki-editor (2; its edge signs are dropped,
so it is an unsigned graph here). 22 conditions:

| family | conditions | what it models |
|---|---|---|
| global | h = 0, 0.1, …, 1 | homogeneous mixing, every node has the same target homophily |
| beta | (m, c) = (0.5, 0.5), (0.5, 5), (0.2, 0.5), (0.8, 0.5) | heterogeneous: per-node homophily ~ Beta; c = 0.5 is U-shaped (nodes are either homophilic or heterophilic) |
| uniform | — | per-node homophily ~ U(0, 1), independent |
| logistic | — | per-node homophily from the logistic map x ← 4x(1−x): deterministic chaos |
| regional | 20 or 200 seed regions, alternately h = 0.9 / 0.1 | spatially coherent homophilic and heterophilic regions |
| community | half the classes h = 0.9, half 0.1 | class-dependent mixing |
| permutation | h = 0.1, 0.3; heterophilic edges go to one fixed other class | structured (non-uniform) heterophily |

3 re-wiring seeds × the first 3 splits = 9 paired runs per condition. Methods:
EDSP-default, EDSP-tuned (the E1 grid) and the E1 Track-A baselines.

**Provenance (R1).** 9,240 records in `results/perturbed/runs/`, Actions run
[42214181850](https://github.com/Dellos-code/information_theoretic_dynamic_damping_in_node_classification/actions/runs/42214181850)
(commit 2ff5600). Full tables (achieved h_edge, adjusted homophily, label
informativeness, spread of local homophily, every method ± std):
`results/perturbed/aggregates/MIX_SUMMARY.md`.

## 0. The pre-registered test could not reject anything — a design error

A2 fixed a Wilcoxon test over the 9 paired runs per condition, Holm-corrected over
the 22 conditions of a base graph. With 9 pairs the smallest two-sided p is
2/2⁹ = 0.0039, so the smallest Holm-adjusted p is 22 × 0.0039 = 0.086 > 0.05:
**no condition could come out significant, whatever the data.** That is what
happened (all adjusted p ≥ 0.078). The error is in the design I wrote, not in the
data. Everything below is therefore **descriptive and exploratory**; amendment A4
fixes a confirmatory run with fresh seeds and enough pairs, decided before it runs.

As a descriptive measure the tables give, per condition, in how many of the 9
paired runs EDSP-tuned beat the best baseline (the baseline with the highest mean
in that condition, chosen after the fact, which favours the baselines).

## 1. Where EDSP-tuned is ahead (all 9 of 9 paired runs)

| base | condition | EDSP-tuned | best baseline | Δ |
|---|---|---|---|---|
| wiki-editor (k = 2) | global h = 0.1 / 0.2 / 0.3 / 0.4 | 95.7 / 92.1 / 85.4 / 73.5 | LinBP 95.2 / 91.3 / 84.7 / 72.8 | +0.5 / +0.8 / +0.7 / +0.6 |
| wiki-editor | beta m=0.2 c=0.5 | 91.6 | LinBP 86.1 | +5.5 |
| wiki-editor | beta m=0.5 c=0.5 | 88.7 | LGC 74.5 | +14.2 |
| wiki-editor | beta m=0.5 c=5 | 70.8 | LGC 64.2 | +6.6 |
| wiki-editor | beta m=0.8 c=0.5 | 90.8 | LGC 88.7 | +2.1 |
| wiki-editor | uniform | 78.8 | LGC 68.0 | +10.8 |
| wiki-editor | logistic (chaotic) | 84.6 | LGC 71.6 | +13.0 |
| wiki-editor | regional, 20 regions | 91.2 | LinBP 73.4 | +17.7 |
| wiki-editor | community | 72.8 | LinBP 69.2 | +3.6 |
| wiki-editor | permutation h = 0.1 / 0.3 | 96.0 / 85.5 | LinBP 95.3 / 84.8 | +0.6 / +0.7 |
| amazon-ratings (k = 5) | global h = 0 | 76.1 | LinBP 75.2 | +1.0 |
| amazon-ratings | beta m=0.2 c=0.5 | 60.6 | LinBP 53.9 | +6.7 |
| amazon-ratings | beta m=0.5 c=0.5 | 64.2 | LGC 61.4 | +2.8 |

Also ahead on average: wiki-editor regional with 200 regions (+6.9, 6 of 9 runs),
citeseer beta m=0.2 c=0.5 (+0.5, 6 of 9).

## 2. Where it is behind

- **Homophilic mixing (global h ≥ 0.5 on every base, wiki-editor included):**
  EDSP ≈ plain harmonic LP, as the α = 1 reduction predicts; harmonic+CMN or LGC is
  ahead by 0–5.8 points (largest: citeseer h = 0.8, −5.8; wiki-editor h = 0.6, −5.3).
- **Structured heterophily with k > 2 (permutation, community on cora, citeseer,
  amazon-ratings, roman-empire):** LinBP-MCE, which estimates the full
  compatibility matrix, is ahead by 6.5–25.4 points. EDSP's complement operator
  assumes heterophilic mass is spread uniformly over the other k − 1 classes; when
  it goes to one class, the operator throws that information away. With k = 2 the
  two coincide, which is why the same conditions on wiki-editor are EDSP wins.
  This is hypothesis H3 (EDSP-fullH), still to be run in E5.
- **Heterogeneous mixing with k ≥ 6 (cora, citeseer, roman-empire):** behind LGC or
  harmonic+CMN by 0.8–4.5 points in every beta / uniform / logistic / regional
  condition except citeseer beta m=0.2 c=0.5 (+0.5).
- **Weak uniform heterophily with k > 2 (global h = 0.1–0.3):** on cora, citeseer
  and roman-empire (label informativeness 0.01–0.09) every method is near the
  class prior and EDSP-tuned is 0.4–5.5 points behind the best baseline; on
  amazon-ratings LinBP is far ahead at h = 0.1 and 0.2 (58.9 and 43.8 vs 47.3 and
  29.6).

## 3. Why EDSP-tuned wins where it wins (and a caveat)

- EDSP-default (H-TRAINONLY with homophilic fill) behaves like harmonic LP in every
  condition. All the gains come from the polarity regime EDSP-tuned selects on
  validation (counts over the run records):
  - heterogeneous mixing on wiki-editor (beta, uniform, logistic, regional,
    community): **H-LEARNED in 81 of 81 runs** — per-edge polarity predicted by a
    classifier trained on train–train edges from label-free structural features
    (degrees and a spectral embedding);
  - uniform low-h mixing on wiki-editor (permutation 17 of 18 runs; global h = 0.1–0.4: 31 of 36):
    "all edges heterophilic", which for k = 2 is the exact class flip;
  - amazon-ratings heterogeneous conditions: H-LEARNED in 39 of 81 runs.
- **Caveat on the generator.** To keep heterogeneous mixing local, `perturb.py` draws
  80 % of partners from the anchor's own stratum (nodes with similar target
  homophily, or the same region). That makes the strata structurally visible, so
  the spectral embedding can tell heterophilic regions from homophilic ones, which is
  exactly what H-LEARNED exploits. The result is therefore: *when per-edge polarity is
  predictable from structure, EDSP's per-edge polarity mechanism turns it into
  accuracy, which LGC / LinBP cannot*. It is not evidence that real heterogeneous
  graphs have such structure; E1 (§5, §6 of E1_REPORT) says the standard real
  benchmarks do not expose it to a label-only method.
- The k = 2 vs k > 2 pattern matches the theory: with two classes "not my
  neighbour's class" determines the class, with five it leaves four candidates.

## 4. Confirmatory follow-up (PREREGISTRATION A4)

Same 22 conditions × 5 base graphs, **fresh re-wiring seeds 10–19** × 3 splits = 30
paired runs per condition (smallest Holm-adjusted p ≈ 22 × 2/2³⁰ ≪ 0.05); test and
decision rule fixed in A4 before that run starts.
