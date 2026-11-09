# Final report — EDSP evaluation (state of 2026-11-09)

The algorithm is **EDSP**: information-theoretic per-node dynamic damping with
trust-calibrated homophily/heterophily for node classification in stochastic graphs. It is
evaluated against the label-propagation state of the art (Track A: labels and graph only)
and against Correct & Smooth and tuned GNNs (Track B: on top of a feature model). Every
number below is read from a committed aggregate file. `scripts/final_tables.py` regenerates
the LaTeX tables in `docs/final/` together with `docs/final/DIGEST.txt`, a plain list of
every number this report uses.

What is evaluated:
- **The EDSP core is the algorithm as given (R5, unchanged).** Variants are named, never
  silently substituted:
  - EDSP-default: zero tuning.
  - EDSP-tuned: 9 validation-selected configurations of p₀ × polarity regime.
  - EDSP-fullH (H3): an estimated heterophilic operator in place of the complement.
- **H-LEARNED is part of the evaluation, not of the algorithm.** It is an edge-polarity
  classifier trained on train–train edges, used as one polarity regime. Because it turned out
  to carry most of the wins, A10, A11 and A13 test its contribution separately: they give
  simple signed propagation the same learned polarity (`+HL`).
- **Protocol.** Preregistered hypotheses H1–H7 plus amendments A1–A17 (`docs/PREREGISTRATION.md`),
  each written before the runs it governs. Wilcoxon over splits, Holm across datasets,
  Friedman/Nemenyi for ranks. Real data first. PERTURBED-REAL results (MIX) are kept apart
  from the real-world tables.

## 1. Bottom line

1. **On real graphs EDSP is not #1 against the strongest comparator anywhere.**
   - **Unsigned graphs (E1, 14):** plain LGC is significantly better on 10 and tied on 4. Average
     rank: LGC 1.50, EDSP-tuned 4.14.
   - **Graphs with given signs (E2):** EDSP-tuned ties the signed-propagation baselines
     (signed vote 1 win / 8 ties / 0 losses; signed LGC 0 / 9 / 0).
   - **Against the signed GNNs (B5, A17; 9 graphs):** SGCN 1 / 5 / 3, SNEA 2 / 7 / 0,
     SDGNN 1 / 8 / 0, SiGAT 3 / 4 / 2. Every EDSP win is on wiki-editor or a Senate graph;
     on wiki-editor simple signed LGC does as well. EDSP is #1 by mean on 1 of the 9 graphs
     (congress-senate-100). Exploratory average rank: 5th of 7 (SGCN 1st; Friedman p = 0.28).
   - **Against the fraud GNNs (B6, A17):** CARE-GNN 1 / 0 / 1 and PC-GNN 0 / 1 / 1. On yelpchi
     EDSP-B is 6th of 8, 10.5 AUC points below PC-GNN.
     `docs/B5B6_REPORT.md`.
   - **On top of a feature model (E3):** tuned GNNs beat it almost everywhere (0 wins against
     GCN+, SAGE+, GAT+, GPR-GNN).
2. **Where EDSP's own mechanism significantly beats a strong comparator of its own kind:**
   - **As a post-hoc smoother on top of a feature model (Track B), against C&S** on the same base
     predictor (4 / 3 / 6 overall), notably under heterophily:
     - roman-empire +13.0
     - squirrel-filtered +14.5
     - amazon-fraud AUC 96.62 vs 95.65

     There C&S, which assumes homophily, drags the base predictor down (roman-empire 64.3 → 51.3),
     while EDSP-B never falls significantly below its base predictor on any of the 13 real
     graphs (9 / 4 / 0). This last point is exploratory, not preregistered. On yelpchi, run
     later for B6, EDSP-B is 0.16 below its base predictor (raw p = 0.037). That is not
     significant after Holm, but it is not a gain either.
   - **With a noisy learned polarity on fraud graphs (A10)**, against simple signed propagation
     fed the same polarity: amazon-fraud +1.74 AUC, yelpchi +4.64. Here EDSP is itself only
     tied with (amazon-fraud) or below (yelpchi) plain LGC.
   - **On perturbed-real graphs (A11), 9 of the 17 confirmed MIX wins survive** the same-polarity
     comparison. Six of them are free of the generator-locality caveat: wiki-editor with edges
     re-wired uniformly at heterophily h = 0.1–0.4 (`global` / `permutation`), +0.2 to +2.3
     points. They come from a single base graph, and none of the `loc = 0` wins is EDSP's own.
3. **Every other "win" belongs to the learned polarity (H-LEARNED), not to EDSP's
   propagation:**
   - the 8 sign-blind Congress / wiki-elec wins (E2U, A10)
   - 6 of the 17 MIX wins (A11)
   - the 4 Facebook100 wins on natively mixed graphs (A13)
4. **H5 is supported, on one graph (ogbn-proteins, 112 tasks, A15).**
   - EDSP with the native edge confidences beats EDSP with a constant edge probability:
     +1.01 ROC-AUC points, better on 108 of 112 tasks (p = 1.9 × 10⁻¹⁹).
   - The gain is generic edge weighting. Harmonic propagation gains as much from the same
     weights (+0.85), and EDSP's untuned gain is indistinguishable from harmonic's.
   - EDSP stays 4.7 points below PPR-LP+w and LGC+w, and below them on 108–109 of 112 tasks.
   - `docs/H5_REPORT.md`.

## 2. Preregistered hypotheses

| | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| H1 | Track-A EDSP ≈ harmonic LP / LGC on homophilous graphs | **descriptive** (no "≈" rule was preregistered) | Close to harmonic LP (D2 reduction), clearly below LGC; `docs/E1_REPORT.md` |
| H2 | Track-A near chance on minesweeper, tolokers, questions | **descriptive** | Near chance on minesweeper only; tolokers and questions are clearly above chance; E1 |
| H3 | EDSP-fullH > EDSP on structured-compatibility graphs, gap growing with k | **not supported** | C3d: 1 win / 2 ties / 2 losses; gap vs k Spearman ρ = −0.36 (p = 0.89); roman-empire +16.6, wikics −23.5; `docs/E5_REPORT.md` |
| H4 | Track-B EDSP ≈ C&S < tuned GNNs on feature-rich heterophilous graphs | **inconclusive** (both A14 modes) | EDSP-B ≈ C&S on 2 / 6 C2 graphs, best GNN significantly better on 5 / 6; `docs/E3_REPORT.md` |
| H5 | P-NATIVE > P-CONST on ogbn-proteins / uncertain DBLP | **supported** (ogbn-proteins only) | 112 tasks (A15): EDSP-native-tuned − EDSP-const-tuned +1.01 AUC [0.85, 1.18], 108 / 4, p = 1.9 × 10⁻¹⁹. One graph, one split. The effect matches weighted vs unweighted harmonic propagation (+0.85). Uncertain DBLP not obtainable; `docs/H5_REPORT.md` |
| H6 | Finite T_L helps under label noise | **not supported** | 14 graphs × 11 noise conditions (A16): EDSP-TL vs EDSP-default 1 win / 13 ties in both primary conditions (uniform and pairwise 30 %); mean effect ≤ 0.05 points; `docs/H6_REPORT.md` |
| H7 | Convergence / oscillation / fixed-point non-uniqueness (descriptive) | **reported** | EDSP-default did not converge in 3 of 153 E1 real-graph runs (Planetoid public splits); no period-2 oscillation flagged; E1 report §H7 |

Later confirmatory tests, each preregistered:

| Test | Verdict |
|---|---|
| A4: MIX confirmatory | 17 of 110 perturbed conditions are confirmed wins: 14 on wiki-editor (k = 2), 3 on amazon-ratings (k = 5); 0 on cora, citeseer, roman-empire |
| A8 (i): real graphs with polarity hidden | Supported: wins on 7 Congress graphs + wiki-elec |
| A8 (ii): without generator locality | Partly cleared: 5 of 8 wiki-editor wins survive `loc = 0` |
| A10: whose win (real) | The learned polarity's: 9 / 9 ties vs `+HL` on the Congress graphs and wiki-editor; EDSP wins only on the fraud graphs, where `+HL` collapses |
| A11: whose win (MIX) | 9 of 17 EDSP's own (6 locality-free); 0 of the `loc = 0` wins |
| A13: native near-random mixing (13 graphs) | **inconclusive**: 4 wins (all FB100) / 9 losses; the 4 wins belong to the learned polarity (2 ties, 2 losses vs `+HL`) |

## 3. Win / tie / loss of EDSP-tuned against each baseline (real graphs)

Wilcoxon over splits, Holm across datasets per baseline. `docs/final/tab_wtl.tex`; B5 / B6 in
`docs/final/tab_b5b6.tex`.

| Suite | Baseline: win / tie / loss |
|---|---|
| E1 (unsigned, 14) | Harmonic 4/9/1 · Harmonic+CMN 2/10/2 · **LGC 0/4/10** · PPR-LP 1/9/4 · LinBP 10/2/2 · train prior 12/0/2 |
| E2 (signs or relations given) | Harmonic 9/1/1 · Harmonic+CMN 9/1/1 · LGC 9/1/1 · PPR-LP 10/0/1 · LinBP 11/0/0 · train prior 11/0/0 · **Signed vote 1/8/0 · Signed LGC 0/9/0** |
| E3 (Track B, 13) | **LogReg base 9/4/0** · **C&S 4/3/6** · MLP 7/3/3 · GCN+ 0/3/9 · SAGE+ 0/1/11 · GAT+ 0/3/9 · GPR-GNN 0/2/10 · BWGNN 1/2/10 |
| B5 (signed, 9; A17) | SGCN 1/5/3 · SNEA 2/7/0 · SDGNN 1/8/0 · SiGAT 3/4/2 |
| B6 (fraud, 2; A17; EDSP-B-tuned) | CARE-GNN 1/0/1 · PC-GNN 0/1/1 |

Average ranks (Friedman over the graphs where every ranked method exists; `tab_ranks.tex`):
- **E1 (14 graphs):** LGC 1.50, Harmonic+CMN 3.32, PPR-LP 3.71, **EDSP-tuned 4.14**,
  Harmonic 4.68, EDSP-default 5.64, LinBP 6.00, train prior 7.00.
- **E2 (11 graphs):** **EDSP-tuned 1.73**, EDSP-default 2.45, LGC 4.36, … The signed baselines
  do not run on the two relation-typed fraud graphs, so they cannot enter a ranking that needs
  every method on every graph. On the 9 graphs where they do run, EDSP-tuned ties them (above).
- **E3 (12 graphs):** SAGE+ 2.92, GAT+ 3.08, BWGNN 3.17, GCN+ 3.75, GPR-GNN 3.92, C&S 5.83,
  **EDSP-tuned 7.17**, MLP 7.83, EDSP-default 8.25, LogReg 9.08.

## 4. Practical reading
- **Use EDSP as a post-hoc graph smoother on top of an existing classifier when it is unknown
  whether the graph is homophilous.** It does not damage the base predictor where C&S does, and
  it sometimes adds a lot (minesweeper +10.7 AUC, a constructed benchmark). If a GNN can be
  trained end to end, the GNN is better.
- **For label-only propagation on unsigned graphs, plain LGC is the better choice.**
- **On graphs with edge polarity (signed, or polarity learnable from structure), the value is
  in the polarity, not in EDSP.** Simple signed propagation fed that polarity does as well.
- **Probabilities:** EDSP's NLL is higher than the best baseline's on every E3 graph; no
  calibration advantage is claimed.

## 5. Not run, and limitations
- **H5** rests on one graph and one split: ogbn-proteins, 112 tasks. Uncertain DBLP, the
  US-Patent assignee graph and raw STRING + GO-slim could not be obtained as specified. The
  single-channel P-NATIVE modes are substituted by max- and mean-channel modes, because OGB does
  not name the columns.
- **Signed GNNs (B5)** run as label-free link-sign embeddings plus logistic regression, on
  undirected graphs. The direction-specific parts of SDGNN and SiGAT are inert, which favours
  EDSP.
- **CARE-GNN and PC-GNN (B6)** are re-implementations; parity with the published numbers is
  TO VERIFY. GAGA and H²-FDetector are not run.
- **E3 on coauthor-cs / coauthor-physics (A12)** is reported in `docs/E3_REPORT.md`
  (homophilous, not in H4). EDSP-B-tuned is within 0.35 points of the best GNN, and GPR-GNN
  beats it on both graphs. On physics, SAGE+ (9 / 10 splits) and GAT+ (3 / 10) hit the
  330-minute CI limit and are NOT-RUN.
- **E3 on fraud (A9)** runs only MLP and BWGNN, at hidden 64. On tolokers, GAT+ is at hidden 64
  (A14), which favours EDSP, and the verdict is unchanged with GAT+ left out.
- **GNNs** are re-implementations with 9-point grids, trained on CPU runners with early
  stopping; re-runs differ by up to 0.5 points. Published numbers may differ.
- **The locality-free MIX wins** come from one base graph (wiki-editor).
- **Track A uses labels only.** Feature-based published results on the A13 graphs are not
  comparable and are not used.

## 6. Where the details are
| Document | Covers |
|---|---|
| `docs/E1_REPORT.md` | Unsigned graphs, H1, H2, H7 |
| `docs/E2_REPORT.md` | Signed and relation graphs |
| `docs/CAVEATS_REPORT.md` | A8, A10 |
| `docs/E5_REPORT.md` | H3 |
| `docs/MIX_REPORT.md` | A2, A4, A11 |
| `docs/A13_REPORT.md` | Native mixing |
| `docs/E3_REPORT.md` | Track B, H4 |
| `docs/B5B6_REPORT.md` | Signed GNNs and fraud GNNs (A17) |
| `docs/H5_REPORT.md` | H5, native edge probabilities (ogbn-proteins) |
| `docs/H6_REPORT.md` | H6, label noise |
| `docs/AUDIT_2026-11-07.md` | Independent audit |

LaTeX:
- `docs/final/tab_*.tex`: win/tie/loss, ranks, H4, A11, A13, B5 / B6.
- `docs/final/tab_headline_*.tex`: per-suite mean ± std tables.
