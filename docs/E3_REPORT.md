# E3 report — Track B: EDSP on top of a feature model, against C&S and tuned GNNs (H4)

**Question.** Give EDSP the same feature-based prior as Correct & Smooth: the
logistic-regression base predictor `feature_lr`, i.e. Track B. How does it compare with:
- the base predictor itself,
- C&S on the same predictor,
- six GNNs trained end to end?

**Answer in brief:**
- **H4 is INCONCLUSIVE** by the A7 rule, with the same verdict under both A14 variants for
  GAT+ on tolokers.
  - The GNN half of the prediction holds: the best tuned GNN is significantly better than
    EDSP-B-tuned on 5 of the 6 C2 graphs.
  - The "EDSP-B ≈ C&S" half fails: they are indistinguishable on only 2 of 6. EDSP-B is
    clearly better on two (roman-empire +13.0, squirrel-filtered +14.5) and clearly worse on
    two (amazon-ratings −6.6, tolokers −2.0).
- **Against its own base predictor, EDSP-B-tuned never loses:** 9 wins, 4 ties, 0 losses on
  the 13 real graphs. This is an exploratory observation, not a preregistered test.
- **Against C&S it is mixed (4 / 3 / 6).** It wins on amazon-fraud, amazon-photo,
  roman-empire and squirrel-filtered. It loses on the homophilous graphs cora, pubmed,
  amazon-computers and wikics, and on amazon-ratings and tolokers.
- **Against the tuned GNNs it essentially always loses:** 0 wins against GCN+, SAGE+, GAT+
  and GPR-GNN (8–11 losses each), and 1 against BWGNN (citeseer). Its average rank is 7.17
  of 10, behind all five message-passing GNNs and C&S.
- **Its probabilities are worse calibrated:** EDSP-B-tuned's NLL is higher than the best
  baseline's on every graph.

## Setup (PREREGISTRATION A7, A9, A12, A14)
- **Methods.** EDSP-B-default / -tuned with Track-B priors, `feature_lr`, C&S (3 × 3 grid), and
  the GNNs of `edsp/baselines/gnn.py`: MLP, GCN+, SAGE+, GAT+, GPR-GNN, BWGNN.
  - The GNNs are re-implementations, not the authors' code.
  - Each GNN has a 9-point grid, validation early stopping and selection on the headline
    metric, at hidden width 256.
- **Datasets and splits.** The real E1 datasets except sampson (no features). One CI job per
  split, or per split and GNN group for the heavy graphs.
- **Fraud (A9).** amazon-fraud runs feature_lr, C&S, EDSP-B, and MLP and BWGNN at hidden 64.
  GCN+, SAGE+, GAT+ and GPR-GNN are NOT-RUN there for compute reasons. Four early hidden-256
  GCN records are left out of the assembly on purpose (`--drop amazon-fraud:gcn`).
- **GAT+ fallback (A14, written before any of these jobs finished).**
  - tolokers: full-size GAT+ exceeded the 330-minute CI limit on 5 splits and finished on
    only 1 of 10. GAT+ is therefore reported at hidden 64 for all 10 splits (column `gat`
    in mode `h64`, `results/aggregates/e3`). In mode `omit` it is left out of "best GNN"
    (`results/aggregates/e3-gat-omitted`).
  - questions: full-size GAT+ finished on all 10 splits, so it is used. Its hidden-64 run is
    shown as an extra column `gat(h64)`.
- **Coauthor (A12).** coauthor-cs and coauthor-physics are running separately and will be
  added in a later PR. They are homophilous and do not enter H4.
- **minesweeper** is a constructed benchmark (†) and is excluded from all real-world
  aggregates.

Records are in `results/runs/e3/`, assembled from 76 per-dispatch results branches plus
main by `scripts/assemble_runs.py`; see `ASSEMBLY.md` there for sources, conflicts and drops.
- 28 records exist twice with different values:
  - Most are amazon-fraud records from the pre-A9 runs; the official A9 records on main are
    kept.
  - The others are GNN re-runs on questions and amazon-computers that differ by at most 0.5
    points (CPU training is not bit-reproducible across machines); the earliest run is kept. The
tables are `results/aggregates/e3/E3_SUMMARY.md` and the `e3-gat-omitted` variant.

## H4 (C2 graphs; Wilcoxon over splits, Holm across the 6)

Best GNN is the best of GCN+ / SAGE+ / GAT+ / GPR-GNN by mean. Choosing it after the fact
favours the GNNs, as stated in A7.

| graph | EDSP − C&S | p (Holm) | best GNN | GNN − EDSP | p (Holm) |
|---|---|---|---|---|---|
| amazon-ratings | −6.63 | 0.012 | SAGE+ | +10.96 | 0.012 |
| chameleon-filtered | −2.46 | 0.055 (≈) | SAGE+ | +7.15 | 0.012 |
| questions | −0.67 | 0.105 (≈) | GAT+ | +4.23 | 0.012 |
| roman-empire | +13.02 | 0.012 | SAGE+ | +24.21 | 0.012 |
| squirrel-filtered | +14.52 | 0.012 | GAT+ | +0.86 | 0.232 (tie) |
| tolokers | −1.97 | 0.012 | GAT+(h64); GCN+ in mode omit | +9.95 / +9.39 | 0.012 |

The result:
- EDSP-B ≈ C&S on 2 of 6 graphs.
- The best GNN is significantly better on 5 of 6.
- **H4 is inconclusive** in both A14 modes.

On squirrel-filtered, BWGNN (42.11) is the best GNN overall, but A7 excludes it from "best
GNN" (the set is GCN+ / SAGE+ / GAT+ / GPR-GNN). EDSP-B-tuned (39.17) ties GAT+ (40.03).

## EDSP-B-tuned against every baseline (13 real graphs; Wilcoxon over splits, Holm across graphs)

| baseline | win | tie | loss |
|---|---|---|---|
| feature_lr (its own base predictor) | 9 | 4 | **0** |
| C&S | 4 | 3 | 6 |
| MLP | 7 | 3 | 3 |
| GCN+ | 0 | 3 | 9 |
| SAGE+ | 0 | 1 | 11 |
| GAT+ (mode h64 / omit) | 0 | 3 | 9 / 8 |
| GPR-GNN | 0 | 2 | 10 |
| BWGNN | 1 | 2 | 10 |

On amazon-fraud only feature_lr, C&S, MLP and BWGNN exist (A9). Average ranks, Friedman
over the graphs where every ranked method exists:
- **Mode h64** (12 graphs): SAGE+ 2.92, GAT+ 3.08, BWGNN 3.17, GCN+ 3.75, GPR-GNN 3.92,
  C&S 5.83, **EDSP-B-tuned 7.17**, MLP 7.83, EDSP-B-default 8.25, feature_lr 9.08.
- **Mode omit** (11 graphs): EDSP-B-tuned 7.18, same order.

## Where EDSP-B is useful, and where it is not
- **Under heterophily it is safe where C&S is not.** C&S assumes homophily and pulls the base
  predictor down:
  - roman-empire: 64.33 → 51.29
  - squirrel-filtered: 39.40 → 24.65

  EDSP-B-tuned stays at the level of the base predictor (64.31; 39.17). On minesweeper (†), EDSP-B adds +10.7 AUC over the base predictor, while
  C&S adds nothing.
- **Under homophily C&S is better** (cora, pubmed, amazon-computers, wikics), and so are all
  the GNNs.
- **It never falls significantly below its base predictor** on any real graph (9 / 4 / 0).
  MLP, a feature model without the graph, is beaten on 7 graphs and wins on 3, all of them
  heterophilous: amazon-ratings, chameleon-filtered, roman-empire.

## What this does not show
- **Track B only.** End-to-end GNNs are the stronger tool whenever they can be trained. EDSP-B
  is a post-hoc smoother on top of a fixed predictor, comparable to C&S, not to the GNNs.
- **The GNN numbers are from re-implementations** with 9-point grids on CPU runners. Published
  numbers for the same graphs may differ.

## Coauthor graphs (A12, not part of H4)

These are homophilous graphs with dense bag-of-words features (6,805 and 8,415 dimensions).
They are run per split and per method group on branch `pr-21-e3-results`. Records are in
`results/runs/e3-coauthor/` (assembly listing in `ASSEMBLY.md`) and tables in
`results/aggregates/e3-coauthor/E3_SUMMARY.md`. Accuracy is ×100, mean ± std over 10 splits.

| graph | EDSP-B-tuned | LogReg (base) | C&S | MLP | GCN+ | SAGE+ | GAT+ | GPR-GNN | BWGNN |
|---|---|---|---|---|---|---|---|---|---|
| coauthor-cs | 95.78 ± 0.22 | 94.72 ± 0.23 | 95.83 ± 0.21 | 95.40 ± 0.22 | 95.52 ± 0.20 | 95.38 ± 0.38 | 95.93 ± 0.30 | **95.97 ± 0.20** | 95.88 ± 0.29 |
| coauthor-physics | 97.03 ± 0.22 | 96.25 ± 0.19 | 97.21 ± 0.23 | 96.98 ± 0.23 | 97.12 ± 0.17 | NOT-RUN | NOT-RUN | **97.37 ± 0.21** | 97.31 ± 0.24 |

**NOT-RUN on coauthor-physics (330-minute CI limit, A12).** A row is kept only if it is
complete (the principle of A14), so these partial rows are left out of the tables and tests
(`--drop`):
- **SAGE+** finished 9 of 10 splits, mean 97.13. On the same 9 splits EDSP-B-tuned has 97.06
  and is higher on 4 of them.
- **GAT+** finished 3 of 10 splits, mean 97.47. On the same 3 splits EDSP-B-tuned has 97.17
  and is higher on none.

**EDSP-B-tuned win / tie / loss** (Wilcoxon over splits, Holm across the 2 graphs):

| baseline | win / tie / loss |
|---|---|
| LogReg (base) | 2 / 0 / 0 |
| C&S | 0 / 1 / 1 (loss on physics) |
| MLP | 1 / 1 / 0 |
| GCN+ | 1 / 1 / 0 |
| SAGE+ | 1 / 0 / 0 (cs only) |
| GAT+ | 0 / 1 / 0 (cs only) |
| GPR-GNN | 0 / 0 / 2 |
| BWGNN | 0 / 1 / 1 |

**Reading.** On these feature-rich homophilous graphs every method is within about 1.3
points. EDSP-B adds about 1 point over its base predictor, as C&S does, and is within 0.2
(cs) and 0.35 (physics) points of the best GNN (GPR-GNN), which beats it on both graphs.
This matches the homophilous E3 graphs above: no win over the best comparator.
