# E2 report — native-polarity real graphs (signed, relation-typed), Track-A

**Scope.** Track-A (labels and graph only) on the real graphs whose edges carry
their own polarity information:

| dataset | cell | polarity input to EDSP | headline | status |
|---|---|---|---|---|
| wiki-editor (Mercado et al. 2019; 20,198 nodes, labelled LCC) | C5 signed | H-NATIVE (edge signs) | accuracy | ranked |
| amazon-fraud (CARE-GNN; 11,944 nodes, 3 relations) | C3a relation-typed | H-RELATION (one polarity per relation combination, learned on train–train edges) | ROC-AUC | ranked |
| yelpchi (CARE-GNN; 45,954 nodes, 3,846,979 edges) | C3a relation-typed | H-RELATION | ROC-AUC | **missing** — see §4 |
| wiki-rfa, wiki-elec (SNAP) | C5 signed | H-NATIVE | accuracy | case studies: the label is decided by the incoming votes (§3.1 C5.3), excluded from ranks |
| sampson (18 nodes) | C5 signed | H-NATIVE | accuracy | case study (leave-one-out), excluded from ranks |

Methods: EDSP-default (§2.5; H-NATIVE / H-RELATION instead of H-TRAINONLY), EDSP-tuned
(9 configurations: p0 ∈ {0.1, 0.5, 0.9} × {native polarity, H-LEARNED, all
homophilic}), and the Track-A baselines with 9-point grids: harmonic, harmonic+CMN,
LGC, PPR-LP, LinBP-MCE, train prior; on signed graphs also `signed_vote` and
`signed_lgc` (signed-Laplacian label spreading — the label-only state of the art
for signed graphs that runs without a GNN). GNN baselines for signed graphs
(SGCN, SDGNN, SiGAT) and fraud graphs (CARE-GNN, PC-GNN, BWGNN, GAGA) need PR-6
(torch) and are **not** in this report.

**Provenance (R1).** 648 run records in `results/runs/e2/` from Actions run
[42214156507](https://github.com/Dellos-code/information_theoretic_dynamic_damping_in_node_classification/actions/runs/42214156507)
(commit 2ff5600). Tables: `results/aggregates/e2/E2_SUMMARY.md`; every number below
is copied from it.

## 1. Results (mean ± std over 10 splits, ×100)

| dataset | EDSP-default | EDSP-tuned | best unsigned baseline | signed_vote | signed_lgc |
|---|---|---|---|---|---|
| wiki-editor [acc] | 86.15 ± 0.65 | 86.12 ± 0.67 | LGC 76.54 ± 0.48 | 84.33 ± 0.63 | **86.43 ± 0.42** |
| amazon-fraud [AUC] | 80.95 ± 1.14 | **82.87 ± 1.13** | LGC 82.68 ± 1.02 | — | — |
| wiki-elec† [acc] | 80.54 ± 1.45 | 80.58 ± 1.42 | harmonic+CMN 60.58 ± 2.55 | 78.09 ± 1.43 | **81.66 ± 0.78** |
| wiki-rfa† [acc] | 72.22 ± 1.22 | 72.23 ± 1.18 | harmonic+CMN 64.12 ± 2.23 | 71.73 ± 1.57 | **76.59 ± 0.81** |
| sampson† [acc] | **66.67** | **66.67** | harmonic+CMN 50.00 | **66.67** | 61.11 |

† case studies, not ranked.

Wilcoxon over splits, Holm across datasets (real, ranked datasets only — here
wiki-editor and amazon-fraud):

| EDSP-tuned vs | win | tie | loss |
|---|---|---|---|
| harmonic, harmonic+CMN, LGC | 1 (wiki-editor) | 1 (amazon-fraud) | 0 |
| PPR-LP, LinBP-MCE, train prior | 2 | 0 | 0 |
| signed_vote | 1 (wiki-editor) | 0 | 0 |
| signed_lgc | 0 | 0 | 1 (wiki-editor, −0.31 points) |

The Friedman ranks over two datasets (p = 0.165) are not informative and are not
reported as a result.

## 2. Accuracy by local homophily (§5.7)

Low = test nodes whose local homophily (full labels) is < 0.3; high = ≥ 0.7.
Accuracy ×100, pooled over splits.

| dataset | low-h test nodes | EDSP-tuned low / high | signed_lgc | LGC | harmonic+CMN | LinBP |
|---|---|---|---|---|---|---|
| wiki-editor | 4,847 | **71.8** / 91.3 | 72.0 / 91.4 | 24.7 / 94.2 | 10.9 / 97.0 | 6.2 / 97.0 |
| wiki-elec† | 1,122 | **64.4** / 97.3 | 61.5 / 98.4 | 4.3 / 98.7 | 19.4 / 89.9 | 0.3 / 99.1 |
| wiki-rfa† | 657 | **70.0** / 86.8 | 70.2 / 94.9 | 6.8 / 98.5 | 13.1 / 98.6 | 2.6 / 97.8 |
| amazon-fraud | 2,979 | 1.7 / 99.8 | — | 0.0 / 100.0 | 0.0 / 100.0 | 0.1 / 100.0 |

## 3. Reading

- **Signed graphs: this is where the polarity mechanism works.** With native signs
  EDSP classifies heterophilic nodes (local homophily < 0.3) at 64–72 %, where every
  unsigned propagation method is at 0–25 %. On wiki-editor that lifts it 9.6 points
  above the best unsigned baseline (LGC) and 1.8 above the signed majority vote.
  It does **not** beat the signed-Laplacian label spreading baseline: −0.31 on
  wiki-editor (significant), −1.1 and −4.4 on the two case studies. Both use the
  same information (signs + train labels); signed_lgc is better on the
  homophilic side (wiki-rfa high-h nodes 94.9 vs 86.8), i.e. EDSP loses accuracy
  on the nodes that plain propagation already gets right.
- **Relation-typed fraud graph (amazon-fraud):** EDSP-tuned has the best mean
  ROC-AUC (82.87 vs LGC 82.68) but the difference is a tie; it is significantly
  better than PPR-LP and LinBP. Neither EDSP nor any baseline classifies the
  heterophilic (fraud-adjacent) nodes: low-h accuracy ≤ 1.7 %. With labels only, the
  per-relation polarities learned from train–train edges carry little
  information beyond the base rate.
- **EDSP-default vs EDSP-tuned:** identical on signed graphs (the native signs fix
  the polarity, tuning p0 changes nothing); on amazon-fraud tuning adds 1.9 AUC
  points.
- **Calibration:** on the signed graphs EDSP has the lowest NLL of all methods
  (wiki-editor: EDSP-tuned 0.41, LGC 0.52, signed_vote 2.52, signed_lgc 2.81 — the
  two signed baselines turn scores into probabilities by a shift, so their
  probabilities are poor even where their accuracy is high); on amazon-fraud
  harmonic+CMN is better (0.268 vs EDSP-default 0.276).
- **H7:** no non-convergence and no oscillation on any E2 graph; GS and damped
  fixed points coincide.

## 4. Missing / limitations

- **YelpChi:** its job ran 116 minutes and the runner was killed ("lost
  communication with the server"), which on GitHub-hosted runners (16 GB RAM)
  indicates memory exhaustion; no record was saved. A diagnostic re-run with a
  memory cap and progress logging is under way; YelpChi results will be added to
  this report, or reported as NOT-RUN with the cause.
- Only two ranked datasets so far; any claim about "signed graphs in general"
  needs Bitcoin OTC / Alpha and the Congress networks (acquisition pending).
- GNN baselines for signed and fraud graphs (PR-6) and Track-B (features) are not
  run.
- Mercado et al.'s GL (diffuse-interface) method is not implemented; `signed_lgc`
  is the signed-Laplacian label-propagation member of the family they compare.
