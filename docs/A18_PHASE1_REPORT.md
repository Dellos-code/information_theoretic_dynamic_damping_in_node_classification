# A18 Phase 1 — EDSP-B against the post-hoc comparators a reviewer would demand (exploratory)

**Question.** Is "EDSP-B is a safe post-hoc smoother: it never harms its base predictor, unlike
C&S" a claim worth a confirmatory test on unused graphs?

**Data.**
- 17 graphs already used in E3: the 14 E3 graphs, coauthor-cs and -physics, and yelpchi.
- Track B: every method refines the same `feature_lr` soft outputs.
- This phase is exploratory: these graphs informed the hypothesis.
- Every number below is read from `results/aggregates/a18-phase1/A18_SUMMARY.md` / `a18.json`,
  produced by `scripts/a18_posthoc.py`.
- Records:
  - new comparators: Actions run 42707731268, in `results/runs/a18-phase1`;
  - EDSP-B and base: `results/runs/e3`, `e3-coauthor`, `b6`.
- One amazon-fraud split (rand8) is left out for every method: its base predictor chose a
  different l2 in the new run than in the original one, so the two base predictions are not
  identical. The other 182 (graph, split) pairs have identical base-predictor choices.

**Statistics.**
- d = method − base per split (headline metric ×100).
- Non-inferiority (NI) with margin δ = 1 point, one-sided α = 0.025, Nadeau–Bengio-corrected t
  (n_test/n_train: 0.5 on the Platonov graphs, 1 on the fraud graphs, 1/3 otherwise).
- Harm = significantly more than δ below the base, Holm across graphs.
- EDSP-B vs comparator: two-sided corrected t, Holm across graphs, win / tie / loss.
- Heterophilous = adjusted homophily < 0.2. Nine graphs: amazon-fraud, amazon-ratings,
  chameleon-filtered, minesweeper, questions, roman-empire, squirrel-filtered, tolokers, yelpchi.

## 1. Result

| method | NI holds (of 17) | significant harm | mean gain over base: heterophilous (worst graph) | mean gain: homophilous | EDSP-B-tuned vs it: W / T / L |
|---|---|---|---|---|---|
| **EDSP-B-tuned** | 15 (fails: chameleon, squirrel — noise, mean −0.2 / +0.2) | none | +1.66 (−0.23) | +3.72 | — |
| C&S (repository) | 11 | roman-empire, squirrel, yelpchi | −2.06 (−14.75) | +4.71 | 4 / 11 / 2 |
| `cs_ref` (Huang / PyG semantics) | 11 | roman-empire, squirrel, yelpchi | −1.85 (−15.29) | +4.73 | 4 / 8 / 5 |
| `cs_2hop_ref` (Lim et al.) | 12 | roman-empire, squirrel, yelpchi | −0.75 (−16.94) | +3.98 | 4 / 11 / 2 |
| `clp` (Zhong et al., TMLR 2022) | **17** | none | +2.16 (−0.08) | +4.10 | 1 / 11 / 5 |
| **`clp_es`** (CLP, layer chosen on validation) | **17** | **none** | **+3.32 (−0.11)** | +4.10 | **0 / 12 / 5** |
| `linbp_b` (our construction) | 14 | questions | +0.32 (−3.53) | +1.21 | 9 / 6 / 2 |

**What this shows:**
- **The C&S result is not an artefact of our implementation.** With the reference semantics
  (`cs_ref` matches PyG exactly), C&S still collapses on roman-empire (−13.3), squirrel-filtered
  (−15.3) and yelpchi (−5.0). C&S 2-hop also collapses on roman-empire and squirrel-filtered.
- **CLP is safe and stronger.**
  - Neither CLP variant significantly harms the base on any of the 17 graphs.
  - CLP-ES gains more than EDSP-B on average, both on heterophilous and on homophilous graphs.
  - EDSP-B never beats CLP-ES: 0 wins, 12 ties, 5 losses (amazon-computers, amazon-ratings,
    tolokers, wikics, yelpchi).
  - Where EDSP-B's gain is largest (minesweeper +10.65), CLP-ES gains +11.87.

## 2. Decision on the confirmatory phase

The GO rule announced before this run read: *GO iff EDSP-B is never significantly harmful and at
least one strong comparator (`cs_ref`, `cs_2hop_ref`, `clp_es`) significantly harms the base
somewhere.*

Taken literally it is satisfied, since `cs_ref` and `cs_2hop_ref` harm the base on three graphs.
But the rule did not foresee a comparator that is both safe and better. CLP is published prior
art (TMLR 2022) and does exactly that here. A confirmatory test could at best establish that
EDSP-B is as safe as CLP and gains less; that is not a contribution.

**Recommendation: NO-GO for the confirmatory phase (A18) unless the owner decides otherwise.**
Nothing has been run on the unused graphs.

## 3. Caveats

- This phase is exploratory, and the graphs are the ones that suggested the hypothesis.
- Every comparator has a 9-point validation grid. The references tune more freely with Optuna;
  that would favour them further.
- The official CLP code selects its configuration on test accuracy. Ours selects on validation
  only, so our CLP numbers are not optimistic in that way.
- On the noisiest graphs (chameleon-, squirrel-filtered: about 900 / 2,200 nodes, split sd
  0.8–2.3) a 1-point margin cannot be shown for EDSP-B, even though its mean is within 0.25 of
  the base.
