# H5 — P-NATIVE vs P-CONST on ogbn-proteins (PREREGISTRATION A15)

Hypothesis H5: EDSP with the graph's native edge probabilities (P-NATIVE) beats EDSP with a
constant edge probability (P-CONST). It is tested as registered in A15, which was written
before any method ran on ogbn-proteins.

Every number here is read from three files:
- `results/aggregates/c4/H5_VERDICT.md` / `h5_verdict.json` (`scripts/h5_verdict.py`);
- `results/aggregates/c4/C4_SUMMARY.md` / `c4_aggregates.json` (`scripts/aggregate_c4.py`);
- the assembled records in `results/runs/c4/`, with `ASSEMBLY.md`.

**Setup.**
- **Data:** ogbn-proteins, official species split.
- **Tasks:** 112 binary GO tasks; the headline metric is test ROC-AUC.
- **Labels:** Track A (structure and train labels only).
- **Tuning:** 9 configurations per tuned variant, selected on validation ROC-AUC.

## 1. Verdict: **SUPPORTED**

| A vs B | tasks | mean Δ (AUC ×100) [95% CI] | median Δ | Wilcoxon p | A better / worse |
|---|---|---|---|---|---|
| **EDSP-native-tuned vs EDSP-const-tuned (primary)** | 112 | **+1.01** [+0.85, +1.18] | +0.75 | 1.9 × 10⁻¹⁹ | 108 / 4 |
| EDSP-native-default vs EDSP-const-default | 112 | +0.82 [+0.70, +0.96] | +0.67 | 1.4 × 10⁻¹⁹ | 105 / 7 |

- **Completeness:** no task was NOT-RUN. All 112 tasks have every record (1,792 = 16 methods
  × 112 tasks).
- **Mean ROC-AUC over tasks:** EDSP-native-tuned 78.81, EDSP-const-tuned 77.79.
- **Defaults:** the gain is about the same without any tuning. EDSP-native-default (P-NATIVE
  noisy-OR of the 8 channels, H-TRAINONLY) is 78.56 against 77.73 for EDSP-const-default.

## 2. How to read it (reading notes fixed in A15, and what the data add)

**One graph, one split.** The 112 tasks share one graph and one split. The p-value therefore
measures how consistently the effect appears across tasks; it says nothing about other graphs.
The other C4 sources (uncertain DBLP, the US-Patent assignee graph, raw STRING + GO-slim) could
not be obtained as specified (`docs/SPEC_AMBIGUITIES.md`). H5 rests on this graph alone.

**What the contrast is.** A15 predicted the following:
- At ogbn-proteins' mean degree (≈ 597), the escape probability is ≈ 1 at almost every node
  under both P-CONST and P-NATIVE.
- So the contrast is in practice "edges weighted by their native confidence" against "equally
  weighted edges".

The data agree, and they show the effect is not specific to EDSP. All three comparisons below
are exploratory (not preregistered), computed from the committed records:

| Comparison | Result |
|---|---|
| Harmonic propagation, weighted (noisy-OR p) vs unweighted | +0.85 AUC, 109 / 3 tasks, the same size as EDSP's |
| EDSP's default gain (native − const) vs harmonic's gain (+w − unweighted) | 0.03 apart (p = 0.36); per-task correlation of the two gains 0.95 |
| EDSP-const-default vs harmonic | −0.05 (p = 0.44) |

The other weighted baselines also gain from the weights, by less: LinBP +0.42, PPR-LP +0.15,
LGC +0.14.

- At this degree EDSP-const behaves like harmonic propagation. The native edge confidences help
  it as much as they help harmonic propagation.
- The tuned gain is 0.16 points larger than harmonic's (p = 0.004). Unlike `harmonic+w`, the
  tuned grid can also choose the max-channel and mean-channel modes.
  - The validation selection chose the max-channel mode on 77 of 112 tasks, the mean-channel
    mode on 30, and noisy-OR on 5.
  - These two modes are this repository's substitutes for the unnamed single channels of the
    Master Prompt (§2.2; see `docs/SPEC_AMBIGUITIES.md`).

**H5 is supported as registered. What it shows is that native edge confidence beats a constant
probability inside EDSP on this graph. It does not show that EDSP's escape-probability
mechanism uses native probabilities better than ordinary weighted propagation does.**

## 3. EDSP against the baselines on the same tasks (descriptive, A15 secondary)

Mean ROC-AUC over the 112 tasks (×100):

| Method | Unweighted | Weighted by the noisy-OR p (`+w`) |
|---|---|---|
| PPR-LP | 83.32 | 83.47 |
| LGC | 83.20 | 83.34 |
| **EDSP-native-tuned** | — | **78.81** |
| harmonic / harmonic+CMN | 77.78 | 78.63 |
| EDSP-const-tuned | 77.79 | — |
| LinBP | 74.73 | 75.15 |
| majority / train prior | 50.00 | — |

| EDSP-native-tuned vs | mean Δ | better / worse | p |
|---|---|---|---|
| PPR-LP+w (best weighted) | −4.67 | 3 / 109 | 6.3 × 10⁻²⁰ |
| PPR-LP (best unweighted) | −4.51 | 3 / 109 | 6.5 × 10⁻²⁰ |
| LGC+w | −4.53 | 4 / 108 | — |
| harmonic+w | +0.17 | 69 / 43 | 0.005 |
| LinBP+w | +3.65 | 82 / 30 | — |

The pattern is the same as in E1: EDSP lies between harmonic propagation and LGC / PPR-LP, and
well below LGC / PPR-LP. The p-values are uncorrected and descriptive.

## 4. Convergence (H7, descriptive)

- **EDSP-const-default, EDSP-const-tuned and EDSP-native-default** converged on every task.
  Median iterations: 156, 155 and 140.
- **EDSP-native-tuned** did not converge on 17 of 112 tasks. Oscillation was flagged on 6 of
  them; the median was 128 iterations.
  - On those 17 tasks the mean ROC-AUC of the last iterate and of the mean of the last two
    iterates are equal (77.81). The headline does not depend on which one is reported.
  - Restricted to the 95 tasks where EDSP-native-tuned converged (exploratory), the primary
    comparison stays the same: +1.01, 91 / 4, p = 1.3 × 10⁻¹⁶.

## 5. Runs, assembly and reproducibility

| Run | Tasks | Notes |
|---|---|---|
| Pilot 42614818317 | 0–1 | |
| 42626280592 | 0–59 | Cancelled after its own ranges were done; its later ranges duplicated the next run. Completed records of the cancelled jobs were still uploaded. |
| 42636369947 | 56–111 | |

- **Commits.** All runs ran at commit d55872b or d501eec. They differ only in `e1.yml`.
- **Duplicates.** 178 (task, method) records were computed twice: tasks 0–1 by the pilot and by
  run 1, and tasks 56–84 by both full runs. 169 of them are bit-identical. 9 differ by at most
  1.6 × 10⁻⁷ in ROC-AUC, on harmonic, harmonic+CMN, LGC and EDSP-native-default for tasks 56,
  57 and 64.
  - That is about the size of one swapped pair of tied scores; floating-point differences
    between runner CPUs are the likely cause.
  - No tuned variant chose a different configuration.
  - `assemble_runs.py` keeps the earlier run's record; the list is in `results/runs/c4/ASSEMBLY.md`.
- **The pilot is not representative.** Tasks 0 and 1 are among the hardest: EDSP-native-tuned
  scores 48.8 and 56.6 there, against 78.8 on average over all tasks.

## 6. Reproduce

```
python3 scripts/assemble_runs.py --subdir results/runs/c4 --out results/runs/c4 \
    --refs origin/results/c4-42614818317 origin/results/c4-42626280592 origin/results/c4-42636369947
python3 scripts/aggregate_c4.py      # → results/aggregates/c4/C4_SUMMARY.md
python3 scripts/h5_verdict.py        # → results/aggregates/c4/H5_VERDICT.md
```
