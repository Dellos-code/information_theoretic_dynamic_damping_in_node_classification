# H6 report — does finite label trust T_L help under label noise? (PREREGISTRATION A16)

**Answer: H6 is not supported.**
- In both primary conditions (`uniform:0.3`, `pairwise:0.3`), EDSP-TL beats EDSP-default
  significantly on 1 of the 14 real graphs (wikics) and ties on the other 13. The
  preregistered "not supported" rule is ≤ 1 win in both conditions.
- Across all 11 conditions (clean, and 10–50 % uniform or pairwise noise), the selected T_L
  changes test accuracy by at most a few hundredths of a point on average.
- The one graph where finite T_L helps is wikics. There it helps just as much without any
  noise: T_L = 0.5 is chosen in 20 of 20 clean splits. The gain is therefore not a response
  to label noise.

## Setup (A16)
- **Graphs:** the 14 real graphs of the E1 Friedman set, with the same 10 splits as E1.
- **Noise:** train and validation labels are corrupted (`edsp/data/label_noise.py`):
  - `uniform:r` replaces the label with another class with probability r.
  - `pairwise:r` maps c → c+1 mod k with probability r.
  - r ∈ {0.1, …, 0.5}; the clean control is `uniform:0`. Noise seed 0, one random stream per
    split.
  - Test labels are never altered, and the vault refuses an observed array that differs on
    a test node.
- **Methods (Track A):**
  - EDSP-default (T_L = ∞).
  - EDSP-TL: the same configuration with T_L chosen on the noisy validation labels from
    {0.5, 1, 2, 4, ∞}; ties go to ∞.
  - The fixed EDSP-TL=T sweep.
  - For context: EDSP-tuned and every E1 baseline on the same noisy labels.
- **Test:** Wilcoxon over splits, Holm across the 14 graphs within each condition.
- **Data:** 11 Actions runs on branch `pr-28-h6-label-noise`. Records are in
  `results/runs/e6/<run_id>/` and tables in `results/aggregates/e6/H6_SUMMARY.md`
  (`scripts/h6_label_noise.py`). The smoke run 42603215359 (cora, 1 split) is not part of
  the analysis.

## Results
See `results/aggregates/e6/H6_SUMMARY.md` for every condition. The pattern, read from that
table:
- **EDSP-TL vs EDSP-default:** 0–2 wins and 0 losses per condition, all others ties. The
  mean Δ over graphs is between −0.01 and +0.05 points. The clean control has as many wins
  as the noisy conditions.
- **Fixed T_L sweep:** even T_L = 0.5, the strongest down-weighting, moves the mean accuracy
  by only a few hundredths of a point.
- **What validation picks:** as the noise rate grows, validation chooses T_L = ∞ *more*
  often, not less. Under uniform noise, ∞ is chosen in 77 of 153 clean splits and in 92 of
  153 at 50 %. This is the opposite of the hypothesised mechanism.
- **Context:** under noise, EDSP-TL is significantly worse than LGC on 11–14 of the 14
  graphs in every noisy condition except `pairwise:0.5`, where it loses on 2 and ties on 12.
  It is worse than harmonic LP on up to 9.

## Why (reading, not a test)
Trust calibration (§2.4, `edsp/core.py: trust_calibrate`) mixes a training node's one-hot
prior with the uniform distribution:
v = (1 − e^(−T_L)) · onehot + e^(−T_L)/k.
- This lowers how strongly a training node speaks, but not *what* it says. A wrong label
  still points to the wrong class, only more softly.
- The other training nodes in its neighbourhood are softened by the same factor. The
  relative influence of correct and wrong labels on an unlabelled node therefore barely
  changes, and neither do its predicted classes.
- Methods that are robust to label noise act differently. LGC's α lets graph smoothing
  override individual seeds. Other methods explicitly re-estimate or down-weight the labels
  that disagree with their neighbourhood. Uniform tempering has neither effect.

## What this means for the algorithm
The trust component, as specified, gives no measurable protection against label noise on
these 14 graphs. If robustness to noisy labels is a goal, the trust would have to depend on
the evidence against each training label (for example, its disagreement with the
neighbourhood). A single global tempering constant does not achieve it. Such a variant would
be a new hypothesis and would need its own preregistration.
