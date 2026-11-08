# A13 report — real graphs with native near-random mixing

**Question (PREREGISTRATION A13, written before any run).** The MIX wins came from
re-wired graphs. Do they also appear on real graphs whose two classes mix across edges by
themselves, close to randomly?

**Answer.**
- **Verdict: INCONCLUSIVE.** EDSP-tuned has 4 significant wins over the best sign-blind
  propagation baseline and 9 significant losses, with no ties, across the 13 k = 2 graphs.
  The verdict is inconclusive by the fixed rule: there are more than 1 win, but fewer wins
  than losses and only one source family.
- **The wins belong to the learned polarity.** All 4 wins are on Facebook100 schools. On
  all four, simple signed propagation fed the same learned polarity (A10's +HL baselines)
  ties EDSP (2) or beats it (2).
- **Everywhere else EDSP loses to plain LGC, LinBP or PPR:** genius, deezer-europe, all six
  twitch graphs and fb100-reed98.

## Data
These are the 14 graphs added in PR #30, loaded from pinned sources and verified by
dataset-stats run 42478975498: sizes match the published ones. All statistics below come
from the DATASET-STATS records.

| graph | nodes | edges | edge homophily | adjusted homophily | label informativeness |
|---|---|---|---|---|---|
| fb100-penn94 | 41,554 | 1,362,229 | 0.511 | 0.021 | 0.000 |
| fb100-amherst41 | 2,235 | 90,954 | 0.531 | 0.060 | 0.003 |
| fb100-cornell5 | 18,660 | 790,777 | 0.546 | 0.091 | 0.006 |
| fb100-johnshopkins55 | 5,180 | 186,586 | 0.549 | 0.097 | 0.007 |
| fb100-reed98 | 962 | 18,812 | 0.522 | 0.022 | 0.000 |
| genius | 421,961 | 922,868 | 0.593 | −0.053 | 0.002 |
| deezer-europe | 28,281 | 92,752 | 0.525 | 0.030 | 0.001 |
| twitch-de / en / es / fr / pt / ru | 9,498 / 7,126 / 4,648 / 6,549 / 1,912 / 4,385 | 153,138 / 35,324 / 59,382 / 112,666 / 31,299 / 37,304 | 0.56–0.63 | 0.03–0.14 | 0.001–0.015 |
| actor (k = 5, reported only) | 7,600 | 26,659 | 0.217 | 0.003 | 0.000 |

Adjusted homophily and label informativeness are close to 0 on every graph. One-hop label
agreement therefore carries almost no information about a node's class. This is the
"chaotic mixing" regime, and it is hard for every label-propagation method.

## Protocol
- Suite E1, Track A (labels only, no node features).
- 10 stratified 60/20/20 splits per graph.
- Headline metric from the registry: accuracy for FB100, deezer and actor; ROC-AUC for
  genius and twitch.
- `--h7-splits 0` and `--learned-sign-baselines`.

Actions runs:
- 42480280847: the five FB100 schools.
- 42490815688: genius split 9. The rest of this run was cancelled to let a PR's CI through.
- 42493127755: everything else.

Records are in `results/runs/native-mixed/` and the table in
`results/aggregates/native-mixed/A13_SUMMARY.md`. The analysis is
`scripts/native_mixed_a13.py`, which applies the A13 rules as written.

## Results (means ×100 over 10 splits)

| graph | EDSP-tuned | best sign-blind | Δ (a) | (a) | best +HL | Δ (b) | (b) |
|---|---|---|---|---|---|---|---|
| fb100-penn94 | 71.94 | LGC 63.51 | +8.43 | **win** | signed_lgc+HL 72.59 | −0.64 | loss |
| fb100-amherst41 | 71.65 | LGC 63.76 | +7.89 | **win** | signed_vote+HL 71.13 | +0.52 | tie |
| fb100-cornell5 | 71.53 | LGC 67.92 | +3.61 | **win** | signed_lgc+HL 72.29 | −0.76 | loss |
| fb100-johnshopkins55 | 70.03 | LGC 67.26 | +2.77 | **win** | signed_lgc+HL 71.08 | −1.05 | tie |
| fb100-reed98 | 59.42 | harmonic 60.81 | −1.39 | loss | | | |
| genius (AUC) | 59.16 | LGC 70.37 | −11.21 | loss | | | |
| deezer-europe | 54.89 | LGC 56.40 | −1.51 | loss | | | |
| twitch-de (AUC) | 67.72 | LinBP 70.40 | −2.68 | loss | | | |
| twitch-en (AUC) | 57.57 | LinBP 61.29 | −3.72 | loss | | | |
| twitch-es (AUC) | 66.41 | PPR-LP 67.74 | −1.32 | loss | | | |
| twitch-fr (AUC) | 64.76 | LGC 66.43 | −1.68 | loss | | | |
| twitch-pt (AUC) | 65.12 | LGC 66.39 | −1.27 | loss | | | |
| twitch-ru (AUC) | 56.94 | LGC 58.36 | −1.42 | loss | | | |
| actor (not tested) | 23.33 | LGC 25.27 | −1.94 | — | | | |

Tests and corrections:
- (a) Wilcoxon over splits, Holm across the 13 graphs. Every (a) result is significant at
  α = 0.05 after Holm (p ≤ 0.03).
- (b) Holm over the 4 (a)-wins.

## Reading
- **Where EDSP gains, H-LEARNED is doing the work.** The H-LEARNED edge-polarity classifier
  was selected by EDSP-tuned in 10 of 10 splits on penn94, amherst41, cornell5,
  johnshopkins55 and genius.
  - On the FB100 schools it lifts accuracy from about 63–68 % (sign-blind) to about 70–72 %.
  - Simple signed LGC fed the same polarity gets the same or more. This repeats A10 (Congress)
    and A11 (MIX): the gain belongs to the learned polarity, not to EDSP's propagation.
- **On genius the learned polarity hurts** (EDSP-tuned 59.2 vs LGC 70.4 AUC), although
  validation selected it in all 10 splits.
- **On twitch and deezer EDSP-tuned is within 0.35 of harmonic** and below LGC, LinBP or
  PPR-LP. H-TRAINONLY was chosen in most splits.
- **Caveats on the comparison:**
  - All methods here use labels only.
  - Published feature-based results on these graphs (LINKX and others) are higher, but they
    are not comparable to Track A and are not used here.
  - The +HL baselines are part of this evaluation, not of John's algorithm.

## What this changes
Nothing in the earlier verdicts is reversed. The one new negative finding is that on real
graphs with native near-random two-class mixing, EDSP does not beat simple label propagation
except where the learned polarity helps, and there the learned polarity alone is as good.
