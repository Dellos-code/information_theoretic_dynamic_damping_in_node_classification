# Testing the two caveats of the MIX wins (PREREGISTRATION A8)

The confirmed MIX wins of EDSP-tuned (A4: 14 conditions on re-wired wiki-editor, k = 2,
plus 3 on amazon-ratings) came with two caveats: (i) shown only on re-wired graphs;
(ii) the re-wiring generator's locality rule makes edge polarity predictable from
structure. Both were tested with rules fixed in A8 before the runs.

## (i) Real graphs with their polarity hidden — **supported**

Suite E2U: the real k = 2 graphs that carry edge signs or relation types, run with the
signs / relation types **removed** (`run_e1.py --drop-signs`; same nodes, labels, splits
and edges). EDSP has to recover polarity from structure (H-LEARNED), as in MIX.
1,183 records in `results/runs/e2u/`, Actions run
[42377245757](https://github.com/Dellos-code/information_theoretic_dynamic_damping_in_node_classification/actions/runs/42377245757)
(branch pr-13-caveats, commit with `--drop-signs` only), no NOT-RUN. Tables:
`results/aggregates/e2u/E2U_SUMMARY.md`.

EDSP-tuned vs the best of harmonic, harmonic+CMN, LGC, PPR-LP, LinBP (by mean); Wilcoxon
over 10 splits; Holm over the 11 ranked datasets (×100):

| dataset (signs hidden) | EDSP-tuned | best baseline | Δ | p (Holm) | verdict |
|---|---|---|---|---|---|
| congress-house-93 | 80.22 | LGC 67.08 | +13.15 | 0.022 | **win** |
| congress-house-100 | 87.19 | LinBP 73.71 | +13.48 | 0.022 | **win** |
| congress-house-107 | 95.84 | LinBP 91.01 | +4.83 | 0.022 | **win** |
| congress-house-114 | 96.18 | LinBP 91.80 | +4.38 | 0.022 | **win** |
| congress-senate-93 | 70.48 | LinBP 62.86 | +7.62 | 0.15 | tie |
| congress-senate-100 | 76.50 | LinBP 62.50 | +14.00 | 0.031 | **win** |
| congress-senate-107 | 93.16 | LinBP 79.47 | +13.68 | 0.031 | **win** |
| congress-senate-114 | 98.00 | LinBP 83.00 | +15.00 | 0.022 | **win** |
| wiki-editor | 74.01 | LGC 76.54 | −2.52 | 0.022 | loss |
| amazon-fraud [AUC] | 82.79 | LGC 82.68 | +0.11 | 0.92 | tie |
| yelpchi [AUC] | 62.86 | LGC 63.54 | −0.68 | 0.022 | loss |

**7 wins / 2 ties / 2 losses → supported** (rule: ≥ 3 wins and more wins than losses).
Case studies (not ranked): wiki-elec +12.19 over harmonic+CMN (p = 0.002), wiki-rfa −0.17.

Reading: on the Congress co-sponsorship networks with their signs hidden, EDSP-tuned
(H-LEARNED, train labels only) recovers most of what the signs carry — e.g. house-114
96.2 % without signs vs 97.1 % with them — while the sign-blind baselines stay at
50–92 %. **Caveat:** 7 of the 7 wins come from one data source (8 chamber-sessions of
the US Congress, not independent); on the other real graphs the result is 2 losses and
1 tie, plus one strong case-study win (wiki-elec).

## (ii) Without the generator's locality — **partly cleared**

The same heterogeneous conditions re-run with `loc = 0` (partners drawn from the whole
target class; no structural strata), fresh seeds 20–29 × 3 splits = 30 pairs; A4 test.
2,800 records in `results/perturbed/loc0/runs/` (runs 42377254640, 42377263914); tables in
`results/perturbed/loc0/aggregates/MIX_SUMMARY.md`.

| base | condition | loc = 0.8 (A4) | loc = 0 (A8) |
|---|---|---|---|
| wiki-editor | beta (0.5, 0.5) | win +13.7 | **win +5.3** |
| wiki-editor | uniform | win +9.2 | **win +1.1** |
| wiki-editor | logistic (chaotic) | win +11.7 | **win +2.9** |
| wiki-editor | regional 20 | win +12.0 | **win +4.9** |
| wiki-editor | community | win +7.9 | **win +8.5** |
| wiki-editor | beta (0.2, 0.5) | win +5.5 | tie +0.1 |
| wiki-editor | beta (0.5, 5) | win +7.3 | loss −1.7 |
| wiki-editor | beta (0.8, 0.5) | win +1.9 | loss −1.2 |
| amazon-ratings | beta (0.2, 0.5) | win +7.8 | loss −10.9 |
| amazon-ratings | beta (0.5, 0.5) | win +1.6 | loss −0.4 |

The locality caveat is **cleared for 5 of the 8 k = 2 conditions** (still significant
wins, with smaller margins: +1.1 to +8.5 instead of +7.9 to +13.7) and **not cleared**
for the other 3 nor for either k = 5 condition. Most of the size of the MIX wins came
from the generator's locality; a smaller, still significant advantage remains for k = 2
with strongly heterogeneous, chaotic, regional or class-dependent mixing.

## (iii) Whose win is it? Ablation A10 — **the learned polarity's, not EDSP's propagation**

The A8 wins used H-LEARNED: a train-only edge classifier predicting which edges are
heterophilic. A10 gives the *same* predicted polarity, as signs, to the two simple
signed baselines (`signed_vote+HL`, `signed_lgc+HL`). Records:
`results/runs/e2u-ablation/` (Actions run
[42414234730](https://github.com/Dellos-code/information_theoretic_dynamic_damping_in_node_classification/actions/runs/42414234730),
branch pr-16-ablation), no NOT-RUN. EDSP-tuned vs the better of the two (×100; Wilcoxon over
splits, Holm over the 11 ranked datasets):

| dataset (signs hidden) | EDSP-tuned | signed_vote+HL | signed_lgc+HL | verdict |
|---|---|---|---|---|
| congress-house-93 | 80.22 | 80.79 | 82.81 | tie |
| congress-house-100 | 87.19 | 87.08 | 87.08 | tie |
| congress-house-107 | 95.84 | 95.73 | 95.62 | tie |
| congress-house-114 | 96.18 | 96.29 | 96.18 | tie |
| congress-senate-93 | 70.48 | 69.52 | 71.90 | tie |
| congress-senate-100 | 76.50 | 74.50 | 75.50 | tie |
| congress-senate-107 | 93.16 | 94.21 | 94.21 | tie |
| congress-senate-114 | 98.00 | 98.50 | 98.50 | tie |
| wiki-editor | 74.01 | 73.65 | 74.01 | tie |
| amazon-fraud [AUC] | 82.79 | 81.05 | 51.86 | win (+1.74) |
| yelpchi [AUC] | 62.86 | 58.22 | 52.59 | win (+4.64) |

**On every Congress graph the A8 advantage disappears**: once the simple signed
methods receive the same learned polarity, they match EDSP-tuned (differences −2.6 to
+1.0, none significant). The A8 wins are therefore due to the **learned edge polarity**
(H-LEARNED, part of the evaluation framework's regimes, not of EDSP), and any signed
propagation using it would get them. EDSP's propagation adds value only on the two
fraud graphs, where the predicted polarity is poor and a vote/spreading on it fails
(signed_lgc+HL ≈ 52 AUC); there, however, EDSP-tuned is itself only tied with (amazon-fraud)
or below (yelpchi) the plain sign-blind LGC. The same mechanism (H-LEARNED selected in
81 of 81 runs) drives the MIX k = 2 wins, which therefore most likely have the same origin;
this was not tested separately.

## Bottom line

The niche found in (i)–(ii) — **two classes, edge polarity present but not recorded,
and recoverable from structure** — is real, but (iii) shows that the advantage there
belongs to the **learned edge polarity** (H-LEARNED), not to EDSP's propagation: simple
signed propagation fed with the same predicted polarity does as well. As a method,
EDSP is therefore *competitive* (it is one of the propagation schemes that can use
predicted or given polarity), not the source of the gains.
