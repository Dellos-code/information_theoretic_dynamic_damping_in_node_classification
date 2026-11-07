# Pre-registration

Status: **APPROVED; frozen at the merge commit of the PR that introduced this
version (PR-2b).** From that commit on, this file may only be appended to
(amendments section). Rule R2: it precedes every full experiment run (PR-8, E1).

Decision record: on 2026-11-05 the owner (Ioannis Konstantellos) delegated the
three open decisions of the PR-2 draft to the implementing assistant ("do what is
best for the project"). The choices below and their reasons are recorded so that
the owner can amend them before the first full run.

## 1. Hypotheses (Master Prompt §5.11, H7 revised)

| id | hypothesis | outcome labels |
|---|---|---|
| H1 | Track-A EDSP ≈ harmonic LP / LGC on homophilic graphs (C1). | supported / not supported / inconclusive |
| H2 | Track-A EDSP is near chance on minesweeper, tolokers, questions (label informativeness ≈ 0). | same |
| H3 | EDSP-fullH > EDSP on structured-compatibility graphs (C3d); the gap grows with k. | same |
| H4 | Track-B EDSP ≈ C&S < tuned GNNs on feature-rich heterophilous graphs. | same |
| H5 | P-NATIVE > P-CONST on ogbn-proteins / uncertain DBLP. | same |
| H6 | Finite T_L helps under label noise. | same |
| H7 (revised) | Descriptive outcome, no directional prediction. Report per dataset and track: (a) the fraction of EDSP-canonical runs that hit `max_iter` without converging; (b) the fraction of nodes flagged as period-2 oscillating; (c) for every run, the per-node L1 distance between the final states of EDSP-GS (id order) and EDSP-damped(β=0.5), summarised by mean and 95th percentile, as a measure of fixed-point non-uniqueness. | reported values |

Reason for revising H7: the original H7 ("k=2 Jacobi oscillation only in
label-free components") is already falsified by the toy cases in
`docs/FINDINGS.md` (one homophilic edge between two informed nodes oscillates for
P_esc above about 0.55). Keeping a falsified directional hypothesis would be
uninformative; the descriptive version measures how often the phenomenon occurs
on real graphs, which is what the paper needs.

Hypotheses added after seeing the PR-2 dataset statistics are not allowed; the
statistics in `results/dataset_stats` describe datasets only and no method has
been run on them.

## 2. Primary metrics, tracks and non-convergence

Metrics follow Master Prompt §5.4; the headline metric per dataset is the one in
`data/REGISTRY.json` (`metric`). Track-A and Track-B are reported separately and
never compared as like-for-like (§4).

Non-convergence policy:
- **Headline column:** the literal last iterate at `max_iter = 1000`
  (`docs/SPEC_AMBIGUITIES.md`). This is what EDSP-canonical outputs.
- **Secondary column (labelled "EDSP-canonical, mean of last two iterates"):**
  the average of the last two iterates, reported for every metric whenever at
  least one run of that dataset did not converge.
- Reason: in a 2-cycle the last iterate depends on the parity of `max_iter`, so
  the headline number alone could change with an arbitrary constant. The
  secondary column exposes that sensitivity without replacing the literal
  algorithm. It is a reporting device, not a method, and it is never used in
  rankings.

## 3. Pre-registered exclusions (rule R2)

| dataset / setting | treatment | reason |
|---|---|---|
| chameleon, squirrel (unfiltered) | never in headline tables | duplicate nodes (Platonov et al., ICLR 2023); filtered versions are used |
| texas, cornell, wisconsin | appendix only | tiny, unstable splits (Master Prompt §3.1 C2.4) |
| sampson | case study, excluded from rank statistics | 18 nodes (Master Prompt §3.1 C5.4) |
| H-ORACLE results | separate table, never ranked | uses test labels (§2.3g) |
| minesweeper | kept and reported, labelled "constructed benchmark" in every table; excluded from real-world aggregates (mean ranks, CD diagrams, win/tie/loss counts and headline claims), which are computed over real networks only | it is a grid game constructed by Platonov et al. (2023), not an observed network; the owner's rule is real data first (R4), but the dataset is a standard heterophily benchmark, so dropping it would make results less comparable with the literature |

## 4. Splits

Official splits where they exist (Planetoid public; Platonov 10 splits; WikiCS
20 splits; OGB species split). Otherwise 10 stratified random 60/20/20 splits
with seeds 0–9 (`edsp.data.graph.random_splits`). Planetoid datasets also get the
10 random splits. Sampson: leave-one-out.

## 5. Freeze record

- Draft committed: PR-2 (commit 9c07f2e).
- Decisions delegated by the owner and recorded: 2026-11-05 (this version, PR-2b).
- Frozen at: the merge commit of PR-2b.

## 6. Amendments

Every amendment must state its date, reason and whether any experiment had
already been run when it was made.

### A1 (2026-11-06, PR-8) — empirical mixed cells C3c / C3d

Reason: §3.1 defines C3c ("top-5 C1/C2 datasets by std of local homophily") and
C3d (compatibility matrix "far from both I and (J−I)/(k−1)") as determined
empirically, without a numeric rule. Rule fixed here: among the real (not
constructed) C1/C2 datasets with run records, C3c = the 5 largest full-label
`local_homophily_std`; C3d = the 5 largest min(‖H − I‖_F, ‖H − (J−I)/(k−1)‖_F).
Both are reported as extra tables in the E1 aggregate; they do not change the
headline ranks. Already run: E1 (all 16 datasets) — its per-dataset results were
known when this rule was written; the rule uses only label statistics, not
method results.

### A2 (2026-11-06, PR-8) — PERTURBED-REAL mixing sweep (suite MIX)

Reason: owner request (comparison on every kind of mixing, "even chaotic").
Design: `edsp/data/perturb.py` (modes, locality, degree-preserving partner
draws); base graphs cora, citeseer, amazon-ratings, roman-empire, wiki-editor;
22 conditions (`.github/workflows/mix.yml`); 3 re-wiring seeds × the first 3
splits; EDSP-default, EDSP-tuned (the 9-config E1 grid), and the E1 Track-A
baselines; H7 GS/damped runs skipped. Primary comparison: EDSP-tuned vs the best
baseline per condition (Wilcoxon over the 9 paired runs, Holm per base graph);
with 9 pairs the smallest attainable two-sided p is 0.0039, so a condition with
Holm-adjusted p ≥ 0.05 is reported as a tie, not as evidence of equality.
Results are PERTURBED-REAL, kept under `results/perturbed/`, never ranked with
real data. Already run when written: only pipeline smoke tests on cora
(≤ 2 seeds × 2 splits, conditions global:h=0.1 and permutation:h=0.1).

### A4 (2026-11-06, results PR of E1-rerun / E2 / MIX) — confirmatory mixing run

Reason: the A2 test could not reject at α = 0.05 by construction (9 pairs, Holm
over 22 conditions: smallest adjusted p = 0.086); see docs/MIX_REPORT.md §0. The
MIX run of A2 (seeds 0–2) is therefore exploratory, and its results were known
when this amendment was written.
Confirmatory run: the same 5 base graphs and 22 conditions, the same methods and
grids, **re-wiring seeds 10–19** (never used before) × the first 3 splits = 30
paired runs per condition (`mix.yml` with seeds = 10, seed_offset = 10).
Test: per condition, two-sided Wilcoxon signed-rank over the 30 pairs, EDSP-tuned
vs the baseline (harmonic, harmonic+CMN, LGC, PPR-LP, LinBP-MCE) with the highest
mean headline metric in that condition of the confirmatory run; Holm over the 22
conditions of each base graph; win / loss if adjusted p < 0.05, else tie. Every
condition is reported, whatever its outcome; the exploratory "wins" of MIX_REPORT
§1 are confirmed only where the confirmatory test gives a win.

### A6 (2026-11-07, PR-11) — Congress signed networks in E2

Datasets: the signed co-sponsorship backbones of Neal (2020) for the House and the
Senate in the 93rd, 100th, 107th and 114th Congress (one session about every 14
years, fixed before any of them was loaded): 8 graphs, labels D / R
(independents unlabelled), 10 stratified 60/20/20 splits, headline accuracy,
H-NATIVE. They enter E2 like wiki-editor (ranked, Holm across datasets); because
the 8 graphs are 2 chambers × 4 sessions and not independent samples, the E2
report also gives the Friedman ranks with and without them. Bitcoin OTC / Alpha
stay unregistered: their REV2 labels are defined from the ratings themselves
(C5.3 leakage) and the label files are not reachable from the dev sandbox.
Already run when written: nothing on these graphs (fixture test only).

### A5 (2026-11-07, PR-10) — H3 decision rule and the E5 run

Reason: H3 had no numeric decision rule. Variant: EDSP-fullH (Master Prompt §2.5)
— on edges with h_ij = 1 the message is V Ĥ instead of C(V); Ĥ = row-normalised,
Laplace-smoothed (+1) class counts on the train–train edges with h_ij = 1
(`edsp/regimes/compat.py`); canonical complement if there are none.
EDSP-fullH-default uses the EDSP-default inputs, EDSP-fullH-tuned the same
9-configuration grid as EDSP-tuned, validation-selected (R6).
Run (suite E5): every E1 dataset, all splits, `--fullh`.
Part 1: on the five C3d datasets of A1 (wikics, questions, amazon-computers,
roman-empire, coauthor-cs), EDSP-fullH-tuned vs EDSP-tuned, two-sided Wilcoxon
over splits, Holm across the five; **supported** if ≥ 3 wins and no loss, **not
supported** if ≤ 1 win, else **inconclusive**.
Part 2 ("the gap grows with k"): Spearman correlation between k and the mean
paired gap over all real E1 datasets, one-sided (ρ > 0), α = 0.05.
Secondary, exploratory: EDSP-fullH vs LinBP-MCE on the same data, and a MIX run
with `--fullh` on global h = 0 and 0.2, permutation h = 0.1 and 0.3, and community
(seeds 10–19 × 3 splits; PERTURBED-REAL).
Already run when written: only smoke tests (cora / citeseer / chameleon-filtered,
2 splits; cora permutation h = 0.1, 1 seed).

### A3 (2026-11-06, PR-8) — stratified accuracy bins

§5.7 specifies local-homophily bins of width 0.1 and per-degree bins; PR-3 had
5 bins. Now 10 bins of width 0.1 plus 11 logarithmic degree bins. No experiment
with the 5-bin layout was run beyond PR smoke tests (the E2 dispatch of
2026-11-06 was cancelled for this reason before any job finished).
