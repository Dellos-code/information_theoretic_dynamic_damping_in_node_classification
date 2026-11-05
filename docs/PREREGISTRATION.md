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

_None yet. Every amendment must state its date, reason and whether any experiment
had already been run when it was made._
