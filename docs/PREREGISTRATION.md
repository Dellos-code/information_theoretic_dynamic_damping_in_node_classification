# Pre-registration (DRAFT — requires owner sign-off before any full run)

Status: **DRAFT, not frozen.** Rule R2 requires this file to be committed and
frozen before the first full experiment run (PR-8, suite E1). The owner must
approve or edit the items marked **OWNER DECISION**; after approval the commit
SHA that freezes it is recorded here and the file is only appended to.

## 1. Hypotheses (Master Prompt §5.11, with one revision)

| id | hypothesis | outcome labels |
|---|---|---|
| H1 | Track-A EDSP ≈ harmonic LP / LGC on homophilic graphs (C1). | supported / not supported / inconclusive |
| H2 | Track-A EDSP is near chance on minesweeper, tolokers, questions (label informativeness ≈ 0). | same |
| H3 | EDSP-fullH > EDSP on structured-compatibility graphs (C3d); the gap grows with k. | same |
| H4 | Track-B EDSP ≈ C&S < tuned GNNs on feature-rich heterophilous graphs. | same |
| H5 | P-NATIVE > P-CONST on ogbn-proteins / uncertain DBLP. | same |
| H6 | Finite T_L helps under label noise. | same |
| H7 (revised) | **OWNER DECISION.** The original H7 ("k=2 Jacobi oscillation only in label-free components") is already falsified by toy cases in `docs/FINDINGS.md` (two informed nodes on one homophilic edge oscillate for P_esc above ≈0.55). Proposed replacement: *descriptive outcome, no directional prediction*: report the non-convergence rate of EDSP-canonical per dataset and track, the fraction of oscillating nodes, and whether EDSP-GS and EDSP-damped(0.5) reach different fixed points (‖V_GS − V_damped‖₁ per node). | reported values |

## 2. Primary metrics and tracks

Metrics follow Master Prompt §5.4; the headline metric per dataset is the one in
`data/REGISTRY.json` (`metric`). Track-A and Track-B are reported separately and
never compared as like-for-like (§4).

**OWNER DECISION — non-convergence policy.** EDSP-canonical is evaluated on its
last iterate when it hits `max_iter=1000` without converging
(`docs/SPEC_AMBIGUITIES.md`). Because a 2-cycle makes the last iterate depend on
the parity of `max_iter`, the proposal is to *also* report, as a labelled
secondary column, the average of the last two iterates. The headline column stays
the literal last iterate.

## 3. Pre-registered exclusions (rule R2)

| dataset / setting | treatment | reason |
|---|---|---|
| chameleon, squirrel (unfiltered) | never in headline tables | duplicate nodes (Platonov et al., ICLR 2023); filtered versions are used |
| texas, cornell, wisconsin | appendix only | tiny, unstable splits (Master Prompt §3.1 C2.4) |
| sampson | case study, excluded from rank statistics | 18 nodes (Master Prompt §3.1 C5.4) |
| H-ORACLE results | separate table, never ranked | uses test labels (§2.3g) |
| minesweeper | **OWNER DECISION** | it is a constructed grid game from Platonov et al. (2023), i.e. not a real-world network; proposal: keep it as an established benchmark but label it "constructed benchmark" in every table and exclude it from real-world aggregate claims, in the spirit of R4 |

## 4. Splits

Official splits where they exist (Planetoid public; Platonov 10 splits; WikiCS
20 splits; OGB species split). Otherwise 10 stratified random 60/20/20 splits
with seeds 0–9 (`edsp.data.graph.random_splits`). Planetoid datasets also get the
10 random splits. Sampson: leave-one-out.

## 5. Freeze record

- Draft committed: PR-2.
- Approved by owner: _pending_.
- Frozen at commit: _pending_.
