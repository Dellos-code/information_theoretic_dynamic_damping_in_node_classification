# Progress

## Done

- Read the Master Prompt in full.
- Created `AGENTS.md` and `.github/copilot-instructions.md`.
- Added the Python package scaffold and literal NumPy EDSP-canonical implementation.
- Added PR-1 unit/property/toy/benchmark coverage, including explicit T11 xfail.
- Added the ambiguity register.
- Regenerated `uv.lock` with `uv lock` (no manual lockfile edits).
- PR-1b follow-up: torch test gating via `EDSP_REQUIRE_TORCH`, CI sets `EDSP_REQUIRE_TORCH=1`, GS sweep uses CSR offsets, canonical rejects `beta != 1`, regime diagnostics use H/H'/Hmax definitions, and GS diagnostics are now computed (not fabricated).
- Added §1.8 follow-up coverage: explicit Regime 1/3 tests, E7 specified setup (unlabelled 4-cycle with one non-uniform start), case (b) `p=0.3` convergence check, and P1/D1 hypothesis tests.
- Added `scripts/findings_pr1b.py`, JSONL findings records at `results/runs/findings_pr1b.jsonl`, and regenerated `docs/FINDINGS.md` with final states and `||Φ(V)-V||_1`.
- Validation:
  - `uv sync --frozen --group dev && uv run pytest -q tests/unit`
  - `24 passed, 1 xfailed in 5.43s`
  - `uv run --with ruff ruff check edsp tests`
  - `uv run --with mypy mypy edsp/core.py`

- PR-2 (data layer): `edsp/data/` (graph format + canonicalisation, LabelVault with caller checks, registry, loaders, acquisition log, §3.2 statistics), `edsp/eval/evaluate.py` and `edsp/selection/` as the only val/test label consumers, `data/REGISTRY.json`, `data/PUBLISHED_STATS.json`, `scripts/dataset_stats.py`, workflow `dataset-stats.yml`, draft `docs/PREREGISTRATION.md`, new `tests/unit/test_data.py`.
  - Loaders: cora, citeseer, pubmed, amazon-computers, amazon-photo, coauthor-cs, coauthor-physics, wikics, roman-empire, amazon-ratings, minesweeper, tolokers, questions, chameleon-filtered, squirrel-filtered, sampson, ogbn-proteins (the last one untested on the real archive until its first Actions run).
  - DoD evidence: the `dataset-stats` workflow run on the PR (job summary + artifact). Numbers are cited only from that run.

- PR-2b: owner delegated the three pre-registration decisions; `docs/PREREGISTRATION.md` approved (H7 descriptive, secondary last-two-iterates column, minesweeper labelled and excluded from real-world aggregates) and frozen at the PR-2b merge commit.

- PR-4 (regimes, Master Prompt §2): `edsp/regimes/` — Track-A/Track-B priors and trusts, P-CONST/P-NATIVE/P-COUNT/P-SIGNMAG/P-LEARNED, H-TRAINONLY(b1–b4)/H-NATIVE/H-LEARNED/H-ALL0/H-ALL1/H-RELATION, H-ORACLE isolated in `edsp/regimes/oracle.py` behind `ORACLE=1`; `edsp/methods.run_edsp`; selection helpers that keep validation labels inside `edsp.selection`; `edge_polarity_auc` in `edsp.eval.evaluate`; numpy/scipy logistic regression, temperature scaling, randomized SVD and ROC-AUC in `edsp/utils/numerics.py`.
  - T11 now passes: every Track × H × P combination runs in strict vault mode, never touches test/oracle/descriptive labels, and gives bit-identical output when the test labels are changed (`tests/unit/test_regimes.py`).
  - DoD evidence (edge-classifier AUCs): the `edge-classifier` workflow run on the PR (job summary + artifact).

- PR-4b: H-LEARNED decision rule — default `prior-corrected` (train-edge base rate restored by Bayes prior shift), options `val-threshold` and `balanced`; edge-classifier report now compares the three rules with H-ALL0/H-ALL1 on test edges.

- PR-5 (B0/B1 baselines, Master Prompt §4): `edsp/baselines/` — majority, train prior, signed vote, feature LR (= Track-B base predictor), harmonic (exact; plain and class-mass-normalised), LGC, PPR-LP (exact solves), LinBP* with MCE, Correct & Smooth on the Track-B base predictor, possible-worlds harmonic; `select_and_predict` chooses hyper-parameters on validation through edsp.selection; `docs/BASELINES.md`; `scripts/baseline_parity.py` + workflow `baseline-parity.yml`; tests incl. test-label invariance for every baseline.

- PR-7 (E1 pipeline): full §5.4 metric set (`edsp/eval/metrics.py`), §5.9 statistics (`edsp/eval/stats_tests.py`: bootstrap CI, Wilcoxon, Holm, Friedman + Nemenyi CD), validation selection by headline metric, 9-point baseline grids (R6), `scripts/run_e1.py`, `scripts/aggregate_e1.py`, workflow `e1.yml` (dispatch: one job per dataset → aggregate → results branch; PR: 1-split smoke run).

- E1 (Track-A, 16 datasets, 1,743 run records) completed: `docs/E1_REPORT.md`, `results/aggregates/e1/`. EDSP-tuned: 0 wins / 4 ties / 10 losses vs LGC; ≈ harmonic LP (H1 supported for harmonic, not for LGC); H2 supported only for minesweeper; H7: non-convergence 3/134 (Planetoid public splits), no oscillation, GS and damped fixed points coincide.

- PR-3 (mixed / signed acquisition + E2 pipeline): loaders for YelpChi and Amazon fraud (C3a, relation-typed; edge relation bitmask; Amazon unlabelled prefix excluded), wiki-editor, wiki-RfA, wiki-Elec (C5, signed + node labels; Mercado convention; RfA/Elec are leakage case studies); homophily statistics ignore unlabelled nodes; `signed_lgc` baseline; per-record stratified accuracy by local-homophily bin (§5.7); `run_e1.py --suite E2` (H-NATIVE on signed, H-RELATION on relation-typed graphs) and `aggregate_e1.py --suite E2`; `e1.yml` takes a `suite` input.

- PR-8 (mixing): stratified accuracy with §5.7 bins (10 local-homophily bins of width 0.1 + 11 degree bins); a DATASET-STATS record per dataset; empirical mixed cells C3c / C3d in the E1/E2 aggregate (PREREGISTRATION A1); PERTURBED-REAL mixing sweep — `edsp/data/perturb.py` (global, beta, uniform, logistic-map chaos, regional, class-community, permutation-structured mixing), `run_e1.py --perturb`, `scripts/aggregate_mix.py`, workflow `mix.yml` (PREREGISTRATION A2).

- Results (2026-11-06): E1 re-run with §5.7 bins and C3c/C3d (docs/E1_REPORT.md), E2 on signed and relation-typed graphs (docs/E2_REPORT.md; YelpChi pending), exploratory MIX sweep (docs/MIX_REPORT.md); PREREGISTRATION A4 fixes the confirmatory MIX run.

- PR-10 (H3): named variant EDSP-fullH in the core (`heterophilic_operator`, canonical unchanged), train-only Ĥ estimator, runner `--fullh`, H3 table in the aggregate, `extra_args` workflow input, PREREGISTRATION A5.

- PR-11: Congress signed networks (8 graphs). PR-12: Track-B / E3 — PyTorch GNN baselines (MLP, GCN+, SAGE+, GAT+, GPR-GNN, BWGNN), `run_e1.py --track B --gnn`, H4 analysis (PREREGISTRATION A7).

- Results (2026-11-08): E3 / H4 on 13 real graphs (docs/E3_REPORT.md; H4 inconclusive; A9 fraud, A14 GAT fallback on tolokers); A11 attribution of the MIX wins (MIX_REPORT §5); A13 native-mixing graphs (docs/A13_REPORT.md; inconclusive, FB100 wins belong to the learned polarity). E3 on coauthor-cs / -physics (A12) done (2026-11-09).

- PR-27 (C4 / H5; run 2026-11-09, results in docs/H5_REPORT.md): `GraphData.task_view` (binary task views of multi-label graphs), `scripts/run_c4.py` (EDSP-const/native default + tuned, 9 configs each; B0/B1 baselines unweighted and weighted by the noisy-OR p), `scripts/aggregate_c4.py` (descriptive, no verdict), suite C4 in `e1.yml`, CSR backend `edsp.sparse_backend` (parity with the reference), P-NATIVE max/mean modes (the ogbn-proteins columns are unnamed, SPEC_AMBIGUITIES), batched H-LEARNED above 5M edges. Cost probes (agent sandbox, synthetic graphs, n = 132,534, k = 2): `results/runs/c4_iteration_probe_39.6m.json` — 0.40 s per EDSP iteration at 39.56M edges, 7.4 s setup per run, 19 s directed-edge plan per job; `results/runs/c4_cost_probe_10.8m.json` — one task with all methods at 10.8M edges took 1,435 s (EDSP grids 1,142 s, with the 6 H-ALL1 configurations at the 1,000-iteration cap; H-TRAINONLY(b1) 223 iterations, H-LEARNED ≲ 95 judging by wall clock; baselines 208 s), tracemalloc peak 1.78 GiB per EDSP block. Extrapolated to ogbn-proteins (39.56M edges): ≈ 7,900 EDSP iterations × 0.40 s + 18 × 7.4 s ≈ 55 min, baselines ≈ 13 min, so ≈ 70 min per task if the real graph converges like the synthetic one, ≈ 2.3 h per task if no EDSP configuration converges (18,000 iterations); peak memory ≈ 8.5–10.5 GB (16 GB runner). Recommended: pilot `chunks = ogbn-proteins:2:1`, then `ogbn-proteins:112:2` (56 jobs; `--time-budget-min 290` records tasks that would not fit as NOT-RUN).
- C4 items 2–4 (uncertain DBLP, US-Patent assignee graph, raw STRING + GO-slim): availability checked 2026-11-09, none run (evidence in `data/REGISTRY.json`, SPEC_AMBIGUITIES).
- PR-29 (B5 / B6, A17; 2026-11-09): signed GNNs SGCN / SNEA / SDGNN / SiGAT (PyGSD 1.2.0, embeddings + LR) on the 12 E2 signed graphs, CARE-GNN / PC-GNN re-implementations on amazon-fraud and yelpchi, yelpchi Track-B set. EDSP-tuned vs signed GNNs on 9 ranked graphs: 7 wins / 24 ties / 5 losses over 36 comparisons, #1 by mean on 1 graph; EDSP-B-tuned vs CARE-GNN 1 / 0 / 1, vs PC-GNN 0 / 1 / 1 (docs/B5B6_REPORT.md, `scripts/b5b6_compare.py`).

- PR-40 (A18 graphs, 2026-11-10, stopped early on the coordinator's request): loaders for GraphLand hm-categories / artnet-exp / city-reviews / web-fraud / web-topics, ogbn-arxiv, ogbn-products, arxiv-year, twitch-gamers, snap-patents, pokec, registered as opt-in extension "a18" (no default list picks them up), md5 check and HTML-page rejection in `acquisition.fetch`, fixture tests `tests/unit/test_a18_loaders.py`. Not exercised on the real files: no dataset-stats run was dispatched, so every download and every count is TO VERIFY; no method has been run on these graphs.

## Remaining
- H6 (A16, 2026-11-09): not supported (docs/H6_REPORT.md).
- H5 (A15, 2026-11-09): supported on ogbn-proteins, 112 tasks (docs/H5_REPORT.md). The effect equals generic edge weighting. Uncertain DBLP, US-Patent and raw STRING could not be obtained as specified.
- Fraud SOTA not run: GAGA, H²-FDetector; parity of CARE-GNN / PC-GNN with published numbers.

- Verify the sources marked TO VERIFY in data/PUBLISHED_STATS.json (Planetoid/Amazon/Coauthor/WikiCS reference values).
- PR-3: fetch-data.yml and acquisition run for the remaining cells (LINKX, fraud, C4/C5 sources).
- `scripts/oracle_upper_bound.py` (E10) and H-SOFT (extension) — not implemented yet.
- Track-B 2-layer MLP / LightGBM base predictors.
- LinBP-DCE, Mercado GL, uBayes, HAPPY, REV2.
- Re-run E1 so its records carry the stratified accuracy (§5.7).
- Remaining acquisitions: Bitcoin OTC/Alpha (+ REV2 labels), Congress signed backbones, C4 (DBLP / patents); E2 on ogbn-proteins (multi-label, per task, P-NATIVE).

## Blockers

- New dependencies (scikit-learn, LightGBM, PyG) cannot be locked from the Claude session because download.pytorch.org is unreachable there; `uv lock` must run in Actions or a Copilot session.
- snap.stanford.edu (ogbn-proteins) is unreachable from the development session; the loader is first exercised by `dataset-stats.yml` with `include_large=true`.
