# Master Prompt for an Autonomous GitHub Agent: Real-World Evaluation of Entropy-Damped Signed Propagation for Node Classification

Your algorithm can be evaluated rigorously and autonomously on GitHub. The only setup that works is this: the Copilot cloud agent writes the code and runs smoke tests in its session, which is capped at 59 minutes. Every heavy run happens in ordinary GitHub Actions workflows, which allow 6 hours per job and are not behind the agent firewall.\[1\] All realistic results must come from leakage-free polarity regimes; "oracle-h" numbers count only as an upper bound.

## TL;DR

- **What to run:** Give the agent the Master Prompt below as one master issue split into sub-issues of under 45 minutes each. The agent builds a leakage-guarded harness and a dataset registry whose rule is real data first and synthetic data only as an audited fallback. Full matrices run in `workflow_dispatch` Actions jobs; the agent session only does development and smoke tests.
- **What your method actually is:** In the label-only setting it reduces exactly to soft-clamped harmonic label propagation over signed complement edges with log-odds weights. Unlabelled nodes always get α=1, and one-hot labelled nodes get α≤1/2. The decisive baselines are therefore Zhu–Ghahramani, LGC, LinBP with estimated compatibility, and C&S, not only GNNs.
- **Who to beat, and where you will probably lose:** No single "#1 SOTA" exists. The strongest documented reference is tuned classic GNNs: Luo, Shi & Wu ("Classic GNNs are Strong Baselines", NeurIPS 2024 Datasets & Benchmarks, arXiv 2406.08993) report that "classic GNNs outperform state-of-the-art GTs, achieving the top rank on 17 out of 18 datasets", and NodeGround (2026) again favours carefully tuned GNNs. Polynormer is the strongest graph transformer. Expect the label-only track to be near chance on minesweeper, tolokers and questions (label informativeness ≈0.00–0.01). Your realistic arenas are the feature-prior track (against C&S) and graphs with native signs or native probabilities.

## Key Findings that shaped the prompt

**1. The algorithm, analysed as a peer reviewer would.**
- **Exact reduction.** If v_j is uniform, then H=Hmax and α_j=1 whenever P_esc>0, so V_j=S_j. If v_j is one-hot, then H=0 and α = (Hmax−H′)P/((Hmax−H′)P+Hmax) ≤ 1/2. The unlabelled subsystem is therefore affine (Jacobi iteration of a signed random-walk operator); only labelled nodes are nonlinear. In the label-only track P_esc and p0 act only through labelled nodes, so the prompt makes the agent test whether p0 is nearly inert there.
- **The complement operator is a compatibility matrix.** C = (J−I)/(k−1) on the simplex, with eigenvalue 1 on the all-ones direction and −1/(k−1) on the zero-sum subspace. Writing V=u+δ, deviations evolve as δ_j ← Σ w_ij s_ij δ_i with s_ij ∈ {1, −1/(k−1)}.
  - For k=2 the heterophilic factor is −1, so a component with no labelled node and bipartite or unbalanced structure can have spectral radius 1, which predicts oscillation under synchronous updates.
  - For k≥3 each heterophilic hop shrinks the signal by 1/(k−1), which is why structured-compatibility methods should win on graphs such as roman-empire (18 classes).
- **Existence but not convergence.** The update map is continuous on the product of simplices (at H=Hmax the boundary rule returns uniform, which equals S_j), so Brouwer guarantees a fixed point; nothing guarantees uniqueness or that the iteration reaches it.
- **A free numerical identity.** With den_j = Σ_i −ln(1−p_ij), P_esc(j) = 1 − exp(−den_j) = −expm1(−den_j).

**2. Leakage is the central threat.**
- **Oracle polarity.** Computing h_ij from both endpoints' labels makes k=2 trivial: one label per component determines all the others.
- **Wiki-RfA/Elec.** The node label is the adminship outcome and the edges are the support/oppose votes on that candidate.\[2\]
- **Bitcoin OTC/Alpha.** REV2 ground truth starts from the platform founder and the users he rated highly, so the founder's rating edges encode labels.\[3\]
- **Required controls.** Trivial "majority in-vote sign" baselines and founder-edge-removed variants; otherwise a "win" on these graphs means nothing.

**3. Datasets that genuinely match the paper's semantics, all verified to exist.**
- **Native probabilistic edges with labels:** ogbn-proteins (132,534 nodes, 39.56M edges with 8 STRING v11 channel scores, 112 binary GO tasks); uncertain DBLP (922,673 nodes, 3,389,272 edges, 14 classes, 16% labelled) and Patent (108,658 nodes, 1,059,822 edges, 5 classes) from Dallachiesa, Aggarwal and Palpanas.\[4\]\[5\]\[6\]\[7\]
- **Natively signed edges with labels:** Sampson's monastery (25 novices, 4 factions, labelled in PyG Signed Directed); Neal's US Congress signed co-sponsorship backbones (1973–2016, party labels); Bitcoin OTC/Alpha with REV2 labels; Wiki-RfA/Elec/Editor as built by Mercado et al.\[2\]\[8\]\[9\]\[10\]
- **Mixed by relation type:** YelpChi's R-U-R relation has label similarity 0.90 while R-T-R and R-S-R are 0.05; Amazon's U-P-U is 0.19 and U-S-U/U-V-U are 0.04/0.03.\[11\]

**4. The competitive landscape (as of November 2026).** Tuned classic GNNs (`LUOyk1999/tunedGNN`) and Polynormer (`cornell-zhang/Polynormer`) lead the Platonov benchmarks at roughly 92 on roman-empire; SGFormer trails at 73.91.\[12\]\[13\]\[14\] Newer claimants (HAPPY, IM-MPNN, TANGO, GraphTARIF, fixed-aggregation features, graph foundation models) are treated as claims until reproduced.\[15\]\[16\]\[17\]\[18\] Luan, Lu, Hua, Wang, Zhu, Chang, Wolf & Tang (arXiv 2409.05755) "train and fine-tune baseline models on 27 most widely used benchmark datasets, and categorize them into three distinct groups: malignant, benign and ambiguous heterophilic datasets"; the prompt uses this taxonomy so wins on "benign" heterophily are not over-claimed.

**5. What the platform allows (verified).**
- **Firewall:** custom allowlist entries go under Settings → Copilot → Internet access. The firewall does not apply to processes started in setup steps.\[19\]
- **`copilot-setup-steps.yml`:** the job must be named `copilot-setup-steps`; GitHub Docs list the customisable fields as "steps, permissions, runs-on, services, snapshot, timeout-minutes (maximum value: 59)" (some doc versions also list `container`).
- **Runners:** larger runners via `runs-on`; current docs say the agent runs on Ubuntu x64 or Windows x64, while one docs version says it "is only compatible with Ubuntu x64 Linux runners" — use Ubuntu x64.
- **Limits:** GitHub Docs (as quoted by a secondary source) state "Each Copilot cloud agent session has a maximum execution time of 59 minutes. This limit cannot be extended or bypassed"; GitHub's "Actions limits" page states "Each job in a workflow can run for up to 6 hours of execution time" and "A job matrix can generate a maximum of 256 jobs per workflow run"; the GitHub Changelog of 8 July 2024 announced that "GitHub Actions GPU hosted runners are now generally available for Windows and Linux, providing T4 GPU access", configured through organization/enterprise runner groups; and the GitHub Changelog of 4 February 2026 announced that "Claude by Anthropic and OpenAI Codex are now available as coding agents for Copilot Pro+ and Copilot Enterprise customers" (Agent HQ public preview, one premium request per session), which must be enabled in settings.

## Preface for John (Greek)

Γιάννη, οι βασικές αποφάσεις: (1) Ο agent γράφει κώδικα και τρέχει μόνο smoke tests· τα βαριά πειράματα τρέχουν σε GitHub Actions (όριο 6 ωρών/job, εκτός firewall). (2) Τα «oracle-h» αποτελέσματα είναι μόνο άνω φράγμα· οι πραγματικοί αριθμοί προέρχονται από native/train-only/learned polarity. (3) Συνθετικά δεδομένα μόνο με καταγεγραμμένη εξάντληση όλων των πραγματικών υποψηφίων, ποτέ στους κύριους πίνακες. (4) Δύο tracks: label-only (εκεί ο αλγόριθμός σου είναι signed harmonic LP — αντίπαλοι LP/LinBP/C&S) και feature-prior (αντίπαλοι tuned GNNs, Polynormer).
Χειροκίνητες ενέργειες:
1. Repo (ιδανικά public για δωρεάν Actions minutes) με ενεργό Copilot cloud agent (πληρωμένο πλάνο· για Claude/Codex μέσω Agent HQ: Pro+ ή Enterprise και ενεργοποίηση στα settings).
2. Settings → Copilot → Internet access → Custom allowlist: snap.stanford.edu, ogb.stanford.edu, data.pyg.org, raw.githubusercontent.com, objects.githubusercontent.com, codeload.github.com, zenodo.org, figshare.com, ndownloader.figshare.com, string-db.org, networks.skewed.de, konect.cc, vlado.fmf.uni-lj.si, helios2.mi.parisdescartes.fr, cs.stanford.edu, dblp.org, dblp.uni-trier.de, drive.google.com, drive.usercontent.google.com, download.pytorch.org.
3. Settings → Actions → General: «Read and write permissions» και «Allow GitHub Actions to create and approve pull requests».
4. Στα PR του agent πάτα «Approve and run workflows» όταν ζητηθεί· αν ο agent δεν μπορεί να κάνει dispatch, ξεκίνα τα full runs από Actions → Run workflow.
5. Προαιρετικά larger/GPU runners (απαιτούν organization σε Team/Enterprise — έλεγξε) και άλλαξε το `runs-on`.
6. Δημιούργησε issue με το Master Prompt και ανάθεσέ το στο Copilot (Assignees → Copilot ή Agents panel)· secrets δεν χρειάζονται.

## The Master Prompt (paste into a GitHub issue)

````markdown
# MASTER TASK: Publication-grade evaluation of "Information-Theoretic Per-Node Dynamic Damping and Trust-Calibrated Homophily/Heterophily for Node Classification in Stochastic Graphs" (method id: EDSP — Entropy-Damped Signed Propagation)

## 0. ROLE, MISSION, NON-NEGOTIABLE RULES
You are a senior research engineer and an adversarial reviewer. Mission: implement §1 exactly, instantiate it leak-free on REAL-WORLD graphs covering homophilic, heterophilic and every kind of mixed network plus the subcategories of §3, compare it fairly against the strongest methods of each kind (§4), EXECUTE all experiments (§5) via GitHub Actions (§6), and deliver a publication-ready report (§8), working incrementally across sessions and PRs (§7).

R1 NO FABRICATED NUMBERS. Every number in any table, figure, README, PR or report must trace to a logged run record: git SHA, config hash (SHA-256 of resolved config), dataset hash (SHA-256 of processed tensors), seed, split id, runner type, wall-clock, peak RSS. Paper numbers are labelled REPORTED with a citation and never mixed into REPRODUCED columns.
R2 HONEST NEGATIVE RESULTS. Losses, ties, non-convergence are first-class results. Never drop a dataset, seed or baseline because the method loses. Exclusions are pre-registered in `docs/PREREGISTRATION.md` (committed before any full run).
R3 ZERO TEST-LABEL LEAKAGE. Test labels are readable only inside `evaluate()`. Polarity, p, T, priors, tuning, early stopping, calibration and compatibility estimation may use TRAIN labels (VALIDATION labels for selection only). Enforced by code and test (§2.7).
R4 REAL-DATA-FIRST. Synthetic graphs are allowed in a (sub)category IF AND ONLY IF every ranked real candidate has failed the logged procedure of §3.3. Synthetic results are labelled SYNTHETIC-SUBSTITUTE, stored in `results/synthetic/`, and NEVER enter real-world aggregate tables, ranks, CD diagrams or headline claims. Perturbations of real graphs are PERTURBED-REAL, not synthetic.
R5 NEVER SILENTLY "FIX" THE ALGORITHM. `EDSP-canonical` follows §1 literally. Any deviation is a named variant in `edsp/variants.py`, reported separately. Ambiguities: literal reading in canonical, alternative as a variant, entry in `docs/SPEC_AMBIGUITIES.md`.
R6 EQUAL TUNING BUDGETS for EDSP and every tuned baseline (trials, space breadth, selection metric, validation-only), plus a separately reported ZERO-TUNING DEFAULT.
R7 Never claim what you did not run; list it in a "Not run" table with the reason.
R8 Mark anything unverified (URLs, licenses, splits) TO VERIFY until verified; record evidence (URL, date, checksum).

## 1. ALGORITHM SPECIFICATION (canonical)
### 1.1 Inputs
Undirected G=(V,E), n=|V|, k≥2 classes. Prior $v_j\in\Delta^{k-1}$. Edge probability $p_{ij}=p_{ji}\in[0,1]$ (degree of homophilic identification / heterophilic opposition and independent existence probability). Polarity $h_{ij}\in\{0,1\}$ (0 homophilic, 1 heterophilic). Trust $T_{ij},T_j\ge0$, ∞ allowed.
### 1.2 Step 0 — trust calibration (once)
$$p_{ij}\leftarrow p_{ij}(1-e^{-T_{ij}}),\qquad v_j\leftarrow(1-e^{-T_j})v_j+\tfrac{e^{-T_j}}{k}\mathbf 1_k.$$
### 1.3 Step 1 — static quantities
$$L_{ij}=-\ln(1-p_{ij}),\ \mathrm{den}_j=\textstyle\sum_{i\in N(j)}L_{ij},\ w_{ij}=L_{ij}/\mathrm{den}_j,\ P_{esc}(j)=1-\prod_{i\in N(j)}(1-p_{ij})=1-e^{-\mathrm{den}_j}.$$
Implement $L$=`-log1p(-p)`, $P_{esc}$=`-expm1(-den)`. In general $w_{ij}\ne w_{ji}$.
### 1.4 Step 2 — heterophily operator
$\tilde V_i^{(j)}=V_i$ if $h_{ij}=0$; $\tilde V_i^{(j)}=C(V_i)=(\mathbf 1_k-V_i)/(k-1)$ if $h_{ij}=1$. $C=(J-I)/(k-1)$; $C(u+\delta)=u-\delta/(k-1)$; $C\circ C(u+\delta)=u+\delta/(k-1)^2$.
### 1.5 Step 3 — synchronous (Jacobi) iteration from $V^{(0)}_j=v_j$
$$S_j^{(t)}=\textstyle\sum_{i\in N(j)}w_{ij}\tilde V_i^{(j,t)},\ H=H(v_j),\ H'=H(S_j^{(t)}),\ H_{max}=\ln k,$$
$$\alpha_j^{(t)}=\frac{(H_{max}-H')P_{esc}(j)}{(H_{max}-H')P_{esc}(j)+(H_{max}-H)},\quad V_j^{(t+1)}=(1-\alpha_j^{(t)})v_j+\alpha_j^{(t)}S_j^{(t)}.$$
Boundary: if $H=H'=H_{max}$, or $P_{esc}(j)=0$ and $H=H_{max}$, set $V_j^{(t+1)}=u$. Stop when $\max_j\|V_j^{(t+1)}-V_j^{(t)}\|_1<\varepsilon$ (1e-6) or t=max_iter (1000). Predict argmax; probabilistic output $V_j$.
### 1.6 Claims to test
P1 $H_{max}-H(P)=\mathrm{KL}(P\|u)$ ⇒ $\alpha/(1-\alpha)=[\mathrm{KL}(S_j\|u)/\mathrm{KL}(v_j\|u)]P_{esc}(j)$. P2 base invariance. P3 Regime 1 exact. P4 Regime 2 ⇒ α=1. P5 Regime 3 ⇒ α=0. P6 O(|E|k+|V|k) time, O(|E|+|V|k) memory per iteration. P7 training-free.
Derived facts (test and report): D1 one-hot prior ⇒ α≤1/2. D2 uniform prior, $P_{esc}>0$ ⇒ α=1 regardless of $P_{esc}$ (label-only EDSP = soft-clamped nonlinear harmonic LP with complement edges and log-odds weights). D3 unlabelled deviation dynamics $\delta_j\leftarrow\sum_i w_{ij}s_{ij}\delta_i$, $s_{ij}\in\{1,-1/(k-1)\}$; k=2 label-free bipartite/unbalanced components may oscillate. D4 fixed point exists (Brouwer); uniqueness/convergence not guaranteed — measure. D5 constant p0 ⇒ w=1/deg, $P_{esc}=1-(1-p_0)^{\deg}$. D6 $p=1-e^{-c/\mu}$ ⇒ $w\propto c$.
### 1.7 Implementation
- Package `edsp/` (Python ≥3.11); backends numpy/scipy.sparse (float64 reference) and torch (CPU default, CUDA optional) via `torch.sparse`/`index_add_`/`scatter_reduce`; `torch_scatter`/`torch_sparse` optional only. Backends agree to 1e-10.
- Directed edge list with both directions; per-edge p, h, L; per-node den, P_esc; V is n×k.
- Numerics: clip p to [0,1−1e-12] (log the count); entropy via `xlogy`; H==Hmax via tolerance 1e-12·ln k, never float equality; if α numerator and denominator < 1e-15 apply the boundary rule.
- Degenerate nodes (degree 0 or den=0): S=u, P_esc=0 ⇒ V=v if H<Hmax, uniform otherwise. Label-free components stay uniform: report "uncovered test nodes", predict by lowest-class tie-break AND variant `EDSP+prior-fallback` (train-prior argmax); report accuracy with/without uncovered nodes.
- Per-run logs: iterations, converged, final residual, residual trace, oscillation flag (non-decreasing residual over 20 iterations and δ sign alternation on ≥1% nodes), regime fractions per iteration, α histograms at iterations {1,5,last}, mean entropy trajectory.
- Variants: `EDSP-GS` (Gauss–Seidel, id order and seeded random order), `EDSP-fixedalpha(a)`, `EDSP-noPesc`, `EDSP-noTrust`, `EDSP-uniformW`, `EDSP-fullH` (C replaced by row-normalised compatibility Ĥ estimated from train–train edges, Laplace-smoothed), `EDSP-softpol(q)` (EXTENSION: Ṽ=(1−q)V+qC(V)), `EDSP-damped(β)` (labelled oscillation remedy), `EDSP+prior-fallback`.
### 1.8 Tests (pytest + hypothesis, ≥500 examples, k∈{2..20}; must pass before experiments)
T1 C preserves the simplex. T2 C eigen-structure and C∘C. T3 KL identity and P1. T4 base invariance (ln, log2, log10). T5 Regimes 1–3, D1, D2. T6 trust: T=0 ⇒ uniform prior and p=0; T=∞ unchanged; KL(v‖u) monotone in T. T7 Σw=1, P_esc identity. T8 iterates stay in simplex. T9 backend parity. T10 degenerate nodes. T11 leakage guard.
Toy examples (tolerance 1e-12): E1 k=2 C([0.7,0.3])=[0.3,0.7]. E2 k=3 C([1/2,1/4,1/4])=[1/4,3/8,3/8]. E3 k=2, v_j=[1,0], one neighbour p=0.5, h=0, V_i=[0,1] fixed: P_esc=0.5, α=1/3, V_j=[2/3,1/3]. E4 same with h=1: V_j=[1,0]. E5 uniform v_j ⇒ V_j=S_j after one step for any p∈(0,1). E6 two-node h=1 graph with one labelled node ⇒ complementary labels. E7 unlabelled 4-cycle, k=2, all h=1, one non-uniform start: document whether Jacobi oscillates and whether GS converges (a finding, not a bug).

## 2. INSTANTIATING v, p, h, T (each regime named and reported separately)
2.1 Tracks. TRACK-A LABEL/STRUCTURE-ONLY: features ignored; train nodes one-hot (trust T_L), others uniform. TRACK-B FEATURE-PRIOR: priors from a base predictor trained on train nodes only (logistic regression, 2-layer MLP, LightGBM; selected on validation; temperature-scaled on validation); overwriting train nodes with one-hot is a tuned option; T_j = τ·(ln k − H(prior_j))/ln k (τ tuned) or ∞ for train nodes. Track-B EDSP is directly comparable to Correct&Smooth.
2.2 Edge probability p. P-NATIVE (ogbn-proteins: (i) noisy-OR of the 8 channels, (ii) single channels "experimental", "textmining", (iii) STRING's official combined-score formula with prior correction — TO VERIFY at string-db.org; uncertain DBLP/Patent probabilities). P-COUNT p=1−exp(−c/μ), μ∈{median(c), tuned}. P-CONST(p0): default 0.5; tuned p0∈{0.01,0.05,0.1,0.2,0.3,0.5,0.7,0.9,0.99}; sweep always reported. P-SIGNMAG for signed weights: p=1−exp(−|w|/μ) and p=|w|/max|w|. P-LEARNED: edge-classifier confidence.
2.3 Polarity h. (a) H-NATIVE: exogenous signs; directed→undirected by summing signed weights (h=1 iff sum<0; |sum| feeds p); log sign-conflicting reciprocal pairs and report a "drop conflicting" variant. (b) H-TRAINONLY: h=1[y_i≠y_j] on train–train edges; remaining edges (b1) h=0, (b2) h=1, (b3) train-edge majority, (b4) dropped (report coverage). (c) H-LEARNED: edge classifier (LR, 2-layer MLP) trained ONLY on train–train edges with inputs [x_i,x_j,|x_i−x_j|,x_i⊙x_j] (Track-B) or label-free structural features (Track-A: degrees, common neighbours, Jaccard, SVD/node2vec); class-balanced; selection on train–val/val–val edges; decision → h, confidence → p. (d) H-ALL0 / H-ALL1. (e) H-SOFT(q): EXTENSION, not the paper's algorithm. (f) H-RELATION: multi-relational graphs; h_r=1 iff train–train homophily of relation r < Σ_c π_c². (g) H-ORACLE: both endpoints' true labels; UPPER BOUND ONLY, separate table "Oracle polarity (uses test labels; not a valid method)", never ranked.
2.4 Trust. Track-A: T_L∈{∞ (default),0.5,1,2,4}; unlabelled T=0; T_ij=∞ or tuned. Pre-registered hypothesis: finite T_L helps under label noise.
2.5 EDSP tuning space: p-regime parameter, T_L, H-regime within the registered family (reported as "EDSP-tuned"; each regime also reported), Track-B base model and τ; ε and max_iter fixed. Zero-tuning default: P-CONST(0.5) or P-NATIVE, H-TRAINONLY(b1) or H-NATIVE, T_L=∞.
2.6 Directed graphs: symmetrise A←A+Aᵀ and report it; Dir-GNN keeps directions.
2.7 Leakage guard: `edsp/data/vault.py` `LabelVault` exposes `train_labels()`, `val_labels()` (only to `selection/`) and `test_labels()` only to `edsp/eval/evaluate.py` (caller check + flag). T11 runs every method and regime on a toy graph in strict mode and asserts no test-label access. H-ORACLE runs only in `scripts/oracle_upper_bound.py` with a recorded `ORACLE=1` flag.

## 3. DATASET REGISTRY
### 3.1 Taxonomy grid (ranked REAL candidates; verify every TO VERIFY and log evidence)
C1 HOMOPHILIC: 1. Cora, CiteSeer, PubMed (PyG Planetoid; public split and 10 random 60/20/20; accuracy; count CiteSeer isolated nodes). 2. Amazon-Computers/Photo, Coauthor-CS/Physics (splits as in `LUOyk1999/tunedGNN`, TO VERIFY). 3. WikiCS (20 official splits). 4. ogbn-arxiv (symmetrised), ogbn-products (official splits).
C2 HETEROPHILIC: 1. roman-empire, amazon-ratings, minesweeper, tolokers, questions (Platonov et al., ICLR 2023; PyG `HeterophilousGraphDataset`; `https://github.com/yandex-research/heterophilous-graphs/raw/main/data/{name}.npz`; MIT; 10 splits; accuracy for the first two, ROC-AUC otherwise). Sanity check: edge homophily 0.05/0.38/0.68/0.59/0.84, adjusted −0.05/0.14/0.01/0.09/0.02, label informativeness 0.11/0.04/0.00/0.01/0.00.\[20\]\[21\] 2. chameleon-filtered, squirrel-filtered (never unfiltered in headline tables). 3. LINKX: arxiv-year, snap-patents, pokec, genius, twitch-gamers, Penn94 (LINKX repo loaders, some on Google Drive → gdown; TO VERIFY; 5 splits 50/25/25; genius ROC-AUC). Luan et al. classify squirrel-filtered, Penn94, pokec, snap-patents, twitch-gamers as "ambiguous".\[22\] 4. Texas/Cornell/Wisconsin: appendix only.
C3 MIXED: C3a by relation type — YelpChi (45,954 reviews, 14.5% fraud; R-U-R 49,315 edges, similarity 0.90; R-T-R 573,616 / R-S-R 3,402,743, ≈0.05) and Amazon fraud (11,944 users; U-P-U 175,608 ≈0.19; U-S-U 3,566,479 ≈0.04; U-V-U 1,036,737 ≈0.03),\[11\] from the CARE-GNN repo (pin commit, TO VERIFY) or DGL `FraudDataset`; Amazon fraud ratio is reported as 9.5% and 6.87% in different papers (3,305 unlabelled nodes)\[11\]\[23\]\[24\] — compute and document; ROC-AUC, AP, macro-F1; natural regime H-RELATION. C3b by sign: C5. C3c by local structure (Mao et al., NeurIPS 2023):\[25\] top-5 C1/C2 datasets by std of local homophily, determined empirically. C3d structured compatibility: datasets whose compatibility matrix is far (Frobenius) from both I and (J−I)/(k−1), determined empirically.
C4 NATIVE PROBABILISTIC + LABELS: 1. ogbn-proteins (132,534 nodes, 39,561,252 edges, 8 STRING v11 scores, no node features, 112 binary GO tasks, species split, mean ROC-AUC; EDSP per task with k=2; `http://snap.stanford.edu/ogb/data/nodeproppred/proteins.zip` via `ogb`, CC0 — TO VERIFY). 2. Uncertain DBLP (project page `https://helios2.mi.parisdescartes.fr/~themisp/collectiveclassification/`, code + `xml2dat.py`, rebuild from the 2012 DBLP XML; paper: 922,673 nodes / 3,389,272 edges / 14 classes / 16% labelled; project page: 677,098 / 4,122,556 / 153,022 labelled — document which you reproduce). 3. US Patent assignee graph (108,658 nodes, 1,059,822 edges, 5 classes, 66% labelled; source TO VERIFY).\[7\] 4. Raw STRING for yeast/human + GO-slim labels (URLs, licenses TO VERIFY). 5. Krogan PPI, BioMine, Flickr uncertain graphs (labels TO VERIFY; skip and log if none).
C5 NATIVE SIGNED + LABELS: 1. US Congress signed co-sponsorship backbones 1973–2016 (Neal, figshare DOI 10.6084/m9.figshare.8096429; 1/−1/0 matrix per chamber per session; party in header; signs are SDSM-inferred, not votes; independents dropped or k=3; license TO VERIFY). 2. Bitcoin OTC/Alpha (SNAP; OTC 5,881 nodes, 35,592 edges, weights −10..+10, 89% positive) + REV2 labels (`https://cs.stanford.edu/~srijan/rev2/`; Alpha 102 fraud / 138 benign, OTC 136 / 180 — TO VERIFY);\[10\]\[26\]\[27\] main setting removes edges incident to label-defining founder/seed nodes (TO VERIFY from REV2 code); ROC-AUC, AP. 3. Wiki-RfA/Elec/Editor per Mercado et al. (`melopeo/GL`; LCC 3,470/2,325/20,198 nodes; labelled files TO VERIFY, else rebuild from SNAP and UMD/VEWS data); mandatory baseline: sign of summed incoming votes. 4. Sampson monastery (PyG Signed Directed `sampson`, 25 nodes, 4 groups; Pajek `Sampson.net` + `Sampson_factions_T4.clu`): leave-one-out case study, excluded from rank statistics. 5. Highland tribes (KONECT `ucidata-gama`, Netzschleuder `new_guinea_tribes`; 16 nodes, 58 edges; no shipped labels — include only with citable labels, Hage & Harary 1983, TO VERIFY). 6. Search for more (log queries): gene regulatory networks with activation/repression and gene labels (RegulonDB, TRRUST), Conover et al. 2011 Twitter retweet/mention, Correlates of War, Slovene parliament, 2023–2026 signed node-classification benchmarks.
C6 FEATURELESS: ogbn-proteins, C5 graphs, plus every C1/C2 graph in Track-A. C7 LARGE-SCALE: ogbn-products, ogbn-proteins, pokec, snap-patents, twitch-gamers (ogbn-papers100M only with a larger runner). C8 BINARY vs HIGH-k and IMBALANCE: report macro-F1 and balanced accuracy too. C9 REAL LABEL NOISE: search and log; else PERTURBED-REAL noise injection. C10 MANY COMPONENTS/ISOLATED NODES: CiteSeer, roman-empire (diameter 6,824), Congress House, Bitcoin; report component coverage. C11 DOMAINS: citation, co-purchase, social, web/wiki, language, biology, fraud/finance, politics, crowdsourcing, Q&A.
### 3.2 Per-dataset statistics (`scripts/dataset_stats.py` → JSON + LaTeX)
n, |E|, k, class distribution, components, isolated nodes, degree stats, edge/node/class-adjusted homophily and label informativeness (Platonov 2023), local-homophily histogram (train-label version and full-label "descriptive" version), empirical compatibility matrix and its distances to I and (J−I)/(k−1), negative-edge fraction, p distribution, Luan category.
### 3.3 Acquisition protocol (enforces R4)
Append-only `data/ACQUISITION_LOG.jsonl`, one record per attempt: {timestamp_utc, cell_id, candidate_rank, dataset, source_url, method, environment (agent-session | actions-run-id), http_status, bytes, sha256_raw, sha256_processed, outcome, failure_reason (firewall|404|license|format|size|checksum|other), next_action}. `data/REGISTRY.yaml` holds the ranked lists and status. Per cell: try candidates in rank order, ≥2 routes each (loader, mirror, original source, release asset); on firewall blocks retry in an Actions workflow before declaring failure; then run ≥5 logged literature searches (`data/SEARCH_LOG.md`). Only if all fail: mark EXHAUSTED, post a PR comment "SYNTHETIC-SUBSTITUTE REQUEST: <cell>" listing all attempts, then generate §3.4 data (the human may veto).
### 3.4 Synthetic generators (fixed in advance; EXHAUSTED cells only; labelled SYNTHETIC-SUBSTITUTE)
G1 contextual SBM (n=10,000, k∈{2,5}, avg degree 10, homophily 0.1–0.9, Gaussian features, separation {0.5,1,2}). G2 signed SBM (sign noise η∈{0,0.1,0.2,0.3}). G3 planted partition with polarity flips φ∈{0.1,0.3,0.5}. G4 uncertain edges: true p~Beta(5,2) or Beta(2,2), spurious edges (1–3×|E|) with p~Beta(1,5). G5 mixed-homophily blocks {0.9,0.5,0.1}. G6 planted compatibility via `northeastern-datalab/factorized-graphs` `graphGenerator.py`.\[28\]

## 4. BASELINES (official code pinned to a commit, or reimplementation with parity check)
Record per baseline: repo, commit, license, PyG/DGL version, patches (`third_party/patches/`), and REPRODUCED vs REPORTED side by side (parity within 1 std or 1 point, else "parity failed").
B0 Trivial: majority class, train prior, feature-only MLP/LR/LightGBM, sign-majority rule for signed graphs.
B1 MANDATORY PROPAGATION: Zhu–Ghahramani harmonic function; Zhou et al. LGC; PPR/APPNP-style label propagation; Correct & Smooth (PyG `CorrectAndSmooth`; identical base predictor to Track-B EDSP — THE direct comparison); LinBP with SSLH compatibility estimation and DCE (SIGMOD 2020, `northeastern-datalab/factorized-graphs`);\[28\]\[29\] ZooBP (code TO VERIFY); signed-Laplacian LP and Mercado et al. GL (Python reimplementation, parity on their Wiki numbers); uBayes/uBayes+/uBayes+RN (Dallachiesa et al., C++ built in Actions);\[7\] possible-worlds sampling LP (S∈{10,50}); HAPPY/H-PPR (IJCAI 2025, code TO VERIFY); fixed aggregation features (arXiv 2601.19449, TO VERIFY); REV2 (Bitcoin only).
B2 TUNED CLASSIC GNNs (Luo et al., NeurIPS 2024, `LUOyk1999/tunedGNN`; top rank on 17/18 datasets): GCN, GraphSAGE, GAT with residual/normalisation/dropout/JK tuned. Reference REPORTED: roman-empire GraphSAGE+ 92.19 / GCN+ 91.35; amazon-ratings GAT+ 54.99; minesweeper GCN+ 97.77; questions GAT+ 79.04.
B3 GRAPH TRANSFORMERS: Polynormer (`cornell-zhang/Polynormer`; REPORTED roman-empire 92.13, amazon-ratings 54.46, minesweeper 96.96, tolokers 84.83, questions 77.95; ReLU variant ogbn-arxiv 73.46, ogbn-products 83.82, pokec 86.10); SGFormer (REPORTED roman-empire 73.91, questions 77.06, ogbn-arxiv 72.63, ogbn-products 74.16, pokec 73.76);\[13\]\[14\]\[30\]\[31\] NodeFormer; 2025–2026 GTs (GraphTARIF, DUALFormer) as CLAIMs.
B4 HETEROPHILY GNNs: H2GCN, GloGNN, ACM-GCN, LINKX, FAGCN, GGCN, GPR-GNN, BernNet, ChebNetII, OrderedGNN, CPGNN, GBK-GNN, Dir-GNN; IM-MPNN, TANGO as CLAIMs.
B5 SIGNED GNNs (embeddings + LR): SGCN, SNEA, SDGNN, SiGAT via PyTorch Geometric Signed Directed (TO VERIFY versions).
B6 FRAUD: CARE-GNN, PC-GNN, BWGNN, GAGA, H²-FDetector (reproduce at least CARE-GNN and BWGNN).
B7 GFMs (appendix, optional): GraphAny, GraphPFN; cite NodeGround (arXiv 2609.39673; 51 datasets, 6 GFMs, 15 supervised methods; favours tuned GNNs).\[32\]
Track-A EDSP is compared with B0/B1 (Track-A); Track-B EDSP with B0–B4 on identical splits. Never present cross-track comparisons as like-for-like.

## 5. EXPERIMENTAL PROTOCOL
5.1 Official splits where they exist; else 10 stratified random splits; ≥10 seeds for stochastic methods (EDSP deterministic: 1 run per split).
5.2 Label-rate sweep: {1,3,5,10,20} labels per class and {1,5,10,50}% of nodes.
5.3 Tuning: Optuna TPE, validation-only, 50 trials (n<100k) or 20 (larger), identical for EDSP-tuned; search spaces committed in `configs/search_spaces/`.
5.4 Metrics: accuracy, macro-F1, balanced accuracy, ROC-AUC, AP, NLL, Brier, ECE (15 equal-mass bins) and adaptive ECE, wall-clock, peak RSS/GPU memory, iterations, convergence rate, uncovered-node fraction.
5.5 Robustness (PERTURBED-REAL): train-label noise {10–50}% uniform and pairwise; p'=clip(p+N(0,σ²)), σ∈{0.05,0.1,0.2}; spurious low-p edges (Dallachiesa protocol ϕ=3, σ=0.25);\[7\] h-flips {5,10,20,30}%.
5.6 Ablations: fixed vs dynamic α; no P_esc; no trust; uniform vs −ln(1−p) weights; complement vs estimated full compatibility (gap vs k and vs compatibility distance); sync vs async; T and p0 sweeps; regimes (a)–(f) vs oracle (g); possible-worlds sampling vs expected-weight propagation.
5.7 Accuracy per local-homophily bin (width 0.1; descriptive full-label computation) and per degree bin, for EDSP vs tuned GCN vs C&S (Mao et al.-style disparity plots).
5.8 Scalability: runtime/memory vs |E| on 10/25/50/100% node-induced subgraphs of ogbn-arxiv/products/proteins/pokec; CPU threads {1,4}; GPU if available.
5.9 Statistics: mean ± std, 95% bootstrap CI (10,000 resamples); Wilcoxon signed-rank with Holm; Friedman + Nemenyi CD diagrams per track and per grid cell; win/tie/loss counts.
5.10 Mechanistic diagnostics: α distributions, regime fractions, entropy trajectories, oscillation on real label-free components, iteration distributions, p0-inertness (Kendall τ of Track-A predictions across p0).
5.11 Pre-registered hypotheses: H1 Track-A EDSP ≈ harmonic LP/LGC on homophilic graphs. H2 Track-A near chance on minesweeper/tolokers/questions. H3 EDSP-fullH > EDSP on structured-compatibility graphs, gap growing with k. H4 Track-B EDSP ≈ C&S < tuned GNNs on feature-rich heterophilous graphs. H5 P-NATIVE > P-CONST on ogbn-proteins/uncertain DBLP. H6 finite T_L helps under label noise. H7 k=2 Jacobi oscillation only in label-free components. Report each as supported / not supported / inconclusive.

## 6. ENGINEERING
6.1 Layout: `edsp/` (core, variants, regimes/, data/ {registry, loaders, vault, stats}, baselines/, eval/ {metrics, calibration, evaluate, stats_tests}, selection/, utils/); `configs/` (Hydra); `scripts/` (run, sweep, dataset_stats, aggregate, make_tables, make_figures, oracle_upper_bound, fetch_data); `third_party/` (pinned submodules + patches); `tests/`; `results/` (runs/, aggregates/, tables/, figures/, synthetic/); `docs/`; `data/` (REGISTRY.yaml, ACQUISITION_LOG.jsonl, SEARCH_LOG.md; no raw data in git).
6.2 Environment: `uv` + `uv.lock` (Python 3.11/3.12); CPU torch from download.pytorch.org; torch_geometric, ogb, optuna, hydra-core, scipy, pandas, pyarrow, scikit-learn, lightgbm, hypothesis, pytest, matplotlib, autorank or scikit-posthocs; Dockerfile (CPU, optional CUDA); conflicting baselines get their own `third_party/<name>/.venv`.
6.3 Determinism: seed everything, deterministic torch where possible; float64 CPU EDSP bit-reproducible.
6.4 Each run writes one JSONL record (config, hashes, SHA, environment hash, metrics, timings, diagnostics); aggregation to Parquet.
6.5 `make_tables.py` → booktabs LaTeX (mean ± std; bold only when significant; REPORTED in grey italics with citation key); `make_figures.py` → vector PDFs (CD diagrams, label-rate curves, stratified bars, robustness, scalability, diagnostics, compatibility heatmaps); English captions plus a `captions_el.yaml` placeholder for the author's Greek captions.
6.6 CI (`ci.yml`): ruff, mypy on core, unit + property tests, smoke runs (Cora, roman-empire, Sampson, 2k-node ogbn-proteins subgraph) in <15 min.
6.7 Workflows: `fetch-data.yml` (workflow_dispatch; downloads all candidates outside the agent firewall, verifies checksums, writes log records, caches by sha256, Release `data-v<N>` for large files). `run-matrix.yml` (workflow_dispatch inputs experiment/track/datasets/methods/shard_count; ≤256 jobs; `timeout-minutes: 330`; `fail-fast: false`; `sweep.py --resume` with checkpoints uploaded every 20 minutes). `aggregate.yml` (workflow_dispatch + `workflow_run`; merges artifacts, builds tables/figures, opens a results-only PR via pinned `peter-evans/create-pull-request`; contents/pull-requests: write). `gpu-matrix.yml` (`runs-on: ${{ vars.GPU_RUNNER_LABEL }}`, skipped if unset).
6.8 `copilot-setup-steps.yml`: single job named `copilot-setup-steps`, timeout ≤59, steps: checkout, install uv, `uv sync --frozen`, restore smoke-dataset cache, `pytest -x -q tests/unit`.
6.9 `AGENTS.md` (portable across Copilot, Claude, Codex) and `.github/copilot-instructions.md` (pointer): commands, rules R1–R8, leakage rules, "never run full matrices in the agent session; dispatch workflows", definition of done.\[33\]\[34\]

## 7. EXECUTION PLAN (PRs ≤ ~45 min of agent work each; sub-issue per PR)
Commit and push work-in-progress before ~45 minutes in any session, with a remaining-work checklist; maintain `docs/PROGRESS.md`.
PR-0 Bootstrap (skeleton, lock, Dockerfile, AGENTS.md, setup steps, CI). DoD: CI green.
PR-1 Core + all §1.8 tests. DoD: T1–T11, E1–E7 green; 1M-edge iteration benchmark logged.
PR-2 Registry, C1/C2/Sampson/ogbn-proteins loaders, LabelVault, dataset_stats, PREREGISTRATION.md, SPEC_AMBIGUITIES.md. DoD: ≥10 datasets reproduce published homophily/LI within 0.01 or discrepancy documented.
PR-3 fetch-data.yml and acquisition run. DoD: every cell has ≥1 OK real dataset or an EXHAUSTED record.
PR-4 Regimes incl. H-LEARNED/H-RELATION. DoD: T11 passes; edge-classifier AUCs logged.
PR-5 B0/B1 with parity. DoD: harmonic/LGC/C&S parity on Planetoid or documented.
PR-6 B2–B6 wrappers. DoD: BASELINES.md complete with parity status.
PR-7 run-matrix/aggregate/gpu workflows. DoD: smoke aggregate PR auto-opened.
PR-8..n suites in PRIORITY ORDER (also the overrun order): E1 Track-A all real datasets → E2 C4/C5 native cells → E3 Track-B vs C&S and tuned GNNs → E4 label-rate → E5 ablations → E6 robustness → E7 diagnostics → E8 large-scale/scalability → E9 remaining SOTA reproductions → E10 oracle upper bound → E11 synthetic substitutes (EXHAUSTED cells only).
PR-final REPORT.md, CLAIMS_LEDGER.md, LaTeX/figure bundle, `make reproduce`.
If you cannot dispatch workflows, put exact "Actions → <workflow> → Run workflow" inputs in the PR description for the human.

## 8. FINAL REPORT (`docs/REPORT.md` + `paper_assets/`)
Summary linked to claim IDs; method as implemented (ambiguities, variants); dataset grid (REAL vs SYNTHETIC-SUBSTITUTE), statistics, acquisition summary; results per track and cell (REPRODUCED vs REPORTED separated), CD diagrams, label-rate curves; mandatory "Where EDSP loses and why" tied to D1–D7/H1–H7; ablations, robustness, calibration, scalability, diagnostics; oracle upper bound (flagged); threats to validity (Wiki-RfA/Bitcoin leakage, symmetrisation, split conventions, budgets, reported-vs-reproduced, omissions, synthetic substitutes); claims ledger {claim_id, claim, evidence, run_ids, status}; not-run table; exact reproduction commands and environment hashes.

## 9. ESCALATION RULES
- Irreproducible baseline: official code at pinned commit → minimal documented patches → PyG reimplementation with parity; if parity fails report "REPORTED only" plus our flagged run.
- Download failure: log, next route/candidate; synthetic only for EXHAUSTED cells, after the SYNTHETIC-SUBSTITUTE REQUEST comment.
- Firewall block in session: never bypass; move to fetch-data.yml; list needed hosts under "Allowlist request" in the PR.
- Compute overrun: follow §7 priority; cut tuning trials uniformly for ALL methods; shard further; drop optional large datasets first.
- Spec ambiguity: literal canonical + variant + SPEC_AMBIGUITIES.md entry.
- Non-convergence: report the rate; EDSP-damped only as a labelled variant.
- Anything that would change a headline conclusion: open a "DECISION NEEDED" issue tagging the owner and continue other suites.
````

## Caveats

- **Platform details move fast.** The 59-minute session cap and the firewall/setup-steps behaviour were verified against GitHub documentation and statements by GitHub staff. The ~5–7 concurrent agent sessions figure is a user observation, not documented.\[35\] Whether Agent HQ's Claude and Codex agents share the same firewall and setup-steps mechanism was not verified. Whether the agent's token can dispatch workflows is uncertain, hence the manual-dispatch fallback.
- **Larger and GPU runners** are configured through organization runner groups and billed per minute;\[36\] availability for a personal account is unverified, so GPU jobs are optional.\[37\]\[38\]
- **Items marked TO VERIFY** (LINKX hosting, STRING's combined-score prior correction, REV2's founder-seed definition, Highland-tribe labels, shipped Mercado Wiki labels, Congress/OGB/STRING licences, real label-noise datasets) were not confirmed here; the agent must verify and log evidence.
- **SOTA numbers** are REPORTED values from different tuning protocols and split subsets: parity references, not a leaderboard. The 2025–2026 claimants are claims to reproduce.
- **Expected outcome:** the label-only track will likely lose to feature-using GNNs on most standard benchmarks. The defensible contribution is more plausibly calibration and robustness in the feature-prior track, performance on graphs with exogenous signs or probabilities, and the information-theoretic interpretation of the damping. Frame the manuscript accordingly.

## Sources

1. [Actions limits - GitHub Docs](https://docs.github.com/en/actions/reference/limits)
2. [Node Classification for Signed Social Networks Using Diffuse Interface Methods](https://arxiv.org/pdf/1809.06432)
3. [FairJudge: Trustworthy User Prediction in Rating Platforms](https://arxiv.org/pdf/1703.10545)
4. [Improving Classification Quality in Uncertain Graphs](https://dl.acm.org/doi/fullHtml/10.1145/3242095)
5. [(PDF) Node Classification in Uncertain Graphs](https://researchgate.net/publication/262568688_Node_Classification_in_Uncertain_Graphs)
6. [Emulated-Inc/ogbn-proteins · Datasets at Hugging Face](https://huggingface.co/datasets/Emulated-Inc/ogbn-proteins)
7. <https://helios2.mi.parisdescartes.fr/~themisp/publications/jdiq18.pdf>
8. [A Sign of the Times: Dataset of US Congress signed network backbones from co-sponsorship data, 1973-2016](https://figshare.com/articles/dataset/A_Sign_of_the_Times/8096429)
9. [Real-World Data Set Descriptions — PyTorch Geometric Signed Directed documentation](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/notes/datasets.html)
10. [SNAP: Signed network datasets: Bitcoin OTC web of trust network](https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html)
11. [Improving Fraud Detection via Hierarchical Attention-based Graph Neural Network](https://arxiv.org/pdf/2202.06096)
12. [Classic GNNs are Strong Baselines: Reassessing GNNs for Node Classification](https://arxiv.org/pdf/2406.08993)
13. [Published as a conference paper at ICLR 2025 DUALFORMER: DUAL GRAPH TRANSFORMER](https://proceedings.iclr.cc/paper_files/paper/2025/file/128911cc894d57bcae78074a9551c132-Paper-Conference.pdf)
14. [Polynormer: Polynomial-Expressive Graph Transformer in Linear Time](https://arxiv.org/html/2403.01232)
15. [Heterophily-Aware Personalized PageRank for Node Classification Giuseppe Pirr\`o](https://www.ijcai.org/proceedings/2025/0676.pdf)
16. [Improving the Effective Receptive Field of Message-Passing Neural Networks](https://arxiv.org/pdf/2505.23185)
17. [TANGO: Graph Neural Dynamics via Learned Energy and Tangential Flows](https://arxiv.org/pdf/2508.05070)
18. [Fixed Aggregation Features Can Rival GNNs](https://arxiv.org/pdf/2601.19449)
19. [Customizing or disabling the firewall for GitHub Copilot - GitHub Docs](https://docs.github.com/copilot/customizing-copilot/customizing-or-disabling-the-firewall-for-copilot-coding-agent)
20. [GitHub - yandex-research/heterophilous-graphs: A Critical Look at the Evaluation of GNNs under Heterophily: Are We Really Making Progress? · GitHub](https://github.com/yandex-research/heterophilous-graphs)
21. [Source code for dgl.data.heterophilous\_graphs](https://www.dgl.ai/dgl_docs/_modules/dgl/data/heterophilous_graphs.html)
22. [The Heterophilic Graph Learning Handbook:](https://arxiv.org/pdf/2407.09618)
23. [Pivot to YelpChi/Amazon graph fraud datasets, implement Model 1 by caelenm · Pull Request #33 · caelenm/fraud-detect-gnn · GitHub](https://github.com/caelenm/fraud-detect-gnn/pull/33)
24. [Addressing Noise and Stochasticity in Fraud Detection for Service Networks](https://arxiv.org/pdf/2505.00946)
25. [NeurIPS Poster Demystifying Structural Disparity in Graph Neural Networks: Can One Size Fit All?](https://neurips.cc/virtual/2023/poster/70449)
26. [Cross-Network Detection of Fraudulent Behavior in Real-World ...](https://ksiresearch.org/seke/seke25paper/paper080.pdf)
27. [graph convolutional network with expanded balance theory ...](https://d-nb.info/1222027763/34)
28. [GitHub - northeastern-datalab/factorized-graphs: Semi-supervised learning and inference for sparsely labelled graphs](https://github.com/northeastern-datalab/factorized-graphs)
29. [Wolfgang Gatterbauer](https://gatterbauer.name/publications)
30. [Polynormer: Polynomial-Expressive Graph Transformer in Linear Time](https://arxiv.org/pdf/2403.01232)
31. [PolyFormer: Scalable Node-wise Filters via Polynomial Graph Transformer](https://arxiv.org/pdf/2407.14459)
32. [\[2609.39673\] NodeGround: A Node Classification Benchmark in the Graph Foundation Model Era](https://arxiv.org/abs/2609.39673)
33. [GitHub Copilot Instructions vs Prompts vs Custom Agents vs Skills vs X vs WHY? - DEV Community](https://dev.to/pwd9000/github-copilot-instructions-vs-prompts-vs-custom-agents-vs-skills-vs-x-vs-why-339l)
34. [AGENTS.md vs CLAUDE.md vs Copilot instructions: a practical setup guide for AI coding agents](https://indieseek.co/blogs/agents-md-vs-claude-md-copilot-instructions/)
35. [Copilot coding agent: is there a max concurrent task limit? (\~5-7 observed, rest queue) · community · Discussion #199770 · GitHub](https://github.com/orgs/community/discussions/199770)
36. [Configuring runners for GitHub Copilot code review - GitHub Docs](https://docs.github.com/en/copilot/how-tos/copilot-on-github/set-up-copilot/configure-runners)
37. [GitHub Actions runners: where do your workflows run?](https://blog.stephane-robert.info/en/docs/pipeline-cicd/github/runners/)
38. [GitHub Actions: GPU hosted runners are now generally available - GitHub Changelog](https://github.blog/changelog/2024-07-08-github-actions-gpu-hosted-runners-are-now-generally-available/)
