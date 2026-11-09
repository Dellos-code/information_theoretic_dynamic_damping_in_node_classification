"""PR-27 (suite C4 / H5): task views, P-NATIVE modes, weighted baselines, the CSR backend,
batched H-LEARNED scoring, scripts/run_c4.py and the C4 branch of e1.yml."""

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from edsp.baselines import propagation as BP
from edsp.core import edsp
from edsp.data import GraphData, LabelAccessError, Split
from edsp.regimes import edge_features as EF
from edsp.regimes import polarity as H
from edsp.regimes import probability as P
from edsp.sparse_backend import DirectedPlan, edsp_csr
from edsp.utils.memo import memo, shared_structure

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


def multilabel_graph(n: int = 80, tasks: int = 3, seed: int = 0, name: str = "ogbn-proteins") -> GraphData:
    rng = np.random.default_rng(seed)
    comm = rng.integers(0, 4, n)
    rows = set()
    while len(rows) < 4 * n:
        i, j = (int(x) for x in rng.integers(0, n, 2))
        if i != j and (comm[i] == comm[j] or rng.random() < 0.25):
            rows.add((min(i, j), max(i, j)))
    edges = np.array(sorted(rows))
    attr = np.where(rng.random((len(edges), 8)) < 0.4, rng.beta(2, 4, (len(edges), 8)), 0.0).astype(np.float32)
    pos = rng.random((tasks, 4)) < 0.5
    pos[:, 0], pos[:, 1] = True, False  # every task has both classes
    Y = (pos[:, comm].T ^ (rng.random((n, tasks)) < 0.1)).astype(np.int8)
    perm = rng.permutation(n)
    split = Split("species", np.sort(perm[: n // 2]), np.sort(perm[n // 2: 3 * n // 4]), np.sort(perm[3 * n // 4:]), "toy")
    return GraphData(name, n, edges, 2, [split], Y, edge_attr=attr, multilabel=True)


# ------------------------------------------------------------------ task views
def test_task_view_shares_structure_and_exposes_one_task_through_the_vault():
    g = multilabel_graph()
    for t in range(g.num_tasks):
        v = g.task_view(t)
        assert v.name == f"ogbn-proteins#t{t:03d}" and v.num_classes == 2 and not v.multilabel
        assert v.edges is g.edges and v.edge_attr is g.edge_attr and v.splits is g.splits and v.num_nodes == g.num_nodes
        assert v.meta["task"] == t and v.meta["parent"] == "ogbn-proteins"
        _, y = v.vault("species").train_labels()
        assert y.dtype == np.int64 and set(np.unique(y)) <= {0, 1}
        assert np.array_equal(y, g.vault("species").train_labels()[1][:, t])
        with pytest.raises(LabelAccessError):  # strict vault: test labels only inside edsp.eval.evaluate
            v.vault("species").test_labels()
        with pytest.raises(LabelAccessError):
            v.vault("species").val_labels()
    assert g.num_tasks == 3 and g.task_view(0).num_tasks == 1
    with pytest.raises(IndexError):
        g.task_view(3)
    with pytest.raises(ValueError):
        g.task_view(0).task_view(0)


def test_task_view_runs_through_edsp_and_multilabel_graph_still_refuses():
    from edsp.methods import run_edsp

    g = multilabel_graph()
    with pytest.raises(NotImplementedError):
        run_edsp(g, "species")
    run = run_edsp(g.task_view(1), "species", {"track": "A", "p": {"name": "native", "mode": "noisy-or"}})
    assert run.probabilities.shape == (g.num_nodes, 2)
    assert not run.vault.test_accesses_outside_evaluate()


# ------------------------------------------------------------------ P-NATIVE
def test_p_native_modes_are_columnwise_exact():
    g = multilabel_graph()
    S = np.clip(g.edge_attr.astype(np.float64), 0, 1)
    assert np.array_equal(P.p_native(g, "noisy-or"), 1 - np.prod(1 - S, axis=1))
    assert np.array_equal(P.p_native(g, "max"), S.max(axis=1))
    assert np.allclose(P.p_native(g, "mean"), S.mean(axis=1), atol=1e-15)
    assert np.array_equal(P.p_native(g, "channel:4"), S[:, 4])
    with pytest.raises(ValueError):
        P.p_native(g, "median")


# ------------------------------------------------------------------ weighted baselines
def _legacy_linbp(graph, vault, scale=0.5):
    """linbp as it was before PR-27 (no weights argument, no memo)."""
    from scipy.sparse.linalg import eigsh

    k = graph.num_classes
    Hm = BP.estimate_compatibility(graph, vault)
    Hc = Hm - 1.0 / k
    e = graph.edges
    import scipy.sparse as sp
    W = sp.csr_matrix((np.ones(2 * len(e)), (np.concatenate([e[:, 0], e[:, 1]]), np.concatenate([e[:, 1], e[:, 0]]))),
                      shape=(graph.num_nodes,) * 2)
    rho_w = float(abs(eigsh(W.astype(np.float64), k=1, which="LM", return_eigenvectors=False, v0=np.ones(W.shape[0]))[0]))
    rho_h = float(np.max(np.abs(np.linalg.eigvals(Hc)))) or 1.0
    eps = scale / (rho_w * rho_h)
    Y, idx, _ = BP._one_hot_train(graph, vault)
    X = np.zeros_like(Y)
    X[idx] = Y[idx] - 1.0 / k
    F, _ = BP._iterate(lambda F: X + eps * (W @ F) @ Hc, X.copy())
    return BP._to_probs(F + 1.0 / k)


@pytest.mark.parametrize("name", list(BP.WEIGHTED))
def test_weights_none_reproduces_the_unweighted_baselines_exactly(name):
    g = multilabel_graph().task_view(0)
    v = g.vault("species", strict=False)
    a, ia = BP.select_and_predict(name, g, v, "roc_auc")
    b, ib = BP.select_and_predict(name, g, v, "roc_auc", weights=None)
    assert np.array_equal(a, b) and ia.get("chosen") == ib.get("chosen")
    fn = BP.BASELINES[name][2]
    assert np.array_equal(fn(g, v)[0], fn(g, v, weights=None)[0])
    if name == "linbp":
        for s in BP.GRIDS["linbp"]["scale"]:
            assert np.array_equal(BP.linbp(g, v, scale=s)[0], _legacy_linbp(g, v, scale=s))
    ones = BP.select_and_predict(name, g, v, "roc_auc", weights=np.ones(len(g.edges)))[0]
    assert np.allclose(ones, a, atol=1e-8)
    w = P.p_native(g, "noisy-or")
    assert not np.allclose(BP.select_and_predict(name, g, v, "roc_auc", weights=w)[0], a)


def test_weights_are_refused_by_graph_free_baselines():
    g = multilabel_graph().task_view(0)
    with pytest.raises(ValueError):
        BP.select_and_predict("majority", g, g.vault("species"), "roc_auc", weights=np.ones(len(g.edges)))


def test_shared_structure_builds_operators_once_and_changes_no_output(monkeypatch):
    g = multilabel_graph()
    outside = {(b, t): BP.select_and_predict(b, g.task_view(t), g.task_view(t).vault("species"), "roc_auc")[0]
               for b in BP.WEIGHTED for t in (0, 1)}
    calls = []
    real = BP.sp.csr_matrix

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    with shared_structure():
        monkeypatch.setattr(BP.sp, "csr_matrix", counting)
        for b in BP.WEIGHTED:
            for t in (0, 1):
                gt = g.task_view(t)
                assert np.array_equal(BP.select_and_predict(b, gt, gt.vault("species"), "roc_auc")[0], outside[(b, t)])
        monkeypatch.setattr(BP.sp, "csr_matrix", real)
    assert len(calls) == 1  # one adjacency for all tasks and grid points
    assert memo("x", (object(),), lambda: 1) == 1  # outside the scope: plain call


# ------------------------------------------------------------------ CSR backend
@pytest.mark.parametrize("seed", range(12))
def test_csr_backend_matches_the_reference_iteration(seed):
    rng = np.random.default_rng(seed)
    n, k = int(rng.integers(2, 50)), int(rng.integers(2, 5))
    m = int(rng.integers(0, 3 * n))
    u, w = rng.integers(0, n, m), rng.integers(0, n, m)
    keep = u != w
    e = np.unique(np.stack([np.minimum(u, w), np.maximum(u, w)], 1)[keep], axis=0).reshape(-1, 2)
    p = rng.uniform(0, 1, len(e))
    p[rng.random(len(e)) < 0.1] = 0.0
    h = (rng.random(len(e)) < 0.3).astype(np.int8)
    pri = rng.dirichlet(np.ones(k), n)
    pri[rng.random(n) < 0.4] = 1.0 / k
    kw = {"prior_trust": rng.choice([0.0, 0.5, 2.0, math.inf], n), "edge_trust": float(rng.choice([math.inf, 1.0])),
          "heterophilic_operator": rng.dirichlet(np.ones(k), k) if seed % 3 == 0 else None, "max_iter": 150}
    if seed % 4 == 0:
        kw |= {"variant": "EDSP-damped", "beta": 0.5}
    a = edsp(e, p, h, pri, **kw)
    b = edsp_csr(e, p, h, pri, **kw)
    c = edsp_csr(e, p, h, pri, plan=DirectedPlan(n, e), **kw)
    for r in (b, c):
        assert np.allclose(r.probabilities, a.probabilities, atol=1e-12, rtol=0)
        assert (r.iterations, r.converged, r.oscillation) == (a.iterations, a.converged, a.oscillation)
        assert np.allclose(r.weights, a.weights, atol=1e-15) and np.array_equal(r.p_esc, a.p_esc)
        assert np.allclose(r.last_iterates[0], a.last_iterates[0], atol=1e-12)
        assert np.isclose(r.phi_residual_l1, a.phi_residual_l1, atol=1e-12)
        assert len(r.residual_trace) == len(a.residual_trace)


def test_csr_backend_refuses_gauss_seidel_and_replans_on_other_edges():
    e = np.array([[0, 1], [1, 2]])
    pri = np.array([[1.0, 0.0], [0.5, 0.5], [0.5, 0.5]])
    with pytest.raises(ValueError):
        edsp_csr(e, [0.5, 0.5], [0, 0], pri, variant="EDSP-GS")
    plan = DirectedPlan(3, np.array([[0, 2]]))
    r = edsp_csr(e, [0.5, 0.5], [0, 0], pri, plan=plan)  # plan does not match: rebuilt
    assert np.allclose(r.probabilities, edsp(e, [0.5, 0.5], [0, 0], pri).probabilities, atol=1e-12)


# ------------------------------------------------------------------ batched H-LEARNED
def test_h_learned_batched_scoring_equals_the_materialised_features(monkeypatch):
    g = multilabel_graph(n=120, seed=3).task_view(0)
    monkeypatch.setattr(EF, "LARGE_EDGE_COUNT", 10)  # no neighbourhood features in either path
    monkeypatch.setattr(H, "LARGE_EDGE_COUNT", 10**9)
    ref = H.h_learned(g, g.vault("species"), "A")
    monkeypatch.setattr(H, "LARGE_EDGE_COUNT", 10)
    monkeypatch.setattr(EF.batched_edge_scores, "__defaults__", (7,))  # many small batches
    bat = H.h_learned(g, g.vault("species"), "A")
    assert bat.diagnostics["features"].startswith("batched")
    assert np.array_equal(bat.h, ref.h)
    assert np.allclose(bat.score, ref.score, atol=1e-12) and np.allclose(bat.confidence, ref.confidence, atol=1e-12)
    assert bat.diagnostics["chosen_l2"] == ref.diagnostics["chosen_l2"]


# ------------------------------------------------------------------ run_c4 end to end
def _run_c4(tmp_path, monkeypatch, g, extra=()):
    import run_c4

    monkeypatch.setattr(run_c4, "load", lambda name: g)
    monkeypatch.setattr(sys, "argv", ["run_c4.py", "--out", str(tmp_path), "--tasks", "0:2", *extra])
    assert run_c4.main() == 0
    return [json.loads(x) for f in sorted(tmp_path.glob("*.jsonl")) for x in f.read_text().splitlines()]


def test_run_c4_records_methods_configs_and_equal_budgets(tmp_path, monkeypatch):
    import run_c4
    from run_e1 import default_config, tuned_grid

    recs = _run_c4(tmp_path, monkeypatch, multilabel_graph())
    stats = [r for r in recs if r["method"] == "DATASET-STATS"]
    assert len(stats) == 1 and stats[0]["stats"]["num_tasks"] == 3
    assert set(stats[0]["stats"]["splits"]["species"]["edges_by_part"]) >= {"train-test", "val-test"}
    runs = [r for r in recs if r["method"] != "DATASET-STATS"]
    expected = {"EDSP-const-default", "EDSP-native-default", "EDSP-const-tuned", "EDSP-native-tuned",
                *run_c4.TRACK_A_BASELINES, *(f"{b}+w" for b in BP.WEIGHTED)}
    assert {r["method"] for r in runs} == expected and len(runs) == 2 * len(expected)
    assert {r["task"] for r in runs} == {0, 1} and all(r["suite"] == "C4" and r["dataset"] == "ogbn-proteins" for r in runs)
    assert all(r["headline_metric"] == "roc_auc" and "roc_auc" in r["metrics"] for r in runs)
    by = {(r["task"], r["method"]): r for r in runs}
    for t in (0, 1):
        assert by[(t, "EDSP-const-default")]["config"] == json.loads(json.dumps(default_config(False)))
        nd = by[(t, "EDSP-native-default")]["config"]
        assert nd["p"] == {"name": "native", "mode": "noisy-or"} and nd["h"] == {"name": "trainonly", "fill": "b1"}
        cs, ns = by[(t, "EDSP-const-tuned")]["selection"], by[(t, "EDSP-native-tuned")]["selection"]
        assert len(cs["grid"]) == len(ns["grid"]) == 9  # rule R6
        assert cs["grid"] == json.loads(json.dumps(tuned_grid(False)))
        assert [c["h"] for c in cs["grid"]] == [c["h"] for c in ns["grid"]]
        assert {c["p"]["mode"] for c in ns["grid"]} == set(run_c4.NATIVE_MODES)
        # the defaults are grid members, reused rather than re-run
        for fam in ("const", "native"):
            d, tu = by[(t, f"EDSP-{fam}-default")], by[(t, f"EDSP-{fam}-tuned")]
            assert tu["selection"]["grid"][d["reused_from_grid"]["index"]] == d["config"]
        assert by[(t, "lgc+w")]["config"]["weights"] == "P-NATIVE noisy-or" and by[(t, "lgc")]["config"]["weights"] is None
        assert by[(t, "EDSP-native-tuned")]["diagnostics"]["backend"].startswith("scipy-csr")
    assert all(r["run_record"]["dataset_hash"].endswith(f":task{r['task']}") for r in runs)


def test_run_c4_selection_is_test_label_invariant(tmp_path, monkeypatch):
    g = multilabel_graph()
    rng = np.random.default_rng(7)
    Y = g._labels.copy()  # test fixture: build a twin graph whose test labels differ
    te = g.splits[0].test_idx
    Y[te] = rng.integers(0, 2, Y[te].shape)
    g2 = GraphData(g.name, g.num_nodes, g.edges, 2, g.splits, Y, edge_attr=g.edge_attr, multilabel=True)
    a = _run_c4(tmp_path / "a", monkeypatch, g, ("--methods", "EDSP-const-tuned,EDSP-native-tuned,lgc+w,linbp"))
    b = _run_c4(tmp_path / "b", monkeypatch, g2, ("--methods", "EDSP-const-tuned,EDSP-native-tuned,lgc+w,linbp"))
    key = lambda r: (r["task"], r["method"])
    for x, y in zip(sorted(a, key=key), sorted(b, key=key), strict=True):
        assert x["config"] == y["config"]
        assert x.get("selection", {}).get("val_scores") == y.get("selection", {}).get("val_scores")


def test_aggregate_c4_reads_the_records(tmp_path, monkeypatch):
    import aggregate_c4

    _run_c4(tmp_path / "runs", monkeypatch, multilabel_graph())
    monkeypatch.setattr(sys, "argv", ["aggregate_c4.py", "--runs", str(tmp_path / "runs"), "--out", str(tmp_path / "agg")])
    assert aggregate_c4.main() == 0
    agg = json.loads((tmp_path / "agg" / "c4_aggregates.json").read_text())
    entry = agg["datasets"]["ogbn-proteins|species"]
    assert entry["methods"]["EDSP-native-tuned"]["tasks"] == 2
    assert entry["pairs"]["EDSP-native-default vs EDSP-const-default"]["tasks"] == 2


# ------------------------------------------------------------------ workflow
def _plan(inputs: dict, out: Path) -> dict:
    text = (ROOT / ".github" / "workflows" / "e1.yml").read_text()
    body = re.search(r"      - id: p\n        run: \|\n(.*?)\n\n  run:", text, re.DOTALL).group(1)
    body = "\n".join(line[10:] for line in body.splitlines())

    def sub(mo: re.Match) -> str:
        e = mo.group(1).strip()
        if e == "inputs.suite || 'E1'":
            return inputs.get("suite") or "E1"
        if e == "github.event_name":
            return "workflow_dispatch"
        return inputs.get(re.fullmatch(r"inputs\.(\w+)", e).group(1), "")

    out.unlink(missing_ok=True)
    subprocess.run(["bash", "-e", "-c", re.sub(r"\$\{\{(.*?)\}\}", sub, body)], cwd=ROOT, check=True,
                   env=dict(os.environ, GITHUB_OUTPUT=str(out)))
    return dict(line.split("=", 1) for line in out.read_text().splitlines())


def test_e1_workflow_plans_c4_task_chunks_and_keeps_e1_defaults(tmp_path):
    c4 = _plan({"suite": "C4", "chunks": "ogbn-proteins:112:4"}, tmp_path / "o1")
    jobs = json.loads(c4["datasets"])
    assert c4["suite"] == "C4" and len(jobs) == 28
    assert jobs[0] == {"dataset": "ogbn-proteins", "range": "0:4", "tag": "ogbn-proteins-00"}
    assert jobs[-1]["range"] == "108:112"
    e1 = json.loads(_plan({"suite": "E1", "chunks": "questions:10:2"}, tmp_path / "o2")["datasets"])
    assert "ogbn-proteins" not in {j["dataset"] for j in e1} and {"cora", "questions"} <= {j["dataset"] for j in e1}
    text = (ROOT / ".github" / "workflows" / "e1.yml").read_text()
    assert "scripts/run_c4.py" in text and "--tasks {0}" in text and "scripts/aggregate_c4.py" in text
