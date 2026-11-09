"""scripts/run_e1.py hooks for B5 (--signed-gnn) and B6 (--fraud-gnn): off by default, isolated
with --gnn-only, one record per method in the common schema, NOT-RUN on failure."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from test_fraud_gnn import needs_torch, toy_fraud
from test_regimes import toy_graph
from test_signed_gnn import needs_pygsd

from edsp.baselines import fraud_gnn as fg
from edsp.baselines import signed_gnn as sg

ROOT = Path(__file__).resolve().parents[2]
B56 = {"sgcn", "snea", "sdgnn", "sigat", "care_gnn", "pc_gnn"}


def _module():
    spec = importlib.util.spec_from_file_location("run_e1_under_test", ROOT / "scripts" / "run_e1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(monkeypatch, tmp_path, graph, metric, *argv) -> list[dict]:
    mod = _module()
    monkeypatch.setattr(mod, "load", lambda name: graph)
    monkeypatch.setattr(mod, "dataset_entry", lambda name: {"metric": metric, "cell": "test"})
    monkeypatch.setattr(sys, "argv", ["run_e1.py", "--datasets", graph.name, "--out", str(tmp_path), *argv])
    mod.main()
    path = tmp_path / f"{graph.name}.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def test_flags_default_off_leave_the_suite_unchanged(monkeypatch, tmp_path):
    recs = _run(monkeypatch, tmp_path, toy_graph(signed=True), "accuracy", "--h7-splits", "0")
    methods = {r["method"] for r in recs}
    assert "EDSP-tuned" in methods and "signed_lgc" in methods
    assert not methods & B56 and not any(r.get("group") in ("B5", "B6") for r in recs)


def test_unknown_method_is_rejected(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        _run(monkeypatch, tmp_path, toy_graph(signed=True), "accuracy", "--signed-gnn", "sgcn,gcn")


def test_failing_method_gives_a_not_run_record(monkeypatch, tmp_path):
    mod = _module()

    def boom(*a, **k):
        raise ImportError("no module named torch_geometric_signed_directed")

    monkeypatch.setattr(mod, "select_and_predict_signed_gnn", boom)
    g = toy_graph(signed=True)
    monkeypatch.setattr(mod, "load", lambda name: g)
    monkeypatch.setattr(mod, "dataset_entry", lambda name: {"metric": "accuracy", "cell": "test"})
    monkeypatch.setattr(sys, "argv", ["run_e1.py", "--datasets", g.name, "--out", str(tmp_path), "--gnn-only",
                                      "--signed-gnn", "sgcn,snea"])
    mod.main()
    recs = [json.loads(x) for x in (tmp_path / "toy.jsonl").read_text().splitlines()]
    assert [r["method"] for r in recs] == ["NOT-RUN", "NOT-RUN"]
    assert [r["baseline"] for r in recs] == ["sgcn", "snea"] and "ImportError" in recs[0]["reason"]


@needs_pygsd
def test_signed_gnn_only_job(monkeypatch, tmp_path):
    monkeypatch.setattr(sg, "MAX_EPOCHS", 10)
    recs = _run(monkeypatch, tmp_path, toy_graph(signed=True), "accuracy", "--gnn-only", "--signed-gnn", "sgcn")
    assert [r["method"] for r in recs] == ["sgcn"]
    r = recs[0]
    assert r["group"] == "B5" and r["track"] == "A" and r["suite"] == "E1"
    assert r["config"]["chosen"] in r["config"]["grid"] and len(r["config"]["grid"]) == 9
    assert {"accuracy", "roc_auc", "stratified"} <= set(r["metrics"]) and "run_record" in r
    # a graph without signs is not applicable: nothing runs
    assert _run(monkeypatch, tmp_path / "u", toy_graph(), "accuracy", "--gnn-only", "--signed-gnn", "sgcn") == []


@needs_torch
def test_fraud_gnn_only_job(monkeypatch, tmp_path):
    monkeypatch.setattr(fg, "MAX_EPOCHS", 5)
    recs = _run(monkeypatch, tmp_path, toy_fraud(), "roc_auc", "--track", "B", "--gnn-only", "--gnn=",
                "--fraud-gnn", "care,pcgnn")
    assert [r["method"] for r in recs] == ["care_gnn", "pc_gnn"]
    for r in recs:
        assert r["group"] == "B6" and r["track"] == "B" and len(r["config"]["grid"]) == 9
        assert 0.0 <= r["metrics"]["roc_auc"] <= 1.0
    # Track A runs never start the Track-B fraud GNNs
    assert _run(monkeypatch, tmp_path / "a", toy_fraud(), "roc_auc", "--gnn-only", "--fraud-gnn", "care") == []
