"""scripts/run_e1.py --posthoc (PR-39): off by default (records unchanged), Track B only, isolated with
--gnn-only, one record per method on the shared base predictor, per-method NOT-RUN on failure; and
scripts/aggregate_e1.py adds the columns only when such records exist."""

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from test_regimes import toy_graph

ROOT = Path(__file__).resolve().parents[2]
POSTHOC = ["cs_2hop", "clp", "linbp_b"]
FLAG = "cs2hop,clp,linbp_b"


def _module(name: str = "run_e1"):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(monkeypatch, out, graph, *argv, mod=None) -> list[dict]:
    mod = mod or _module()
    monkeypatch.setattr(mod, "load", lambda name: graph)
    monkeypatch.setattr(mod, "dataset_entry", lambda name: {"metric": "accuracy", "cell": "test"})
    monkeypatch.setattr(sys, "argv", ["run_e1.py", "--datasets", graph.name, "--out", str(out), *argv])
    mod.main()
    path = out / f"{graph.name}.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def _strip(recs: list[dict]) -> list[dict]:
    """Records without the fields that vary between runs (timings, memory)."""
    out = []
    for r in recs:
        r = json.loads(json.dumps(r))
        for k in ("wall_clock_seconds", "peak_rss_mib"):
            r.get("run_record", {}).pop(k, None)
        r.get("diagnostics", {}).pop("base_wall_clock_seconds", None)
        out.append(r)
    return out


TRACK_B = ("--track", "B", "--gnn=", "--h7-splits", "0")


def test_posthoc_off_by_default_and_other_records_unchanged(monkeypatch, tmp_path):
    g = toy_graph()
    default = _run(monkeypatch, tmp_path / "a", g, *TRACK_B)
    assert not {r["method"] for r in default} & set(POSTHOC)
    assert "correct_and_smooth" in {r["method"] for r in default}
    withp = _run(monkeypatch, tmp_path / "b", g, *TRACK_B, "--posthoc", FLAG)
    assert _strip(withp[: len(default)]) == _strip(default)
    extra = withp[len(default):]
    assert [r["method"] for r in extra] == POSTHOC
    flr = next(r for r in default if r["method"] == "feature_lr")
    for r in extra:
        assert r["group"] == "B1" and r["track"] == "B" and r["split"] == flr["split"]
        c = r["config"]
        assert c["baseline"] == r["method"] and len(c["grid"]) == 9 and c["chosen"] in c["grid"]
        assert c["base_predictor"]["chosen"] == flr["config"]["chosen"]  # same base predictor as feature_lr / C&S
        assert {"accuracy", "roc_auc", "nll", "stratified"} <= set(r["metrics"]) and "run_record" in r
        assert len(r["diagnostics"]["val_scores"]) == 9 and len(r["diagnostics"]["base_sha256"]) == 64


def test_gnn_only_job_runs_just_the_posthoc_methods(monkeypatch, tmp_path):
    g = toy_graph()
    full = _run(monkeypatch, tmp_path / "a", g, *TRACK_B, "--posthoc", FLAG)
    only = _run(monkeypatch, tmp_path / "b", g, "--track", "B", "--gnn=", "--gnn-only", "--posthoc", FLAG)
    assert [r["method"] for r in only] == POSTHOC
    assert _strip(only) == _strip([r for r in full if r["method"] in POSTHOC])


def test_track_a_never_runs_posthoc_and_unknown_names_are_rejected(monkeypatch, tmp_path):
    assert _run(monkeypatch, tmp_path / "a", toy_graph(), "--gnn-only", "--posthoc", FLAG) == []
    with pytest.raises(SystemExit):
        _run(monkeypatch, tmp_path / "b", toy_graph(), "--track", "B", "--gnn-only", "--posthoc", "cs2hop,cs")


def test_failing_method_gives_its_own_not_run_record(monkeypatch, tmp_path):
    mod = _module()
    real = mod.select_and_predict_posthoc

    def flaky(name, *a, **k):
        if name == "clp":
            raise MemoryError("simulated")
        return real(name, *a, **k)

    monkeypatch.setattr(mod, "select_and_predict_posthoc", flaky)
    recs = _run(monkeypatch, tmp_path, toy_graph(), "--track", "B", "--gnn=", "--gnn-only", "--posthoc", FLAG, mod=mod)
    assert [r["method"] for r in recs] == ["cs_2hop", "NOT-RUN", "linbp_b"]
    assert recs[1]["baseline"] == "clp" and "MemoryError" in recs[1]["reason"]


def test_base_predictor_failure_gives_one_not_run_per_method(monkeypatch, tmp_path):
    g = dataclasses.replace(toy_graph(), features=None)
    recs = _run(monkeypatch, tmp_path, g, "--track", "B", "--gnn=", "--gnn-only", "--posthoc", FLAG)
    assert [(r["method"], r["baseline"]) for r in recs] == [("NOT-RUN", m) for m in POSTHOC]
    assert all("feature_lr" in r["reason"] for r in recs)


def test_aggregate_adds_posthoc_columns_only_when_present(monkeypatch, tmp_path):
    g = toy_graph()
    for sub, extra in (("plain", ()), ("posthoc", ("--posthoc", FLAG))):
        _run(monkeypatch, tmp_path / sub / "runs", g, "--suite", "E3", *TRACK_B, *extra)
        agg = _module("aggregate_e1")
        monkeypatch.setattr(sys, "argv", ["aggregate_e1.py", "--suite", "E3", "--runs", str(tmp_path / sub / "runs"),
                                          "--out", str(tmp_path / sub / "agg")])
        agg.main()
        md = (tmp_path / sub / "agg" / "E3_SUMMARY.md").read_text()
        header = next(line for line in md.splitlines() if line.startswith("| dataset [metric]"))
        columns = [c.strip() for c in header.strip("|").split("|")]
        rows = {line.split(" | ")[0][2:] for line in md.splitlines() if line.startswith("| ")}
        for m in POSTHOC:
            assert (m in columns) == (sub == "posthoc")
            assert (m in rows) == (sub == "posthoc")  # win / tie / loss row
        assert all(m not in agg.RANKED for m in POSTHOC)
