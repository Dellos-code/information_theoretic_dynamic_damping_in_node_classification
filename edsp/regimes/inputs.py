"""Assemble EDSP inputs (edges, p, h, priors, trusts) from a named configuration.

Configuration (all keys optional except ``track``)::

    {"track": "A" | "B",
     "p": {"name": "const", "p0": 0.5} | {"name": "native", "mode": "noisy-or"}
          | {"name": "count", "mu": "median"} | {"name": "signmag", "mode": "exp", "mu": "median"}
          | {"name": "learned"},                      # requires h = learned
     "h": {"name": "trainonly", "fill": "b1"} | {"name": "native"} | {"name": "learned"}
          | {"name": "all0"} | {"name": "all1"} | {"name": "relation"},
     "T_L": inf, "edge_trust": inf,                    # Track-A trusts (§2.4)
     "tau": 1.0, "train_onehot": false}                # Track-B (§2.1)

The zero-tuning default of §2.5 is ``DEFAULT_CONFIG`` (P-CONST(0.5),
H-TRAINONLY(b1), T_L = inf, Track-A). H-ORACLE is deliberately not reachable from
here (see edsp.regimes.oracle and scripts/oracle_upper_bound.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from edsp.data.graph import GraphData
from edsp.data.vault import LabelVault

from . import polarity as H
from . import probability as P
from .priors import track_a_priors, track_b_priors

DEFAULT_CONFIG: dict = {"track": "A", "p": {"name": "const", "p0": 0.5}, "h": {"name": "trainonly", "fill": "b1"}, "T_L": math.inf}


@dataclass
class EDSPInputs:
    edges: np.ndarray
    probabilities: np.ndarray
    heterophily: np.ndarray
    priors: np.ndarray
    prior_trust: np.ndarray
    edge_trust: np.ndarray | float
    config: dict
    diagnostics: dict = field(default_factory=dict)
    het_score: np.ndarray | None = None  # learned P(heterophilic) per kept edge (H-LEARNED)


def _polarity(graph: GraphData, vault: LabelVault, spec: dict, track: str) -> H.Polarity:
    name = spec.get("name", "trainonly")
    if name == "trainonly":
        return H.h_trainonly(graph, vault, spec.get("fill", "b1"))
    if name == "native":
        return H.h_native(graph)
    if name == "learned":
        return H.h_learned(graph, vault, track, **{k: v for k, v in spec.items() if k != "name"})
    if name in ("all0", "all1"):
        return H.h_all(graph, int(name[-1]))
    if name == "relation":
        return H.h_relation(graph, vault)
    if name == "oracle":
        raise PermissionError("H-ORACLE is only available through scripts/oracle_upper_bound.py")
    raise ValueError(f"unknown polarity regime {name!r}")


def _probability(graph: GraphData, spec: dict, pol: H.Polarity) -> np.ndarray:
    name = spec.get("name", "const")
    if name == "const":
        return P.p_const(graph, spec.get("p0", 0.5))
    if name == "native":
        return P.p_native(graph, spec.get("mode", "noisy-or"))
    if name == "count":
        return P.p_count(graph, spec.get("mu", "median"))
    if name == "signmag":
        return P.p_signmag(graph, spec.get("mode", "exp"), spec.get("mu", "median"))
    if name == "learned":
        if pol.confidence is None:
            raise ValueError("P-LEARNED requires H-LEARNED")
        return pol.confidence
    raise ValueError(f"unknown probability regime {name!r}")


def build_inputs(graph: GraphData, vault: LabelVault, config: dict | None = None, *, polarity: H.Polarity | None = None) -> EDSPInputs:
    """``polarity`` may be supplied (e.g. H-ORACLE from the oracle script)."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if graph.multilabel:
        raise NotImplementedError("multi-label graphs are run per task (binary sub-graphs); see PR for E2")
    track = cfg["track"]
    if track == "A":
        priors, trust, pinfo = track_a_priors(graph, vault, cfg.get("T_L", math.inf))
    elif track == "B":
        priors, trust, pinfo = track_b_priors(graph, vault, tau=cfg.get("tau", 1.0), train_onehot=cfg.get("train_onehot", False))
    else:
        raise ValueError("track must be 'A' or 'B'")
    pol = polarity if polarity is not None else _polarity(graph, vault, cfg.get("h", {}), track)
    p = _probability(graph, cfg.get("p", {}), pol)
    keep = pol.keep if pol.keep is not None else np.ones(len(graph.edges), dtype=bool)
    edge_trust = cfg.get("edge_trust", math.inf)
    diag = {"priors": pinfo, "polarity": pol.diagnostics, "edges_kept": int(keep.sum()), "edges_total": len(keep)}
    return EDSPInputs(
        graph.edges[keep], np.asarray(p)[keep], np.asarray(pol.h)[keep], priors, trust,
        edge_trust if np.isscalar(edge_trust) else np.asarray(edge_trust)[keep], cfg, diag,
        None if pol.score is None else pol.score[keep],
    )
