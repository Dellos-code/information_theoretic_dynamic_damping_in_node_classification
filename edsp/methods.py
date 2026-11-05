"""Run EDSP end-to-end on a GraphData split (regimes -> core -> probabilities)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from edsp.core import EDSPResult, edsp
from edsp.data.graph import GraphData
from edsp.data.vault import LabelVault
from edsp.regimes import EDSPInputs, build_inputs


@dataclass
class EDSPRun:
    probabilities: np.ndarray
    result: EDSPResult
    inputs: EDSPInputs
    vault: LabelVault


def run_edsp(
    graph: GraphData,
    split: int | str = 0,
    config: dict | None = None,
    *,
    variant: str = "EDSP-canonical",
    strict: bool = True,
    vault: LabelVault | None = None,
    **core_kwargs: object,
) -> EDSPRun:
    vault = vault if vault is not None else graph.vault(split, strict=strict)
    inputs = build_inputs(graph, vault, config)
    result = edsp(
        inputs.edges, inputs.probabilities, inputs.heterophily, inputs.priors,
        prior_trust=inputs.prior_trust, edge_trust=inputs.edge_trust, variant=variant, **core_kwargs,  # type: ignore[arg-type]
    )
    return EDSPRun(result.probabilities, result, inputs, vault)
