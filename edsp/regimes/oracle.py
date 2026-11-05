"""H-ORACLE (Master Prompt §2.3g): polarity from BOTH endpoints' true labels.

UPPER BOUND ONLY. It reads test labels, so it is not a valid method: results go
to the separate table "Oracle polarity (uses test labels; not a valid method)"
and are never ranked. The vault releases labels to this module only when the
environment variable ORACLE=1 is set, which scripts/oracle_upper_bound.py does
and records in its run records.
"""

from __future__ import annotations

import numpy as np

from edsp.data.graph import GraphData
from edsp.data.vault import LabelVault

from .polarity import Polarity


def h_oracle(graph: GraphData, vault: LabelVault) -> Polarity:
    y = vault.oracle_labels()
    u, v = graph.edges[:, 0], graph.edges[:, 1]
    return Polarity((y[u] != y[v]).astype(np.int8), diagnostics={"regime": "H-ORACLE", "uses_test_labels": True})
