"""B0 trivial and B1 propagation baselines (Master Prompt §4).

Every baseline is ``f(graph, vault, **hyper) -> (probabilities (n, k), info)`` and
reads TRAIN labels only; hyper-parameters are chosen on validation labels through
:func:`select_and_predict` (which calls edsp.selection). Test labels are read
only by edsp.eval.evaluate.
"""

from .propagation import BASELINES, select_and_predict

__all__ = ["BASELINES", "select_and_predict"]
