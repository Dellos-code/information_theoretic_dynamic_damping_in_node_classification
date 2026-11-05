# PR-1b findings

These are toy-graph run records from the canonical NumPy implementation. No
dataset results are included.

| Case | EDSP-canonical | EDSP-GS (id) | EDSP-GS (seeded random) | EDSP-damped(beta=0.5) |
|---|---|---|---|---|
| (a) one homophilic edge, p=0.5 | converged in 261 iterations | converged in 101 | converged in 131 | converged in 5 |
| (a) one homophilic edge, p=0.9 | 2-cycle, max_iter=1000, amplitude 0.7521601134 | converged in 7 | converged in 8 | converged in 8 |
| (b) heterophilic 4-cycle, p=0.3 | converged in 321 iterations | converged in 117 | converged in 157 | converged in 5 |
| (b) heterophilic 4-cycle, p=0.5 | 2-cycle, max_iter=1000, amplitude 0.6332320112 | converged in 20 | converged in 18 | converged in 6 |
| (b) heterophilic 4-cycle, p=0.7 | 2-cycle, max_iter=1000, amplitude 0.7577581928 | converged in 11 | converged in 8 | converged in 8 |
| (b) heterophilic 4-cycle, p=0.9 | 2-cycle, max_iter=1000, amplitude 0.7959321616 | converged in 8 | converged in 3 | converged in 8 |
| (c) one-hot priors, one homophilic edge, p=0.9 | 2-cycle, max_iter=1000, amplitude 0.9365404153 | converged in 9 | converged in 11 | converged in 7 |

Run command:

```text
PYTHONPATH=. /tmp/edsp-venv/bin/python - <<'PY'
from edsp import edsp, edsp_damped, edsp_gs
# cases (a)-(c) as specified in tests/unit/test_core.py
PY
```

The canonical detector is diagnostic only: `EDSP-canonical` is unchanged.
When `max_iter` is reached, predictions use the last iterate and the last two
iterates are retained in `EDSPResult.last_iterates`.
