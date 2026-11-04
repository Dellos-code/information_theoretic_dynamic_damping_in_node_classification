"""PR-1 1M-edge benchmark; no datasets are downloaded."""

from __future__ import annotations

import time

import numpy as np

from edsp import edsp


def main() -> None:
    rng = np.random.default_rng(0)
    n, m, k = 100_000, 1_000_000, 3
    edges = rng.integers(0, n, size=(m, 2), dtype=np.int64)
    edges = edges[edges[:, 0] != edges[:, 1]]
    p = rng.uniform(0.01, 0.2, len(edges))
    h = rng.integers(0, 2, len(edges), dtype=np.int8)
    priors = np.full((n, k), 1.0 / k)
    started = time.perf_counter()
    result = edsp(edges, p, h, priors, max_iter=1)
    elapsed = time.perf_counter() - started
    print(f"n={n} edges={len(edges)} iterations={result.iterations} seconds={elapsed:.6f}")


if __name__ == "__main__":
    main()
