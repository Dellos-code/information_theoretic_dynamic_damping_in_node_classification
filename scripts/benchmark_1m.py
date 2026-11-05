"""PR-1 1M-edge benchmark; no datasets are downloaded."""

from __future__ import annotations

import time
import resource

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
    one_hot = rng.choice(n, size=n // 10, replace=False)
    priors[one_hot] = 0.0
    priors[one_hot, rng.integers(0, k, size=len(one_hot))] = 1.0
    started = time.perf_counter()
    result = edsp(edges, p, h, priors, max_iter=50)
    elapsed = time.perf_counter() - started
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(
        f"n={n} edges={len(edges)} iterations={result.iterations} "
        f"seconds={elapsed:.6f} peak_rss_mb={peak_rss_mb:.1f}"
    )


if __name__ == "__main__":
    main()
