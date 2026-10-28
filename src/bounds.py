"""Exact one-sided finite-population count bounds and paired risk certificates."""

import numpy as np
from scipy.stats import hypergeom, beta


def hg_upper(pop, n, k, alpha):
    """Largest compatible population count K with P_K(X <= k) > alpha.
    Conditional on n uniformly sampled items without replacement, coverage >=1-alpha.
    Arrays broadcast. Zero observations give pop; a census gives k exactly.
    """
    pop, n, k = np.broadcast_arrays(
        np.asarray(pop, dtype=np.int64),
        np.asarray(n, dtype=np.int64),
        np.asarray(k, dtype=np.int64),
    )
    assert np.all((0 <= k) & (k <= n) & (n <= pop)) and 0 < alpha < 1
    lo = k.copy()
    hi = pop - n + k
    while np.any(lo < hi):
        mid = (lo + hi + 1) // 2
        valid = hypergeom.cdf(k, pop, mid, n) > alpha
        lo = np.where(valid, mid, lo)
        hi = np.where(valid, hi, mid - 1)
    return np.where(n == 0, pop, lo)


def cp_upper(n, k, alpha):
    n, k = np.broadcast_arrays(n, k)
    return np.where(k == n, 1.0, beta.ppf(1 - alpha, k + 1, np.maximum(n - k, 1)))


def paired_upper(pop, n, plus, minus, total, alpha, method="hypergeom"):
    """Upper bound on full-pool excess error; pop is the audited stratum size."""
    pop, n, plus, minus = np.broadcast_arrays(pop, n, plus, minus)
    if method == "binomial":
        up = pop * cp_upper(n, plus, alpha)
        low = pop * (1 - cp_upper(n, n - minus, alpha))
        up = np.where(n == pop, plus, up)
        low = np.where(n == pop, minus, low)
    else:
        up = hg_upper(pop, n, plus, alpha)
        low = 0 if method == "harm" else pop - hg_upper(pop, n, n - minus, alpha)
    return (up - low) / total
