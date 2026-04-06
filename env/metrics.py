"""Standard metrics (Jain fairness, etc.)."""

from __future__ import annotations

import numpy as np


def jains_fairness_index(rates: np.ndarray | list[float] | tuple[float, ...]) -> float:
    """
    Jain's fairness index for nonnegative allocations:

        J = (sum_i x_i)^2 / (N * sum_i x_i^2)

    For x_i >= 0, J lies in [1/N, 1]: J = 1 iff equal shares; J = 1/N iff one user
    gets all (others zero). Using (sum x)^2 / (sum x^2) without N is wrong and can
    reach 2 when N = 2 and x_1 = x_2.
    """
    x = np.asarray(rates, dtype=np.float64).ravel()
    x = np.maximum(x, 0.0)
    n = int(x.size)
    if n == 0:
        return 1.0
    s = float(x.sum())
    sq = float(np.dot(x, x))
    if sq <= 0.0:
        return 1.0
    j = (s * s) / (float(n) * sq)
    lo = 1.0 / float(n)
    return float(np.clip(j, lo, 1.0))
