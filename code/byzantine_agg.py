"""
Byzantine-robust aggregation rules for federated tensor factorization.

Standard federated aggregation (sum/mean of per-client contributions) is
fragile to a Byzantine client: a single corrupted contribution can be given
unbounded influence over the shared model. This module implements classic
robust-aggregation rules from the Byzantine-robust FL literature —
coordinate-wise median, trimmed mean, Krum — adapted to aggregate per-client
MATRIX contributions (scatter matrices, see mpca_byzantine.py) rather than
the per-client gradient/parameter VECTORS they were originally proposed for.

References (methods are standard and well-known; exact citations should be
pulled and verified before use in the chapter — not yet done):
  - Coordinate-wise median / trimmed mean: Yin, Chen, Kannan, Bartlett,
    "Byzantine-Robust Distributed Learning: Towards Optimal Statistical
    Rates," ICML 2018.
  - Krum: Blanchard, El Mhamdi, Guerraoui, Stainer, "Machine Learning with
    Adversaries: Byzantine Tolerant Gradient Descent," NeurIPS 2017.
  - Small-magnitude/"stealthy" attacks defeating robust aggregation: Baruch,
    Baruch, Goldberg, "A Little Is Enough: Circumventing Defenses For
    Distributed Learning," NeurIPS 2019 — the literature analogue of our
    monitoring-pilot finding that small-fraction, spatially-correlated
    artifacts evade residual-based detection (experiment_log.md, Session 7,
    pilot v1-v6).

All aggregation functions take a list of n same-shaped matrices (one per
client) and return one aggregated matrix of the same shape.
"""

import numpy as np


def aggregate_sum(matrices):
    """Standard (non-robust) federated aggregation: plain sum.

    Equivalent (up to the numerical algorithm) to pooling all clients'
    data centrally — the existing MPCA_FD/MPCA_FD_Weighted behavior. Zero
    Byzantine tolerance: a single corrupted client can dominate the
    aggregate.
    """
    return np.sum(matrices, axis=0)


def aggregate_coordinate_median(matrices):
    """Coordinate-wise median across clients.

    Robust to up to floor((n-1)/2) Byzantine clients — each matrix entry
    is a median over n scalars, and a scalar median tolerates up to half
    of them being arbitrary. Well-defined and non-degenerate even at
    small n (n=3 gives a real, non-trivial median at every coordinate,
    unlike two-sided trimming — see aggregate_trimmed_mean).
    """
    stacked = np.stack(matrices, axis=0)
    return np.median(stacked, axis=0)


def aggregate_trimmed_mean(matrices, n_trim=1):
    """Coordinate-wise trimmed mean: drop the n_trim highest and n_trim
    lowest values at each coordinate, average what remains.

    Requires n > 2 * n_trim clients (else the trimmed set is empty). At
    n=3, n_trim=1 leaves exactly 1 value per coordinate — mathematically
    identical to aggregate_coordinate_median in that regime (verified in
    the pilot smoke test). This function exists for when the client count
    grows beyond 3 (e.g. Direction 2's heterogeneity experiments may add
    synthetic clients), where trimmed mean and median diverge and trimmed
    mean is the more standard Byzantine-robust FL baseline to report.
    """
    n = len(matrices)
    if n <= 2 * n_trim:
        raise ValueError(
            f"n_trim={n_trim} requires more than {2 * n_trim} clients, got "
            f"n={n}. At n=3 with n_trim=1, use aggregate_coordinate_median "
            "instead (equivalent, but needs no trim-fraction assumption)."
        )
    stacked = np.stack(matrices, axis=0)
    sorted_vals = np.sort(stacked, axis=0)
    trimmed = sorted_vals[n_trim: n - n_trim]
    return np.mean(trimmed, axis=0)


def _krum_scores(matrices, f):
    """Shared scoring step for Krum/Multi-Krum: for each client, the
    summed squared distance to its (n - f - 2) nearest OTHER clients — a
    low score means "centrally consistent with the rest."
    """
    n = len(matrices)
    if n < 2 * f + 3:
        raise ValueError(
            f"Krum requires n >= 2f+3 clients for f={f} Byzantine (need "
            f"n>={2 * f + 3}), got n={n}. Use aggregate_coordinate_median "
            "or aggregate_geometric_median instead, or increase client "
            "count."
        )
    flat = [m.flatten() for m in matrices]
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dists[i, j] = np.sum((flat[i] - flat[j]) ** 2)
    n_closest = n - f - 2
    scores = np.zeros(n)
    for i in range(n):
        nearest = np.sort(dists[i])[1:n_closest + 1]  # exclude self (dist 0)
        scores[i] = np.sum(nearest)
    return scores


def aggregate_krum(matrices, f):
    """Krum: select the single client contribution with the lowest
    _krum_scores value — the most "centrally consistent" contribution —
    and use it alone as the aggregate (discarding all others, including
    honest ones).

    Requires n >= 2*f + 3 for the standard guarantee (Blanchard et al.
    2017); NOT satisfied at n=3, f=1 (needs n>=5) — raises rather than
    silently producing an unguaranteed result.
    """
    scores = _krum_scores(matrices, f)
    best = int(np.argmin(scores))
    return matrices[best]


def aggregate_multi_krum(matrices, f, m=None):
    """Multi-Krum: like Krum, but averages the m clients with the lowest
    Krum scores (default m = n - f) instead of keeping a single one — a
    common variant that discards less honest signal than plain Krum while
    keeping the same distance-based filtering. Same n >= 2f+3 requirement.
    """
    n = len(matrices)
    if m is None:
        m = n - f
    scores = _krum_scores(matrices, f)  # validates n >= 2f+3
    order = np.argsort(scores)[:m]
    return np.mean([matrices[i] for i in order], axis=0)


def aggregate_geometric_median(matrices, max_iter=200, tol=1e-8, eps=1e-8):
    """Geometric (spatial) median via Weiszfeld's algorithm: treats each
    client's matrix as one point in R^(rows*cols) and finds the point
    minimizing the sum of Euclidean distances to all of them.

    Distance-based, NOT coordinate-wise: the result is a weighted AVERAGE
    of the input matrices (weights = inverse distance to the current
    estimate), so it preserves each client's internal entry-to-entry
    correlation structure — unlike aggregate_coordinate_median, which
    picks entries independently per coordinate and can assemble a
    combination that doesn't resemble any real client's matrix (see the
    Direction-1 foundation pilot finding, experiment_log.md: at n=3,
    coordinate median diverged further from sum than the "attack" itself
    did, purely from cross-client heterogeneity). Same asymptotic
    breakdown point as coordinate median (~1/2 of clients), but the
    combination step itself is fundamentally different.
    """
    stacked = np.stack(matrices, axis=0)
    shape = stacked.shape[1:]
    flat = stacked.reshape(len(matrices), -1)
    y = flat.mean(axis=0)
    for _ in range(max_iter):
        dists = np.maximum(np.linalg.norm(flat - y, axis=1), eps)
        weights = 1.0 / dists
        y_new = (weights[:, None] * flat).sum(axis=0) / weights.sum()
        if np.linalg.norm(y_new - y) < tol:
            y = y_new
            break
        y = y_new
    return y.reshape(shape)


AGGREGATORS = {
    'sum': aggregate_sum,
    'median': aggregate_coordinate_median,
    'trimmed_mean': aggregate_trimmed_mean,
    'krum': aggregate_krum,
    'multi_krum': aggregate_multi_krum,
    'geometric_median': aggregate_geometric_median,
}
