"""
Non-IID client partitioning for Direction 2 (statistical heterogeneity),
experiment_log.md Session 8+.

Standard Dirichlet-based non-IID partitioning (Hsu, Qi & Brown 2019,
"Measuring the Effects of Non-Identical Data Distribution for Federated
Visual Classification" — the canonical construction for non-IID FL
benchmarks; exact citation to be verified before chapter use). For each
class/regime, sample proportions are drawn once from Dir(alpha) and used
to split that regime's pool across clients. alpha -> large gives a
near-uniform (IID-like) split; alpha -> small concentrates each regime
into one or a few clients (highly non-IID).

Regimes here are TTF-based degradation stages (tertiles by default), not
attacker-controlled — this is legitimate cross-client distribution shift,
not contamination. Uses a SINGLE crop size (prepare_data 'C', 20x20)
applied to the FULL 284-sample pool, isolating regime heterogeneity from
the sensor/crop-size heterogeneity already baked into the real A/B/C
user split (see Direction 1 findings: sum-clean vs other rules'
attack-free deviation was partly attributable to that confound).
"""

import numpy as np


def ttf_regime_labels(y, n_regimes=3):
    """Assign each sample a regime index 0..n_regimes-1 by TTF quantile
    (0 = shortest remaining life / most degraded, n_regimes-1 = longest).
    """
    edges = np.percentile(y, np.linspace(0, 100, n_regimes + 1))
    edges[0] -= 1e-9  # include the minimum in bin 0
    labels = np.digitize(y, edges[1:-1], right=True)
    return np.clip(labels, 0, n_regimes - 1)


def _dirichlet_partition_once(n_samples, regime_labels, n_clients, alpha, rng):
    n_regimes = int(regime_labels.max()) + 1
    client_indices = [[] for _ in range(n_clients)]
    for r in range(n_regimes):
        idx_r = np.where(regime_labels == r)[0].copy()
        rng.shuffle(idx_r)
        if len(idx_r) == 0:
            continue
        proportions = rng.dirichlet(alpha * np.ones(n_clients))
        counts = (proportions * len(idx_r)).astype(int)
        remainder = len(idx_r) - counts.sum()
        for i in range(remainder):
            counts[i % n_clients] += 1
        start = 0
        for c in range(n_clients):
            client_indices[c].extend(idx_r[start:start + counts[c]].tolist())
            start += counts[c]
    return [np.array(sorted(idx)) for idx in client_indices]


def dirichlet_partition(n_samples, regime_labels, n_clients, alpha, rng,
                        min_client_size=5, max_attempts=50):
    """Partition sample indices [0, n_samples) into n_clients groups via
    a Dirichlet(alpha) draw per regime. Returns a list of n_clients
    integer index arrays (disjoint, covering all n_samples).

    Low alpha can produce empty or near-empty clients (Vinitial's
    eigendecomposition needs at least a handful of samples per client to
    be non-degenerate) — retries the draw up to max_attempts times,
    requiring every client to have >= min_client_size samples. Raises if
    no attempt succeeds (can happen at very low alpha with many clients
    relative to sample count — lower n_clients or raise alpha instead of
    silently proceeding with a degenerate client).
    """
    for _ in range(max_attempts):
        parts = _dirichlet_partition_once(
            n_samples, regime_labels, n_clients, alpha, rng)
        if min(len(p) for p in parts) >= min_client_size:
            return parts
    raise RuntimeError(
        f"Could not produce a partition with every client >= "
        f"{min_client_size} samples after {max_attempts} attempts at "
        f"alpha={alpha}, n_clients={n_clients}. Try a higher alpha or "
        f"fewer clients."
    )


def regime_composition(client_indices, regime_labels, n_regimes):
    """Per-client fraction of samples in each regime — for reporting how
    non-IID a given alpha actually produced (a sanity/calibration check,
    since small n_samples-per-regime can make low alpha noisier than
    intended).
    """
    comp = np.zeros((len(client_indices), n_regimes))
    for c, idx in enumerate(client_indices):
        if len(idx) == 0:
            continue
        labels = regime_labels[idx]
        for r in range(n_regimes):
            comp[c, r] = np.mean(labels == r)
    return comp
