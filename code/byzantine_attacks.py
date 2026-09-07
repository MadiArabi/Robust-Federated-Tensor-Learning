"""
Adversarial (worst-case) Byzantine attacks for federated tensor
factorization, to be injected via MPCA_FD_Robust's adversarial_client /
adversarial_fn hook (mpca_byzantine.py).

Motivation (experiment_log.md, Direction 1, pilot v1/v2): physically
-motivated "faulty camera" attacks (contamination.py — hot_block, stripe)
turned out to be largely absorbed by every aggregation rule tested,
including plain sum, regardless of amplitude or contamination fraction.
Real corrupted images carry limited, incidentally-shaped energy that tends
to settle into its own eigenvector slot rather than competing for the top
of the shared subspace. The classical Byzantine-robust FL literature does
NOT constrain the attacker this way: a compromised client can send an
arbitrary contribution at the aggregation step. This module implements
that literature-standard threat model directly in scatter-matrix space.

Attack design ("hijack" attack): the classic failure mode of unweighted
averaging is that a single large, well-aligned outlier can dominate the
sum and steer the estimated top eigenvector wherever the attacker wants.
Concretely: eigendecompose the honest clients' aggregated (summed) scatter
matrix, pick a target direction (default: the honest data's OWN weakest
eigendirection — the direction real data says matters least), and inject
a rank-1 malicious scatter matrix along that direction scaled to
`amplitude_mult` times the honest top eigenvalue. Large amplitude_mult
should hijack sum's estimated top eigenvector toward the target direction;
robust aggregation rules (median, trimmed_mean, krum, multi_krum,
geometric_median) are specifically designed to reject a single such
outlier regardless of its magnitude, at n >= their required minimum.

Assumes an "omniscient" attacker (can observe/estimate the honest
clients' aggregate) — a standard, if strong, assumption used throughout
the Byzantine-robust FL literature to derive worst-case guarantees.
"""

import numpy as np


def _eig_sorted(S):
    S = np.real_if_close(S, tol=1)
    eigval, eigvec = np.linalg.eig(S)
    eigval = np.real_if_close(eigval, tol=1)
    eigvec = np.real_if_close(eigvec, tol=1)
    order = np.argsort(eigval)[::-1]
    return eigval[order], eigvec[:, order]


def hijack_attack(amplitude_mult=5.0, target='weakest'):
    """Returns adversarial_fn(honest_scatters, MODE) -> malicious_scatter.

    target: 'weakest' (default) — inject along the honest aggregate's
    smallest eigendirection, the most adversarially "wrong" choice
    available from real structure. 'random' — inject along a random
    direction orthogonal to nothing in particular (a rougher, attacker
    -knowledge-agnostic sanity check that the effect isn't an artifact of
    specifically picking the weakest honest direction).
    """
    def fn(honest_scatters, MODE):
        S_honest = np.sum(honest_scatters, axis=0)
        eigval, eigvec = _eig_sorted(S_honest)
        lam1 = max(float(eigval[0]), 1e-8)
        d = S_honest.shape[0]
        if target == 'weakest':
            direction = eigvec[:, -1]
        elif target == 'random':
            rng = np.random.RandomState(abs(hash((MODE, 'hijack'))) % (2**31))
            v = rng.randn(d)
            direction = v / np.linalg.norm(v)
        else:
            raise ValueError(f"Unknown target: {target}")
        c = amplitude_mult * lam1
        return c * np.outer(direction, direction)
    return fn
