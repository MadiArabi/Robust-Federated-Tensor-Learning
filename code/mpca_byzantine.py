"""
Byzantine-robust variant of federated MPCA_FD (my_mpca_02_27_nomean.py).

Local per-client factor matrices (V) are UNCHANGED from MPCA_FD — each
client's local projection is fit entirely from its own data, so there is
nothing to aggregate there (a Byzantine client can only corrupt its own
local V, which never touches other clients).

The GLOBAL factor matrices (U) are shared across all clients — this is
exactly the aggregation point a Byzantine client can poison. MPCA_FD
estimates U via incremental SVD over samples pooled directly across
clients, which is mathematically equivalent (different numerical
algorithm, same target) to eigendecomposing the SUM of all clients'
per-mode scatter matrices. This module makes that aggregation step
explicit: each client computes its own local scatter-matrix contribution
to the shared subspace, and the "server" combines the n contributions
with a configurable aggregation rule (byzantine_agg.py) before
eigendecomposing to get U. Passing agg_fn=aggregate_sum recovers the
behavior of MPCA_FD (verified in pilot_byzantine_agg.py: small principal
angle between the two on clean data); passing a robust rule (e.g.
aggregate_coordinate_median) bounds the influence any single corrupted
client's contribution can have on the shared subspace.

See experiment_log.md, Session 8 ("Byzantine-robust federated
aggregation") for the motivation: this reframes RFTL-S's existing
federated-MAD + Huber-weighting machinery as one instance of a
robust-aggregation rule, and positions it against the classic FL
Byzantine-robustness baselines implemented in byzantine_agg.py.
"""

import numpy as np
from tensorly import unfold
from tensorly.tenalg import multi_mode_dot

from byzantine_agg import aggregate_sum

LAM = 0.00001


class MPCA_FD_Robust:
    """Federated MPCA with a pluggable Byzantine-robust aggregation rule
    for the shared global factor matrices U.

    Structurally identical to MPCA_FD (local V update is byte-for-byte
    the same math); differs only in how the n clients' contributions to
    U are combined.

    Optional adversarial injection (adversarial_client, adversarial_fn):
    a true Byzantine client is not obligated to derive its aggregation
    contribution from real (even corrupted) sensor data at all — it can
    send an arbitrary, hand-crafted contribution directly. The physically
    -motivated "faulty camera" attacks (contamination.py) turned out to
    be absorbed by every aggregation rule tested (experiment_log.md,
    Direction 1 pilot v1/v2), because real corrupted images still carry
    limited, incidentally-shaped energy. This hook lets a client's
    scatter-matrix contribution at each aggregation step be REPLACED by
    adversarial_fn(honest_scatters, MODE) -> malicious_scatter — see
    byzantine_attacks.py for the actual attack construction. That
    client's own local V is still fit from whatever client_data it was
    given (irrelevant to the shared U once its aggregation contribution
    is overridden; kept only for interface symmetry).
    """

    def __init__(self, I, P, iterations=150, agg_fn=aggregate_sum,
                adversarial_client=None, adversarial_fn=None):
        self.I = I
        self.P = P
        self.iterations = iterations
        self.agg_fn = agg_fn
        self.adversarial_client = adversarial_client
        self.adversarial_fn = adversarial_fn

    def _scatter_list(self, per_client_scatter, MODE):
        """Apply the adversarial override, if configured, to one client's
        scatter contribution before aggregation. MODE matters: each
        tensor mode has a differently-sized scatter matrix (I_COMMON
        varies per mode), and the honest top/bottom eigendirections the
        attack targets differ by mode too.
        """
        if self.adversarial_client is None or self.adversarial_fn is None:
            return per_client_scatter
        honest = [s for i, s in enumerate(per_client_scatter)
                 if i != self.adversarial_client]
        out = list(per_client_scatter)
        out[self.adversarial_client] = self.adversarial_fn(honest, MODE)
        return out

    def projection(self, data, matrix):
        return multi_mode_dot(data, [m.T for m in matrix], modes=[1, 2, 3])

    def _client_scatter(self, client_data, MODE, kron_other):
        """This client's contribution to the mode-MODE global scatter
        matrix: sum over its own samples of
        (unfold(sample)@kron_other) @ (unfold(sample)@kron_other).T
        """
        unfolded = np.array([
            unfold(client_data[i], mode=MODE) @ kron_other
            for i in range(client_data.shape[0])
        ])
        return np.einsum("nij,nkj->ik", unfolded, unfolded)

    def _client_scatter_uncontracted(self, client_data, MODE):
        """Initialization-time contribution: no other-mode U estimate
        exists yet, so use the raw per-mode unfolding (matches
        MPCA_FD.Uinitial's un-contracted scatter).
        """
        unfolded = np.array([
            unfold(client_data[i], mode=MODE)
            for i in range(client_data.shape[0])
        ])
        return np.einsum("nij,nkj->ik", unfolded, unfolded)

    @staticmethod
    def _top_eigvecs(S, P):
        S = np.real_if_close(S, tol=1)
        eigenvalue, u = np.linalg.eig(S)
        u = np.real_if_close(u, tol=1)
        eigenvalue = np.real_if_close(eigenvalue, tol=1)
        u = u[:, np.argsort(eigenvalue)[::-1]]
        return u[:, 0:P]

    def _aggregated_U(self, projected_clients, U_mat, MODE, P):
        first, second = min((MODE + 1) % 3, (MODE + 2) % 3), \
            max((MODE + 1) % 3, (MODE + 2) % 3)
        kron_other = np.kron(U_mat[first], U_mat[second])
        per_client_scatter = [
            self._client_scatter(data, MODE, kron_other)
            for data in projected_clients
        ]
        per_client_scatter = self._scatter_list(per_client_scatter, MODE)
        S_agg = self.agg_fn(per_client_scatter)
        return self._top_eigvecs(S_agg, P)

    def _initial_U(self, projected_clients, MODE, P):
        per_client_scatter = [
            self._client_scatter_uncontracted(data, MODE)
            for data in projected_clients
        ]
        per_client_scatter = self._scatter_list(per_client_scatter, MODE)
        S_agg = self.agg_fn(per_client_scatter)
        return self._top_eigvecs(S_agg, P)

    def train(self, client_data, shared_v=False):
        """client_data: list of n arrays, each (J_m, I1, I2, I3).

        shared_v (Direction 2 / heterogeneity ablation): when True, V is
        NOT fit per-client — it is fit ONCE on the pooled data across all
        clients and applied identically to every client. This is the
        "no personalization" baseline: it isolates whether the
        federated architecture's per-client local V is what protects the
        shared U from degrading under non-IID client heterogeneity (the
        standard personalized-FL hypothesis), by removing exactly that
        one component while keeping everything else — including the
        aggregation step at U — identical. Requires all clients to share
        the same raw tensor shape (a single crop size), since pooling
        concatenates along the sample axis.
        """

        def Vinitial(x, MODE, S):
            unfolded_x = np.array(
                [unfold(x[i], mode=MODE).T for i in range(x.shape[0])])
            scatter = np.einsum("nij,nik->jk", unfolded_x, unfolded_x)
            return self._top_eigvecs(scatter, S)

        def V_update(x2, U_mat, V_mat_i, P1, MODE, S):
            u1, u2, u3 = U_mat
            first, second = min((MODE + 1) % 3, (MODE + 2) % 3), \
                max((MODE + 1) % 3, (MODE + 2) % 3)
            v1, v2 = V_mat_i[first], V_mat_i[second]
            if MODE == 0:
                c = np.kron(v1 @ u2, v2 @ u3)
            elif MODE == 1:
                c = np.kron(v1 @ u1, v2 @ u3)
            else:
                c = np.kron(v1 @ u1, v2 @ u2)
            unfolded_x = np.array(
                [(unfold(x2[i], mode=MODE) @ c).T for i in range(x2.shape[0])])
            scatter = np.einsum("nij,nik->jk", unfolded_x, unfolded_x)
            u = self._top_eigvecs(scatter, P1)
            u_mode = U_mat[MODE]
            I2 = S
            return u @ u_mode.T @ np.linalg.inv(
                u_mode @ u_mode.T + LAM * np.eye(I2))

        n = len(client_data)

        if shared_v:
            pooled = np.concatenate(client_data, axis=0)
            V_shared = [Vinitial(pooled, j, self.I[j]) for j in range(3)]
            V_mat = [V_shared for _ in range(n)]
        else:
            # ─── Local V: identical math to MPCA_FD, purely per-client ───
            V_mat = [None] * n
            for i, data in enumerate(client_data):
                V_mat[i] = [Vinitial(data, j, self.I[j]) for j in range(3)]

        projected = [self.projection(data, V_mat[i])
                    for i, data in enumerate(client_data)]

        # ─── Global U: aggregated across clients (the Byzantine-robust point) ───
        U_mat = [self._initial_U(projected, j, self.P[j]) for j in range(3)]

        for _ in range(self.iterations):
            for i in range(3):
                U_mat[i] = self._aggregated_U(projected, U_mat, i, self.P[i])
            if shared_v:
                V_shared = [V_update(pooled, U_mat, V_shared, self.P[j], j,
                                     self.I[j]) for j in range(3)]
                V_mat = [V_shared for _ in range(n)]
                projected = [self.projection(data, V_mat[i])
                            for i, data in enumerate(client_data)]
                continue
            for i, data in enumerate(client_data):
                for j in range(3):
                    V_mat[i][j] = V_update(
                        data, U_mat, V_mat[i], self.P[j], j, self.I[j])
            projected = [self.projection(data, V_mat[i])
                        for i, data in enumerate(client_data)]

        self.U_mat = U_mat
        self.V_mat = V_mat
        return U_mat, V_mat
