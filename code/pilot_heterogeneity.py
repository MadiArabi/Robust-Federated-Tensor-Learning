"""
Direction 2 foundation pilot: does the federated architecture's local-V /
shared-U split protect the shared subspace from non-IID client
heterogeneity, compared to a "no personalization" ablation (single shared
V across all clients)?

Motivation (experiment_log.md, Direction 2): Chapters 2/3's federated MPCA
already has a personalization-style architecture (local V per client,
shared global U) — directly analogous to personalized-FL schemes (e.g.
FedPer, LG-FedAvg) that keep a local layer to absorb client-specific
structure while a shared layer captures common structure. Direction 1
established the infrastructure (n=8 synthetic clients) and, in its
attack-free heterogeneity checks, already saw that aggregation-rule choice
alone can be sensitive to non-IID clients (e.g. coordinate-median vs sum).
This pilot asks the architectural question directly, with a legitimate
(non-adversarial) heterogeneity source: does the existing local-V/global-U
split degrade more gracefully under increasing client heterogeneity than
a no-personalization (shared-V) alternative would?

Heterogeneity mechanism: Dirichlet(alpha) partition (heterogeneity.py) of
the FULL 284-sample pool, single crop size ('C', 20x20 — isolates
degradation-REGIME heterogeneity from the sensor/crop-size heterogeneity
already baked into the real A/B/C split), by TTF-tertile "regime" label,
into n=8 synthetic clients. Low alpha = high heterogeneity (each client
dominated by one regime); high alpha = near-IID (balanced regime mix).

Two architectures compared at each alpha level, both agg_fn=sum (no
Byzantine robustness question here — this is about heterogeneity, not
attacks):
  - federated: MPCA_FD_Robust.train(..., shared_v=False) — the existing
    architecture, local V per client.
  - no_personalization: MPCA_FD_Robust.train(..., shared_v=True) — V fit
    once on pooled data, identical across all clients.

Both are compared via principal angle against a POOLED REFERENCE: the
same architecture's own n=1 fit on all 284 samples with no client
boundaries at all (the "no privacy constraint, fully homogeneous" upper
bound). Hypothesis: federated (local V) should stay closer to the pooled
reference than no_personalization does, and the gap should widen as
alpha decreases.

Usage:
    python pilot_heterogeneity.py --n-repeats 5
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import copy

import numpy as np
import pandas as pd

from mpca_byzantine import MPCA_FD_Robust
from byzantine_agg import aggregate_sum
from heterogeneity import ttf_regime_labels, dirichlet_partition, regime_composition
from motivation_pilot import principal_angles_deg
from rftl_s_real import load_data, prepare_data, I_COMMON

RANK = [5, 5, 6]
FIT_ITERATIONS = 30
CROP = 'C'          # single crop size for the whole pool (20x20)
N_CLIENTS = 8
N_REGIMES = 3
ALPHA_LEVELS = [0.2, 0.5, 1.0, 3.0, 10.0, 100.0]


def max_principal_angle(U_a, U_b):
    return max(np.max(principal_angles_deg(U_a[n], U_b[n])) for n in range(3))


def run_repeat(rep_seed):
    data, y = _GLOBAL_DATA, _GLOBAL_Y
    full = prepare_data(data, len(data), CROP)
    full_mean = np.mean(full, axis=0)
    full_centered = full - full_mean
    labels = ttf_regime_labels(y, n_regimes=N_REGIMES)

    rows = []

    # ─── Pooled reference: no client boundaries at all, same for every
    # alpha level this repeat ───
    ref_model = MPCA_FD_Robust(I_COMMON, RANK, iterations=FIT_ITERATIONS,
                               agg_fn=aggregate_sum)
    U_ref, _ = ref_model.train([copy.deepcopy(full_centered)])

    rng = np.random.RandomState(rep_seed)
    for alpha in ALPHA_LEVELS:
        parts = dirichlet_partition(len(y), labels, N_CLIENTS, alpha, rng)
        clients = [full_centered[idx] for idx in parts]
        comp = regime_composition(parts, labels, N_REGIMES)
        avg_max_frac = float(comp.max(axis=1).mean())

        for arch, shared_v in [('federated', False),
                               ('no_personalization', True)]:
            model = MPCA_FD_Robust(I_COMMON, RANK, iterations=FIT_ITERATIONS,
                                   agg_fn=aggregate_sum)
            U_fit, _ = model.train([copy.deepcopy(c) for c in clients],
                                   shared_v=shared_v)
            angle = max_principal_angle(U_ref, U_fit)
            rows.append({'repeat': rep_seed, 'alpha': alpha, 'arch': arch,
                        'avg_max_regime_frac': avg_max_frac,
                        'angle_vs_pooled_deg': angle,
                        'min_client_size': int(min(len(p) for p in parts))})

    return rows


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default=None)
    parser.add_argument('--n-repeats', type=int, default=5)
    parser.add_argument('--output', default='output/pilot_heterogeneity.csv')
    args = parser.parse_args()

    data_path = args.data_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'data')

    print("Loading real degradation data...", flush=True)
    _GLOBAL_DATA, _GLOBAL_Y = load_data(data_path)
    print(f"  Loaded {len(_GLOBAL_DATA)} samples, single crop='{CROP}'",
          flush=True)
    print(f"  Alpha levels: {ALPHA_LEVELS}", flush=True)

    master_rng = np.random.RandomState(2024)
    rep_seeds = [int(master_rng.randint(1, 100000))
                for _ in range(args.n_repeats)]

    all_rows = []
    for i, seed in enumerate(rep_seeds):
        rows = run_repeat(seed)
        all_rows.extend(rows)
        print(f"  Repeat {i + 1} done (seed={seed}).", flush=True)

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} rows to {args.output}", flush=True)

    print("\n" + "=" * 96, flush=True)
    print("HETEROGENEITY: does local-V (federated) beat shared-V "
          "(no_personalization) as heterogeneity increases (alpha "
          "decreases)?", flush=True)
    print(f"\n  {'alpha':>7}  {'avg_max_regime_frac':>20}  "
         f"{'federated':>12}  {'no_personal':>12}  {'gap':>8}", flush=True)
    for alpha in ALPHA_LEVELS:
        sub = df[df.alpha == alpha]
        frac = sub['avg_max_regime_frac'].mean()
        fed = sub[sub['arch'] == 'federated']['angle_vs_pooled_deg'].mean()
        nop = sub[sub['arch'] == 'no_personalization'][
            'angle_vs_pooled_deg'].mean()
        print(f"  {alpha:7.2f}  {frac:20.3f}  {fed:12.2f}  {nop:12.2f}  "
             f"{nop - fed:+8.2f}", flush=True)
