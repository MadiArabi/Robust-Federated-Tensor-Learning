"""
Foundation pilot for the Byzantine-robust aggregation reframing (Direction 1,
experiment_log.md Session 8+).

v1 of this pilot (n=3 real users/clients — same split as the rest of the
project) found that coordinate-wise median aggregation is WORSE than plain
sum, even attack-free, purely from cross-client heterogeneity: at n=3 a
"coordinate-wise median" is a raw per-entry SELECTION among only 3 values,
not a smoothed estimate, so it can assemble an incoherent combined matrix
that resembles no real client. This also meant Krum (needs n>=2f+3=5) and a
genuinely-averaging trimmed mean (needs n > 2*n_trim with slack) weren't
even well-posed at n=3.

v2 (this version) attacks both problems at once:
  1. CLIENT COUNT: splits each of the 3 real users (A/B/C, distinct crop
     sizes from Chapter 2's design) into several sub-cohorts, reaching
     n=8 total synthetic clients. These are honestly synthetic — same
     underlying sensor/site data, randomly subsampled — not independent
     physical sites; flagged clearly, and this expansion is *also* exactly
     the foundation Direction 2 (non-IID heterogeneity) will need.
  2. AGGREGATION FAMILY: compares COORDINATE-WISE rules (sum, median,
     trimmed_mean) against DISTANCE-BASED rules (geometric_median, krum,
     multi_krum) side by side, to see whether preserving each client's
     internal entry-correlation structure (distance-based) fixes the
     heterogeneity fragility that coordinate-wise aggregation showed.

Two checks per aggregation rule, matched to each rule's OWN clean
reference (comparing a dirty fit under rule X to a clean fit under rule Y
would conflate contamination damage with rule X vs Y's baseline
difference — the bug the n=3 pilot's first version had, fixed there and
carried forward here):
  (a) HETEROGENEITY (attack-free): principal angle of rule's clean fit vs
      sum's clean fit. Measures how much a rule distorts the shared
      subspace purely from non-IID clients, with zero attacker.
  (b) ROBUSTNESS: principal angle of rule's dirty fit (one client
      contaminated) vs that SAME rule's own clean fit. Measures how much
      the attack itself moves the rule's aggregate.

Usage:
    python pilot_byzantine_agg.py --n-repeats 5
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import copy
import random
import functools

import numpy as np
import pandas as pd

from mpca_byzantine import MPCA_FD_Robust
from byzantine_agg import (aggregate_sum, aggregate_coordinate_median,
                           aggregate_trimmed_mean, aggregate_krum,
                           aggregate_multi_krum, aggregate_geometric_median)
from contamination import contaminate
from motivation_pilot import principal_angles_deg
from rftl_s_real import prepare_data, load_data, SIZES, I_COMMON

RANK = [5, 5, 6]
FIT_ITERATIONS = 30     # foundation check, not a converged production fit

# Split each real user (A/B/C) into this many synthetic sub-cohorts.
# 2 + 3 + 3 = 8 clients, the minimum requested. Sub-cohort sizes (from the
# FULL per-user pool, not just a train split — no held-out test needed for
# this pilot): A(45)/2 ~ 22-23, B(50)/3 ~ 16-17, C(55)/3 ~ 18-19 samples.
SPLIT_PLAN = [('A', 2), ('B', 3), ('C', 3)]

TARGET_CLIENT = 0        # first sub-cohort of A — the "Byzantine" client
F_BYZANTINE = 1          # matches TARGET_CLIENT count: exactly 1 corrupted
CONDITIONS = [
    # (mode, amplitude, pi_s)
    ('hot_block', 10.0, 0.60),
    ('hot_block', 10.0, 1.00),
]
BLOCK_SIZE = 9

AGG_RULES = {
    # coordinate-wise family
    'sum': aggregate_sum,
    'median': aggregate_coordinate_median,
    'trimmed_mean': functools.partial(aggregate_trimmed_mean, n_trim=1),
    # distance-based family
    'geometric_median': aggregate_geometric_median,
    'krum': functools.partial(aggregate_krum, f=F_BYZANTINE),
    'multi_krum': functools.partial(aggregate_multi_krum, f=F_BYZANTINE),
}
COORDWISE = ['median', 'trimmed_mean']
DISTANCE = ['geometric_median', 'krum', 'multi_krum']


def max_principal_angle(U_a, U_b):
    return max(np.max(principal_angles_deg(U_a[n], U_b[n])) for n in range(3))


def build_synthetic_clients(data, y, split_plan, seed):
    """Split each real user (A/B/C) into several sub-cohorts. Returns a
    flat list of mean-centered client tensors and a parallel list of
    origin labels (e.g. 'A0','A1','B0',...) for traceability.
    """
    rng = np.random.RandomState(seed)
    sample = np.arange(len(data))
    clients, labels = [], []
    for size, which in zip(SIZES, ['A', 'B', 'C']):
        rng.shuffle(sample)
        user_data = prepare_data(data[sample], size, which)
        n_splits = dict(split_plan)[which]
        idx = rng.permutation(user_data.shape[0])
        for k, chunk in enumerate(np.array_split(idx, n_splits)):
            sub = user_data[chunk]
            clients.append(sub - np.mean(sub, axis=0))
            labels.append(f"{which}{k}")
    return clients, labels


def clean_client_residual(clients, U_mat, V_mat, clean_idx):
    from rftl_s import reconstruction_residual
    vals = [reconstruction_residual(clients[m], V_mat[m], U_mat)
           for m in clean_idx]
    return float(np.mean(np.concatenate(vals)))


def run_repeat(rep_seed):
    clients, labels = build_synthetic_clients(
        _GLOBAL_DATA, _GLOBAL_Y, SPLIT_PLAN, rep_seed)
    n = len(clients)
    clean_idx = [m for m in range(n) if m != TARGET_CLIENT]

    rows = []

    # ─── Clean reference U for EVERY rule (matched-reference design) ───
    U_clean = {}
    for agg_name, agg_fn in AGG_RULES.items():
        model = MPCA_FD_Robust(I_COMMON, RANK, iterations=FIT_ITERATIONS,
                               agg_fn=agg_fn)
        U_clean[agg_name], _ = model.train([copy.deepcopy(c) for c in clients])

    for agg_name in AGG_RULES:
        het_angle = max_principal_angle(U_clean['sum'], U_clean[agg_name])
        rows.append({'repeat': rep_seed, 'n_clients': n, 'check': 'heterogeneity',
                    'agg': agg_name, 'mode': 'none', 'pi_s': 0.0,
                    'angle_deg': het_angle})

    # ─── Robustness: contaminate TARGET_CLIENT, compare each rule to its
    # OWN clean reference ───
    for cond_idx, (mode, amplitude, pi_s) in enumerate(CONDITIONS):
        contam_rng = np.random.RandomState(
            rep_seed + 1000 * (cond_idx + 1) + int(pi_s * 100))
        dirty_clients, _ = contaminate(
            clients, pi_s, mode, contam_rng, amplitude=amplitude,
            block_size=BLOCK_SIZE, target_user=TARGET_CLIENT)

        for agg_name, agg_fn in AGG_RULES.items():
            model = MPCA_FD_Robust(I_COMMON, RANK, iterations=FIT_ITERATIONS,
                                   agg_fn=agg_fn)
            U_dirty, V_dirty = model.train(
                [copy.deepcopy(d) for d in dirty_clients])

            angle = max_principal_angle(U_clean[agg_name], U_dirty)
            resid = clean_client_residual(dirty_clients, U_dirty, V_dirty,
                                          clean_idx)
            rows.append({
                'repeat': rep_seed, 'n_clients': n, 'check': 'robustness',
                'agg': agg_name, 'mode': mode, 'pi_s': pi_s,
                'angle_deg': angle, 'clean_client_residual': resid,
            })

    return rows


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default=None)
    parser.add_argument('--n-repeats', type=int, default=5)
    parser.add_argument('--output', default='output/pilot_byzantine_agg.csv')
    args = parser.parse_args()

    data_path = args.data_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'data')

    print("Loading real degradation data...", flush=True)
    _GLOBAL_DATA, _GLOBAL_Y = load_data(data_path)
    print(f"  Loaded {len(_GLOBAL_DATA)} samples", flush=True)
    n_clients = sum(k for _, k in SPLIT_PLAN)
    print(f"  Split plan: {SPLIT_PLAN} -> {n_clients} synthetic clients "
          f"(target={TARGET_CLIENT}, f={F_BYZANTINE})", flush=True)

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
    print("HETEROGENEITY (attack-free): angle of each rule's clean fit vs "
          "sum's clean fit", flush=True)
    het = df[df.check == 'heterogeneity']
    print("\n  coordinate-wise:", flush=True)
    for agg_name in COORDWISE:
        sub = het[het['agg'] == agg_name]
        print(f"    {agg_name:<17}: {sub['angle_deg'].mean():6.2f} deg "
              f"+/- {sub['angle_deg'].std():.2f}", flush=True)
    print("\n  distance-based:", flush=True)
    for agg_name in DISTANCE:
        sub = het[het['agg'] == agg_name]
        print(f"    {agg_name:<17}: {sub['angle_deg'].mean():6.2f} deg "
              f"+/- {sub['angle_deg'].std():.2f}", flush=True)

    print("\n" + "=" * 96, flush=True)
    print("ROBUSTNESS: angle of each rule's dirty fit vs ITS OWN clean fit "
          "(one client contaminated, hot_block)", flush=True)
    rob = df[df.check == 'robustness']
    for mode, amplitude, pi_s in CONDITIONS:
        print(f"\n{mode} amp={amplitude} pi_s={pi_s}:", flush=True)
        print("  coordinate-wise:", flush=True)
        for agg_name in ['sum'] + COORDWISE:
            sub = rob[(rob['mode'] == mode) & (rob.pi_s == pi_s)
                     & (rob['agg'] == agg_name)]
            print(f"    {agg_name:<17}: {sub['angle_deg'].mean():6.2f} deg   "
                  f"clean-client residual "
                  f"{sub['clean_client_residual'].mean():8.4f}", flush=True)
        print("  distance-based:", flush=True)
        for agg_name in DISTANCE:
            sub = rob[(rob['mode'] == mode) & (rob.pi_s == pi_s)
                     & (rob['agg'] == agg_name)]
            print(f"    {agg_name:<17}: {sub['angle_deg'].mean():6.2f} deg   "
                  f"clean-client residual "
                  f"{sub['clean_client_residual'].mean():8.4f}", flush=True)
