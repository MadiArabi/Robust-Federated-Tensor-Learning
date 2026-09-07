"""
Foundation pilot: the "hijack" adversarial attack (byzantine_attacks.py) —
the harder stress test proposed after pilot v2 (image-based contamination,
n=8, 6 rules) showed every aggregation rule, including plain sum, barely
reacted to a physically-motivated faulty-camera artifact. This pilot tests
the literature-standard worst-case instead: one client sends an arbitrary,
hand-crafted scatter-matrix contribution (not derived from real, even
corrupted, image data) designed specifically to hijack the shared
subspace's top eigenvector toward the honest data's own weakest direction.

Same 8-client split as pilot_byzantine_agg.py (reused directly), same
matched-clean-reference design (each rule compared to ITS OWN clean fit,
not a shared one), same 6 aggregation rules. New axis: amplitude_mult
sweep — how many multiples of the honest top eigenvalue does the
malicious contribution need before each rule's estimate breaks?

IMPORTANT METHODOLOGICAL FIX (found during first run's investigation,
experiment_log.md): comparing a defended (attacked-but-rejected) fit
against a clean reference that INCLUDES the target client's own real
data conflates two different things — whether the attack was actually
rejected, and the unavoidable cost of losing that client's real,
legitimate signal once a defense correctly excludes it (clients here are
non-IID, so excluding one always costs something, attack or not). This
pilot therefore reports THREE numbers per rule/amplitude:
  - angle_vs_loo_clean_deg: dirty fit vs a clean fit trained on the 7
    OTHER clients ONLY (target absent entirely). This is the metric that
    actually answers "did the attack succeed" — direct traced-call
    verification (experiment_log.md) confirmed every rule's raw
    aggregation step correctly suppresses the malicious contribution to
    near-zero influence per call.
  - angle_vs_full_clean_deg: dirty fit vs a clean fit on all 8 real
    clients (the total real-world cost, attack + exclusion combined).
  - angle_vs_loo_clean_deg under target='no_attack_baseline': the SAME
    LOO comparison with NO attacker at all (just the full-8 clean fit vs
    the 7-client-only clean fit) — the noise floor from this non-convex
    alternating (V,U) optimization's sensitivity to which clients are
    included, independent of any attack. A rule whose attacked-LOO angle
    doesn't exceed this baseline hasn't really been beaten.

Usage:
    python pilot_byzantine_hijack.py --n-repeats 5
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import copy
import functools

import numpy as np
import pandas as pd

from mpca_byzantine import MPCA_FD_Robust
from byzantine_agg import (aggregate_sum, aggregate_coordinate_median,
                           aggregate_trimmed_mean, aggregate_krum,
                           aggregate_multi_krum, aggregate_geometric_median)
from byzantine_attacks import hijack_attack
from motivation_pilot import principal_angles_deg
from rftl_s_real import load_data, I_COMMON
from pilot_byzantine_agg import build_synthetic_clients, SPLIT_PLAN

RANK = [5, 5, 6]
FIT_ITERATIONS = 30
TARGET_CLIENT = 0
F_BYZANTINE = 1
AMPLITUDE_LEVELS = [2.0, 5.0, 20.0]   # multiples of honest top eigenvalue

AGG_RULES = {
    'sum': aggregate_sum,
    'median': aggregate_coordinate_median,
    'trimmed_mean': functools.partial(aggregate_trimmed_mean, n_trim=1),
    'geometric_median': aggregate_geometric_median,
    'krum': functools.partial(aggregate_krum, f=F_BYZANTINE),
    'multi_krum': functools.partial(aggregate_multi_krum, f=F_BYZANTINE),
}
COORDWISE = ['median', 'trimmed_mean']
DISTANCE = ['geometric_median', 'krum', 'multi_krum']


def max_principal_angle(U_a, U_b):
    return max(np.max(principal_angles_deg(U_a[n], U_b[n])) for n in range(3))


def run_repeat(rep_seed):
    clients, labels = build_synthetic_clients(
        _GLOBAL_DATA, _GLOBAL_Y, SPLIT_PLAN, rep_seed)
    honest_only = [c for i, c in enumerate(clients) if i != TARGET_CLIENT]

    rows = []

    # ─── Two clean references per rule (see module docstring for why both
    # are needed):
    #   U_clean_full: fit on ALL 8 real clients, including the target's
    #     own real (unattacked) data. Comparing a defended fit against
    #     this measures the TOTAL cost of the attack scenario, including
    #     the unavoidable cost of losing the target client's real signal
    #     once a defense correctly excludes it — NOT purely attack damage.
    #   U_clean_loo: fit on the 7 OTHER (honest) clients only, target
    #     entirely absent. Comparing a defended fit against THIS isolates
    #     whether the rule correctly rejected the malicious contribution,
    #     separate from the unavoidable exclusion cost. This is the
    #     metric that actually answers "did the attack succeed."
    U_clean_full, U_clean_loo = {}, {}
    for agg_name, agg_fn in AGG_RULES.items():
        model_full = MPCA_FD_Robust(I_COMMON, RANK, iterations=FIT_ITERATIONS,
                                    agg_fn=agg_fn)
        U_clean_full[agg_name], _ = model_full.train(
            [copy.deepcopy(c) for c in clients])
        model_loo = MPCA_FD_Robust(I_COMMON, RANK, iterations=FIT_ITERATIONS,
                                   agg_fn=agg_fn)
        U_clean_loo[agg_name], _ = model_loo.train(
            [copy.deepcopy(c) for c in honest_only])

        # NO-ATTACK baseline noise floor: both references above are
        # already fit with zero attacker involved (U_clean_loo simply
        # never saw client 0 at all; U_clean_full has client 0's REAL
        # data). Comparing them directly measures how much this
        # non-convex alternating (V,U) optimization's converged solution
        # naturally shifts from including/excluding one real client —
        # with NO attack anywhere in this comparison. Essential context:
        # a rule whose LOO-vs-attacked angle is no bigger than this
        # baseline hasn't really been beaten by the attack at all.
        baseline_angle = max_principal_angle(U_clean_full[agg_name],
                                             U_clean_loo[agg_name])
        rows.append({'repeat': rep_seed, 'agg': agg_name,
                    'amplitude_mult': 0.0, 'target': 'no_attack_baseline',
                    'angle_vs_full_clean_deg': 0.0,
                    'angle_vs_loo_clean_deg': baseline_angle})

    # ─── Hijack attack at each amplitude level, each rule ───
    for amplitude_mult in AMPLITUDE_LEVELS:
        attack_fn = hijack_attack(amplitude_mult=amplitude_mult, target='weakest')
        for agg_name, agg_fn in AGG_RULES.items():
            model = MPCA_FD_Robust(
                I_COMMON, RANK, iterations=FIT_ITERATIONS, agg_fn=agg_fn,
                adversarial_client=TARGET_CLIENT, adversarial_fn=attack_fn)
            U_dirty, _ = model.train([copy.deepcopy(c) for c in clients])

            angle_full = max_principal_angle(U_clean_full[agg_name], U_dirty)
            angle_loo = max_principal_angle(U_clean_loo[agg_name], U_dirty)
            rows.append({'repeat': rep_seed, 'agg': agg_name,
                        'amplitude_mult': amplitude_mult, 'target': 'weakest',
                        'angle_vs_full_clean_deg': angle_full,
                        'angle_vs_loo_clean_deg': angle_loo})

    # ─── One 'random' direction check at the highest amplitude, sanity
    # that the finding isn't an artifact of targeting the weakest honest
    # eigendirection specifically ───
    attack_fn_rand = hijack_attack(amplitude_mult=AMPLITUDE_LEVELS[-1],
                                   target='random')
    for agg_name, agg_fn in AGG_RULES.items():
        model = MPCA_FD_Robust(
            I_COMMON, RANK, iterations=FIT_ITERATIONS, agg_fn=agg_fn,
            adversarial_client=TARGET_CLIENT, adversarial_fn=attack_fn_rand)
        U_dirty, _ = model.train([copy.deepcopy(c) for c in clients])
        angle_full = max_principal_angle(U_clean_full[agg_name], U_dirty)
        angle_loo = max_principal_angle(U_clean_loo[agg_name], U_dirty)
        rows.append({'repeat': rep_seed, 'agg': agg_name,
                    'amplitude_mult': AMPLITUDE_LEVELS[-1], 'target': 'random',
                    'angle_vs_full_clean_deg': angle_full,
                    'angle_vs_loo_clean_deg': angle_loo})

    return rows


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default=None)
    parser.add_argument('--n-repeats', type=int, default=5)
    parser.add_argument('--output', default='output/pilot_byzantine_hijack.csv')
    args = parser.parse_args()

    data_path = args.data_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'data')

    print("Loading real degradation data...", flush=True)
    _GLOBAL_DATA, _GLOBAL_Y = load_data(data_path)
    print(f"  Loaded {len(_GLOBAL_DATA)} samples", flush=True)
    print(f"  Amplitude levels (x honest top eigenvalue): {AMPLITUDE_LEVELS}",
          flush=True)

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
    print("HIJACK ATTACK: LOO metric = angle vs a clean fit on the 7 OTHER "
          "clients only (isolates whether the attack was actually "
          "rejected). FULL metric = angle vs a clean fit on all 8 real "
          "clients (also includes the unavoidable cost of losing the "
          "target client's own real signal once correctly excluded — "
          "NOT purely attack damage). (target='weakest')", flush=True)
    weak = df[df.target == 'weakest']

    baseline = df[df.target == 'no_attack_baseline']

    def _print_group(name, rules):
        print(f"\n  {name}:", flush=True)
        header = (f"    {'rule':<17} {'noattack':>10} | " +
                 " ".join(f"{'LOO/' + str(a) + 'x':>12}" for a in AMPLITUDE_LEVELS) +
                 "   " + " ".join(f"{'full/' + str(a) + 'x':>13}"
                                  for a in AMPLITUDE_LEVELS))
        print(header, flush=True)
        for agg_name in rules:
            base_val = baseline[baseline['agg'] == agg_name][
                'angle_vs_loo_clean_deg'].mean()
            loo_vals = [weak[(weak['agg'] == agg_name) & (weak.amplitude_mult == a)]
                       ['angle_vs_loo_clean_deg'].mean() for a in AMPLITUDE_LEVELS]
            full_vals = [weak[(weak['agg'] == agg_name) & (weak.amplitude_mult == a)]
                        ['angle_vs_full_clean_deg'].mean() for a in AMPLITUDE_LEVELS]
            print(f"    {agg_name:<17} {base_val:10.2f} | "
                  + " ".join(f"{v:12.2f}" for v in loo_vals) + "   "
                  + " ".join(f"{v:13.2f}" for v in full_vals), flush=True)

    _print_group("sum (non-robust baseline)", ['sum'])
    _print_group("coordinate-wise", COORDWISE)
    _print_group("distance-based", DISTANCE)

    print("\n" + "=" * 96, flush=True)
    print(f"RANDOM-DIRECTION sanity check (amplitude={AMPLITUDE_LEVELS[-1]}x), "
          "LOO metric:", flush=True)
    rand = df[df.target == 'random']
    for agg_name in ['sum'] + COORDWISE + DISTANCE:
        sub = rand[rand['agg'] == agg_name]
        print(f"    {agg_name:<17}: {sub['angle_vs_loo_clean_deg'].mean():6.2f} "
              f"deg +/- {sub['angle_vs_loo_clean_deg'].std():.2f}", flush=True)
