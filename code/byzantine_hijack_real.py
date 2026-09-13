"""
Byzantine hijack-attack benchmark on real degradation data — PRODUCTION
(HPC) version of pilot_byzantine_hijack.py.

The harder stress test after pilot_byzantine_agg.py/byzantine_agg_real.py
showed every rule, including plain sum, barely reacting to a
physically-motivated faulty-camera artifact: one client sends an
arbitrary, hand-crafted scatter-matrix contribution (not derived from
real image data) designed to hijack the shared subspace's top
eigenvector toward the honest data's own weakest direction.

Same 8-client split as byzantine_agg_real.py (reused directly), same
matched-clean-reference design, same 6 aggregation rules. New axis:
amplitude_mult sweep — how many multiples of the honest top eigenvalue
does the malicious contribution need before each rule's estimate breaks?

Reports THREE numbers per rule/amplitude (see original pilot's docstring
for the full methodological rationale on why both LOO and full references
are needed to separate "attack rejected" from "unavoidable exclusion
cost"):
  - angle_vs_loo_clean_deg: dirty fit vs a clean fit on the 7 OTHER
    clients only (target absent). Answers "did the attack succeed."
  - angle_vs_full_clean_deg: dirty fit vs a clean fit on all 8 real
    clients. Total real-world cost (attack + exclusion combined).
  - no_attack_baseline: the same LOO comparison with NO attacker at all —
    the noise floor from this non-convex alternating (V,U) optimization's
    sensitivity to which clients are included.

This was previously only a 5-repeat local pilot; this is the properly
powered (n=50) HPC version, hardened with the same
checkpointing/resume/defensive-error-handling machinery already proven
necessary for the prediction and monitoring HPC jobs.

Usage:
    python byzantine_hijack_real.py --data-path /path/to/data --n-repeats 50 \
        --n-workers 10 --output-dir output_byzantine_hijack/
"""

import sys
import os

# Each multiprocessing.Pool worker must be single-threaded internally --
# otherwise every worker's BLAS calls spawn their own thread pool (observed
# locally: 8 threads/process by default) and N worker processes oversubscribe
# the machine's cores N-fold, making --n-workers parallelism far slower than
# sequential. Must be set before numpy (or anything importing numpy) loads.
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_var] = '1'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import copy
import json
import time
import functools
import multiprocessing

import numpy as np
import pandas as pd

from mpca_byzantine import MPCA_FD_Robust
from byzantine_agg import (aggregate_sum, aggregate_coordinate_median,
                           aggregate_trimmed_mean, aggregate_krum,
                           aggregate_multi_krum, aggregate_geometric_median)
from byzantine_attacks import hijack_attack
from motivation_pilot import principal_angles_deg
from rftl_s_real import load_data, I_COMMON
from byzantine_agg_real import build_synthetic_clients, SPLIT_PLAN

RANK = [5, 5, 6]
FIT_ITERATIONS = 150     # converged production fit (pilot used 30)
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

_GLOBAL_DATA = None
_GLOBAL_Y = None
_GLOBAL_OUTPUT_DIR = None


def _init_worker(data, y, output_dir):
    """Pool initializer — required on Windows (spawn), harmless under fork."""
    global _GLOBAL_DATA, _GLOBAL_Y, _GLOBAL_OUTPUT_DIR
    _GLOBAL_DATA, _GLOBAL_Y, _GLOBAL_OUTPUT_DIR = data, y, output_dir


def max_principal_angle(U_a, U_b):
    return max(np.max(principal_angles_deg(U_a[n], U_b[n])) for n in range(3))


# ─── Checkpointing (same pattern as monitoring_real.py / byzantine_agg_real.py) ─

def _ckpt_path(output_dir, rep_idx, rep_seed):
    return os.path.join(output_dir, 'checkpoints',
                        f'repeat_{rep_idx:03d}_seed{rep_seed}.json')


def _save_repeat(rep_idx, rep_seed, rows, output_dir):
    if output_dir is None:
        return
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    with open(_ckpt_path(output_dir, rep_idx, rep_seed), 'w') as f:
        json.dump(rows, f)


def _load_checkpoints(output_dir):
    ckpt_dir = os.path.join(output_dir, 'checkpoints')
    rows, completed = [], set()
    if not os.path.isdir(ckpt_dir):
        return rows, completed
    for fname in sorted(os.listdir(ckpt_dir)):
        if not fname.endswith('.json'):
            continue
        completed.add(int(fname.split('_')[1]))
        with open(os.path.join(ckpt_dir, fname)) as f:
            rows.extend(json.load(f))
    return rows, completed


# ─── Worker ──────────────────────────────────────────────────────────────────

def _run_single_repeat(args):
    rep_idx, rep_seed = args
    data, y = _GLOBAL_DATA, _GLOBAL_Y

    clients, labels = build_synthetic_clients(data, y, SPLIT_PLAN, rep_seed)
    honest_only = [c for i, c in enumerate(clients) if i != TARGET_CLIENT]

    rows = []
    good_rules = dict(AGG_RULES)  # rules that fit cleanly this repeat

    # ─── Two clean references per rule (full + leave-one-out) ───
    U_clean_full, U_clean_loo = {}, {}
    for agg_name, agg_fn in AGG_RULES.items():
        try:
            model_full = MPCA_FD_Robust(I_COMMON, RANK,
                                        iterations=FIT_ITERATIONS, agg_fn=agg_fn)
            U_clean_full[agg_name], _ = model_full.train(
                [copy.deepcopy(c) for c in clients])
            model_loo = MPCA_FD_Robust(I_COMMON, RANK,
                                       iterations=FIT_ITERATIONS, agg_fn=agg_fn)
            U_clean_loo[agg_name], _ = model_loo.train(
                [copy.deepcopy(c) for c in honest_only])
        except Exception as e:
            print(f"  Repeat {rep_idx + 1} (seed={rep_seed}) rule="
                 f"{agg_name} clean-reference fit FAILED: "
                 f"{type(e).__name__}: {e} -- excluding this rule for the "
                 f"whole repeat", flush=True)
            good_rules.pop(agg_name, None)
            U_clean_full.pop(agg_name, None)
            U_clean_loo.pop(agg_name, None)
            continue

        baseline_angle = max_principal_angle(U_clean_full[agg_name],
                                             U_clean_loo[agg_name])
        rows.append({'repeat': rep_idx, 'rep_seed': rep_seed,
                    'agg': agg_name, 'amplitude_mult': 0.0,
                    'target': 'no_attack_baseline',
                    'angle_vs_full_clean_deg': 0.0,
                    'angle_vs_loo_clean_deg': baseline_angle})

    # ─── Hijack attack at each amplitude level, each surviving rule ───
    for amplitude_mult in AMPLITUDE_LEVELS:
        attack_fn = hijack_attack(amplitude_mult=amplitude_mult,
                                  target='weakest')
        for agg_name, agg_fn in good_rules.items():
            try:
                model = MPCA_FD_Robust(
                    I_COMMON, RANK, iterations=FIT_ITERATIONS, agg_fn=agg_fn,
                    adversarial_client=TARGET_CLIENT, adversarial_fn=attack_fn)
                U_dirty, _ = model.train([copy.deepcopy(c) for c in clients])

                angle_full = max_principal_angle(U_clean_full[agg_name],
                                                 U_dirty)
                angle_loo = max_principal_angle(U_clean_loo[agg_name], U_dirty)
                rows.append({'repeat': rep_idx, 'rep_seed': rep_seed,
                            'agg': agg_name, 'amplitude_mult': amplitude_mult,
                            'target': 'weakest',
                            'angle_vs_full_clean_deg': angle_full,
                            'angle_vs_loo_clean_deg': angle_loo})
            except Exception as e:
                print(f"  Repeat {rep_idx + 1} (seed={rep_seed}) "
                     f"amplitude={amplitude_mult} rule={agg_name} "
                     f"weakest-hijack fit FAILED: {type(e).__name__}: {e} "
                     f"-- skipping", flush=True)

    # ─── Random-direction sanity check at the highest amplitude ───
    attack_fn_rand = hijack_attack(amplitude_mult=AMPLITUDE_LEVELS[-1],
                                   target='random')
    for agg_name, agg_fn in good_rules.items():
        try:
            model = MPCA_FD_Robust(
                I_COMMON, RANK, iterations=FIT_ITERATIONS, agg_fn=agg_fn,
                adversarial_client=TARGET_CLIENT, adversarial_fn=attack_fn_rand)
            U_dirty, _ = model.train([copy.deepcopy(c) for c in clients])
            angle_full = max_principal_angle(U_clean_full[agg_name], U_dirty)
            angle_loo = max_principal_angle(U_clean_loo[agg_name], U_dirty)
            rows.append({'repeat': rep_idx, 'rep_seed': rep_seed,
                        'agg': agg_name, 'amplitude_mult': AMPLITUDE_LEVELS[-1],
                        'target': 'random', 'angle_vs_full_clean_deg': angle_full,
                        'angle_vs_loo_clean_deg': angle_loo})
        except Exception as e:
            print(f"  Repeat {rep_idx + 1} (seed={rep_seed}) rule="
                 f"{agg_name} random-hijack fit FAILED: "
                 f"{type(e).__name__}: {e} -- skipping", flush=True)

    _save_repeat(rep_idx, rep_seed, rows, _GLOBAL_OUTPUT_DIR)
    print(f"  Repeat {rep_idx + 1} done (seed={rep_seed}).", flush=True)
    return rows


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default=None)
    parser.add_argument('--n-repeats', type=int, default=50)
    parser.add_argument('--n-workers', type=int, default=10)
    parser.add_argument('--output-dir', default='output_byzantine_hijack')
    args = parser.parse_args()

    data_path = args.data_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'data')

    print("Loading real degradation data...", flush=True)
    data, y = load_data(data_path)
    print(f"  Loaded {len(data)} samples", flush=True)
    print(f"  Amplitude levels (x honest top eigenvalue): {AMPLITUDE_LEVELS}",
          flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    master_rng = np.random.RandomState(2024)
    rep_seeds = [int(master_rng.randint(1, 100000))
                for _ in range(args.n_repeats)]

    _, completed = _load_checkpoints(args.output_dir)
    worker_args = [(i, s) for i, s in enumerate(rep_seeds)
                   if i not in completed]
    if completed:
        print(f"  Resuming: {len(completed)} repeats already saved, "
              f"{len(worker_args)} remaining.", flush=True)

    n_fits_per_repeat = len(AGG_RULES) * (2 + len(AMPLITUDE_LEVELS) + 1)
    print(f"Running {len(worker_args)} repeats x ~{n_fits_per_repeat} fits, "
          f"{args.n_workers} workers...", flush=True)

    start = time.time()
    if worker_args:
        if args.n_workers > 1:
            pool = multiprocessing.Pool(
                processes=args.n_workers, initializer=_init_worker,
                initargs=(data, y, args.output_dir))
            pool.map(_run_single_repeat, worker_args)
            pool.close()
            pool.join()
        else:
            _init_worker(data, y, args.output_dir)
            for a in worker_args:
                _run_single_repeat(a)
    print(f"\nDone in {(time.time() - start) / 3600:.2f} hours.", flush=True)

    all_rows, _ = _load_checkpoints(args.output_dir)
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(args.output_dir, 'byzantine_hijack_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} rows to {csv_path}", flush=True)

    print("\n" + "=" * 96, flush=True)
    print("HIJACK ATTACK: LOO metric = angle vs a clean fit on the 7 OTHER "
          "clients only (isolates whether the attack was actually "
          "rejected). FULL metric = angle vs a clean fit on all 8 real "
          "clients (also includes the unavoidable cost of losing the "
          "target client's own real signal once correctly excluded — "
          "NOT purely attack damage). (target='weakest')", flush=True)
    weak = df[df['target'] == 'weakest']
    baseline = df[df['target'] == 'no_attack_baseline']

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
    rand = df[df['target'] == 'random']
    for agg_name in ['sum'] + COORDWISE + DISTANCE:
        sub = rand[rand['agg'] == agg_name]
        print(f"    {agg_name:<17}: {sub['angle_vs_loo_clean_deg'].mean():6.2f} "
              f"deg +/- {sub['angle_vs_loo_clean_deg'].std():.2f}  "
              f"(n={len(sub)})", flush=True)
