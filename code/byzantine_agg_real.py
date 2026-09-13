"""
Byzantine-robust aggregation benchmark on real degradation data —
PRODUCTION (HPC) version of pilot_byzantine_agg.py.

Compares 6 aggregation rules (sum, median, trimmed_mean, geometric_median,
krum, multi_krum) for the shared global U in federated MPCA, under a
physically-motivated "faulty camera" attack (hot_block, one of 8 synthetic
clients). See direction1_byzantine_aggregation_findings.md for the full
design history and the n=3-degeneracy / n=8-fix narrative this scales up.

Two checks per rule, each compared to that rule's OWN clean reference
(never a shared one — see the findings doc for why that matters):
  - heterogeneity: attack-free angle of the rule's clean fit vs sum's
    clean fit (how much cross-client heterogeneity alone biases a rule).
  - robustness: angle of the rule's dirty fit vs its own clean fit, at
    hot_block pi_s in {0.6, 1.0}.

NOTE on clean_client_residual (confirmed present identically in the
original 5-repeat pilot's own output, output/pilot_byzantine_agg.csv --
not a hardening regression): this column is nearly CONSTANT (~1e-6
relative) across all 6 aggregation rules and both pi_s levels within a
repeat, even though angle_deg varies by 80+ degrees between rules in the
same row. This is because each clean client's local V is refit every
iteration via a closed-form ridge-regression-style update against
whatever global U it is given (see MPCA_FD_Robust.train's V_update) --
at RANK=[5,5,6] that local V has enough capacity to nearly fully
compensate for whichever subspace U ends up aggregated, so this
"swamping" metric does not actually discriminate between rules or
contamination severity. It is kept in the output (free byproduct of the
fit) but should NOT be used as evidence about swamping damage; angle_deg
is the metric all real Direction 1 conclusions are based on and is
unaffected by this (it compares U matrices directly, before any
per-client V compensation).

This experiment was previously run only as a 5-repeat local pilot; per
Madi's request this is the properly-powered (n=50) HPC version, with the
same checkpointing/resume/defensive-error-handling machinery already
proven necessary for the prediction and monitoring HPC jobs.

Usage:
    python byzantine_agg_real.py --data-path /path/to/data --n-repeats 50 \
        --n-workers 10 --output-dir output_byzantine_agg/
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
from contamination import contaminate
from motivation_pilot import principal_angles_deg
from rftl_s_real import prepare_data, load_data, SIZES, I_COMMON

# ─── Final design constants (matches the validated pilot exactly) ──────────

RANK = [5, 5, 6]
FIT_ITERATIONS = 150     # converged production fit (pilot used 30)

SPLIT_PLAN = [('A', 2), ('B', 3), ('C', 3)]   # 8 synthetic clients
TARGET_CLIENT = 0
F_BYZANTINE = 1
CONDITIONS = [
    # (mode, amplitude, pi_s)
    ('hot_block', 10.0, 0.60),
    ('hot_block', 10.0, 1.00),
]
BLOCK_SIZE = 9

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


# ─── Checkpointing (same pattern as monitoring_real.py) ─────────────────────

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
    n = len(clients)
    clean_idx = [m for m in range(n) if m != TARGET_CLIENT]

    rows = []
    good_rules = dict(AGG_RULES)  # rules that fit cleanly this repeat

    # ─── Clean reference U for EVERY rule (matched-reference design) ───
    U_clean = {}
    for agg_name, agg_fn in AGG_RULES.items():
        try:
            model = MPCA_FD_Robust(I_COMMON, RANK, iterations=FIT_ITERATIONS,
                                   agg_fn=agg_fn)
            U_clean[agg_name], _ = model.train(
                [copy.deepcopy(c) for c in clients])
        except Exception as e:
            print(f"  Repeat {rep_idx + 1} (seed={rep_seed}) rule="
                 f"{agg_name} clean-fit FAILED: {type(e).__name__}: {e} "
                 f"-- excluding this rule for the whole repeat", flush=True)
            good_rules.pop(agg_name, None)

    if 'sum' not in U_clean:
        print(f"  Repeat {rep_idx + 1} (seed={rep_seed}): 'sum' clean-fit "
             f"failed -- no reference available, skipping heterogeneity "
             f"checks entirely for this repeat", flush=True)
    else:
        for agg_name in good_rules:
            het_angle = max_principal_angle(U_clean['sum'], U_clean[agg_name])
            rows.append({'repeat': rep_idx, 'rep_seed': rep_seed,
                        'n_clients': n, 'check': 'heterogeneity',
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

        for agg_name, agg_fn in good_rules.items():
            try:
                model = MPCA_FD_Robust(I_COMMON, RANK,
                                       iterations=FIT_ITERATIONS, agg_fn=agg_fn)
                U_dirty, V_dirty = model.train(
                    [copy.deepcopy(d) for d in dirty_clients])

                angle = max_principal_angle(U_clean[agg_name], U_dirty)
                resid = clean_client_residual(dirty_clients, U_dirty, V_dirty,
                                              clean_idx)
                rows.append({
                    'repeat': rep_idx, 'rep_seed': rep_seed, 'n_clients': n,
                    'check': 'robustness', 'agg': agg_name, 'mode': mode,
                    'pi_s': pi_s, 'angle_deg': angle,
                    'clean_client_residual': resid,
                })
            except Exception as e:
                print(f"  Repeat {rep_idx + 1} (seed={rep_seed}) "
                     f"mode={mode} pi_s={pi_s} rule={agg_name} dirty-fit "
                     f"FAILED: {type(e).__name__}: {e} -- skipping",
                     flush=True)

    _save_repeat(rep_idx, rep_seed, rows, _GLOBAL_OUTPUT_DIR)
    print(f"  Repeat {rep_idx + 1} done (seed={rep_seed}).", flush=True)
    return rows


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default=None)
    parser.add_argument('--n-repeats', type=int, default=50)
    parser.add_argument('--n-workers', type=int, default=10)
    parser.add_argument('--output-dir', default='output_byzantine_agg')
    args = parser.parse_args()

    data_path = args.data_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'data')

    print("Loading real degradation data...", flush=True)
    data, y = load_data(data_path)
    print(f"  Loaded {len(data)} samples", flush=True)
    n_clients = sum(k for _, k in SPLIT_PLAN)
    print(f"  Split plan: {SPLIT_PLAN} -> {n_clients} synthetic clients "
          f"(target={TARGET_CLIENT}, f={F_BYZANTINE})", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    # Same seed derivation as rftl_s_real/monitoring_real so repeats match
    # across all experiments in this project
    master_rng = np.random.RandomState(2024)
    rep_seeds = [int(master_rng.randint(1, 100000))
                for _ in range(args.n_repeats)]

    _, completed = _load_checkpoints(args.output_dir)
    worker_args = [(i, s) for i, s in enumerate(rep_seeds)
                   if i not in completed]
    if completed:
        print(f"  Resuming: {len(completed)} repeats already saved, "
              f"{len(worker_args)} remaining.", flush=True)

    n_fits_per_repeat = len(AGG_RULES) * (1 + len(CONDITIONS))
    print(f"Running {len(worker_args)} repeats x {n_fits_per_repeat} fits, "
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
    csv_path = os.path.join(args.output_dir, 'byzantine_agg_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} rows to {csv_path}", flush=True)

    print("\n" + "=" * 96, flush=True)
    print("HETEROGENEITY (attack-free): angle of each rule's clean fit vs "
          "sum's clean fit", flush=True)
    het = df[df['check'] == 'heterogeneity']
    print("\n  coordinate-wise:", flush=True)
    for agg_name in COORDWISE:
        sub = het[het['agg'] == agg_name]
        print(f"    {agg_name:<17}: {sub['angle_deg'].mean():6.2f} deg "
              f"+/- {sub['angle_deg'].std():.2f}  (n={len(sub)})", flush=True)
    print("\n  distance-based:", flush=True)
    for agg_name in DISTANCE:
        sub = het[het['agg'] == agg_name]
        print(f"    {agg_name:<17}: {sub['angle_deg'].mean():6.2f} deg "
              f"+/- {sub['angle_deg'].std():.2f}  (n={len(sub)})", flush=True)

    print("\n" + "=" * 96, flush=True)
    print("ROBUSTNESS: angle of each rule's dirty fit vs ITS OWN clean fit "
          "(one client contaminated, hot_block)", flush=True)
    rob = df[df['check'] == 'robustness']
    for mode, amplitude, pi_s in CONDITIONS:
        print(f"\n{mode} amp={amplitude} pi_s={pi_s}:", flush=True)
        print("  coordinate-wise:", flush=True)
        for agg_name in ['sum'] + COORDWISE:
            sub = rob[(rob['mode'] == mode) & (rob.pi_s == pi_s)
                     & (rob['agg'] == agg_name)]
            print(f"    {agg_name:<17}: {sub['angle_deg'].mean():6.2f} deg   "
                  f"clean-client residual "
                  f"{sub['clean_client_residual'].mean():8.4f}  (n={len(sub)})",
                  flush=True)
        print("  distance-based:", flush=True)
        for agg_name in DISTANCE:
            sub = rob[(rob['mode'] == mode) & (rob.pi_s == pi_s)
                     & (rob['agg'] == agg_name)]
            print(f"    {agg_name:<17}: {sub['angle_deg'].mean():6.2f} deg   "
                  f"clean-client residual "
                  f"{sub['clean_client_residual'].mean():8.4f}  (n={len(sub)})",
                  flush=True)
