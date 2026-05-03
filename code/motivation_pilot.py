"""
Motivation Pilot — Chapter 3

Goal: confirm that Chapter 2's federated estimator (MPCA_FD) is sensitive to
sample-level contamination. Specifically, show that at pi_S = 0.10 the
estimated global subspace U_n shifts by more than 10 degrees (principal angle)
relative to a clean reference.

Contamination model: a fraction pi_S of samples per user have additive
Gaussian noise at 10x baseline variance injected into the full tensor.

Output: table + plot of max principal angle (degrees) vs. contamination level.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import copy
from scipy.io import loadmat
from my_mpca_02_27_nomean import MPCA_FD, train_test
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def flush_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)


# ─���─ Data loading (mirrors onepass-02-13-0.py) ───────────────────────────────

def load_simulated_data(path, max_files=400):
    """Load simulated data. Each .mat has 10 frames of 21x21 images."""
    all_files = sorted([f for f in os.listdir(path) if f.endswith('.mat')])
    # Sort numerically to get consistent ordering
    all_files.sort(key=lambda x: int(x.replace('SimulateData_', '').replace('.mat', '')))
    files = all_files[:max_files]
    flush_print(f"  Loading {len(files)} of {len(all_files)} .mat files...")
    data = []
    for i, file in enumerate(files):
        meta_data = loadmat(os.path.join(path, file))
        imgs = np.array(meta_data['SimulateData'])
        data.append([np.array(img[0]) for img in imgs])
        if (i + 1) % 100 == 0:
            flush_print(f"    ...loaded {i+1}/{len(files)}")
    return np.array(data)


def setup_users(data, sample_indices, sizes, seed=42):
    """
    Set up 3 users with different spatial dimensions, matching Chapter 2.
    Input data shape: (n_samples, 10_frames, 21, 21)
    Output user shape: (n_samples, height, width, frames)

    User 1: (n, 21, 21, 10) — raw
    User 2: (n, 21, 30, 10) — right-multiplied by orthogonal 21x30
    User 3: (n, 30, 30, 10) — both left and right multiplied
    """
    rng = np.random.RandomState(seed)
    size1, size2, size3 = sizes

    # User 1: just transpose frames to last axis → (n, 21, 21, 10)
    user1 = np.transpose(data[sample_indices[:size1]], (0, 2, 3, 1))

    # User 2: right-multiply each frame to change width: (21,21) → (21,30)
    # Need a 21x30 matrix to right-multiply: frame(21x21) @ M(21x30) → (21x30)
    M2 = np.linalg.svd(rng.randn(21, 30), full_matrices=True)[2][:21, :].T  # 21x30... no
    # SVD of (21x30): U(21x21) S(21x21) Vt(21x30). Vt[:21,:] is 21x30.
    # We want an orthonormal 21x30 matrix. Use V from SVD of random (30x21):
    # SVD of (30x21) → U(30x30), S(21), Vt(21x21). U[:,:21] is 30x21 orthonormal cols.
    # Simpler: just get orthonormal rows from SVD of (30x30):
    Q2 = np.linalg.qr(rng.randn(30, 21))[0]  # 30x21, orthonormal columns
    # frame (21x21) @ Q2.T (21x30) → (21x30) ✓
    raw2 = data[sample_indices[size1:size1 + size2]]  # (n, 10, 21, 21)
    user2 = raw2 @ Q2.T  # (n, 10, 21, 30)
    user2 = np.transpose(user2, (0, 2, 3, 1))  # (n, 21, 30, 10)

    # User 3: left and right multiply → (30, 30) frames
    Q3_left = np.linalg.qr(rng.randn(30, 21))[0]  # 30x21
    Q3_right = np.linalg.qr(rng.randn(30, 21))[0]  # 30x21
    raw3 = data[sample_indices[size1 + size2:size1 + size2 + size3]]
    # Q3_left.T (21x30)... no. We want (21x21) → (30x30).
    # Q3_left (30x21) @ frame (21x21) → (30x21), then @ Q3_right.T(21x30) → (30x30)
    user3 = Q3_left @ raw3 @ Q3_right.T  # (n, 10, 30, 30)
    user3 = np.transpose(user3, (0, 2, 3, 1))  # (n, 30, 30, 10)

    return user1, user2, user3


# ─── Contamination injection ─────────────────────────────────────────────────

def inject_contamination(user_data, pi_s, rng, noise_multiplier=10.0):
    """
    Contaminate a fraction pi_s of samples with additive Gaussian noise
    at noise_multiplier * baseline_std.
    Returns a copy of the data with contamination applied.
    """
    contaminated = user_data.copy()
    n_samples = contaminated.shape[0]
    n_contaminate = int(np.ceil(pi_s * n_samples))

    if n_contaminate == 0:
        return contaminated

    indices = rng.choice(n_samples, size=n_contaminate, replace=False)
    baseline_std = np.std(user_data)
    noise_std = noise_multiplier * baseline_std

    for idx in indices:
        contaminated[idx] += rng.randn(*contaminated[idx].shape) * noise_std

    return contaminated


# ─── Subspace comparison ─────────────────────────────────────────────────────

def principal_angles_deg(U_ref, U_hat):
    """
    Compute principal angles (in degrees) between subspaces spanned by
    columns of U_ref and U_hat.
    """
    cos_angles = np.linalg.svd(U_ref.T @ U_hat, compute_uv=False)
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    angles_rad = np.arccos(cos_angles)
    return np.degrees(angles_rad)


# ─── Run the pilot ───────────────────────────────────────────────────────────

def run_pilot(data_path, n_repeats=5, max_files=350, seed=2024):
    """
    For each contamination level, run MPCA_FD on clean and contaminated data
    across multiple random splits. Report principal angles.
    """
    pi_s_levels = [0.0, 0.05, 0.10, 0.20, 0.30]
    sizes = [70, 100, 130]  # 3 users, matches Chapter 2 simulation setup
    I_common = [21, 21, 10]  # intermediate dimension
    rank = [5, 5, 4]  # a representative rank for the pilot

    flush_print("Loading simulated data...")
    min_needed = sum(sizes)
    data = load_simulated_data(data_path, max_files=max(max_files, min_needed + 10))
    n_total = len(data)
    flush_print(f"  Loaded {n_total} samples, tensor shape per sample: 21x21x10 (frames)")
    if n_total < min_needed:
        flush_print(f"  ERROR: Need at least {min_needed} samples, got {n_total}")
        sys.exit(1)

    master_rng = np.random.RandomState(seed)

    # Storage: results[pi_s] = list of max_angle across repeats
    results = {pi_s: [] for pi_s in pi_s_levels}

    for rep in range(n_repeats):
        rep_seed = master_rng.randint(1, 100000)
        rng = np.random.RandomState(rep_seed)
        sample = np.arange(n_total)
        rng.shuffle(sample)

        user1, user2, user3 = setup_users(data, sample, sizes, seed=rep_seed)

        # Mean-center (as in Chapter 2)
        user1_c = user1 - np.mean(user1, axis=0)
        user2_c = user2 - np.mean(user2, axis=0)
        user3_c = user3 - np.mean(user3, axis=0)

        # Clean reference: run MPCA_FD (fewer iterations suffice for subspace direction)
        mpca_clean = MPCA_FD(I_common, rank)
        mpca_clean.iterations = 30
        mpca_clean.train(
            copy.deepcopy(user1_c),
            copy.deepcopy(user2_c),
            copy.deepcopy(user3_c)
        )
        U_star = [u.copy() for u in mpca_clean.U_mat]

        for pi_s in pi_s_levels:
            contam_rng = np.random.RandomState(rep_seed + int(pi_s * 1000))

            # Contaminate each user independently
            u1_dirty = inject_contamination(user1_c, pi_s, contam_rng)
            u2_dirty = inject_contamination(user2_c, pi_s, contam_rng)
            u3_dirty = inject_contamination(user3_c, pi_s, contam_rng)

            # Run MPCA_FD on contaminated data
            mpca_dirty = MPCA_FD(I_common, rank)
            mpca_dirty.iterations = 30
            mpca_dirty.train(
                copy.deepcopy(u1_dirty),
                copy.deepcopy(u2_dirty),
                copy.deepcopy(u3_dirty)
            )
            U_hat = mpca_dirty.U_mat

            # Compute max principal angle across all 3 modes
            max_angle = 0.0
            for n in range(3):
                angles = principal_angles_deg(U_star[n], U_hat[n])
                max_angle = max(max_angle, np.max(angles))

            results[pi_s].append(max_angle)

        flush_print(f"  Repeat {rep + 1}/{n_repeats} done.")

    return pi_s_levels, results


def report_results(pi_s_levels, results):
    print("\n" + "=" * 60)
    print("MOTIVATION PILOT RESULTS")
    print("Max principal angle (degrees) between clean U* and contaminated U")
    print("=" * 60)
    print(f"{'pi_S':>8} | {'Mean':>8} | {'Std':>8} | {'Min':>8} | {'Max':>8}")
    print("-" * 60)

    means = []
    stds = []
    for pi_s in pi_s_levels:
        angles = results[pi_s]
        m = np.mean(angles)
        s = np.std(angles)
        means.append(m)
        stds.append(s)
        print(f"{pi_s:>8.2f} | {m:>8.2f} | {s:>8.2f} | {np.min(angles):>8.2f} | {np.max(angles):>8.2f}")

    print("-" * 60)
    target_idx = pi_s_levels.index(0.10)
    target_mean = means[target_idx]
    verdict = "PASSED" if target_mean > 10.0 else "FAILED"
    print(f"\nMotivation test at pi_S=0.10: mean angle = {target_mean:.1f} deg "
          f"(target: >10 deg) --> {verdict}")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    ax.errorbar(pi_s_levels, means, yerr=stds, fmt='o-', capsize=5,
                linewidth=2, markersize=8, color='#2c3e50')
    ax.axhline(10.0, color='red', linestyle='--', linewidth=1.5, label='10° threshold')
    ax.set_xlabel('Contamination fraction $\\pi_S$', fontsize=12)
    ax.set_ylabel('Max principal angle (degrees)', fontsize=12)
    ax.set_title('Subspace sensitivity to sample-level contamination\n'
                 '(Chapter 2 estimator, no robustness)', fontsize=11)
    ax.legend(fontsize=11)
    ax.set_xticks(pi_s_levels)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', 'motivation_pilot_results.png'), dpi=150)
    plt.show()
    print("\nPlot saved to motivation_pilot_results.png")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default=None)
    parser.add_argument('--n-repeats', type=int, default=10)
    parser.add_argument('--max-files', type=int, default=350)
    args = parser.parse_args()

    if args.data_path:
        data_path = args.data_path
    else:
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '..', 'data', 'Simulated Data')

    if not os.path.exists(data_path):
        flush_print(f"ERROR: Data path not found: {data_path}")
        sys.exit(1)

    pi_s_levels, results = run_pilot(data_path, n_repeats=args.n_repeats,
                                     max_files=args.max_files, seed=2024)
    report_results(pi_s_levels, results)
