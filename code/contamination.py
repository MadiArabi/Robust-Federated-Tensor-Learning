"""
Contamination models for RFTL experiments.

Three modes:
  - 'gaussian':  additive isotropic Gaussian noise per contaminated sample
                 (the original model — kept for comparison; shown in the
                 June 2026 HPC run to be harmless to downstream prediction)
  - 'hot_block': a b x b pixel block saturated to a fixed hot value,
                 same location in every frame and every contaminated sample
                 (defective IR sensor: cluster of hot/dead pixels)
  - 'stripe':    additive high-amplitude offset on a fixed set of rows or
                 columns, constant across frames and contaminated samples
                 (IR fixed-pattern / readout-stripe noise)

Design rationale: hot_block and stripe use a pattern drawn ONCE per call and
applied identically to all contaminated samples. A faulty camera produces the
same artifact in every image — and correlated contamination looks like real
structure to MPCA, dragging leading factors toward the artifact instead of
averaging out the way independent Gaussian noise does.

Federated scenario: by default contamination is concentrated in a single user
(`target_user=0`, the "faulty camera site"). Pass `target_user=None` to spread
pi_s across all users (the original behavior).
"""

import numpy as np


def _draw_pattern(mode, rng, H, W, block_size, n_stripes):
    """Draw the fixed artifact pattern (location/orientation) for one camera."""
    if mode == 'gaussian':
        return {'mode': 'gaussian'}
    elif mode == 'hot_block':
        b = min(block_size, H, W)
        return {'mode': 'hot_block', 'b': b,
                'r0': rng.randint(0, H - b + 1),
                'c0': rng.randint(0, W - b + 1)}
    elif mode == 'stripe':
        along_rows = bool(rng.rand() < 0.5)
        n_lines = min(n_stripes, H if along_rows else W)
        lines = rng.choice(H if along_rows else W, size=n_lines, replace=False)
        return {'mode': 'stripe', 'along_rows': along_rows, 'lines': lines}
    else:
        raise ValueError(f"Unknown contamination mode: {mode}")


def _apply_pattern(dirty, idx, pattern, rng, amplitude, ref_std, ref_max):
    """Apply a drawn pattern to samples idx of dirty (in place)."""
    if pattern['mode'] == 'gaussian':
        noise_std = amplitude * ref_std
        for i in idx:
            dirty[i] += rng.randn(*dirty[i].shape) * noise_std
    elif pattern['mode'] == 'hot_block':
        b, r0, c0 = pattern['b'], pattern['r0'], pattern['c0']
        hot_value = ref_max + amplitude * ref_std
        for i in idx:
            dirty[i, r0:r0 + b, c0:c0 + b, :] = hot_value
    elif pattern['mode'] == 'stripe':
        offset = amplitude * ref_std
        for i in idx:
            if pattern['along_rows']:
                dirty[i, pattern['lines'], :, :] += offset
            else:
                dirty[i, :, pattern['lines'], :] += offset


def _contaminate_one_user(dirty, n_contam, mode, rng, amplitude,
                          block_size, n_stripes, ref_std, ref_max):
    """Contaminate n_contam samples of one user's array in place.

    dirty: (n_samples, H, W, F) array, modified in place.
    Returns the set of contaminated sample indices.

    RNG call order (sample choice first, then pattern draw) must stay as-is
    so results from the July 2026 gate runs remain reproducible.
    """
    n_samples, H, W, _ = dirty.shape
    idx = rng.choice(n_samples, size=n_contam, replace=False)
    pattern = _draw_pattern(mode, rng, H, W, block_size, n_stripes)
    _apply_pattern(dirty, idx, pattern, rng, amplitude, ref_std, ref_max)
    return set(idx)


def contaminate(train_users, pi_s, mode, rng, amplitude=5.0,
                block_size=5, n_stripes=3, target_user=0):
    """
    Contaminate training data. Returns (train_dirty, contam_indices).

    train_users:   list of per-user arrays, each (n_samples, H, W, F)
    pi_s:          fraction of the targeted user's samples to contaminate
                   (or of every user's samples if target_user is None)
    mode:          'gaussian' | 'hot_block' | 'stripe'
    rng:           np.random.RandomState (pattern + sample choice reproducible)
    amplitude:     artifact strength in units of the user's clean data std
    block_size:    hot_block edge length in pixels
    n_stripes:     number of stripe lines
    target_user:   index of the single contaminated user (faulty-camera
                   scenario), or None to contaminate all users

    contam_indices is a list of sets (one per user) of contaminated sample
    indices — same format the detection metrics in rftl_s_real expect.
    """
    train_dirty = []
    contam_indices = []
    for m, clean in enumerate(train_users):
        dirty = clean.copy()
        if target_user is None or m == target_user:
            n_contam = int(np.ceil(pi_s * dirty.shape[0]))
        else:
            n_contam = 0
        if n_contam > 0:
            flagged = _contaminate_one_user(
                dirty, n_contam, mode, rng, amplitude,
                block_size, n_stripes,
                ref_std=float(np.std(clean)), ref_max=float(np.max(clean)),
            )
        else:
            flagged = set()
        train_dirty.append(dirty)
        contam_indices.append(flagged)
    return train_dirty, contam_indices


def contaminate_train_and_test(train_users, test_users, pi_s, mode, rng,
                               amplitude=5.0, block_size=5, n_stripes=3,
                               target_user=0, pi_s_test=None):
    """
    Faulty-camera scenario where the camera corrupts BOTH train and test
    images: one pattern is drawn per contaminated user and applied
    identically to pi_s of its train samples and pi_s_test of its test
    samples (default: same rate as train — the fault fires at the same
    frequency whenever the camera is used).

    Artifact value/scale (ref_std, ref_max) is computed from the user's
    clean TRAIN data only — the artifact is a physical property of the
    camera, and test statistics must not leak into it.

    Returns (train_dirty, train_indices, test_dirty, test_indices) with
    the same list-of-sets index format as contaminate().
    """
    if pi_s_test is None:
        pi_s_test = pi_s
    train_dirty, test_dirty = [], []
    train_indices, test_indices = [], []
    for m, (clean_tr, clean_te) in enumerate(zip(train_users, test_users)):
        dirty_tr, dirty_te = clean_tr.copy(), clean_te.copy()
        flagged_tr, flagged_te = set(), set()
        if target_user is None or m == target_user:
            ref_std = float(np.std(clean_tr))
            ref_max = float(np.max(clean_tr))
            _, H, W, _ = dirty_tr.shape
            pattern = _draw_pattern(mode, rng, H, W, block_size, n_stripes)
            n_tr = int(np.ceil(pi_s * dirty_tr.shape[0]))
            n_te = int(np.ceil(pi_s_test * dirty_te.shape[0]))
            if n_tr > 0:
                idx = rng.choice(dirty_tr.shape[0], size=n_tr, replace=False)
                _apply_pattern(dirty_tr, idx, pattern, rng, amplitude,
                               ref_std, ref_max)
                flagged_tr = set(idx)
            if n_te > 0:
                idx = rng.choice(dirty_te.shape[0], size=n_te, replace=False)
                _apply_pattern(dirty_te, idx, pattern, rng, amplitude,
                               ref_std, ref_max)
                flagged_te = set(idx)
        train_dirty.append(dirty_tr)
        test_dirty.append(dirty_te)
        train_indices.append(flagged_tr)
        test_indices.append(flagged_te)
    return train_dirty, train_indices, test_dirty, test_indices
