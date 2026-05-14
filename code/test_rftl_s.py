"""Quick smoke test for RFTL-S."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.io import loadmat
from rftl_s import RFTL_S, MPCA_FD_Weighted, reconstruction_residual

print("Testing RFTL-S with small data...", flush=True)

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'Simulated Data')
all_files = sorted(os.listdir(path), key=lambda x: int(x.replace("SimulateData_", "").replace(".mat", "")))
data = []
for f in all_files[:60]:
    meta = loadmat(os.path.join(path, f))
    imgs = np.array(meta["SimulateData"])
    data.append([np.array(img[0]) for img in imgs])
data = np.array(data)
print(f"Data loaded: {data.shape}", flush=True)

rng = np.random.RandomState(42)
sample = np.arange(len(data))
rng.shuffle(sample)
user1 = np.transpose(data[sample[:15]], (0, 2, 3, 1))
user2 = np.transpose(data[sample[15:30]], (0, 2, 3, 1))
user3 = np.transpose(data[sample[30:45]], (0, 2, 3, 1))
users = [u - np.mean(u, 0) for u in [user1, user2, user3]]
print(f"User shapes: {[u.shape for u in users]}", flush=True)

# Contaminate first 3 samples of user 0
n_contam = 3
noise_std = 10 * np.std(users[0])
users_dirty = [u.copy() for u in users]
users_dirty[0][:n_contam] += rng.randn(*users_dirty[0][:n_contam].shape) * noise_std
print(f"Contaminated first {n_contam} samples of user 0", flush=True)

# Test weighted model
print("Testing MPCA_FD_Weighted...", flush=True)
weights = [np.ones(u.shape[0]) for u in users_dirty]
model = MPCA_FD_Weighted([21, 21, 10], [5, 5, 4], iterations=10)
prime, U_mat, V_mat = model.train([u.copy() for u in users_dirty], weights)
print(f"  U shapes: {[u.shape for u in U_mat]}", flush=True)
print(f"  Prime shape: {prime.shape}", flush=True)

# Test residuals
print("Testing reconstruction residuals...", flush=True)
for m in range(3):
    r = reconstruction_residual(users_dirty[m], V_mat[m], U_mat)
    print(f"  User {m}: min={r.min():.2f} max={r.max():.2f} mean={r.mean():.2f}", flush=True)

# Test full RFTL-S (cold re-fit)
print("\nTesting RFTL_S.fit() with rank [5,5,4]...", flush=True)
rftl = RFTL_S([21, 21, 10], [5, 5, 4], threshold_c=3.0, inner_iterations=10)
rftl.fit(users_dirty)
print(f"  Flagged {rftl.n_flagged} samples, threshold={rftl.threshold:.2f}", flush=True)
print(f"  User 0 weights (contaminated 0-2, clean 3+):", flush=True)
for i in range(min(6, len(rftl.weights[0]))):
    tag = "contaminated" if i < n_contam else "clean"
    print(f"    idx {i}: {rftl.weights[0][i]:.0f}  ({tag})", flush=True)
print(f"  User 1 mean weight: {np.mean(rftl.weights[1]):.3f} (all clean)", flush=True)
print(f"  User 2 mean weight: {np.mean(rftl.weights[2]):.3f} (all clean)", flush=True)
print("SUCCESS", flush=True)
