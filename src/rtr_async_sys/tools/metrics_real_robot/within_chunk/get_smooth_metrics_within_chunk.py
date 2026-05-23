import os
import pickle
import numpy as np
import argparse
from tqdm import tqdm

RAD2DEG = 180.0 / np.pi
MIN_LATENCY_SEC = 7.0  # Only trajectories >= 10s count toward latency

def l2_norm(x: np.ndarray, axis=-1, eps=1e-12) -> np.ndarray:
    """Compute L2 norm with numerical stability."""
    return np.sqrt(np.sum(x * x, axis=axis) + eps)

def chunk_diff_metrics(seq: np.ndarray):
    """
    seq: [H, D]
    returns:
      a: [H-2]  acceleration magnitude per step (2nd diff)
      j: [H-3]  jerk magnitude per step (3rd diff)
    """
    # 2nd diff (accel)
    if seq.shape[0] >= 3:
        d2 = np.diff(seq, n=2, axis=0)      # [H-2, D]
        a = l2_norm(d2, axis=-1)            # [H-2]
    else:
        a = np.array([], dtype=np.float32)

    # 3rd diff (jerk)
    if seq.shape[0] >= 4:
        d3 = np.diff(seq, n=3, axis=0)      # [H-3, D]
        j = l2_norm(d3, axis=-1)            # [H-3]
    else:
        j = np.array([], dtype=np.float32)

    return a, j

def summarize(name: str, arr: np.ndarray):
    if arr.size == 0:
        return f"{name}: empty"
    return (f"{name}: mean={arr.mean():.6g}, p95={np.percentile(arr,95):.6g}, "
            f"max={arr.max():.6g}, n={arr.size}")

def remove_outliers_p99(arr: np.ndarray):
    """Remove outliers above p99;arr is usually non-negative (norm)."""
    if arr.size == 0:
        return arr, 0
    p99 = np.percentile(arr, 99)
    mask = arr <= p99
    removed = int(arr.size - mask.sum())
    return arr[mask], removed

parser = argparse.ArgumentParser(
    description="Trajectory-level accel & jerk metrics for xyz(mm) and rpy(deg) over multiple traj_*.pkl"
)
parser.add_argument(
    "--traj_dir",
    type=str,
    default="xarm_outputs/dp/traj/peel_cucumber/dp60hz_async0",
    help="directory containing traj_*.pkl files (each is a list of (action, time))",
)
parser.add_argument(
    "--x_threshold_mm",
    type=float,
    default=-210.0,
    help="end-of-episode x-threshold in mm; end time = first action whose x(mm) < this value",
)
args = parser.parse_args()

traj_dir = args.traj_dir
x_threshold_mm = args.x_threshold_mm
x_threshold_m = x_threshold_mm / 1000.0  # raw action is in meters

# Unit conversion switch (fixed to True here)
xyz_to_mm = True
rpy_to_deg = True

if not os.path.isdir(traj_dir):
    raise NotADirectoryError(f"{traj_dir} is not a directory")

# Find all traj_*.pkl files
traj_files = [
    os.path.join(traj_dir, f)
    for f in os.listdir(traj_dir)
    if f.startswith("traj_") and f.endswith(".pkl")
]
traj_files = sorted(traj_files)

if len(traj_files) == 0:
    raise FileNotFoundError(f"No traj_*.pkl found in directory: {traj_dir}")

# Global accumulators
acc_xyz_all, jerk_xyz_all = [], []
acc_rpy_all, jerk_rpy_all = [], []
latency_list = []  # end2end latency (seconds), only records trajectories >= MIN_LATENCY_SEC

num_traj_used = 0
skip_no_action = 0
skip_dim = 0
skip_short_for_smoothness = 0
total_T = 0

latency_short_count = 0          # number of trajectories shorter than MIN_LATENCY_SEC
latency_no_threshold_count = 0   # number of trajectories without x < threshold; fall back to the last action time

for pkl_path in tqdm(traj_files, desc="Processing trajectories"):
    with open(pkl_path, "rb") as f:
        traj_list = pickle.load(f)

    # traj_list should be [(action, time), (action, time), ...]
    if not isinstance(traj_list, (list, tuple)) or len(traj_list) == 0:
        skip_no_action += 1
        continue

    actions = []
    times = []

    for item in traj_list:
        # Follow the collection format exactly: first item is action, second item is time
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        act, t = item[0], item[1]

        act = np.asarray(act, dtype=np.float32)
        if act.ndim == 1 and act.shape[0] >= 6:
            actions.append(act[:6])
            times.append(float(t))
        elif act.ndim == 2 and act.shape[1] >= 6:
            # Should not happen normally; fall back to the first line
            actions.append(act[0, :6])
            times.append(float(t))
        else:
            # Skip entries with unexpected dimensions.
            continue

    if len(actions) == 0:
        skip_no_action += 1
        continue

    # [T, 6]
    traj = np.stack(actions, axis=0)
    T = traj.shape[0]
    total_T += T
    num_traj_used += 1

    # -------- Compute end-to-end latency with a custom termination condition --------
    if len(times) >= 2:
        start_t = times[0]
        # Raw actions are in meters; compare x in meters against x_threshold_m.
        x_m = traj[:, 0]  # [T]
        idxs = np.where(x_m < x_threshold_m)[0]

        if idxs.size > 0:
            end_idx = int(idxs[0])
            end_t = times[end_idx]
        else:
            # If no step satisfies x < threshold, fall back to the final timestamp.
            end_t = times[-1]
            latency_no_threshold_count += 1

        latency = float(end_t - start_t)

        if latency >= MIN_LATENCY_SEC:
            latency_list.append(latency)
        else:
            latency_short_count += 1

    if traj.shape[1] < 6:
        skip_dim += 1
        continue

    xyz = traj[:, 0:3].copy()
    rpy = traj[:, 3:6].copy()

    # unit conversion
    if xyz_to_mm:
        xyz *= 1000.0           # m -> mm
    if rpy_to_deg:
        rpy *= RAD2DEG          # rad -> deg

    # Smoothness is empty for trajectories shorter than 3/4 samples, but latency is still recorded.
    if T < 3:
        skip_short_for_smoothness += 1
        continue

    a_xyz, j_xyz = chunk_diff_metrics(xyz)
    a_rpy, j_rpy = chunk_diff_metrics(rpy)

    acc_xyz_all.append(a_xyz)
    jerk_xyz_all.append(j_xyz)
    acc_rpy_all.append(a_rpy)
    jerk_rpy_all.append(j_rpy)

# Concatenate all trajectory results.
acc_xyz = np.concatenate(acc_xyz_all) if len(acc_xyz_all) else np.array([], dtype=np.float32)
jerk_xyz = np.concatenate(jerk_xyz_all) if len(jerk_xyz_all) else np.array([], dtype=np.float32)

acc_rpy_raw = np.concatenate(acc_rpy_all) if len(acc_rpy_all) else np.array([], dtype=np.float32)
jerk_rpy_raw = np.concatenate(jerk_rpy_all) if len(jerk_rpy_all) else np.array([], dtype=np.float32)

# Remove outliers in rpy accel/jerk with p99 clipping.
acc_rpy, removed_acc = remove_outliers_p99(acc_rpy_raw)
jerk_rpy, removed_jerk = remove_outliers_p99(jerk_rpy_raw)

latency_arr = np.asarray(latency_list, dtype=np.float32) if len(latency_list) else np.array([], dtype=np.float32)

print("\n=== Trajectory Smoothness (predict-only): Accel & Jerk over multiple trajectories ===")
print(f"traj_dir: {traj_dir}")
print(f"#traj_files (traj_*.pkl found): {len(traj_files)}")
print(f"x-threshold for latency end: {x_threshold_mm} mm (i.e. {x_threshold_m} m, condition: x < threshold)")
print(f"latency threshold: >= {MIN_LATENCY_SEC} s")

print("[End-to-end Latency] (seconds, only traj >= threshold)")
print(summarize("latency", latency_arr))

print("[XYZ]")
print(summarize("jerk_xyz   |Δ³p|", jerk_xyz))

print("[RPY]  (after removing >p99 outliers)")
print(summarize("jerk_rpy   |Δ³rpy|", jerk_rpy))

# ====== Print xyz_jerk and rpy_jerk ======
if jerk_xyz.size > 0:
    xyz_jerk_mean = float(jerk_xyz.mean())
else:
    xyz_jerk_mean = 0.0

if jerk_rpy.size > 0:
    rpy_jerk_mean = float(jerk_rpy.mean())
else:
    rpy_jerk_mean = 0.0

combined_jerk = xyz_jerk_mean + rpy_jerk_mean

# print("\n[Combined Jerk]")
# print(f"mean(jerk_xyz) = {xyz_jerk_mean:.6g}")
# print(f"mean(jerk_rpy) = {rpy_jerk_mean:.6g}")
# print(f"mean(jerk_xyz) + mean(jerk_rpy) = {combined_jerk:.6g}")
