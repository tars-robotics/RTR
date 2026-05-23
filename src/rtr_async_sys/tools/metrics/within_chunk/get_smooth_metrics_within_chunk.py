import pickle
import numpy as np
import argparse
from tqdm import tqdm

RAD2DEG = 180.0 / np.pi

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

parser = argparse.ArgumentParser(description="Chunk-level accel & jerk metrics for xyz(mm) and rpy(deg)")
parser.add_argument("--pkl_path", type=str, default="outputs/vis_outputs/dp_plot_actions.pkl")
# parser.add_argument("--xyz_to_mm", action="store_true", help="Convert xyz (0:3) from meters to mm")
# parser.add_argument("--rpy_to_deg", action="store_true", help="Convert rpy (3:6) from rad to degree")
args = parser.parse_args()

xyz_to_mm = True
rpy_to_deg = True

pkl_path = args.pkl_path

with open(pkl_path, "rb") as f:
    plot_actions = pickle.load(f)

# Collectors
acc_xyz_all, jerk_xyz_all = [], []
acc_rpy_all, jerk_rpy_all = [], []

skip_dim = 0
skip_short = 0

for i, step in enumerate(tqdm(plot_actions, desc="Processing chunks (intra-chunk accel/jerk)")):
    predict = np.asarray(step["predict"], dtype=np.float32)  # [H, D]
    if predict.ndim != 2:
        continue
    if predict.shape[1] < 6:
        skip_dim += 1
        continue

    xyz = predict[:, 0:3].copy()
    rpy = predict[:, 3:6].copy()

    # unit conversion
    if xyz_to_mm:
        xyz *= 1000.0
    if rpy_to_deg:
        rpy *= RAD2DEG

    # need length >= 3 for accel, >=4 for jerk
    if xyz.shape[0] < 3:
        skip_short += 1
        continue

    a_xyz, j_xyz = chunk_diff_metrics(xyz)
    a_rpy, j_rpy = chunk_diff_metrics(rpy)

    acc_xyz_all.append(a_xyz)
    jerk_xyz_all.append(j_xyz)

    acc_rpy_all.append(a_rpy)
    jerk_rpy_all.append(j_rpy)

# concat
acc_xyz = np.concatenate(acc_xyz_all) if len(acc_xyz_all) else np.array([], dtype=np.float32)
jerk_xyz = np.concatenate(jerk_xyz_all) if len(jerk_xyz_all) else np.array([], dtype=np.float32)

acc_rpy_raw = np.concatenate(acc_rpy_all) if len(acc_rpy_all) else np.array([], dtype=np.float32)
jerk_rpy_raw = np.concatenate(jerk_rpy_all) if len(jerk_rpy_all) else np.array([], dtype=np.float32)

# Remove p99 outliers from rpy accel / jerk
acc_rpy, removed_acc = remove_outliers_p99(acc_rpy_raw)
jerk_rpy, removed_jerk = remove_outliers_p99(jerk_rpy_raw)

print("\n=== Intra-chunk Smoothness (predict-only): Accel & Jerk ===")
print(f"pkl_path: {pkl_path}")
print(f"num_chunks: {len(plot_actions)}")
print(f"skip_dim<6: {skip_dim} | skip_short(<3): {skip_short}")
print(f"xyz unit: {'mm' if xyz_to_mm else 'raw'} | rpy unit: {'deg' if rpy_to_deg else 'rad'}")
print(f"rpy accel outliers removed (>p99): {removed_acc} / {acc_rpy_raw.size}")
print(f"rpy jerk  outliers removed (>p99): {removed_jerk} / {jerk_rpy_raw.size}")

print("\n[XYZ]")
print(summarize("accel_xyz  |Δ²p|", acc_xyz))
print(summarize("jerk_xyz   |Δ³p|", jerk_xyz))

print("\n[RPY]  (after removing >p99 outliers)")
print(summarize("accel_rpy  |Δ²rpy|", acc_rpy))
print(summarize("jerk_rpy   |Δ³rpy|", jerk_rpy))
