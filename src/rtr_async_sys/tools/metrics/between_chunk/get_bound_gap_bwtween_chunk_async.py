import pickle
import numpy as np
import argparse
from tqdm import tqdm

RAD2DEG = 180.0 / np.pi

def l1_norm(x: np.ndarray, axis=-1) -> np.ndarray:
    """L1 norm (Manhattan distance) along given axis."""
    return np.sum(np.abs(x), axis=axis)

def summarize(name: str, arr: np.ndarray):
    if arr.size == 0:
        return f"{name}: empty"
    return (f"{name}: mean={arr.mean():.6g}, p95={np.percentile(arr,95):.6g}, "
            f"max={arr.max():.6g}, n={arr.size}")

def remove_outliers_p99(arr: np.ndarray):
    """Remove outliers above p99."""
    if arr.size == 0:
        return arr, 0, None
    p99 = np.percentile(arr, 99)
    mask = arr <= p99
    removed = int(arr.size - mask.sum())
    return arr[mask], removed, p99

def is_switch_traj(prev_predict: np.ndarray, curr_predict: np.ndarray) -> bool:
    # Treat prev start < 0 and curr start > 0 as a trajectory switch
    return (prev_predict[0][0] < 0) and (curr_predict[0][0] > 0)

def d0_async_l1(prev_seq: np.ndarray, cur_seq: np.ndarray, async_num: int) -> float:
    """
    D0 only, L1 distance to previous chunk's last point:
      D0 = || cur[async_num] - prev[-1] ||_1
    """
    d0_vec = cur_seq[async_num] - prev_seq[-1]
    return float(l1_norm(d0_vec, axis=-1))

parser = argparse.ArgumentParser(
    description="Inter-chunk boundary D0 (ASYNC mode): L1 distance between cur[async_num] and prev[-1], separately for xyz(mm) and rpy(deg)"
)
parser.add_argument("--pkl_path", type=str, default="outputs/vis_outputs/dp_plot_actions.pkl")
parser.add_argument("--async_num", type=int, default=24)
parser.add_argument("--xyz_scale_mm", action="store_true", help="Scale xyz (0:3) by 1000 (m->mm)")
parser.add_argument("--rpy_to_deg", action="store_true", help="Convert rpy (3:6) from rad->deg")
args = parser.parse_args()

pkl_path = args.pkl_path
async_num = args.async_num

with open(pkl_path, "rb") as f:
    plot_actions = pickle.load(f)

D0_xyz = []
D0_rpy = []

num_switch = 0
num_skip_short = 0
num_skip_dim = 0
num_pairs = 0

prev_predict_full = None

for i, step in enumerate(tqdm(plot_actions, desc="Processing chunks (async D0 only)")):
    curr_predict_full = np.asarray(step["predict"], dtype=np.float32)  # [H, D]
    if curr_predict_full.ndim != 2:
        prev_predict_full = curr_predict_full
        continue

    # need at least 6 dims for xyz+rpy
    if curr_predict_full.shape[1] < 6:
        num_skip_dim += 1
        prev_predict_full = curr_predict_full
        continue

    # first chunk init
    if prev_predict_full is None:
        prev_predict_full = curr_predict_full
        continue

    # switch traj
    if is_switch_traj(prev_predict_full, curr_predict_full):
        num_switch += 1
        prev_predict_full = curr_predict_full
        continue

    # length check: need current has index async_num, and prev has last step
    if curr_predict_full.shape[0] <= async_num or prev_predict_full.shape[0] < 1:
        num_skip_short += 1
        prev_predict_full = curr_predict_full
        continue

    # slice xyz / rpy
    prev_xyz = prev_predict_full[:, 0:3].copy()
    cur_xyz = curr_predict_full[:, 0:3].copy()

    prev_rpy = prev_predict_full[:, 3:6].copy()
    cur_rpy = curr_predict_full[:, 3:6].copy()

    # unit conversion
    if args.xyz_scale_mm:
        prev_xyz *= 1000.0
        cur_xyz *= 1000.0
    if args.rpy_to_deg:
        prev_rpy *= RAD2DEG
        cur_rpy *= RAD2DEG

    # D0 only (L1)
    D0_xyz.append(d0_async_l1(prev_xyz, cur_xyz, async_num=async_num))
    D0_rpy.append(d0_async_l1(prev_rpy, cur_rpy, async_num=async_num))

    num_pairs += 1
    prev_predict_full = curr_predict_full

D0_xyz = np.asarray(D0_xyz, dtype=np.float32)
D0_rpy_raw = np.asarray(D0_rpy, dtype=np.float32)

# Remove p99 outliers from rpy D0
D0_rpy, removed_rpy, rpy_p99 = remove_outliers_p99(D0_rpy_raw)

print("\n=== Inter-chunk Boundary D0 only (ASYNC mode, L1 distance) ===")
print(f"pkl_path: {pkl_path}")
print(f"num_chunks: {len(plot_actions)}")
print(f"async_num: {async_num}")
print(f"traj_switch_detected: {num_switch}")
print(f"skip_short_pairs: {num_skip_short}")
print(f"skip_dim<6: {num_skip_dim}")
print(f"valid_boundary_pairs: {num_pairs}")
print(f"xyz unit: {'mm' if args.xyz_scale_mm else 'raw'} | rpy unit: {'deg' if args.rpy_to_deg else 'rad'}")
print(f"rpy D0 outliers removed (>p99): {removed_rpy} / {D0_rpy_raw.size}")

print("\n[XYZ D0 | async anchor = cur[async_num]]")
print(summarize("D0_xyz_async_L1", D0_xyz))

print("\n[RPY D0 | async anchor = cur[async_num]]  (after removing >p99)")
print(summarize("D0_rpy_async_L1", D0_rpy))
