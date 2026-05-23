"""
get_delta_between_chunk_xyz_rpy (async overlap)

Compute delta (MAE) over the overlap region between adjacent action chunks:
- fact:   previous step predict[-async_num:, :]
- pred:   current step predict[:async_num, :]

Output:
- delta_xyz: xyz MAE (unit: mm, m -> mm)
- delta_rpy: rpy (roll/pitch/yaw) MAE (unit: degrees, rad -> degrees)
Also print the per-dimension means (dx/dy/dz, droll/dpitch/dyaw).
"""

import pickle
import numpy as np
import argparse
from tqdm import tqdm

RAD2DEG = 180.0 / np.pi

parser = argparse.ArgumentParser(description="get_delta_between_chunk_xyz_rpy (MAE only)")
parser.add_argument("--pkl_path", type=str, default="outputs/vis_outputs/dp_plot_actions.pkl")
parser.add_argument("--async_num", type=int, default=24)
args = parser.parse_args()

pkl_path = args.pkl_path
async_num = args.async_num

with open(pkl_path, "rb") as f:
    plot_actions = pickle.load(f)

# xyz abs deltas (mm)
dx_list, dy_list, dz_list = [], [], []
# rpy abs deltas (deg)
dr_list, dp_list, dyaw_list = [], [], []

prev_predict = None
num_pairs = 0
num_skip_short = 0
num_skip_dim = 0
num_switch = 0

def remove_outliers_p99(arr: np.ndarray):
    """Remove outliers above p99."""
    if arr.size == 0:
        return arr, 0, None
    p99 = np.percentile(arr, 99)
    mask = arr <= p99
    removed = int(arr.size - mask.sum())
    return arr[mask], removed, p99

for i, step in enumerate(tqdm(plot_actions, desc="Processing steps")):
    curr_predict = np.asarray(step["predict"], dtype=np.float32)  # [N, D]

    if prev_predict is None:
        prev_predict = curr_predict
        continue

    # shape / dim check
    if prev_predict.ndim != 2 or curr_predict.ndim != 2:
        prev_predict = curr_predict
        continue
    if prev_predict.shape[1] < 6 or curr_predict.shape[1] < 6:
        num_skip_dim += 1
        prev_predict = curr_predict
        continue

    # length check
    if prev_predict.shape[0] < async_num or curr_predict.shape[0] < async_num:
        num_skip_short += 1
        prev_predict = curr_predict
        continue

    # switch traj
    if prev_predict[0][0] < 0 and curr_predict[0][0] > 0:
        num_switch += 1
        prev_predict = curr_predict
        continue

    # overlap slices (use first 6 dims)
    fact = prev_predict[-async_num:, :6].copy()   # [async_num, 6]
    pred = curr_predict[:async_num, :6].copy()    # [async_num, 6]

    # unit conversion
    # xyz: m -> mm
    fact[:, 0:3] *= 1000.0
    pred[:, 0:3] *= 1000.0
    # rpy: rad -> deg
    fact[:, 3:6] *= RAD2DEG
    pred[:, 3:6] *= RAD2DEG

    diff = np.abs(fact - pred)  # [async_num, 6]

    dx_list.extend(diff[:, 0].tolist())
    dy_list.extend(diff[:, 1].tolist())
    dz_list.extend(diff[:, 2].tolist())

    dr_list.extend(diff[:, 3].tolist())
    dp_list.extend(diff[:, 4].tolist())
    dyaw_list.extend(diff[:, 5].tolist())

    num_pairs += 1
    prev_predict = curr_predict

if len(dx_list) == 0:
    print(
        f"pkl_path is {pkl_path}, but no valid chunk pairs were found "
        f"(async_num={async_num}, skip_dim={num_skip_dim}, skip_short={num_skip_short}, switch={num_switch})."
    )
else:
    def mean(x):
        arr = np.asarray(x, dtype=np.float32)
        if arr.size == 0:
            return 0.0
        return float(arr.mean())

    # ========== xyz:no clipping ==========
    dx_arr = np.asarray(dx_list, dtype=np.float32)
    dy_arr = np.asarray(dy_list, dtype=np.float32)
    dz_arr = np.asarray(dz_list, dtype=np.float32)

    delta_xyz = float(np.mean(np.concatenate([dx_arr, dy_arr, dz_arr])))

    # ========== rpy:clip at p99 before computing statistics ==========
    dr_arr = np.asarray(dr_list, dtype=np.float32)
    dp_arr = np.asarray(dp_list, dtype=np.float32)
    dyaw_arr = np.asarray(dyaw_list, dtype=np.float32)

    dr_clean, removed_dr, dr_p99 = remove_outliers_p99(dr_arr)
    dp_clean, removed_dp, dp_p99 = remove_outliers_p99(dp_arr)
    dyaw_clean, removed_dyaw, dyaw_p99 = remove_outliers_p99(dyaw_arr)

    droll_mean = float(dr_clean.mean()) if dr_clean.size > 0 else 0.0
    dpitch_mean = float(dp_clean.mean()) if dp_clean.size > 0 else 0.0
    dyaw_mean = float(dyaw_clean.mean()) if dyaw_clean.size > 0 else 0.0

    if dr_clean.size + dp_clean.size + dyaw_clean.size > 0:
        delta_rpy = float(
            np.mean(
                np.concatenate([dr_clean, dp_clean, dyaw_clean])
            )
        )
    else:
        delta_rpy = 0.0

    print(
        f"[get_delta_between_chunk_xyz_rpy] pkl_path={pkl_path}\n"
        f"  chunk_pairs={num_pairs} (async_num={async_num}) | switch={num_switch} | skip_dim={num_skip_dim} | skip_short={num_skip_short}\n"
        f"  delta_xyz (MAE over xyz, mm) = {delta_xyz:.6g} | "
        f"per-dim: dx={dx_arr.mean():.6g}, dy={dy_arr.mean():.6g}, dz={dz_arr.mean():.6g}\n"
        f"  delta_rpy (MAE over rpy, deg, >p99 removed) = {delta_rpy:.6g} | "
        f"per-dim: droll={droll_mean:.6g}, dpitch={dpitch_mean:.6g}, dyaw={dyaw_mean:.6g}"
    )

    print(
        f"\n[rpy overlap deltas outliers removed (>p99 per-dim)] "
        f"roll: {removed_dr} / {dr_arr.size}, "
        f"pitch: {removed_dp} / {dp_arr.size}, "
        f"yaw: {removed_dyaw} / {dyaw_arr.size}"
    )
