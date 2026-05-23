import os
import pickle
import numpy as np
import argparse
from tqdm import tqdm

RAD2DEG = 180.0 / np.pi

parser = argparse.ArgumentParser(description="Chunk-level delta (MAE) for xyz(mm) and rpy(deg)")
parser.add_argument("--mode", type=int, default=1, help="kept for compatibility; script only computes delta")
parser.add_argument("--pkl_path", type=str, default="outputs/vis_outputs/dp_plot_actions.pkl")
parser.add_argument("--output_dir", type=str, default="outputs/vis_outputs/dp_plot_xyz")
args = parser.parse_args()

pkl_path = args.pkl_path

def remove_outliers_p99(arr: np.ndarray):
    """Remove outliers above p99."""
    if arr.size == 0:
        return arr, 0, None
    p99 = np.percentile(arr, 99)
    mask = arr <= p99
    removed = int(arr.size - mask.sum())
    return arr[mask], removed, p99

# ========= Load pkl =========
with open(pkl_path, "rb") as f:
    plot_actions = pickle.load(f)

# xyz abs errors (mm)
dx_list, dy_list, dz_list = [], [], []
# rpy abs errors (deg)
dr_list, dp_list, dyaw_list = [], [], []

# ========= Iterate over each chunk =========
for i, step in enumerate(tqdm(plot_actions, desc="Processing steps")):
    fact = np.asarray(step["fact"], dtype=np.float32)       # [H, D]
    predict = np.asarray(step["predict"], dtype=np.float32) # [H, D]

    if fact.ndim != 2 or predict.ndim != 2:
        continue
    if fact.shape[1] < 6 or predict.shape[1] < 6:
        continue
    if fact.shape[0] == 0 or predict.shape[0] == 0:
        continue

    # Align lengths using the shortest sequence to avoid occasional mismatches
    H = min(fact.shape[0], predict.shape[0])
    fact = fact[:H, :6].copy()
    predict = predict[:H, :6].copy()

    # --- unit conversion ---
    # xyz: m -> mm
    fact[:, 0:3] *= 1000.0
    predict[:, 0:3] *= 1000.0
    # rpy: rad -> degree
    fact[:, 3:6] *= RAD2DEG
    predict[:, 3:6] *= RAD2DEG

    # --- abs delta ---
    diff = np.abs(fact - predict)  # [H, 6]

    dx_list.extend(diff[:, 0].tolist())
    dy_list.extend(diff[:, 1].tolist())
    dz_list.extend(diff[:, 2].tolist())

    dr_list.extend(diff[:, 3].tolist())
    dp_list.extend(diff[:, 4].tolist())
    dyaw_list.extend(diff[:, 5].tolist())

# ========= Statistics =========
if len(dx_list) == 0:
    print(f"pkl_path is {pkl_path}, but no valid steps were found (need fact/predict with >=6 dims).")
else:
    # xyz use all values directly
    dx_arr = np.asarray(dx_list, dtype=np.float32)
    dy_arr = np.asarray(dy_list, dtype=np.float32)
    dz_arr = np.asarray(dz_list, dtype=np.float32)

    dx_mean = float(dx_arr.mean())
    dy_mean = float(dy_arr.mean())
    dz_mean = float(dz_arr.mean())

    delta_xyz = float(np.mean(np.concatenate([dx_arr, dy_arr, dz_arr])))

    # rpy: convert to array, then remove p99 outliers
    dr_arr = np.asarray(dr_list, dtype=np.float32)
    dp_arr = np.asarray(dp_list, dtype=np.float32)
    dyaw_arr = np.asarray(dyaw_list, dtype=np.float32)

    dr_clean, removed_dr, dr_p99 = remove_outliers_p99(dr_arr)
    dp_clean, removed_dp, dp_p99 = remove_outliers_p99(dp_arr)
    dyaw_clean, removed_dyaw, dyaw_p99 = remove_outliers_p99(dyaw_arr)

    # avoid NaN if the filtered result is empty
    dr_mean = float(dr_clean.mean()) if dr_clean.size > 0 else 0.0
    dp_mean = float(dp_clean.mean()) if dp_clean.size > 0 else 0.0
    dyaw_mean = float(dyaw_clean.mean()) if dyaw_clean.size > 0 else 0.0

    # delta_rpy: compute MAE jointly over the three filtered dimensions
    if dr_clean.size + dp_clean.size + dyaw_clean.size > 0:
        delta_rpy = float(
            np.mean(
                np.concatenate([dr_clean, dp_clean, dyaw_clean])
            )
        )
    else:
        delta_rpy = 0.0

    print(
        f"pkl_path is {pkl_path}, for {len(plot_actions)} steps,\n"
        f"  xyz MAE per-dim (mm): dx_mean={dx_mean:.6g}, dy_mean={dy_mean:.6g}, dz_mean={dz_mean:.6g}\n"
        f"  rpy MAE per-dim (deg, >p99 removed): "
        f"droll_mean={dr_mean:.6g}, dpitch_mean={dp_mean:.6g}, dyaw_mean={dyaw_mean:.6g}\n"
        f"  delta_xyz (MAE over xyz, mm) = {delta_xyz:.6g}\n"
        f"  delta_rpy (MAE over rpy, deg, >p99 removed) = {delta_rpy:.6g}"
    )

    # Optionally print how many points were removed:
    # print(
    #     f"\n[rpy outliers removed (>p99 per-dim)] "
    #     f"roll: {removed_dr} / {dr_arr.size}, "
    #     f"pitch: {removed_dp} / {dp_arr.size}, "
    #     f"yaw: {removed_dyaw} / {dyaw_arr.size}"
    # )
