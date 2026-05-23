"""
Original process_data_all: downsamples action from 60Hz to 15Hz (matching image rate).
process_data_all_zarr_60hz: keeps action at 60Hz (no downsampling); rest of the logic is unchanged.
"""


import os
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
from scipy.spatial.transform import Rotation as R
import gc
import time
import zarr
import open3d as o3d
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

import argparse
import tqdm

CAM1_INTRINSIC = np.array([[604.307, 0, 310.155],
                            [0, 604.662, 251.013],
                            [0, 0, 1]])
CAM1_EXTRINSIC = np.array([[-1, 0, 0, 5],
                           [0, -0.9659, -0.2588, 96.678],
                           [0, -0.2588, 0.9659, -26.625],
                           [0, 0, 0, 1]])
CAM2_INTRINSIC = np.array([[905.998, 0, 651.417],
                           [0, 905.892, 360.766],
                           [0, 0, 1]])
CAM2_EXTRINSIC = np.array([[-0.667935, 0.509141, -0.517803, 612.403564],
                           [0.725919, 0.455896, -0.487969, 1034.735474],
                           [-0.013807, -0.709331, -0.682814, 1016.627197],
                           [0, 0, 0, 1]])

def visualize_depth(save_path, depth):
    depth_clipped = np.clip(depth, 0, 1500)
    
    mask = (depth_clipped > 0)
    norm = np.zeros_like(depth_clipped, dtype=np.float32)
    if np.any(mask):
        norm[mask] = (depth_clipped[mask] / 1500.0)
    
    cmap = plt.get_cmap('jet')
    color_img = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    if np.any(mask):
        colored = cmap(norm)
        color_img[mask] = (colored[mask, :3] * 255).astype(np.uint8)

    cv2.imwrite(save_path, cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR))

def visualize_tactile(save_path, tactile):
    tactile = tactile.reshape(2, -1, 3)
    xy0 = tactile[0, :, :2]
    xy1 = tactile[1, :, :2]

    fig = plt.figure(figsize=(8, 15))
    plt.scatter(xy1[:, 0], xy1[:, 1], c='red', s=10)
    plt.scatter(xy0[:, 0], xy0[:, 1], c='blue', s=10)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    plt.box(True)
    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    time.sleep(0.05)
    gc.collect()

def visualize_two_tactile(save_path, left_tactile, right_tactile):
    fig, axs = plt.subplots(1, 2, figsize=(15, 15))
    # left hand
    xy0_left = left_tactile[0, :, :2]
    xy1_left = left_tactile[1, :, :2]
    axs[0].scatter(xy1_left[:, 0], xy1_left[:, 1], c='red', s=15)
    axs[0].scatter(xy0_left[:, 0], xy0_left[:, 1], c='blue', s=15)
    axs[0].set_title('Left Hand Tactile')
    # axs[0].set_xticks([])
    # axs[0].set_yticks([])
    axs[0].set_xlabel('')
    axs[0].set_ylabel('')
    axs[0].set_box_aspect(1)
    axs[0].set_aspect('equal', adjustable='datalim')
    axs[0].spines['top'].set_visible(True)
    axs[0].spines['right'].set_visible(True)

    # right hand
    xy0_right = right_tactile[0, :, :2]
    xy1_right = right_tactile[1, :, :2]
    axs[1].scatter(xy1_right[:, 0], xy1_right[:, 1], c='red', s=15)
    axs[1].scatter(xy0_right[:, 0], xy0_right[:, 1], c='blue', s=15)
    axs[1].set_title('Right Hand Tactile')
    # axs[1].set_xticks([])
    # axs[1].set_yticks([])
    axs[1].set_xlabel('')
    axs[1].set_ylabel('')
    axs[1].set_box_aspect(1)
    axs[1].set_aspect('equal', adjustable='datalim')
    axs[1].spines['top'].set_visible(True)
    axs[1].spines['right'].set_visible(True)

    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, dpi=300)
    plt.close()

def visualize_two_tactile_normal(save_path, tac1_n, tac2_n):
    heatmap1 = np.abs(tac1_n.reshape(35, 20))
    heatmap2 = np.abs(tac2_n.reshape(35, 20))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im1 = axes[0].imshow(heatmap1, cmap='hot', aspect='auto')
    axes[0].set_title('Heatmap 1')
    axes[0].set_xlabel('Width')
    axes[0].set_ylabel('Height')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(heatmap2, cmap='hot', aspect='auto')
    axes[1].set_title('Heatmap 2')
    axes[1].set_xlabel('Width')
    axes[1].set_ylabel('Height')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, dpi=300)
    plt.close()

def visual_tac_info(save_path, tac_data):
    """
    Plot per-frame point-cloud statistics along the x/y/z axes and save as an image.
    
    Args:
        tac_data: numpy array of shape (t, 700, 3)
        save_path: output image path (e.g. 'result.png')
    """
    t = tac_data.shape[0]
    labels = ['x', 'y', 'z']
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    for i in range(3):
        axis_data = tac_data[:, :, i]  # (t, 700)
        # mean and abs-mean over nonzero entries
        mean_vals = []
        abs_mean_vals = []
        for row in axis_data:
            nonzero = row[row != 0]
            if len(nonzero) > 0:
                mean_vals.append(nonzero.mean())
                abs_mean_vals.append(np.abs(nonzero).mean())
            else:
                mean_vals.append(0)
                abs_mean_vals.append(0)
        mean_vals = np.array(mean_vals)
        abs_mean_vals = np.array(abs_mean_vals)

        median_vals = np.median(axis_data, axis=1)
        max_vals = np.max(axis_data, axis=1)
        min_vals = np.min(axis_data, axis=1)
        nonzero_ratio = np.count_nonzero(axis_data, axis=1) / 700.0

        ax = axes[i]
        time = np.arange(1, t + 1)
        ax.plot(time, mean_vals, label='Mean (nonzero)')
        ax.plot(time, abs_mean_vals, label='Abs Mean (nonzero)')
        ax.plot(time, median_vals, label='Median')
        ax.plot(time, max_vals, label='Max')
        ax.plot(time, min_vals, label='Min')
        ax.plot(time, nonzero_ratio, label='Nonzero Ratio')
        ax.set_ylabel(f'{labels[i]} value')
        ax.set_title(f'{labels[i]} axis statistics over time')
        ax.legend()
        ax.grid(True)

    axes[-1].set_xlabel('Time Frame')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)  # free memory

def plot_traj_velocity_acceleration(save_path, traj, timestamps):
    """
    Plot velocity and acceleration curves of the xyz and rpy trajectory components (one subplot per group, 6 curves total).
    Args:
        traj: shape=(N,6), columns are [x, y, z, r, p, y]
        dt: sampling interval (default 1); pass the actual interval if available
        save_path: if given, the figure is saved here
    """
    traj = np.asarray(traj)
    timestamps = np.asarray(timestamps)

    # convert to seconds; align to the first timestamp
    t = (timestamps - timestamps[0]) / 1000.0

    traj[:, 1, 0] *= 0
    xyz = traj[:, :3, 0]
    rpy = traj[:, 3:, 0]

    # unwrap rpy to avoid jumps
    rpy_unwrapped = np.unwrap(rpy, axis=0)

    # compute velocity and acceleration
    xyz_vel = np.gradient(xyz, t, axis=0)
    xyz_acc = np.gradient(xyz_vel, t, axis=0)
    rpy_vel = np.gradient(rpy_unwrapped, t, axis=0)
    rpy_acc = np.gradient(rpy_vel, t, axis=0)

    labels_xyz = ['x', 'y', 'z']
    labels_rpy = ['roll', 'pitch', 'yaw']

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    # top-left: xyz velocity
    for i in range(3):
        axes[0,0].plot(t, xyz_vel[:, i], label=f'{labels_xyz[i]} Velocity')
    axes[0,0].set_title('XYZ Velocity')
    axes[0,0].set_ylabel('Speed (mm/s²)')
    axes[0,0].legend()
    axes[0,0].grid(True)

    # top-right: xyz acceleration
    for i in range(3):
        axes[0,1].plot(t, xyz_acc[:, i], label=f'{labels_xyz[i]} Acceleration')
    axes[0,1].set_title('XYZ Acceleration')
    axes[0,1].set_ylabel('Acceleration (mm/s²)')
    axes[0,1].legend()
    axes[0,1].grid(True)

    # bottom-left: rpy velocity
    for i in range(3):
        axes[1,0].plot(t, rpy_vel[:, i], label=f'{labels_rpy[i]} Velocity')
    axes[1,0].set_title('RPY Velocity')
    axes[1,0].set_ylabel('Angular Speed (deg/s)')
    axes[1,0].legend()
    axes[1,0].grid(True)

    # bottom-right: rpy acceleration
    for i in range(3):
        axes[1,1].plot(t, rpy_acc[:, i], label=f'{labels_rpy[i]} Acceleration')
    axes[1,1].set_title('RPY Acceleration')
    axes[1,1].set_ylabel('Angular Acc (deg/s²)')
    axes[1,1].legend()
    axes[1,1].grid(True)

    axes[1,0].set_xlabel('Time (s)')
    axes[1,1].set_xlabel('Time (s)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_traj_histograms(traj_list, save_folder):
    """
    Inputs:
        traj_list: list of np.ndarray, each with shape (n, 700, 3)
        save_folder: folder to save the figures into
    Function:
        Produces two figures, each with three subplots:
        1. Histogram of x/y/z mean values over time steps 100-250
        2. Histogram of x/y/z max values over time steps 100-250
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # mean and max of the (x,y,z) coordinates within steps 100-250 for each trajectory
    means = []
    maxs = []
    means_nonzero = []
    for traj in traj_list:
        traj = np.abs(traj)
        # guard against short trajectories
        t_start, t_end = 99, 230  # python uses half-open indexing
        if traj.shape[0] < t_end:
            continue
        seg = traj[t_start:t_end, :, :]  # shape (151, 700, 3)
        # reshape to (-1, 3) for aggregate stats
        seg_flat = seg.reshape(-1, 3)    # (151*700, 3)
        means.append(seg_flat.mean(axis=0))   # (3,)
        maxs.append(seg_flat.max(axis=0))     # (3,)
        nonzero_means = []
        for i in range(3):
            vals = seg_flat[:, i]
            nonzero = vals[vals != 0]
            if len(nonzero) == 0:
                nonzero_means.append(0)
            else:
                nonzero_means.append(nonzero.mean())
        means_nonzero.append(nonzero_means)
    
    means = np.array(means)                 # (m, 3)
    maxs = np.array(maxs)                   # (m, 3)
    means_nonzero = np.array(means_nonzero) # (m, 3)
    
    # ---- mean histograms ----
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4))
    titles = ['Fx(dx) Avg', 'Fy(dy) Avg', 'Fz(dz) Avg']
    for i in range(3):
        axes1[i].hist(means[:, i], bins=20, color='C{}'.format(i), alpha=0.7)
        axes1[i].set_xlabel(['x', 'y', 'z'][i] + 'Avg')
        axes1[i].set_ylabel('Traj Num.')
        axes1[i].set_title(titles[i])
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'mean_histograms.png'))
    plt.close(fig1)
    
    # ---- max histograms ----
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    titles = ['Fx(dx) Max', 'Fy(dy) Max', 'Fz(dz) Max']
    for i in range(3):
        axes2[i].hist(maxs[:, i], bins=20, color='C{}'.format(i), alpha=0.7)
        axes2[i].set_xlabel(['x', 'y', 'z'][i] + 'Max')
        axes2[i].set_ylabel('Traj Num.')
        axes2[i].set_title(titles[i])
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'max_histograms.png'))
    plt.close(fig2)

    # ---- nonzero-mean histograms ----
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4))
    titles = ['Fx(dx) Avg', 'Fy(dy) Avg', 'Fz(dz) Avg']
    for i in range(3):
        axes3[i].hist(means_nonzero[:, i], bins=20, color='C{}'.format(i), alpha=0.7)
        axes3[i].set_xlabel(['x', 'y', 'z'][i] + 'Avg')
        axes3[i].set_ylabel('Traj Num.')
        axes3[i].set_title(titles[i])
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'nonzero_mean_histograms.png'))
    plt.close(fig3)

    print('Saved to: ', save_folder)

def save_pts_to_ply(filename, pointcloud):
    # pts: n*6, xyz in [:,0:3], rgb in [:,3:6] (0-255)
    xyz = pointcloud[:, :3]
    rgb = pointcloud[:, 3:6] / 255.0
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    
    o3d.io.write_point_cloud(filename, pcd)

def get_action_seq(action_data_path):
    with open(action_data_path, 'rb') as f:
        data = pickle.load(f)
    eef_pose = np.array(data['eef_pose'])
    angle_max = np.abs(eef_pose)[:,3:].max()
    if angle_max < 3.15:
        eef_pose[:, 3:] = eef_pose[:, 3:] / np.pi * 180.0
    eef_pose_6d = eef_pose.copy()
    time_stamps = np.array(data['timestamps'])
    rot6d = list()

    for i in range(eef_pose.shape[0]):
        eef_matrix = R.from_euler('xyz', eef_pose[i, 3:, 0], degrees=True).as_matrix()
        rot6d.append(np.concatenate([eef_matrix[:, 0], eef_matrix[:, 1]]))

    eef_pose = np.concatenate([eef_pose[:, :3, 0], np.array(rot6d)], axis=1)

    return eef_pose, eef_pose_6d, time_stamps

def rot6d_to_matrix(rot6d):
    # rot6d: (N,6)
    x = rot6d[:, :3]
    y = rot6d[:, 3:6]
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    z = np.cross(x, y)
    z = z / np.linalg.norm(z, axis=1, keepdims=True)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=-1)  # (N,3,3)
    return R

def rotation_matrix_to_angle(R1, R2):
    # R1, R2: (K,3,3)
    delta_R = np.matmul(R2, np.transpose(R1, (0,2,1)))  # (K,3,3)
    trace = np.trace(delta_R, axis1=1, axis2=2)  # (K,)
    cos_theta = (trace - 1) / 2
    cos_theta = np.clip(cos_theta, -1., 1.)
    angle = np.degrees(np.arccos(cos_theta))  # (K,)
    return angle

def get_h2w(xyz, euler):
    '''
    input:
        euler: rpy of eef pose (unit: deg)
        xyz: eef position (unit: mm)
    output:
        h2w: hand to world transform
    '''
    r = R.from_euler('xyz', euler, degrees=True)
    h2w = np.eye(4)
    h2w[:3, :3] = r.as_matrix()
    h2w[:3, 3] = xyz

    return h2w

def signal_points_to_world(points, s2w):
    xyz = points[:, :3]
    xyz = np.dot(xyz, s2w[:3, :3].T) + s2w[:3, 3]
    if points.shape[1] == 6:
        rgb = points[:, 3:]
        return np.concatenate((xyz, rgb), axis=-1)
    else:
        return xyz

def rot6d_sequence_diff(rot6d_seq):
    """
    rot6d_seq: (N,6) array
    Returns: (N-1,) array of rotation angles (deg) between frame i and frame i+1
    """
    R = rot6d_to_matrix(rot6d_seq)  # (N,3,3)
    R1 = R[:-1]   # (N-1,3,3)
    R2 = R[1:]    # (N-1,3,3)
    angle = rotation_matrix_to_angle(R1, R2)  # (N-1,)
    return angle

def find_nearest_larger(ts_list, start_ts):
    """
    ts_list: ascending list/array of timestamps
    start_ts: starting frame timestamp (scalar)
    Returns: the nearest timestamp greater than start_ts and its index; (None, -1) if none exists
    """
    ts_arr = np.asarray(ts_list)
    idx = np.searchsorted(ts_arr, start_ts, side='right')
    if idx < len(ts_arr):
        return ts_arr[idx], idx
    else:
        return None, -1

def sync_and_truncate_timestamps(ts_dict):
    """
    ts_dict: dict with keys 'camera1','camera2','tactile1','tactile2','action'; values are ascending 1d numpy arrays.

    Returns:
      processed_ts: synchronized timestamp arrays per key
      start_indices: start truncation index per sequence
      end_indices: end truncation index per sequence
      begin: aligned start time on camera1
      end: aligned end time on camera1
    """
    # 1. collect each sequence's start time
    starts = [arr[0] for arr in ts_dict.values()]
    max_start = max(starts)

    # 2. find the nearest camera1 frame with timestamp >= max_start
    camera1 = ts_dict['camera1']
    begin_idx = np.searchsorted(camera1, max_start, side='left')
    if begin_idx >= len(camera1):
        raise ValueError("max_start is later than every camera1 timestamp; cannot synchronize")
    begin = camera1[begin_idx]

    # 3. for each sequence, pick the frame nearest to `begin` as its start
    start_indices = {}
    for k, arr in ts_dict.items():
        if k == 'camera1':
            idx = np.searchsorted(arr, begin, side='left')
        else:
            # find the index in arr closest to `begin`
            idx = np.abs(arr - begin).argmin()
        start_indices[k] = idx

    # 4. collect each sequence's last timestamp
    ends = [arr[-1] for arr in ts_dict.values()]
    min_end = min(ends)

    # 5. find the nearest camera1 frame with timestamp <= min_end
    end_idx = np.searchsorted(camera1, min_end, side='right') - 1
    if end_idx < 0:
        raise ValueError("min_end is earlier than every camera1 timestamp; cannot synchronize")
    end = camera1[end_idx]

    # 6. truncate every other sequence beyond end+60ms
    cut_time = end + 60
    end_indices = {}
    for k, arr in ts_dict.items():
        idx = np.searchsorted(arr, cut_time, side='right')
        end_indices[k] = idx

    # 7. build the output -- processed timestamps per key (half-open interval)
    processed_ts = {}
    for k, arr in ts_dict.items():
        s, e = start_indices[k], end_indices[k]
        processed_ts[k] = arr[s:e]

    return processed_ts, start_indices, end_indices, begin, end

def downsample_evenly(t1, factor=4):
    t1 = np.asarray(t1)
    N = len(t1)
    # average interval
    mean_interval = np.mean(np.diff(t1))
    step = mean_interval * factor
    # number of output samples
    num_out = max(1, int(np.ceil(N / factor)))
    # target sampling timestamps
    target_times = t1[0] + np.arange(num_out) * step
    # greedy: pick the frame nearest each target whose index is greater than the last (no backtracking)
    sel_idx = []
    last = -1
    for t in target_times:
        # only search the remaining frames
        remain = np.arange(last+1, N)
        if remain.size == 0:
            break
        i = remain[np.abs(t1[remain] - t).argmin()]
        sel_idx.append(i)
        last = i
    sel_idx = np.array(sel_idx)
    return t1[sel_idx], sel_idx

def downsample_fixed_fps(timestamp, target_fps):
    timestamp = np.asarray(timestamp)
    start_time = timestamp[0]
    end_time = timestamp[-1]
    step = 1000 / target_fps

    num_out = int(np.floor((end_time - start_time) / step)) + 1
    target_times = start_time + np.arange(num_out) * step

    sel_idx = [np.abs(timestamp - t).argmin() for t in target_times]
    sel_idx = np.array(sel_idx)
    return timestamp[sel_idx], sel_idx

def align_t1_t2_segments(t1, t2):
    """
    t1: 1d array-like, base timestamp sequence of length N
    t2: 1d array-like, high-frequency timestamp sequence of length M
    Returns:
      segment_indices: list of np.ndarray where each entry is the t2 indices falling between t1[i] and t1[i+1]
    """
    t1 = np.asarray(t1)
    t2 = np.asarray(t2)
    segment_indices = []
    for i in range(len(t1)-1):
        # collect t2 frames that fall in [t1[i], t1[i+1])
        start = np.searchsorted(t2, t1[i], side='left')
        end = np.searchsorted(t2, t1[i+1], side='left')
        idx = np.arange(start, end)
        segment_indices.append(idx)
    return segment_indices

def sliding_slices(data_seq, timestamps, window_size=9, stride=1):
    """
    data_seq: input sequence (images, features, ...) of length N
    timestamps: matching timestamp sequence (same length as data_seq)
    window_size: window length (default 9)
    stride: sliding step (default 1)

    Returns:
        slices_list: list of length-`window_size` slices of data_seq (e.g. data_seq[i:i+window_size])
        start_indices: starting index i within data_seq for each slice
        start_time: starting timestamp for each slice
    """
    N = len(data_seq)
    slices_list = []
    start_indices = []
    start_time = []
    for i in range(0, N - window_size + 1, stride):
        slices_list.append(data_seq[i:i+window_size])
        start_indices.append(i)
        start_time.append(timestamps[i])
    return slices_list, np.array(start_indices), np.array(start_time)

def align_timestamps(standard_ts, other_ts):
    standard_ts = np.asarray(standard_ts)
    other_ts = np.asarray(other_ts)
    indices = np.abs(standard_ts[:, None] - other_ts[None, :]).argmin(axis=1)
    aligned_other_ts = other_ts[indices]
    return aligned_other_ts, indices

def get_pca_matrix(data, n_components=15):
    '''
    pca reduction
    '''
    pca = PCA(n_components=n_components)
    pca.fit(data)
    transform_matrix = pca.components_
    center_matrix = pca.mean_
    
    return transform_matrix, center_matrix

def depth_image_to_camera_points(depth_image, color_image, intrinsic, mask=None):
    fx, fy, cx, cy = intrinsic[0,0], intrinsic[1,1], intrinsic[0,2], intrinsic[1,2]
    height, width = depth_image.shape 
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    Z = depth_image
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    point_cloud = np.dstack((X, Y, Z))
    pts = np.concatenate((point_cloud, color_image), axis=-1)
    if mask is not None:
        mask = (mask != 0).astype(np.bool).reshape(-1)
        pts = pts[mask]
    depth_mask = (depth_image > 0) * (depth_image < 1500)
    return pts.reshape(-1, 6), depth_mask

def signal_points_to_world(points, s2w):
    xyz = points[:, :3]
    xyz = np.dot(xyz, s2w[:3, :3].T) + s2w[:3, 3]
    if points.shape[1] == 6:
        rgb = points[:, 3:]
        return np.concatenate((xyz, rgb), axis=-1)
    else:
        return xyz

def pts_downsample(pointcloud, target_num, mode = 'uniform'):
    if mode == 'uniform':
        n = pointcloud.shape[0]
        if target_num >= n:
            return pointcloud.copy()
        idx = np.random.choice(n, target_num, replace=False)
        return pointcloud[idx]

def process_one_episode(data_path, policy, vis_save_path=None, save_camera_vis=False, save_tactile_vis=False):
    """
    Process the data of one episode.
    :param data_path: dataset path
    :param vis_save_path: visualization output path
    :param save_camera_vis: whether to save camera-data visualizations
    :param save_tactile_vis: whether to save tactile-data visualizations
    :return:
    """
    image_data_path = os.path.join(data_path, 'image.pkl')
    depth_data_path = os.path.join(data_path, 'depth.pkl')
    gripper_data_path = os.path.join(data_path, 'gripper.pkl')
    tactile_data_path = os.path.join(data_path, 'tactile.pkl')
    action_data_path = os.path.join(data_path, 'state.pkl')

    try:
        # process camera data
        camera_data_dict = dict()
        image_data = pickle.load(open(image_data_path, 'rb'))
        depth_data = pickle.load(open(depth_data_path, 'rb'))
        for camera in image_data.keys():
            image_list = np.array(image_data[camera]['image'])
            image_stamps = np.array(image_data[camera]['timestamps'])
            depth_list = np.array(depth_data[camera]['depth'])
            depth_stamps = np.array(depth_data[camera]['timestamps'])
            if camera == 'camera2':
                depth_list *= 0.25
            if len(image_list) > len(depth_list):
                image_list = image_list[:len(depth_list)]
                image_stamps = image_stamps[:len(depth_stamps)]
            elif len(image_list) < len(depth_list):
                depth_list = depth_list[:len(image_list)]
                depth_stamps = depth_stamps[:len(image_stamps)]
            camera_data_dict[camera] = {'image': image_list, 'depth': depth_list, 'timestamps': image_stamps}

        # process tactile data
        tac_data = pickle.load(open(tactile_data_path, 'rb'))
        tac1_data_dict = tac_data['tactile1']
        tac2_data_dict = tac_data['tactile2']

        for key in tac1_data_dict.keys():
            tac1_data_dict[key] = np.array(tac1_data_dict[key])
        for key in tac2_data_dict.keys():
            tac2_data_dict[key] = np.array(tac2_data_dict[key])

        # process robot data
        eef_pose, eef_pose_6d, action_timestamps = get_action_seq(action_data_path)

        # process gripper data
        gripper_data_dict = pickle.load(open(gripper_data_path, 'rb'))
        for key in gripper_data_dict.keys():
            gripper_data_dict[key] = np.array(gripper_data_dict[key])


        '''first step: sync_and_truncate_timestamp'''
        sensor_timestamps_dict = {
                                'camera1': camera_data_dict['camera1']['timestamps'], 
                                'camera2': camera_data_dict['camera2']['timestamps'], 
                                'tactile1': tac1_data_dict['timestamps'], 
                                'tactile2': tac2_data_dict['timestamps'],
                                'gripper': gripper_data_dict['timestamps'],
                                'robot': action_timestamps
                                }
        processed_ts, start_indices, end_indices, begin, end = sync_and_truncate_timestamps(sensor_timestamps_dict)
            
        camera1_timestamps = processed_ts['camera1']
        camera2_timestamps = processed_ts['camera2']
        tactile1_timestamps = processed_ts['tactile1']
        tactile2_timestamps = processed_ts['tactile2']
        robot_timestamps = processed_ts['robot']
        gripper_timestamps = processed_ts['gripper']

        for key in camera_data_dict['camera1'].keys():
            camera_data_dict['camera1'][key] = camera_data_dict['camera1'][key][start_indices['camera1']: end_indices['camera1']]
        for key in camera_data_dict['camera2'].keys():
            camera_data_dict['camera2'][key] = camera_data_dict['camera2'][key][start_indices['camera2']: end_indices['camera2']]

        tac1_data = tac1_data_dict['deform'][start_indices['tactile1']: end_indices['tactile1']]
        tac2_data = tac2_data_dict['deform'][start_indices['tactile2']: end_indices['tactile2']]
        tactile1_timestamps, tac1_idx = downsample_fixed_fps(tactile1_timestamps, 60)
        tactile2_timestamps, tac2_idx = downsample_fixed_fps(tactile2_timestamps, 60)
        tac1_data = tac1_data[tac1_idx]
        tac2_data = tac2_data[tac2_idx]
        robot_data = eef_pose[start_indices['robot']: end_indices['robot']]
        robot_6d_data = eef_pose_6d[start_indices['robot']: end_indices['robot']]
        gripper_data =  gripper_data_dict['gripper_pos'][start_indices['gripper']: end_indices['gripper']]


        '''second step: choose mode: tac_wm or dp_zarr'''
        if policy == 'dp_zarr':
            image_resize_shape = (320, 240)
            # print(f"len(camera1_timestamps) is {len(camera1_timestamps)}") # 208
            # print(f"len(camera2_timestamps) is {len(camera2_timestamps)}") # 209
            # print(f"len(tactile1_timestamps) is {len(tactile1_timestamps)}") # 831
            # print(f"len(tactile2_timestamps) is {len(tactile2_timestamps)}") # 831
            # print(f"len(robot_timestamps) is {len(robot_timestamps)}") # 832
            # print(f"len(gripper_timestamps) is {len(gripper_timestamps)}") # 832
            # ==== original ====
            # camera2_timestamps, cam2_indices = align_timestamps(camera1_timestamps, camera2_timestamps)
            # tactile1_timestamps, tac1_indices = align_timestamps(camera1_timestamps, tactile1_timestamps)
            # tactile2_timestamps, tac2_indices = align_timestamps(camera1_timestamps, tactile2_timestamps)
            # robot_state_timestamps, robot_state_indices = align_timestamps(camera1_timestamps, robot_timestamps)
            # gripper_timestamps, gripper_indices = align_timestamps(camera1_timestamps, gripper_timestamps)
            # ==== 60hz: align to action ====
            camera1_timestamps, cam1_indices = align_timestamps(robot_timestamps, camera1_timestamps)
            camera2_timestamps, cam2_indices = align_timestamps(robot_timestamps, camera2_timestamps)
            tactile1_timestamps, tac1_indices = align_timestamps(robot_timestamps, tactile1_timestamps)
            tactile2_timestamps, tac2_indices = align_timestamps(robot_timestamps, tactile2_timestamps)
            # robot_state_timestamps, robot_state_indices = align_timestamps(camera1_timestamps, robot_timestamps)
            gripper_timestamps, gripper_indices = align_timestamps(robot_timestamps, gripper_timestamps)
            
            tac1_arrays = tac1_data[tac1_indices]
            tac2_arrays = tac2_data[tac2_indices]
            tac1_arrays = tac1_arrays.reshape(len(tactile1_timestamps), -1, 3)
            tac2_arrays = tac2_arrays.reshape(len(tactile2_timestamps), -1, 3)
            camera1_image_arrays = camera_data_dict['camera1']['image'][cam1_indices]
            camera1_depth_arrays = camera_data_dict['camera1']['depth'][cam1_indices]
            camera2_image_arrays = camera_data_dict['camera2']['image'][cam2_indices]
            camera2_depth_arrays = camera_data_dict['camera2']['depth'][cam2_indices]
            states_arrays = robot_data#[robot_state_indices]
            states_6d_arrays = robot_6d_data#[robot_state_indices]
            gripper_arrays = gripper_data[gripper_indices][:, None]

            # start_frame = np.where(np.diff(states_arrays[:, 2]) > 2)[0][0] # previously action was downsampled from 60Hz to 15Hz
            start_frame = np.where(np.diff(states_arrays[:, 2]) > 1)[0][0] # action is 60Hz now, so threshold lowered to 0.5
            end_frame = np.argmin(states_arrays[:, 0]) + 20 # take a few extra trailing frames
            print('start:', start_frame, 'end:', end_frame)
            
            # already aligned above, so start_frame/end_frame can be used directly to slice
            camera1_timestamps = camera1_timestamps[start_frame:end_frame]
            camera1_image_arrays = camera1_image_arrays[start_frame:end_frame]
            camera2_image_arrays = camera2_image_arrays[start_frame:end_frame]
            camera2_depth_arrays = camera2_depth_arrays[start_frame:end_frame]
            tac1_n_arrays = tac1_arrays[start_frame:end_frame][:, :, 2]
            tac2_n_arrays = tac2_arrays[start_frame:end_frame][:, :, 2]
            tac1_arrays = tac1_arrays[start_frame:end_frame][:, :, :2]
            tac2_arrays = tac2_arrays[start_frame:end_frame][:, :, :2]
            tcp_pose_arrays = states_arrays[start_frame:end_frame]
            states_6d_arrays = states_6d_arrays[start_frame:end_frame][:, :, 0]
            tcp_pose_arrays[:, :3] /= 1000.0
            gripper_arrays = gripper_arrays[start_frame:end_frame] / 255.0
            
            state_arrays = np.concatenate([tcp_pose_arrays, gripper_arrays], axis=-1)
            new_action_arrays = state_arrays[1:, ...].copy()
            # action equals the next-step state
            action_arrays = np.concatenate([new_action_arrays, new_action_arrays[-1][np.newaxis, :]], axis=0)

            camera2_crop_list = []
            pts_list = []
            for i in range(camera2_image_arrays.shape[0]):
                depth = camera2_depth_arrays[i]
                pts, depth_mask = depth_image_to_camera_points(depth, camera2_image_arrays[i][...,::-1], CAM2_INTRINSIC)
                pts = signal_points_to_world(pts, CAM2_EXTRINSIC)
                pts_mask = (pts[:, 1] > 300) * (pts[:, 1] < 800) * (pts[:, 0] > -440) * (pts[:, 0] < 350) * (pts[:, 2] > 75) * (pts[:, 2] < 550)
                pts = pts[pts_mask * depth_mask.reshape(-1,)]
                image = np.zeros_like(camera2_image_arrays[i])
                pts_mask = pts_mask.reshape(depth_mask.shape[0], depth_mask.shape[1])
                image[pts_mask] = camera2_image_arrays[i][pts_mask]
                camera2_crop_list.append(image[100:628, 200:904, :])
                pts = pts_downsample(pts, 8192)
                pts_list.append(pts)
                
            camera2_image_arrays = np.array(camera2_crop_list)
            camera2_depth_arrays = camera2_depth_arrays[:, 100:628, 200:904]

            pts_arrays = np.array(pts_list)
            pts_arrays[:, :, :3] /= 1000.0
            
            if image_resize_shape is not None:
                camera1_image_arrays = np.array([cv2.resize(img, image_resize_shape) for img in camera1_image_arrays])
                camera2_image_arrays = np.array([cv2.resize(img, image_resize_shape) for img in camera2_image_arrays])
                
            if save_camera_vis:
                save_image1 = os.path.join(vis_save_path, 'camera1', 'image')
                save_image2 = os.path.join(vis_save_path, 'camera2', 'image')
                save_pts = os.path.join(vis_save_path, 'pts')
                os.makedirs(save_image1, exist_ok=True)
                os.makedirs(save_image2, exist_ok=True)
                os.makedirs(save_pts, exist_ok=True)
                for i in range(len(camera1_image_arrays)):
                    cv2.imwrite(os.path.join(save_image1, str(camera1_timestamps[i]) + '.png'), camera1_image_arrays[i])
                for i in range(len(camera2_image_arrays)):
                    cv2.imwrite(os.path.join(save_image2, str(camera1_timestamps[i]) + '.png'), camera2_image_arrays[i])
                for i in range(len(pts_arrays)):
                    save_pts_to_ply(os.path.join(save_pts, str(camera1_timestamps[i]) + '.ply'), pts_arrays[i])
                
            if save_tactile_vis:
                save_tactile_xy = os.path.join(vis_save_path, 'tactile_xy')
                save_tactile_n = os.path.join(vis_save_path, 'tactile_n')
                save_tactile_info = os.path.join(vis_save_path, 'tactile_info')
                save_motion = os.path.join(vis_save_path, 'motion') 
                os.makedirs(save_tactile_xy, exist_ok=True)
                os.makedirs(save_tactile_n, exist_ok=True)
                os.makedirs(save_tactile_info, exist_ok=True)
                os.makedirs(save_motion, exist_ok=True)
                x = np.linspace(-8.5, 8.5, 20)
                y = np.linspace(30, 0, 35)
                X, Y = np.meshgrid(x, y)
                grid = np.stack([X, Y], axis=-1).reshape(-1, 2)
                grid = np.expand_dims(grid, 0)
                tac1_init_arrays = np.repeat(grid, tac1_arrays.shape[0], axis=0)
                tac2_init_arrays = tac1_init_arrays
                visual_tac_info(os.path.join(save_tactile_info, 'tactile1.png'), np.concatenate((tac1_arrays, tac1_n_arrays[..., None]), axis=-1))
                visual_tac_info(os.path.join(save_tactile_info, 'tactile2.png'), np.concatenate((tac2_arrays, tac2_n_arrays[..., None]), axis=-1))
                for i in range(tac1_init_arrays.shape[0]):
                    left_tac = np.stack((tac1_init_arrays[i], tac1_init_arrays[i] + tac1_arrays[i]), axis=0)
                    right_tac = np.stack((tac2_init_arrays[i], tac2_init_arrays[i] + tac2_arrays[i]), axis=0)
                    visualize_two_tactile(os.path.join(save_tactile_xy, str(camera1_timestamps[i]) + '.png'), left_tac, right_tac)
                    visualize_two_tactile_normal(os.path.join(save_tactile_n, str(camera1_timestamps[i]) + '.png'), tac1_n_arrays[i], tac2_n_arrays[i])

            tac1_arrays = np.concatenate((tac1_arrays, tac1_n_arrays[..., None]), axis=-1)
            tac2_arrays = np.concatenate((tac2_arrays, tac2_n_arrays[..., None]), axis=-1)
            return tcp_pose_arrays, gripper_arrays, tac1_arrays, tac2_arrays, camera1_image_arrays[..., ::-1], camera2_image_arrays[..., ::-1], pts_arrays, action_arrays
        

    
    except Exception as e:
        print(f"Error loading data: {data_path}")
        return None


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_camera_vis", action="store_true", help="enable camera visualization")
    parser.add_argument("--save_tactile_vis", action="store_true", help="enable tactile-signal visualization")
    parser.add_argument("--save_action_vis", action="store_true", help="enable action visualization")
    parser.add_argument("--root_path", type=str, default="./data/tactile_dataset", help="where to load")
    parser.add_argument("--save_path", type=str, default="./data/dataset", help="where to save")
    parser.add_argument("--policy", nargs="+", type=str, default=["dp_zarr"], help="policy list")
    parser.add_argument("--task_list", nargs="+", type=str, default=["vase2_new_A"], help="task list") # example: --task_list vase_a_3level dish
    parser.add_argument("--episode_length", type=int, default=-1, help="episode_list=episode_list[:episode_length]. -1 (default) means no limit")

    args = parser.parse_args()

    # save_camera_vis = True
    # save_tactile_vis = False
    # save_action_vis = False
    # policy = ['dp_zarr'] # dp_zarr, tac_wm
    # task_list = ['vase_a_3level'] # board, dish
    # root_path = './data/tactile/vase/'
    # save_path = './data/dataset'
    save_camera_vis = args.save_camera_vis
    save_tactile_vis = args.save_tactile_vis
    save_action_vis = args.save_action_vis
    policy = args.policy # dp_zarr, tac_wm
    task_list = args.task_list # board, dish
    root_path = args.root_path
    save_path = args.save_path

    print(args.save_camera_vis, args.save_tactile_vis, args.save_action_vis, args.episode_length)
    # exit()
    
    if len(task_list) == 1:
        task = task_list[0]
        data_dir = os.path.join(root_path, task)
        episode_list = [os.path.join(data_dir, i) for i in sorted(os.listdir(data_dir))]
        if args.episode_length != -1:
            episode_list = episode_list[:args.episode_length]
    else:
        episode_list = []
        task = ''
        for task_name in task_list:
            data_dir = os.path.join(root_path, task_name)
            task += task_name + '_'
            for episode in sorted(os.listdir(data_dir)):
                episode_list.append(os.path.join(data_dir, episode))
    save_data_path = os.path.join(save_path, task)

    if 'dp_zarr' in policy:
        save_zarr_path1 = os.path.join(save_data_path, 'ours_zarr/replay_buffer.zarr')
        save_zarr_path2 = os.path.join(save_data_path, 'rdp_zarr/replay_buffer.zarr')
        save_pca_path = os.path.join(save_data_path, 'rdp_pca')
        # os.makedirs(save_zarr_path1, exist_ok=True)
        os.makedirs(save_zarr_path2, exist_ok=True)
        os.makedirs(save_pca_path, exist_ok=True)

        episode_end_list = list()
        left_robot_tcp_pose_list = list()
        left_robot_gripper_width_list = list()
        left_gripper1_marker_offset_list = list()
        left_gripper2_marker_offset_list = list()
        left_wrist_img_list = list()
        global_img_list = list()
        global_pts_list = list()
        action_list = list()
        action_assemble_list = list()

        for episode_id in tqdm.tqdm(range(len(episode_list))):
            data_path = episode_list[episode_id]
            vis_save_path = os.path.join(save_data_path, str("%04d"%episode_id))
            print('loading episode:', data_path)
            episode_info = process_one_episode(data_path, 'dp_zarr', vis_save_path, save_camera_vis, save_tactile_vis)
            if episode_info is None:
                continue
            else:
                left_robot_tcp_pose, left_robot_gripper_width, left_gripper1_marker_offset, left_gripper2_marker_offset, \
                left_wrist_img, global_img, global_pts, action = episode_info

            if len(episode_end_list) == 0:
                episode_end_list.append(len(left_robot_tcp_pose))
            else:
                episode_end_list.append(episode_end_list[-1] + len(left_robot_tcp_pose))
            left_robot_tcp_pose_list.append(left_robot_tcp_pose)
            left_robot_gripper_width_list.append(left_robot_gripper_width)
            left_gripper1_marker_offset_list.append(left_gripper1_marker_offset)
            left_gripper2_marker_offset_list.append(left_gripper2_marker_offset)
            left_wrist_img_list.append(left_wrist_img)
            global_img_list.append(global_img)
            global_pts_list.append(global_pts)
            action_list.append(action)

        left_robot_tcp_pose_arrays = np.concatenate(left_robot_tcp_pose_list)
        left_robot_gripper_width_arrays = np.concatenate(left_robot_gripper_width_list)
        left_gripper1_marker_offset_arrays = np.concatenate(left_gripper1_marker_offset_list)
        left_gripper2_marker_offset_arrays = np.concatenate(left_gripper2_marker_offset_list)
        left_wrist_img_arrays = np.concatenate(left_wrist_img_list)
        global_img_arrays = np.concatenate(global_img_list)
        global_pts_arrays = np.concatenate(global_pts_list)
        action_arrays = np.concatenate(action_list)
        episode_ends_arrays = np.array(episode_end_list)

        if save_action_vis:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            from matplotlib import cm
            colors = cm.rainbow(np.linspace(0, 1, len(action_list)))
            cnt = 0
            for traj in action_list:
                ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2], color=colors[cnt], s=10)
                cnt += 1
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.tight_layout()
            plt.savefig(os.path.join(save_pca_path, 'trajectories_3d.png'), dpi=300)
        
            # action_xyz = action_arrays[:, :3]
            # np.savetxt('trajectories_3d.txt', action_xyz)

        left_gripper1_marker_offset_arrays_xy = np.concatenate(left_gripper1_marker_offset_list)[:,:,:2].reshape(episode_end_list[-1], -1)
        left_gripper2_marker_offset_arrays_xy = np.concatenate(left_gripper2_marker_offset_list)[:,:,:2].reshape(episode_end_list[-1], -1)
        pca_matrix1, pca_mean1 = get_pca_matrix(left_gripper1_marker_offset_arrays_xy)
        pca_matrix2, pca_mean2 = get_pca_matrix(left_gripper2_marker_offset_arrays_xy)
        left_gripper1_marker_offset_emb_arrays = (left_gripper1_marker_offset_arrays_xy - pca_mean1) @ pca_matrix1.T
        left_gripper2_marker_offset_emb_arrays = (left_gripper2_marker_offset_arrays_xy - pca_mean2) @ pca_matrix2.T

        np.save(os.path.join(save_pca_path, 'pca_matrix1.npy'), pca_matrix1.T)
        np.save(os.path.join(save_pca_path, 'pca_matrix2.npy'), pca_matrix2.T)
        np.save(os.path.join(save_pca_path, 'pca_mean1.npy'), pca_mean1)
        np.save(os.path.join(save_pca_path, 'pca_mean2.npy'), pca_mean2)

        # zarr_root = zarr.group(save_zarr_path1)
        # zarr_data = zarr_root.create_group('data')
        # zarr_meta = zarr_root.create_group('meta')
        # compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)

        # zarr_data.create_dataset('tcp_pose', data=left_robot_tcp_pose_arrays, chunks=(10000, 9), 
        #                          dtype='float32', overwrite=True, compressor=compressor)
        # zarr_data.create_dataset('gripper1_tactile', data=left_gripper1_marker_offset_arrays, chunks=(100, 700, 3), 
        #                          dtype='float32', overwrite=True, compressor=compressor)
        # zarr_data.create_dataset('gripper2_tactile', data=left_gripper2_marker_offset_arrays, chunks=(100, 700, 3),  
        #                          dtype='float32', overwrite=True, compressor=compressor)
        # zarr_data.create_dataset('action', data=action_arrays, chunks=(10000, 10), dtype='float32', overwrite=True,
        #                          compressor=compressor)
        # zarr_data.create_dataset('action_assemble', data=action_assemble_arrays, chunks=(10000, 4, 10), dtype='float32', overwrite=True,
        #                          compressor=compressor)    
        # zarr_data.create_dataset('wrist_image', data=left_wrist_img_arrays, chunks=(100, 240, 320, 3), dtype='uint8')
        # zarr_data.create_dataset('global_image', data=global_img_arrays, chunks=(100, 240, 320, 3), dtype='uint8')
        # zarr_data.create_dataset('gripper1_marker_offset_emb', data=left_gripper1_marker_offset_emb_arrays, chunks=(10000, 15), 
        #                          dtype='float32', overwrite=True, compressor=compressor)
        # zarr_data.create_dataset('gripper2_marker_offset_emb', data=left_gripper2_marker_offset_emb_arrays, chunks=(10000, 15),  
        #                          dtype='float32', overwrite=True, compressor=compressor)
        # zarr_data.create_dataset('gripper_width', data=left_robot_gripper_width_arrays, chunks=(10000, 1), dtype='float32', overwrite=True,
        #                          compressor=compressor)
        # zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, chunks=(10000,), dtype='int64', overwrite=True,
        #                          compressor=compressor)
        
        zarr_root = zarr.group(save_zarr_path2)
        zarr_data = zarr_root.create_group('data')
        zarr_meta = zarr_root.create_group('meta')
        compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
        zarr_data.create_dataset('left_robot_tcp_pose', data=left_robot_tcp_pose_arrays, chunks=(10000, 9), 
                                 dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('left_gripper1_marker_offset_emb', data=left_gripper1_marker_offset_emb_arrays, chunks=(10000, 15), 
                                 dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('left_gripper2_marker_offset_emb', data=left_gripper2_marker_offset_emb_arrays, chunks=(10000, 15),  
                                 dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('left_gripper1_tactile', data=left_gripper1_marker_offset_arrays, chunks=(100, 700, 3), 
                                 dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('left_gripper2_tactile', data=left_gripper2_marker_offset_arrays, chunks=(100, 700, 3),  
                                 dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('action', data=action_arrays, chunks=(10000, 10), dtype='float32', overwrite=True,
                                 compressor=compressor)
        zarr_data.create_dataset('left_robot_gripper_width', data=left_robot_gripper_width_arrays, chunks=(10000, 1), dtype='float32', overwrite=True,
                                 compressor=compressor)
        zarr_data.create_dataset('left_wrist_img', data=left_wrist_img_arrays, chunks=(100, 240, 320, 3), dtype='uint8')
        zarr_data.create_dataset('global_pts', data=global_pts_arrays[:, :, :3], chunks=(100, 8192, 3), dtype='uint8')
        zarr_data.create_dataset('global_img', data=global_img_arrays, chunks=(100, 240, 320, 3), dtype='uint8')
        zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, chunks=(10000,), dtype='int64', overwrite=True,
                                 compressor=compressor)
        
        del left_robot_tcp_pose_list
        del left_robot_gripper_width_list
        del left_gripper1_marker_offset_list
        del left_gripper2_marker_offset_list
        del left_wrist_img_list
        del global_img_list
        del global_pts_list
        del action_list
        del left_robot_tcp_pose_arrays
        del left_robot_gripper_width_arrays
        del left_gripper1_marker_offset_arrays
        del left_gripper2_marker_offset_arrays
        del left_wrist_img_arrays
        del global_img_arrays
        del global_pts_arrays
        del action_arrays
        del episode_ends_arrays
        gc.collect()

    