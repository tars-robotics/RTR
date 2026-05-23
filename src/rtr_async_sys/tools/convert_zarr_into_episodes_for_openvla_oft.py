import argparse
import hashlib
import shutil
from pathlib import Path

import numpy as np
import zarr
import tqdm

from rtr_async_sys.utils.action_utils import ortho6d_to_rotation_matrix
import transforms3d as t3d


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert RDP zarr replay buffer into per-episode .npy files for openvla-oft fine-tuning.",
    )
    parser.add_argument("--zarr_path", type=str, required=True,
                        help="Path to replay_buffer.zarr.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory; episodes are saved to <output_dir>/train/.")
    parser.add_argument("--stage_in_tmpfs", action=argparse.BooleanOptionalAction, default=True,
                        help="Write episodes to /dev/shm first, then copy to output_dir.")
    parser.add_argument("--tmpfs_root", type=str, default="/dev/shm/openvla_oft_episodes",
                        help="Root directory for tmpfs staging when --stage_in_tmpfs is enabled.")
    parser.add_argument("--cleanup_tmpfs_after_copy", action=argparse.BooleanOptionalAction, default=True,
                        help="Remove tmpfs staging directory after copying to output_dir.")
    parser.add_argument("--num_episodes", type=int, default=-1,
                        help="Number of episodes to convert. Default -1 converts all episodes.")
    return parser.parse_args()


args = parse_args()
zarr_path = args.zarr_path
output_dir = Path(args.output_dir)


def get_staging_dir(output_dir: Path, tmpfs_root: str, stage_in_tmpfs: bool) -> Path:
    if not stage_in_tmpfs:
        return output_dir

    resolved_output = str(output_dir.resolve())
    output_hash = hashlib.sha1(resolved_output.encode("utf-8")).hexdigest()[:10]
    return Path(tmpfs_root) / f"{output_dir.name}_{output_hash}"


staging_dir = get_staging_dir(output_dir, args.tmpfs_root, args.stage_in_tmpfs)
staging_train_dir = staging_dir / "train"
final_train_dir = output_dir / "train"

if args.stage_in_tmpfs:
    print(f"Staging episodes in {staging_dir}")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
else:
    if final_train_dir.exists():
        shutil.rmtree(final_train_dir)

staging_train_dir.mkdir(parents=True, exist_ok=True)

# Load Zarr file
zarr_root = zarr.open(zarr_path, mode='r')

# print(zarr_root["data"].keys())
print((type(zarr_root["data/left_robot_tcp_pose"]), zarr_root["data/left_robot_tcp_pose"].shape, zarr_root["data/left_robot_tcp_pose"].dtype))
print((type(zarr_root["data/left_robot_tcp_pose"][0]), zarr_root["data/left_robot_tcp_pose"][0].shape, zarr_root["data/left_robot_tcp_pose"][0].dtype))
# Input shape is 9 (x, y, z, 6D rotation?), while OpenVLA-OFT uses 8 (x, y, z, yaw, pitch, roll, gripper, gripper?).
# exit()

# image, state, language_instruction

# Extract key data
imgs = zarr_root["data/left_wrist_img"]   # shape: (N_steps, H, W, C)
states = zarr_root["data/left_robot_tcp_pose"]
actions = zarr_root["data/action"]        # shape: (N_steps, A)
episode_ends = zarr_root["meta/episode_ends"]  # shape: (num_episodes,)

print(f"Loaded {imgs.shape[0]} total steps, with {len(episode_ends)} episodes.")
print(imgs[0].dtype)
# exit()

# Fixed language instruction
LANG_INSTRUCTION = "wipe the vase."

print(f"len(episode_ends) is {len(episode_ends)}")
if args.num_episodes == -1:
    num_episodes = len(episode_ends)
else:
    num_episodes = min(args.num_episodes, len(episode_ends))
    print(f"Using {num_episodes} episodes for debug")

def convert_10d_action_to_7d_action(action:np.ndarray):
    if action.shape == (7,):
        return action
    assert action.shape == (10, )

    action = action[None,:]

    left_rot_mat_batch = ortho6d_to_rotation_matrix(action[:, 3:9])  #(action_steps, 3, 3)
    left_rot_mat_batch = np.asarray(left_rot_mat_batch, dtype=np.float64)
    left_euler_batch = np.array([t3d.euler.mat2euler(rot_mat) for rot_mat in left_rot_mat_batch])  # (action_steps, 3)
    left_trans_batch = action[:, :3]  # (action_steps, 3)
    left_action_6d = np.concatenate([left_trans_batch, left_euler_batch, action[:, 9:]], axis=1) # (action_steps, 7)
    
    return left_action_6d[0]

def convert_9d_state_to_8d_state(state:np.ndarray):
    if state.shape == (8,):
        return state
    assert state.shape == (9, )

    state = state[None,:]
    zeros_array = np.zeros((state.shape[0], 2), dtype=state.dtype)

    left_rot_mat_batch = ortho6d_to_rotation_matrix(state[:, 3:9])  #(action_steps, 3, 3)
    left_rot_mat_batch = np.asarray(left_rot_mat_batch, dtype=np.float64)
    left_euler_batch = np.array([t3d.euler.mat2euler(rot_mat) for rot_mat in left_rot_mat_batch])  # (action_steps, 3)
    left_trans_batch = state[:, :3]  # (action_steps, 3)
    left_action_8d = np.concatenate([left_trans_batch, left_euler_batch, zeros_array], axis=1) # (action_steps, 8)
    
    return left_action_8d[0]

# Split by episode and save
prev_end = 0
for i in tqdm.tqdm(range(num_episodes)):
    start = int(prev_end)
    end = int(np.asarray(episode_ends[i]))
    prev_end = end

    episode_images = np.asarray(imgs[start:end], dtype=np.uint8)
    episode_states = np.asarray(states[start:end])
    episode_actions = np.asarray(actions[start:end], dtype=np.float32)

    episode_list = []
    for offset in range(end - start):


        action = convert_10d_action_to_7d_action(episode_actions[offset]).astype(np.float32)
        state = convert_9d_state_to_8d_state(episode_states[offset]).astype(np.float32)
        # print(state.shape)
        episode_list.append({
            "image": episode_images[offset], # (240, 320, 3), int
            "action": action, # (7), units: meters
            "state": state,
            "language_instruction": LANG_INSTRUCTION,
        })

        # print(f"action is {action}")# units: meters

    # Save as .npy
    # break
    np.save(staging_train_dir / f"episode_{i}.npy", episode_list)

if staging_dir != output_dir:
    if final_train_dir.exists():
        shutil.rmtree(final_train_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(staging_train_dir, final_train_dir)
    if args.cleanup_tmpfs_after_copy:
        shutil.rmtree(staging_dir)

print(f"\n🎯 Done. Saved {num_episodes} episodes to {final_train_dir}")
