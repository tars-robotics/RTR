"""
Convert a Zarr replay buffer (rtr_async_sys style) into LeRobot dataset format.

Features in output dataset:
- image:  uint8,  shape (H, W, 3)  (kept identical to convert_zarr_into_episodes)
- state:  float32, shape (8,)      (converted from 9D pose if needed; identical logic)
- actions: float32, shape (10,)    (kept identical to convert_zarr_into_episodes)

Usage:
python convert_zarr_to_lerobot.py \
  --zarr-path /path/to/replay_buffer.zarr \
  --fps 60 \
  --repo-name your_hf_username/libero_zarr

The resulting dataset will be saved under: $HF_LEROBOT_HOME/<repo-name>

For efficiency:
1. First convert the zarr file into episodes and store in memory instead of disk, which is much faster. then copy the dataset in memory to disk. Because image writing is bottleneck, this method can bring 4x speedup.
2. And set image_writer_threads to 8 and image_writer_processes to 4 to speed up the image writing process. Bring 2x speedup. (more multi-process is not beneficial because now image writing is not the main bottleneck.)
"""

from __future__ import annotations

import shutil
import time
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType
from typing import Tuple

import numpy as np
import tqdm
import tyro
import zarr
import transforms3d as t3d

from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# You already used this in your script; keep it the same to ensure identical behavior.
from rtr_async_sys.utils.action_utils import ortho6d_to_rotation_matrix


def convert_9d_state_to_8d_state(state: np.ndarray) -> np.ndarray:
    """
    Keep identical logic to your convert_zarr_into_episodes:
    - If state is already (8,), return as-is.
    - If (9,), treat dims 3:9 as ortho6d rotation, convert to euler, and append 2 zeros.
    Output: (8,)
    """
    state = np.asarray(state)
    if state.shape == (8,):
        return state
    if state.shape != (9,):
        raise ValueError(f"Expected state shape (8,) or (9,), got {state.shape}")

    state_b = state[None, :]
    zeros_array = np.zeros((state_b.shape[0], 2), dtype=state_b.dtype)

    rot_mat_batch = ortho6d_to_rotation_matrix(state_b[:, 3:9])  # (1, 3, 3)
    rot_mat_batch = np.asarray(rot_mat_batch, dtype=np.float64)
    euler_batch = np.array([t3d.euler.mat2euler(rot_mat) for rot_mat in rot_mat_batch])  # (1, 3)
    trans_batch = state_b[:, :3]  # (1, 3)

    out = np.concatenate([trans_batch, euler_batch, zeros_array], axis=1)  # (1, 8)
    return out[0]


def infer_image_shape(imgs: zarr.Array) -> Tuple[int, int, int]:
    """
    imgs is expected to be (N_steps, H, W, C)
    """
    if len(imgs.shape) != 4:
        raise ValueError(f"Expected imgs shape (N, H, W, C), got {imgs.shape}")
    _, h, w, c = imgs.shape
    return int(h), int(w), int(c)


@dataclass
class ProfileStats:
    zarr_open_s: float = 0.0
    metadata_init_s: float = 0.0
    output_cleanup_s: float = 0.0
    staging_cleanup_s: float = 0.0
    dataset_create_s: float = 0.0
    episode_loop_s: float = 0.0
    image_read_s: float = 0.0
    state_read_s: float = 0.0
    action_read_s: float = 0.0
    shape_check_s: float = 0.0
    add_frame_s: float = 0.0
    save_episode_s: float = 0.0
    save_episode_validate_s: float = 0.0
    save_episode_save_tasks_s: float = 0.0
    save_episode_get_task_index_s: float = 0.0
    save_episode_wait_image_writer_s: float = 0.0
    save_episode_compute_stats_s: float = 0.0
    save_episode_data_write_s: float = 0.0
    save_episode_video_s: float = 0.0
    save_episode_meta_save_s: float = 0.0
    save_episode_clear_buffer_s: float = 0.0
    copy_to_final_s: float = 0.0
    tmpfs_cleanup_after_copy_s: float = 0.0
    total_runtime_s: float = 0.0
    frame_count: int = 0
    episode_count: int = 0
    episode_timings: list[tuple[int, float, float, int]] = field(default_factory=list)


def _print_profile_summary(stats: ProfileStats) -> None:
    print("\n=== Profiling Summary ===")

    sections = [
        ("zarr_open", stats.zarr_open_s),
        ("metadata_init", stats.metadata_init_s),
        ("output_cleanup", stats.output_cleanup_s),
        ("staging_cleanup", stats.staging_cleanup_s),
        ("dataset_create", stats.dataset_create_s),
        ("episode_loop", stats.episode_loop_s),
        ("  image_read", stats.image_read_s),
        ("  state_read", stats.state_read_s),
        ("  action_read", stats.action_read_s),
        ("  shape_check", stats.shape_check_s),
        ("  add_frame", stats.add_frame_s),
        ("  save_episode", stats.save_episode_s),
        ("copy_to_final", stats.copy_to_final_s),
        ("tmpfs_cleanup", stats.tmpfs_cleanup_after_copy_s),
    ]

    total_runtime = max(stats.total_runtime_s, 1e-12)
    for name, elapsed_s in sections:
        pct = elapsed_s / total_runtime * 100.0
        print(f"{name:>16}: {elapsed_s:8.3f}s  ({pct:5.1f}%)")

    print(f"{'total_runtime':>16}: {stats.total_runtime_s:8.3f}s  (100.0%)")
    print(f"{'frames':>16}: {stats.frame_count}")
    print(f"{'episodes':>16}: {stats.episode_count}")

    if stats.frame_count > 0:
        print("\nPer-frame averages:")
        print(f"{'image_read':>16}: {stats.image_read_s / stats.frame_count * 1e3:8.3f} ms/frame")
        print(f"{'state_read':>16}: {stats.state_read_s / stats.frame_count * 1e3:8.3f} ms/frame")
        print(f"{'action_read':>16}: {stats.action_read_s / stats.frame_count * 1e3:8.3f} ms/frame")
        print(f"{'shape_check':>16}: {stats.shape_check_s / stats.frame_count * 1e3:8.3f} ms/frame")
        print(f"{'add_frame':>16}: {stats.add_frame_s / stats.frame_count * 1e3:8.3f} ms/frame")

    if stats.episode_count > 0:
        print("\nPer-episode averages:")
        print(f"{'save_episode':>16}: {stats.save_episode_s / stats.episode_count:8.3f} s/episode")
        print(f"{'episode_total':>16}: {stats.episode_loop_s / stats.episode_count:8.3f} s/episode")

    if stats.save_episode_s > 0:
        print("\n`save_episode` breakdown:")
        save_episode_breakdown = [
            ("validate_buffer", stats.save_episode_validate_s),
            ("save_tasks", stats.save_episode_save_tasks_s),
            ("get_task_index", stats.save_episode_get_task_index_s),
            ("wait_image_writer", stats.save_episode_wait_image_writer_s),
            ("compute_stats", stats.save_episode_compute_stats_s),
            ("save_episode_data", stats.save_episode_data_write_s),
            ("save_episode_video", stats.save_episode_video_s),
            ("meta.save_episode", stats.save_episode_meta_save_s),
            ("clear_buffer", stats.save_episode_clear_buffer_s),
        ]
        for name, elapsed_s in save_episode_breakdown:
            pct = elapsed_s / stats.save_episode_s * 100.0
            print(f"{name:>16}: {elapsed_s:8.3f}s  ({pct:5.1f}% of save_episode)")

        accounted_s = sum(elapsed_s for _, elapsed_s in save_episode_breakdown)
        other_s = max(stats.save_episode_s - accounted_s, 0.0)
        print(f"{'other_inline_work':>16}: {other_s:8.3f}s  ({other_s / stats.save_episode_s * 100.0:5.1f}% of save_episode)")

    if stats.episode_timings:
        print("\nSlowest episodes by total time:")
        slowest = sorted(stats.episode_timings, key=lambda x: x[1], reverse=True)[:5]
        for episode_idx, total_s, save_s, length in slowest:
            print(
                f"episode={episode_idx:4d}, frames={length:4d}, "
                f"total={total_s:7.3f}s, save_episode={save_s:7.3f}s"
            )

    print("\nNote: `add_frame` mainly measures enqueue/buffer overhead.")
    print("Async image writing and final flushing may show up more strongly in `save_episode`.")


def _install_save_episode_profiling(dataset: LeRobotDataset, stats: ProfileStats) -> None:
    dataset_module = importlib.import_module("lerobot.datasets.lerobot_dataset")

    original_validate_episode_buffer = dataset_module.validate_episode_buffer
    original_compute_episode_stats = dataset_module.compute_episode_stats
    original_wait_image_writer = dataset._wait_image_writer
    original_save_episode_data = dataset._save_episode_data
    original_save_episode_video = dataset._save_episode_video
    original_meta_save_episode = dataset.meta.save_episode
    original_meta_save_episode_tasks = dataset.meta.save_episode_tasks
    original_meta_get_task_index = dataset.meta.get_task_index
    original_clear_episode_buffer = dataset.clear_episode_buffer

    def profiled_validate_episode_buffer(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return original_validate_episode_buffer(*args, **kwargs)
        finally:
            stats.save_episode_validate_s += time.perf_counter() - t0

    def profiled_compute_episode_stats(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return original_compute_episode_stats(*args, **kwargs)
        finally:
            stats.save_episode_compute_stats_s += time.perf_counter() - t0

    def profiled_wait_image_writer(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return original_wait_image_writer(*args, **kwargs)
        finally:
            stats.save_episode_wait_image_writer_s += time.perf_counter() - t0

    def profiled_save_episode_data(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return original_save_episode_data(*args, **kwargs)
        finally:
            stats.save_episode_data_write_s += time.perf_counter() - t0

    def profiled_save_episode_video(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return original_save_episode_video(*args, **kwargs)
        finally:
            stats.save_episode_video_s += time.perf_counter() - t0

    def profiled_meta_save_episode(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return original_meta_save_episode(*args, **kwargs)
        finally:
            stats.save_episode_meta_save_s += time.perf_counter() - t0

    def profiled_meta_save_episode_tasks(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return original_meta_save_episode_tasks(*args, **kwargs)
        finally:
            stats.save_episode_save_tasks_s += time.perf_counter() - t0

    def profiled_meta_get_task_index(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return original_meta_get_task_index(*args, **kwargs)
        finally:
            stats.save_episode_get_task_index_s += time.perf_counter() - t0

    def profiled_clear_episode_buffer(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return original_clear_episode_buffer(*args, **kwargs)
        finally:
            stats.save_episode_clear_buffer_s += time.perf_counter() - t0

    dataset_module.validate_episode_buffer = profiled_validate_episode_buffer
    dataset_module.compute_episode_stats = profiled_compute_episode_stats
    dataset._wait_image_writer = profiled_wait_image_writer
    dataset._save_episode_data = profiled_save_episode_data
    dataset._save_episode_video = profiled_save_episode_video
    dataset.meta.save_episode = profiled_meta_save_episode
    dataset.meta.save_episode_tasks = profiled_meta_save_episode_tasks
    dataset.meta.get_task_index = profiled_meta_get_task_index
    dataset.clear_episode_buffer = profiled_clear_episode_buffer


def _install_in_memory_image_mode(dataset: LeRobotDataset) -> None:
    dataset_module = importlib.import_module("lerobot.datasets.lerobot_dataset")
    compute_stats_module = importlib.import_module("lerobot.datasets.compute_stats")
    original_sample_images = compute_stats_module.sample_images

    def sample_images_from_memory(image_values) -> np.ndarray:
        if len(image_values) == 0 or isinstance(image_values[0], str):
            return original_sample_images(image_values)

        sampled_indices = compute_stats_module.sample_indices(len(image_values))
        images = None
        for i, idx in enumerate(sampled_indices):
            img = image_values[idx]
            if not isinstance(img, np.ndarray):
                img = np.asarray(img)

            if img.ndim != 3:
                raise ValueError(f"Expected image with 3 dims, got shape {img.shape}")
            if img.shape[0] != 3 and img.shape[-1] == 3:
                img = np.transpose(img, (2, 0, 1))
            elif img.shape[0] != 3:
                raise ValueError(f"Expected CHW or HWC image with 3 channels, got shape {img.shape}")

            if img.dtype != np.uint8:
                if np.issubdtype(img.dtype, np.floating):
                    img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)

            img = compute_stats_module.auto_downsample_height_width(img)
            if images is None:
                images = np.empty((len(sampled_indices), *img.shape), dtype=np.uint8)
            images[i] = img

        return images

    def add_frame_in_memory(self, frame: dict) -> None:
        for name in frame:
            if isinstance(frame[name], dataset_module.torch.Tensor):
                frame[name] = frame[name].numpy()

        dataset_module.validate_frame(frame, self.features)

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        frame_index = self.episode_buffer["size"]
        timestamp = frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)
        self.episode_buffer["task"].append(frame.pop("task"))

        for key in frame:
            if key not in self.features:
                raise ValueError(
                    f"An element of the frame is not in the features. '{key}' not in '{self.features.keys()}'."
                )

            dtype = self.features[key]["dtype"]
            if dtype == "image":
                image = frame[key]
                if not isinstance(image, np.ndarray):
                    image = np.asarray(image)
                self.episode_buffer[key].append(image)
            elif dtype == "video":
                img_path = self._get_image_file_path(
                    episode_index=self.episode_buffer["episode_index"], image_key=key, frame_index=frame_index
                )
                if frame_index == 0:
                    img_path.parent.mkdir(parents=True, exist_ok=True)
                self._save_image(frame[key], img_path, compress_level=1)
                self.episode_buffer[key].append(str(img_path))
            else:
                self.episode_buffer[key].append(frame[key])

        self.episode_buffer["size"] += 1

    compute_stats_module.sample_images = sample_images_from_memory
    dataset.add_frame = MethodType(add_frame_in_memory, dataset)
    dataset.stop_image_writer()


@dataclass
class Args:
    zarr_path: str
    fps: int = 60
    repo_name: str = "sadpiggy/xarm_vase_sponge_test1_60hz"
    robot_type: str = "xarm"
    image_writer_threads: int = 8 #10
    image_writer_processes: int = 4 #5
    stage_in_tmpfs: bool = True
    tmpfs_root: str = "/dev/shm/lerobot"
    cleanup_tmpfs_after_copy: bool = True
    store_images_in_memory: bool = False
    enable_profile: bool = True
    push_to_hub: bool = False  # optional, matches LeRobot example
    num_episodes: int = -1 # default to all episodes


def main(args: Args) -> None:
    stats = ProfileStats()
    total_start_t = time.perf_counter()

    # ---- Load Zarr ----
    t0 = time.perf_counter()
    zarr_root = zarr.open(args.zarr_path, mode="r")
    stats.zarr_open_s += time.perf_counter() - t0

    t0 = time.perf_counter()
    imgs = zarr_root["data/left_wrist_img"]          # (N, H, W, C), uint8
    states = zarr_root["data/left_robot_tcp_pose"]   # (N, 9) or (N, 8)
    actions = zarr_root["data/action"]               # (N, 10) (as in your script)
    episode_ends = zarr_root["meta/episode_ends"]    # (num_episodes,)

    total_steps = int(imgs.shape[0])
    if args.num_episodes == -1:
        num_episodes = int(len(episode_ends))
    else:
        print(f"Using {args.num_episodes} episodes for debug")
        num_episodes = min(args.num_episodes, int(len(episode_ends)))
    h, w, c = infer_image_shape(imgs)
    stats.metadata_init_s += time.perf_counter() - t0

    print(f"Loaded {total_steps} total steps, with {num_episodes} episodes.")
    print(f"Image shape: ({h}, {w}, {c}), imgs dtype: {imgs.dtype}")
    print(f"State shape per step in zarr: {states[0].shape}, actions shape per step: {actions[0].shape}")
    action_shape = np.asarray(actions[0]).shape

    # ---- Prepare output dirs ----
    output_path = HF_LEROBOT_HOME / args.repo_name
    staging_path = Path(args.tmpfs_root) / args.repo_name if args.stage_in_tmpfs else output_path

    if output_path.exists():
        t0 = time.perf_counter()
        shutil.rmtree(output_path)
        stats.output_cleanup_s += time.perf_counter() - t0

    if staging_path != output_path and staging_path.exists():
        t0 = time.perf_counter()
        shutil.rmtree(staging_path)
        stats.staging_cleanup_s += time.perf_counter() - t0

    if staging_path != output_path:
        print(f"Staging dataset in {staging_path}")

    # ---- Create LeRobot dataset ----
    t0 = time.perf_counter()
    dataset = LeRobotDataset.create(
        repo_id=args.repo_name,
        root=staging_path,
        robot_type=args.robot_type,
        fps=int(args.fps),
        features={
            "observation.images.image": {
                "dtype": "image",
                "shape": (h, w, c),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (9,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (int(np.asarray(actions[0]).shape[0]),),
                "names": ["action"],
            },
        },
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )
    stats.dataset_create_s += time.perf_counter() - t0
    if args.store_images_in_memory:
        print("Using in-memory image mode")
        _install_in_memory_image_mode(dataset)
    _install_save_episode_profiling(dataset, stats)

    # ---- Write episodes ----
    prev_end = 0
    for i in tqdm.tqdm(range(num_episodes), desc="Writing episodes"):
        episode_start_t = time.perf_counter()
        end = int(np.asarray(episode_ends[i]))
        start = prev_end
        prev_end = end

        t0 = time.perf_counter()
        episode_images = np.asarray(imgs[start:end], dtype=np.uint8)
        stats.image_read_s += time.perf_counter() - t0

        t0 = time.perf_counter()
        episode_states = np.asarray(states[start:end], dtype=np.float32)
        stats.state_read_s += time.perf_counter() - t0

        t0 = time.perf_counter()
        episode_actions = np.asarray(actions[start:end], dtype=np.float32)
        stats.action_read_s += time.perf_counter() - t0

        for offset, step in enumerate(range(start, end)):
            image = episode_images[offset]  # (H, W, 3)
            # state = convert_9d_state_to_8d_state(episode_states[offset]).astype(np.float32)  # (8,)
            state = episode_states[offset] # (9,)
            act = episode_actions[offset]  # (10,)

            # Basic shape checks to keep consistent with your convert_zarr_into_episodes behavior
            t0 = time.perf_counter()
            if image.shape != (h, w, c):
                raise ValueError(f"Unexpected image shape at step {step}: {image.shape} vs {(h, w, c)}")
            if state.shape != (9,):
                raise ValueError(f"Unexpected state shape at step {step}: {state.shape}")
            # actions shape can be inferred from the first action
            if act.shape != action_shape:
                raise ValueError(f"Unexpected actions shape at step {step}: {act.shape} vs {action_shape}")
            stats.shape_check_s += time.perf_counter() - t0

            t0 = time.perf_counter()
            dataset.add_frame(
                {
                    "observation.images.image": image,
                    "observation.state": state,
                    "action": act,
                    "task": "wipe the vase.",
                }
            )
            stats.add_frame_s += time.perf_counter() - t0
            stats.frame_count += 1

        t0 = time.perf_counter()
        dataset.save_episode()
        save_episode_elapsed_s = time.perf_counter() - t0
        stats.save_episode_s += save_episode_elapsed_s
        stats.episode_count += 1
        episode_elapsed_s = time.perf_counter() - episode_start_t
        stats.episode_loop_s += episode_elapsed_s
        stats.episode_timings.append((i, episode_elapsed_s, save_episode_elapsed_s, end - start))

    dataset._close_writer()
    dataset.meta._close_writer()
    if dataset.image_writer is not None:
        dataset.stop_image_writer()

    if staging_path != output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        shutil.copytree(staging_path, output_path)
        stats.copy_to_final_s += time.perf_counter() - t0
        dataset.root = output_path
        dataset.meta.root = output_path

        if args.cleanup_tmpfs_after_copy:
            t0 = time.perf_counter()
            shutil.rmtree(staging_path)
            stats.tmpfs_cleanup_after_copy_s += time.perf_counter() - t0

    print(f"\n✅ Done! Saved {num_episodes} episodes to: {output_path}")
    stats.total_runtime_s = time.perf_counter() - total_start_t

    if args.enable_profile:
        _print_profile_summary(stats)

    # ---- Optional: push to hub ----
    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["zarr", "lerobot"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print("🚀 Pushed to Hugging Face Hub.")


if __name__ == "__main__":
    tyro.cli(main)
