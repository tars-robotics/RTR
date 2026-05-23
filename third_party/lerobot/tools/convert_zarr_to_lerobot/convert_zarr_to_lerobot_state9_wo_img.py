"""
Convert a Zarr replay buffer into a LeRobot dataset without image/state features.

This is the efficient counterpart of convert_zarr_to_lerobot_state9_wo_img.py:
- reads actions episode-by-episode instead of one zarr row at a time
- optionally stages the output dataset in tmpfs and copies it back at the end
- supports profiling and a num_episodes debug limit
"""

from __future__ import annotations

import importlib
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np
import tqdm
import transforms3d as t3d
import tyro
import zarr

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME

from rtr_async_sys.utils.action_utils import ortho6d_to_rotation_matrix


def convert_9d_state_to_8d_state(state: np.ndarray) -> np.ndarray:
    """
    Keep identical logic to the non-efficient script in case this helper is reused.
    The wo_img converter currently does not write state features.
    """
    state = np.asarray(state)
    if state.shape == (8,):
        return state
    if state.shape != (9,):
        raise ValueError(f"Expected state shape (8,) or (9,), got {state.shape}")

    state_b = state[None, :]
    zeros_array = np.zeros((state_b.shape[0], 2), dtype=state_b.dtype)

    rot_mat_batch = ortho6d_to_rotation_matrix(state_b[:, 3:9])
    rot_mat_batch = np.asarray(rot_mat_batch, dtype=np.float64)
    euler_batch = np.array([t3d.euler.mat2euler(rot_mat) for rot_mat in rot_mat_batch])
    trans_batch = state_b[:, :3]

    out = np.concatenate([trans_batch, euler_batch, zeros_array], axis=1)
    return out[0]


def infer_image_shape(imgs: zarr.Array) -> Tuple[int, int, int]:
    """
    Kept for compatibility with the original wo_img script.
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


@dataclass
class Args:
    zarr_path: str
    fps: int = 60
    repo_name: str = "sadpiggy/xarm_vase_sponge_test1_60hz"
    robot_type: str = "xarm"
    image_writer_threads: int = 10
    image_writer_processes: int = 5
    stage_in_tmpfs: bool = True
    tmpfs_root: str = "/dev/shm/lerobot"
    cleanup_tmpfs_after_copy: bool = True
    enable_profile: bool = True
    push_to_hub: bool = False
    num_episodes: int = -1


def main(args: Args) -> None:
    stats = ProfileStats()
    total_start_t = time.perf_counter()

    t0 = time.perf_counter()
    zarr_root = zarr.open(args.zarr_path, mode="r")
    stats.zarr_open_s += time.perf_counter() - t0

    t0 = time.perf_counter()
    actions = zarr_root["data/action"]
    episode_ends = zarr_root["meta/episode_ends"]
    total_steps = int(actions.shape[0])
    if args.num_episodes == -1:
        num_episodes = int(len(episode_ends))
    else:
        print(f"Using {args.num_episodes} episodes for debug")
        num_episodes = min(args.num_episodes, int(len(episode_ends)))
    action_shape = np.asarray(actions[0]).shape
    stats.metadata_init_s += time.perf_counter() - t0

    print(f"Loaded {total_steps} total steps, with {num_episodes} episodes.")
    print(f"Actions shape per step: {action_shape}, dtype: {actions.dtype}")

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

    t0 = time.perf_counter()
    dataset = LeRobotDataset.create(
        repo_id=args.repo_name,
        root=staging_path,
        robot_type=args.robot_type,
        fps=int(args.fps),
        features={
            "action": {
                "dtype": "float32",
                "shape": (int(action_shape[0]),),
                "names": ["action"],
            },
        },
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )
    stats.dataset_create_s += time.perf_counter() - t0
    _install_save_episode_profiling(dataset, stats)

    prev_end = 0
    for i in tqdm.tqdm(range(num_episodes), desc="Writing episodes"):
        episode_start_t = time.perf_counter()
        end = int(np.asarray(episode_ends[i]))
        start = prev_end
        prev_end = end

        t0 = time.perf_counter()
        episode_actions = np.asarray(actions[start:end], dtype=np.float32)
        stats.action_read_s += time.perf_counter() - t0

        for offset, step in enumerate(range(start, end)):
            act = episode_actions[offset]

            t0 = time.perf_counter()
            if act.shape != action_shape:
                raise ValueError(f"Unexpected actions shape at step {step}: {act.shape} vs {action_shape}")
            stats.shape_check_s += time.perf_counter() - t0

            t0 = time.perf_counter()
            dataset.add_frame(
                {
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

    print(f"\nDone! Saved {num_episodes} episodes to: {output_path}")
    stats.total_runtime_s = time.perf_counter() - total_start_t

    if args.enable_profile:
        _print_profile_summary(stats)

    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["zarr", "lerobot"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print("Pushed to Hugging Face Hub.")


if __name__ == "__main__":
    tyro.cli(main)
