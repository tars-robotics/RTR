#!/usr/bin/env python
# scripts/codes/train_pi0.5/lerobot_train_vae.py

"""
eval rdp vae.
"""

import os
import json
import time
import pathlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import dill
import numpy as np
import torch
import tqdm
import wandb
from torch.utils.data import DataLoader

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging
from lerobot.utils.random_utils import set_seed

from rtr_async_sys.models.reactive_diffusion_policy.model.common.lr_scheduler import get_scheduler
from rtr_async_sys.models.reactive_diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from rtr_async_sys.models.reactive_diffusion_policy.common.json_logger import JsonLogger
from rtr_async_sys.models.reactive_diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
# from rtr_async_sys.models.vla_vae.pi0_5_vae import Pi0_5_VAE
from rtr_async_sys.models.reactive_diffusion_policy.model.vae.model import VAE


from omegaconf import OmegaConf
import hydra
from pathlib import Path
import pickle



# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------------------------
# Hard-coded extras (not in TrainPipelineConfig)
# You can move them into cfg later.
# ---------------------------
# cfg.vae_config_path = "src/rtr_async_sys/configs/user/model_wrapper/model/rdp/pi0_5_vae.yaml"

LR = 1.0e-3
WEIGHT_DECAY = 1.0e-4
LR_WARMUP_STEPS = 100
GRAD_ACCUM_EVERY = 1

NUM_EPOCHS_FALLBACK = 1  # if cfg.steps is large, we loop epochs until global_step reaches cfg.steps
TOPK_K = 1  # keep best 1 by train_loss

TORCH_CUDNN_BENCHMARK = True
TORCH_ALLOW_TF32 = True



def _to_jsonable(obj):
    """
    Recursively convert common non-JSON types (numpy/torch) into JSON-serializable
    Python types.
    """
    # torch
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:
        pass

    # numpy
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):  # np.float32, np.int64, ...
            return obj.item()
    except Exception:
        pass

    # basic containers
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]

    # pathlib
    if isinstance(obj, Path):
        return str(obj)

    # fallback (basic types should pass through)
    return obj


def _safe_json_dump(data, path: str, indent: int = 2):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    jsonable = _to_jsonable(data)
    with path.open("w", encoding="utf-8") as f:
        json.dump(jsonable, f, ensure_ascii=False, indent=indent, sort_keys=True)


def _compute_stats_like_action(x: np.ndarray) -> dict:
    """
    x: np.ndarray, shape [N, D]  (N=all samples across steps/batches, D=latent dim)
    returns: dict with same keys/format as your action stats (lists of floats)
    """
    assert x.ndim == 2, f"expected [N,D], got {x.shape}"
    q01, q10, q50, q90, q99 = np.percentile(x, [1, 10, 50, 90, 99], axis=0)

    stats = {
        "count": [int(x.shape[0])],
        "min":  x.min(axis=0).astype(np.float64).tolist(),
        "max":  x.max(axis=0).astype(np.float64).tolist(),
        "mean": x.mean(axis=0).astype(np.float64).tolist(),
        "std":  x.std(axis=0, ddof=0).astype(np.float64).tolist(),  # ddof=0 to match typical dataset stats
        "q01":  q01.astype(np.float64).tolist(),
        "q10":  q10.astype(np.float64).tolist(),
        "q50":  q50.astype(np.float64).tolist(),
        "q90":  q90.astype(np.float64).tolist(),
        "q99":  q99.astype(np.float64).tolist(),
    }
    return stats

@dataclass
class EvalVaePipelineConfig(TrainPipelineConfig):
    vae_load_path: str | None = None
    get_latent_statistics: bool = False
    vae_config_path: str|None = "src/rtr_async_sys/configs/user/model_wrapper/model/rdp/pi0_5_vae.yaml"
    latent_stats: str|None = None

@parser.wrap()
def main(cfg: EvalVaePipelineConfig):
    """
    Train VAE:
      dataset -> dataloader -> preprocessor(batch) -> vae.compute_loss_and_metric(proc_batch)
    """
    # ---- logging / seed / device ----
    init_logging(accelerator=None)  # no accelerate
    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=None)

    device_str = getattr(cfg.policy, "device", "cuda")
    device = torch.device(device_str if device_str != "cuda" else "cuda:0")

    torch.backends.cudnn.benchmark = TORCH_CUDNN_BENCHMARK
    torch.backends.cuda.matmul.allow_tf32 = TORCH_ALLOW_TF32

    # ---- run dir ----
    # follow your command: --output_dir=...
    run_dir = pathlib.Path(cfg.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # optional: timestamp subdir (comment out if you want exact output_dir only)
    # ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    # run_dir = run_dir / f"{ts}_{cfg.job_name}"
    # run_dir.mkdir(parents=True, exist_ok=True)

    # ---- construct VAE ----
    vae_config = OmegaConf.load(cfg.vae_config_path)
    vae:VAE = hydra.utils.instantiate(vae_config)
    # import pdb;pdb.set_trace()
    print(f"cfg.vae_load_path is {cfg.vae_load_path}")
    payload = torch.load(cfg.vae_load_path, weights_only=False, map_location="cpu")
    # Load the model weights
    vae.load_state_dict(payload['state_dicts']['model'])
    vae.to(device)
    vae.eval()
    print(vae)

    use_latent_stats = (cfg.latent_stats is not None)
    if use_latent_stats:
        vae._load_latent_dataset_statistics(cfg.latent_stats)
    print(f"use_latent_stats is {use_latent_stats}")


    # ---- dataset ----
    dataset = make_dataset(cfg)
    # Save lerobot dataset stats as json (instead of save_dataset_statistics)
    stats_path = run_dir / "dataset_stats.json"
    try:
        # import pdb;pdb.set_trace()
        _safe_json_dump(dataset.meta.stats, str(stats_path))
        print(f"[INFO] Saved dataset.meta.stats to: {stats_path}")
    except Exception as e:
        print(f"[WARN] Failed to dump dataset.meta.stats to json: {e}")

    # vae._load_dataset_statistics(stats_path)

    

    # ---- dataloader (no accelerate, no cycle) ----
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=False,
        )
    else:
        shuffle = False
        sampler = None

    # num_workers: 16->14s; 4->44s
    dataloader = DataLoader(
        dataset,
        num_workers=16,#cfg.num_workers,# default is 4, set to 8
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    if cfg.get_latent_statistics:
        total_steps = len(dataloader)
    else:
        total_steps = int(getattr(cfg, "steps", 0) or 0)
    print(f"total_steps is {total_steps}")
    if total_steps <= 0:
        raise ValueError("cfg.steps must be > 0 (pass --steps=...)")


    global_step = 0


    total_losses = []
    plot_actions = []
    latent_actions = []

    latency_list = []

    for batch in tqdm.tqdm(dataloader):
        if global_step >= total_steps:
            break

        # move to device
        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True) if torch.is_tensor(x) else x)

        # preprocess -> feed VAE
        action = batch['action']
        # naction = vae.normalize_from_dataset(action,is_latent=False)

        if cfg.get_latent_statistics:
            latent_action = vae.encode_to_latent(action)
            latent_actions.append(latent_action.detach().cpu().numpy())
        

        # predicted_action = vae.encode_then_decode(naction)
        # predicted_action = vae.denormalize_from_dataset(predicted_action,is_latent=False)
        start_time = time.time()
        if use_latent_stats:
            latent = vae.encode_to_latent(batch)
            normalization_type="NORMAL"
            # normalization_type="QUANTILES"
            latent = vae.normalize_from_dataset(latent,is_latent=True, normalization_type=normalization_type)
            latent = vae.denormalize_from_dataset(latent,is_latent=True, normalization_type=normalization_type)
            predicted_action = vae.decode_from_latent(latent).to(device)
        else:
            # predicted_action = vae.encode_then_decode(batch).to(device)
            predicted_action = vae.encode_to_latent(batch)
            predicted_action = vae.decode_from_latent(predicted_action).to(device)
        end_time = time.time()
        latency_list.append(end_time-start_time)
        # import pdb;pdb.set_trace()

        l1loss = torch.mean(torch.abs(predicted_action - action))

        predicted_action = predicted_action.detach().cpu().numpy()
        action = action.detach().cpu().numpy()
        plot_action = {
                'fact':[],
                'predict':[] ,
        }
        for i in range(predicted_action.shape[1]):
            plot_action['fact'].append(action[0][i][0:3]*1000)
            plot_action['predict'].append(predicted_action[0][i][0:3]*1000)

        total_losses.append(l1loss)
        plot_actions.append(plot_action)

        global_step += 1

    mean_loss = sum(total_losses)/len(total_losses)
    print(f"mean loss is {mean_loss}")
    print(f"total step is {len(total_losses)}")
    mean_latency = sum(latency_list) / len(latency_list)
    print(f"mean latency is {mean_latency}")
    if not cfg.get_latent_statistics:
        save_path = "outputs/vis_outputs/dp_plot_actions.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(plot_actions, f)

    # TODO: get mean abs delta_x, delta_y, delta_z
    from rtr_async_sys.tools.utils import mean_abs_delta_xyz
    stats_xyz = mean_abs_delta_xyz(plot_actions)
    print("mean abs delta (mm):", stats_xyz)
    
    if cfg.get_latent_statistics:
        # latent_actions: list of np arrays. each element could be [B, T, D] or [B, D] depending on encode_to_latent
        # First reshape uniformly to [N, D]
        latent_np_list = []
        for arr in latent_actions:
            arr = np.asarray(arr)
            if arr.ndim == 3:          # [B, T, D]
                latent_np_list.append(arr.reshape(-1, arr.shape[-1]))
            elif arr.ndim == 2:        # [B, D]
                latent_np_list.append(arr.reshape(-1, arr.shape[-1]))
            elif arr.ndim == 1:        # [D]
                latent_np_list.append(arr[None, :])
            else:
                raise ValueError(f"Unsupported latent_action ndim={arr.ndim}, shape={arr.shape}")

        latent_all = np.concatenate(latent_np_list, axis=0)  # [N, D]
        latent_stats = _compute_stats_like_action(latent_all)

        # Store at the same structural level as action, e.g.:
        # stats_out = {"action": action_stats, "latent_action": latent_stats}
        # Or store latent_action on its own:
        # stats_out = {"latent_action": latent_stats}
        dataset.meta.stats['latent_action'] = latent_stats

        # json.dump no longer fails on ndarrays (everything is converted to python list/float/int)
        # import json
        # latent_stats_path = "/path/to/latent_action_stats.json"
        # with open(latent_stats_path, "w") as f:
        #     json.dump(stats_out, f, indent=2)
        try:
            # import pdb;pdb.set_trace()
            _safe_json_dump(dataset.meta.stats, str(stats_path))
            print(f"[INFO] Saved dataset.meta.stats to: {stats_path}")
        except Exception as e:
            print(f"[WARN] Failed to dump dataset.meta.stats to json: {e}")


if __name__ == "__main__":
    register_third_party_plugins()
    main()
