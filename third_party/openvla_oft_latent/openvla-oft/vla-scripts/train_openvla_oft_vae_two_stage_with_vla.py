"""
finetune.py

finetune vae based on vla generated latent
"""

import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Type

import draccus
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
from accelerate import PartialState
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

from datetime import datetime

import wandb

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map,
)

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import DiffusionActionHead, L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import (  
    NoisyActionProjector,
    ProprioProjector,
)
from prismatic.training.train_utils import (
    compute_actions_l1_loss,
    compute_token_accuracy,
    get_current_action_mask,
    get_next_actions_mask,
)
from prismatic.util.data_utils import PaddedCollatorForActionPrediction, PaddedCollatorForOpenvlaOftVAE_with_vla
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
)
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

import json
from omegaconf import DictConfig, OmegaConf
import hydra

from rtr_async_sys.models.reactive_diffusion_policy.model.common.lr_scheduler import get_scheduler
from rtr_async_sys.models.reactive_diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from rtr_async_sys.models.reactive_diffusion_policy.common.json_logger import JsonLogger
from rtr_async_sys.models.reactive_diffusion_policy.common.pytorch_util import dict_apply, optimizer_to

import numpy as np
import pathlib
import copy
import threading
import dill

from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector, get_vla, get_vla_action, get_noisy_action_projector

import hashlib
from collections import OrderedDict


# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class FinetuneConfig:
    # fmt: off
    vae_config_path: str = "src/rtr_async_sys/configs/user/model_wrapper/model/rdp/openvla_oft_vae.yaml"
    device: str="cuda:0"
    # TODO: add the logic to compute dataset_statistics inside this script; otherwise OFT must be fine-tuned first just to get the statistics — a bit messy.
    vae_latent_dataset_statistics_path:str = "data/ckpts/openvla_oft_latent/vase_sponge_test1_60hz_oft_6drotate_not_aug_normalize_latent/60hz_6drotate_horizon48_latent_compress4_img_aug_normalize_latent--50000_chkpt/dataset_statistics.json"
    vae_load_path:str = "data/ckpts/vase_sponge_test1_60hz/ckpts_abs/openvla_oft/vae/horizon48_compress4_n_embed_10/latest.ckpt"

    # for vla
    vla_ckpt_path:str = "data/ckpts/openvla_oft_latent/vase_sponge_test1_60hz_oft_6drotate_not_aug_normalize_latent/60hz_6drotate_horizon48_latent_compress4_img_aug_normalize_latent--50000_chkpt"
    unnorm_key = "vase_sponge_test1_oft_6drotate_dataset"
    image_aug:bool = True

    # training for vae
    lr_warmup_steps:int = 100
    num_epochs:int = 1001 #1001
    gradient_accumulate_every:int = 1
    checkpoint_every:int = 10
    save_last_ckpt:bool = True
    tqdm_interval_sec:float = 1.0
    max_train_steps:int = None

    # Dataset
    data_root_dir: Path = Path("datasets/rlds")      # Directory containing RLDS datasets
    dataset_name: str = "aloha_scoop_x_into_bowl"    # Name of fine-tuning dataset (e.g., `aloha_scoop_x_into_bowl`)
    run_root_dir: Path = Path("runs")                # Path to directory to store logs & checkpoints
    shuffle_buffer_size: int = 100_000               # Dataloader shuffle buffer size (can reduce if OOM errors occur)


    # Training configuration
    batch_size: int = 64                              # Batch size per device (total batch size = batch_size * num GPUs)
    learning_rate: float = 5e-4                      # Learning rate

    

    # Logging
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "openvla_oft_vae"        # Name of WandB project
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    run_id_override: Optional[str] = None            # Optional string to override the run ID with
    wandb_log_freq: int = 10                         # WandB logging frequency in steps

    # fmt: on


def action_hash_key(action_tensor: torch.Tensor) -> str:
    """
    action_tensor: e.g. shape [T, D] or [D], torch float on GPU/CPU
    key uses raw bytes (stable if values are exactly identical).
    """
    x = action_tensor.detach()
    if x.is_cuda:
        x = x.cpu()
    x = x.contiguous()

    # Hash the bytes; include dtype/shape to prevent collisions
    h = hashlib.blake2b(digest_size=16)
    h.update(str(x.dtype).encode())
    h.update(str(tuple(x.shape)).encode())
    h.update(x.numpy().tobytes())
    return h.hexdigest()


def save_checkpoint(
        vae,
        output_dir,
        path=None, tag='latest', 
    ):
    if path is None:
        path = pathlib.Path(output_dir).joinpath('checkpoints', f'{tag}.ckpt')
    else:
        path = pathlib.Path(path)

    path.parent.mkdir(parents=False, exist_ok=True)
    payload = {
        'state_dicts': dict(),
    } 

    payload['state_dicts']['model'] = vae.state_dict()

    torch.save(payload, path.open('wb'), pickle_module=dill)
    return str(path.absolute())

def get_run_id(cfg) -> str:
    """
    Generates or retrieves an identifier string for an experiment run.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        str: Experiment run ID.
    """
    if cfg.run_id_override is not None:
        # Override the run ID with the user-provided ID
        run_id = cfg.run_id_override
    else:
        run_id = (
            f"openvla_oft_vae+b{cfg.batch_size}"
            f"+lr-{cfg.learning_rate}"
        )

    return run_id



@draccus.wrap()
def train_vae(cfg: FinetuneConfig) -> None:
    # Trim trailing forward slash ('/') in VLA path if it exists
    vae_config_path = cfg.vae_config_path
    vae_config = OmegaConf.load(vae_config_path)
    vae_config.second_stage=True
    vae = hydra.utils.instantiate(vae_config)

    # 
    print("load ckpts for two-stage-vae")
    payload = torch.load(cfg.vae_load_path)
    # Load the model weights
    vae.load_state_dict(payload['state_dicts']['model'])

    vae.second_stage_train()
    vae._load_dataset_statistics(cfg.vae_latent_dataset_statistics_path)
    vae.to(cfg.device)
    vae.train()
    print(vae)



    # optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.optim_params)
    print("maybe you should lower the lr for two-stage-training")
    # lr=1.0e-3#original
    # lr=5.0e-4
    lr = cfg.learning_rate
    optimizer = torch.optim.AdamW(lr=lr, weight_decay=1.0e-4,params=vae.optim_params)

    # Get experiment run ID
    run_id = get_run_id(cfg)
    # Create experiment run directory
    # run_dir = cfg.run_root_dir / run_id
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = cfg.run_root_dir / f"{ts}_{run_id}"


    os.makedirs(run_dir, exist_ok=True)

    # wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{run_id}")
    wandb_run = wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{run_id}")

    # Print detected constants
    print(
        "Detected constants:\n"
        f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
        f"\tACTION_DIM: {ACTION_DIM}\n"
        f"\tPROPRIO_DIM: {PROPRIO_DIM}\n"
        f"\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}"
    )


    train_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        None,
        resize_resolution=(256,256),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        only_for_vae=True,
        two_stage_with_vla=True
    )


    save_dataset_statistics(train_dataset.dataset_statistics, run_dir)


    # vla
    print(f"vla_ckpt_path is {cfg.vla_ckpt_path}")
    use_vla = True
    if use_vla:
        use_diffusion = False
        use_l1_regression = not use_diffusion

        vla_cfg = GenerateConfig(
            # pretrained_checkpoint = "moojink/openvla-7b-oft-finetuned-libero-spatial",
            pretrained_checkpoint=cfg.vla_ckpt_path,
            use_film = False,
            num_images_in_input = 1,
            use_proprio = True,
            load_in_8bit = False,
            load_in_4bit = False,
            unnorm_key=cfg.unnorm_key,
            use_l1_regression=use_l1_regression,
            use_diffusion=use_diffusion
        )
        vla = get_vla(vla_cfg)
        processor = get_processor(vla_cfg)
        action_head = get_action_head(vla_cfg, llm_dim=vla.llm_dim)
        proprio_projector = get_proprio_projector(vla_cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)
        if use_diffusion:
            noisy_action_projector = get_noisy_action_projector(vla_cfg, llm_dim=vla.llm_dim)
        else:
            noisy_action_projector = None

    # Create collator and dataloader
    # collator = PaddedCollatorForActionPrediction(
    #     processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    # )
    collator = PaddedCollatorForOpenvlaOftVAE_with_vla()
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0
        # num_workers=4,  # Important: Set to 0 if using RLDS, which uses its own parallelism
        # pin_memory=True,
        # persistent_workers=True
    )



    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps,
        num_training_steps=(
            len(dataloader) * cfg.num_epochs) \
                // cfg.gradient_accumulate_every,
    )

    # configure checkpoint
    topk_manager = TopKCheckpointManager(
        save_dir=os.path.join(run_dir, 'checkpoints'),
        monitor_key = 'train_loss',
        mode='min',
        k=1,
        format_str='epoch={epoch:04d}-train_loss={train_loss:.6f}.ckpt'
        # **cfg.checkpoint.topk
    )

    # device transfer
    device = torch.device(cfg.device)
    optimizer_to(optimizer, device)

    epoch = 0
    global_step = 0

    train_sampling_batch = None
    steps_per_epoch = len(train_dataset) // cfg.batch_size
    next_epoch_threshold = steps_per_epoch

    # training loop
    log_path = os.path.join(run_dir, "logs.json.txt")

    latent_cache = OrderedDict()   # key: str, value: torch.Tensor(CPU)
    max_cache_items = None#200_000      # Tune up/down based on memory; set to None to disable LRU eviction
    cache_hits = 0
    cache_misses = 0

    with JsonLogger(log_path) as json_logger:
        data_iter = iter(dataloader)  # Key: create once so the dataloader does not restart at every epoch

        for epoch in range(cfg.num_epochs):
            train_losses = []

            pbar = tqdm.tqdm(
                total=steps_per_epoch,
                desc=f"Training epoch {epoch+1}/{cfg.num_epochs}",
                leave=True,
                mininterval=cfg.tqdm_interval_sec,
                dynamic_ncols=True,
            )
            overfit = False
            if overfit:
                batch_iter = next(data_iter)  # Pull from the same infinite stream; no reload
            for step_in_epoch in range(steps_per_epoch):
                if overfit:
                    batch = dict_apply(batch_iter, lambda x: x.clone().to(device, non_blocking=True))
                else:
                    batch = next(data_iter)  # Pull from the same infinite stream; no reload
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                # Proprio must be denormalized; the VLA interface expects raw (un-normalized) proprio
                batch['state'] = vae.denormalize_from_dataset(batch['state'], is_proprio=True)
                action = batch['action']
                # observation = {
                #     'full_image': batch['full_image'], # b,t,h,w,c, uint8
                #     'state': batch['state']
                # }
                # import pdb;pdb.set_trace()
                language_instruction = 'wipe the vase.'
                batch_size = action.shape[0]
                latent_action_chunk = []
                for i in range(batch_size):
                    # Check the cache first
                    # key: only from action (as you requested)
                    k = action_hash_key(action[i])

                    cached = latent_cache.get(k, None)
                    if cached is not None:
                        cache_hits += 1
                        latent_i = cached.to(device, non_blocking=True)
                        # LRU: refresh order on hit
                        latent_cache.move_to_end(k)
                    else:
                        cache_misses += 1

                        obs = {
                            'full_image': batch['full_image'][i,0].cpu().numpy(),
                            'state': batch['state'][i,0].cpu().numpy()
                        }
                        # import pdb;pdb.set_trace()
                        with torch.no_grad():# Not using this interface would be much faster, but the interface makes it much more convenient
                            show_img = False
                            if show_img:
                                img = obs['full_image'] # h,w,c uint8, np.array
                                # TODO: save as image (img{i}.jpg or .png both work)
                                from PIL import Image

                                out_dir = "debug_imgs"#os.path.join(run_dir, "debug_imgs")
                                os.makedirs(out_dir, exist_ok=True)

                                # i comes from the outer `for i in range(batch_size)` loop
                                out_path = os.path.join(out_dir, f"step{step_in_epoch}_img{i}.png")  # or use .jpg

                                # Ensure uint8 and RGB
                                if img.dtype != np.uint8:
                                    img = img.astype(np.uint8)

                                # Some inputs may be (H,W) or (H,W,1) / (H,W,4)
                                if img.ndim == 2:
                                    pil_img = Image.fromarray(img, mode="L")
                                elif img.ndim == 3 and img.shape[2] == 1:
                                    pil_img = Image.fromarray(img[:, :, 0], mode="L")
                                elif img.ndim == 3 and img.shape[2] == 4:
                                    pil_img = Image.fromarray(img, mode="RGBA").convert("RGB")
                                else:
                                    pil_img = Image.fromarray(img, mode="RGB")

                                pil_img.save(out_path)


                            latent_np = get_vla_action(
                                vla_cfg, 
                                vla, 
                                processor, 
                                obs, 
                                language_instruction, 
                                action_head, 
                                proprio_projector,
                                noisy_action_projector = noisy_action_projector
                            )
                            # latent_action = torch.tensor(latent_action).to(device)
                        latent_i_cpu = torch.as_tensor(latent_np).detach().cpu()
                        latent_cache[k] = latent_i_cpu
                        latent_i = latent_i_cpu.to(device, non_blocking=True)

                        # LRU: evict the oldest entry when over capacity
                        if (max_cache_items is not None) and (len(latent_cache) > max_cache_items):
                            latent_cache.popitem(last=False)

                    latent_action_chunk.append(latent_i[None,:])

                latent_action = torch.concatenate(latent_action_chunk,dim=0)
                # print(f"latent_action.shape is {latent_action.shape}")

                # TO DEBUG
                # with torch.no_grad():
                #     encoded_action = vae.encode_to_latent(action)
                #     encoded_action = vae.normalize_from_dataset(encoded_action, is_latent=True)

                #     encoded_action = vae.denormalize_from_dataset(encoded_action, is_latent=True)
                #     encoded_action = vae.decode_from_latent(encoded_action)
                #     encoded_action = vae.denormalize_from_dataset(encoded_action, is_latent=False)

                #     decoded_latent_action = vae.denormalize_from_dataset(latent_action, is_latent=True)
                #     decoded_latent_action = vae.decode_from_latent(decoded_latent_action)
                #     decoded_latent_action = vae.denormalize_from_dataset(decoded_latent_action, is_latent=False)


                if train_sampling_batch is None:
                    train_sampling_batch = batch
                # import pdb;pdb.set_trace()
                loss_metric_dict = vae.compute_loss_and_metric_second_stage(action,vla_action_latent=latent_action)
                raw_loss = loss_metric_dict["loss"]
                loss = raw_loss / cfg.gradient_accumulate_every
                loss.backward()

                if global_step % cfg.gradient_accumulate_every == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()

                raw_loss_cpu = float(raw_loss.item())
                train_losses.append(raw_loss_cpu)

                step_log = {
                    "train_loss": raw_loss_cpu,
                    "global_step": global_step,
                    "epoch": epoch,
                    "lr": lr_scheduler.get_last_lr()[0],
                    "train_encoder_loss": loss_metric_dict["encoder_loss"],
                    "train_vae_recon_loss": loss_metric_dict["vae_recon_loss"],
                }
                if "kl_loss" in loss_metric_dict:
                    step_log["train_kl_loss"] = loss_metric_dict["kl_loss"]

                wandb_run.log(step_log, step=global_step)
                json_logger.log(step_log)
                global_step += 1

                pbar.set_postfix(loss=raw_loss_cpu, refresh=False)
                pbar.update(1)

            pbar.close()

            # ===== epoch end =====
            epoch_train_loss = float(np.mean(train_losses))
            epoch_log = {
                "train_loss": epoch_train_loss,
                "global_step": global_step,
                "epoch": epoch,
                "lr": lr_scheduler.get_last_lr()[0],
            }

            # checkpoint / extra logs (keep the original logic placement)
            if (epoch % cfg.checkpoint_every) == 0:
                if cfg.save_last_ckpt:
                    save_checkpoint(vae, run_dir)

                metric_dict = {k.replace("/", "_"): v for k, v in epoch_log.items()}
                topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                if topk_ckpt_path is not None:
                    save_checkpoint(vae, run_dir, path=topk_ckpt_path)

            wandb_run.log(epoch_log, step=global_step)
            json_logger.log(epoch_log)



if __name__ == "__main__":
    train_vae()
