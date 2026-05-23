"""
finetune.py

Fine-tunes OpenVLA via LoRA.
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
from prismatic.util.data_utils import PaddedCollatorForActionPrediction, PaddedCollatorForOpenvlaOftVAE
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

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class FinetuneConfig:
    # fmt: off
    vae_config_path: str = "src/rtr_async_sys/configs/user/model_wrapper/model/rdp/openvla_oft_vae.yaml"
    device: str="cuda:0"

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
    vae = hydra.utils.instantiate(vae_config)
    vae.to(cfg.device)
    vae.train()
    print(vae)
    # optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.optim_params)
    optimizer = torch.optim.AdamW(lr=1.0e-3, weight_decay=1.0e-4,params=vae.optim_params)

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
        image_aug=False,
        only_for_vae=True
    )


    save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    # Create collator and dataloader
    # collator = PaddedCollatorForActionPrediction(
    #     processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    # )
    collator = PaddedCollatorForOpenvlaOftVAE()
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=4,  # Important: Set to 0 if using RLDS, which uses its own parallelism
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

            for step_in_epoch in range(steps_per_epoch):
                batch = next(data_iter)  # Pull from the same infinite stream; no reload
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                if train_sampling_batch is None:
                    train_sampling_batch = batch

                loss_metric_dict = vae.compute_loss_and_metric(batch)
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
