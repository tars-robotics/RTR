import sys
import os
import pathlib
import hydra
from omegaconf import OmegaConf
from omegaconf import DictConfig
import torch
import numpy as np
# from termcolor import cprint
import copy
from loguru import logger
from hydra.utils import instantiate

from rtr_async_sys.models.reactive_diffusion_policy.policy.latent_diffusion_unet_image_policy import LatentDiffusionUnetImagePolicy

OmegaConf.register_new_resolver("eval", eval, replace=True)

def build_rdp_model(cfg: DictConfig) -> LatentDiffusionUnetImagePolicy:
    logger.info("Instantiating RDP policy...")
    model = instantiate(cfg)      
    return model


if __name__ == "__main__":
    # cli_main()
    cfg = OmegaConf.load(
        "src/rtr_async_sys/configs/model/rdp/rdp.yaml"
    )
    model:LatentDiffusionUnetImagePolicy = build_rdp_model(cfg)
    print(model)