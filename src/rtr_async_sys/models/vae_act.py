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

from rtr_async_sys.models.reactive_diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy

OmegaConf.register_new_resolver("eval", eval, replace=True)

def build_dp_model(cfg: DictConfig) -> DiffusionUnetImagePolicy:
    logger.info("Instantiating Dp policy...")
    model = instantiate(cfg)      # or instantiate(cfg.env), depending on the config structure
    return model

if __name__ == "__main__":
    # cli_main()
    cfg = OmegaConf.load(
        "src/rtr_async_sys/configs/model/rdp/dp.yaml"
    )
    model:DiffusionUnetImagePolicy = build_dp_model(cfg)
    print(model)
    