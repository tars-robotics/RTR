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

from rtr_async_sys.models.reactive_diffusion_policy.policy.kinedex_image_policy import DiffusionUnetImagePolicy

OmegaConf.register_new_resolver("eval", eval, replace=True)

if __name__ == '__main__':
    cfg = OmegaConf.load(
        "src/rtr_async_sys/configs/user/model_wrapper/model/kinedex.yaml"
    )
    model:DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg)
    print(model)