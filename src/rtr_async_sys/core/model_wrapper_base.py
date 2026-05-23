from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Optional
from omegaconf import DictConfig, OmegaConf
import hydra
import numpy as np
import torch
from rtr_async_sys.utils.torch_utils import seed_everything

class AbsModelWrapper:
    def __init__(
            self, 
            model:Union[str, DictConfig], 
            device:Union[str, torch.device], 
            ckpt_path: str,
            return_raw_action:bool = False,
            need_refine:bool = False,
            compress_obs:bool = False,
            build_policy:bool = True
        ) -> None:
        """
        return_raw_action: when True, skip post-processing and return the raw model output, e.g. for L1 loss; when False, post-process and return executable actions such as single-arm (x, y, z, yaw, pitch, roll, gripper_width, gripper_force)
        """
        seed_everything(42)
        print(f"seed_everything 42")
        self.device = device
        self.ckpt_path = ckpt_path
        self.need_refine = need_refine
        self.return_raw_action = return_raw_action
        self.compress_obs = compress_obs

        if build_policy:
            if isinstance(model, str):
                model = OmegaConf.load(model)
            if isinstance(model, DictConfig):
                model = hydra.utils.instantiate(model)
            print("="*100)
            print(f"in model wrapper, ckpt_path is {ckpt_path}")
            self.policy = model
            if ckpt_path != None:
                payload = torch.load(ckpt_path, map_location=self.device)
                # Load the model weights
                self.policy.load_state_dict(payload['state_dicts']['model'])
            # Move policy to the correct device and set to evaluation mode
            self.policy.to(self.device)
            self.policy.eval()
        


    @abstractmethod
    def predict_action_chunk(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Call preprocess_obs and postprocess_action in this function.
        """
        pass

    @abstractmethod
    def preprocess_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def postprocess_action(self, raw_action: np.ndarray) -> np.ndarray:
        """
        Only single-arm actions are currently supported: \\
        Input is a 10D action: xyz + 6D rotation + gripper.\\
        Output: 1. raw model output for dataset_env, used for L1 loss and open-loop metrics; 2. executable action for real_env, [x,y,z,yaw,pitch,roll] or [x,y,z,yaw,pitch,roll,gripper_width,gripper_force] \\
        Angles are in radians and are converted to degrees in env when needed.
        """
        pass

    def refine_action(self, action:np.ndarray, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Input: \\
        1. action to refine \\
        2. obs_dict 
        """
        pass

    @abstractmethod
    def reset(self):
        pass