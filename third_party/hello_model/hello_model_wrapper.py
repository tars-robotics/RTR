from typing import Dict, Any, Union, Optional
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from rtr_async_sys.core.model_wrapper_base import AbsModelWrapper

from hello_model import HelloModel

class HelloModelWrapper(AbsModelWrapper):
    def __init__(self, model, device, ckpt_path, return_raw_action = False, need_refine = False):
        super().__init__(model, device, ckpt_path=None, return_raw_action=return_raw_action, need_refine=need_refine)
        assert isinstance(self.policy, HelloModel), "model should be HelloModel"
        self.policy:HelloModel = self.policy.to(device)
        self.input_key_list = ['left_wrist_img', 'left_robot_tcp_pose', 'left_robot_gripper_width', 'left_gripper1_marker_offset_emb']

    
    def predict_action_chunk(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Call preprocess_obs and postprocess_action inside this function.
        """
        obs_dict = self.preprocess_obs(obs_dict)
        action_chunk = self.policy.predict_action(obs_dict).detach().cpu().numpy()
        action_chunk = self.postprocess_action(action_chunk)

        return action_chunk

    def preprocess_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        # Stack multi-frame observations into a batch
        if isinstance(obs, list):
            obs_processed = {
                key: torch.from_numpy(np.stack([o[key] for o in obs])).unsqueeze(0).to(self.device)
                for key in obs[0].keys()
            }
        else:
            obs_processed = {
                key: torch.from_numpy(obs[key]).unsqueeze(0).to(self.device)
                for key in obs.keys()
            }
        # Data Processing
        for key in obs_processed.keys():
            if 'img' in key:
                # print(obs_processed[key].shape)
                # exit()
                obs_processed[key] = obs_processed[key].permute(0, 1, 4, 2, 3)  # BNHWC -> BNCHW
                obs_processed[key] = obs_processed[key].float() / 255.0 # real image in env is not normalized
        input_dict = dict()
        for key in self.input_key_list:
            input_dict[key] = obs_processed[key]

        return input_dict

    def postprocess_action(self, raw_action_chunk: np.ndarray) -> np.ndarray:
        """
        Currently only single-arm is supported. \\
        Input is a (10,)-shaped action: xyz + 6d_rotation + gripper. \\
        Outputs: 1. raw model output for dataset_env, used to compute L1 loss and other open-loop metrics; 2. executable action for real_env [x,y,z,yaw,pitch,roll] or [x,y,z,yaw,pitch,roll,gripper_width,gripper_force]. \\
        Angles are in radians; convert to degrees inside the env as needed.
        """
        assert len(raw_action_chunk.shape) == 2, "input in postprocess_action should be action chunk"  # (action_steps, d_a)
        assert raw_action_chunk.shape[1] == 10, ""
        if self.return_raw_action:
            return raw_action_chunk
        else:
            return raw_action_chunk[:, 0:6]

@hydra.main(
    config_path="configs/model_wrapper",
    config_name="hello_model_wrapper",
    version_base=None
)
def main(cfg: DictConfig):
    if isinstance(cfg, str):
        cfg = OmegaConf.load(cfg)
    runner = hydra.utils.instantiate(cfg)
    print(runner)

if __name__ == '__main__':
    main()
