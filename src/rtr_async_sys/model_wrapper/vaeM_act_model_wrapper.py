from typing import Dict, Any, Union, Optional, List
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
import hydra
import transforms3d as t3d

from rtr_async_sys.core.model_wrapper_base import AbsModelWrapper
from rtr_async_sys.models.Tactile_Generation_Policy.tactile_generation_policy.model.action.vaeM import ActionVAE
from rtr_async_sys.models.Tactile_Generation_Policy.tools import normalizer_util

from rtr_async_sys.utils.action_utils import ortho6d_to_rotation_matrix
from rtr_async_sys.utils.image_utils import decompress_image, decompress_video_frames
import sys
sys.path.append('src/rtr_async_sys/models/Tactile_Generation_Policy/tools')

class VaeMActModelWrapper(AbsModelWrapper):
    def __init__(
            self, 
            model:Union[str, DictConfig], 
            normalizer_path:str, 
            device:Union[str, torch.device], 
            ckpt_path: str,
            return_raw_action: bool = False,
            need_refine:bool = False,
            compress_obs:bool = False,
            input_key_list:List = ['abs_action'],
            action_space:str = 'abs_action'
        ):
        super().__init__(model, device, ckpt_path, return_raw_action, need_refine, compress_obs=compress_obs,build_policy=False)
        model_cfg = model
        self.policy = ActionVAE(
            input_dim=model_cfg.input_dim,
            horizon=model_cfg.horizon, 
            latent_dim=model_cfg.latent_dim,
            hidden_state=model_cfg.hidden_state,
            n_embed=model_cfg.n_embed,
            mlp_layer_num=model_cfg.mlp_layer_num,
            time_compression_ratio=model_cfg.time_compression_ratio,
            act_scale=model_cfg.act_scale,
        ).to(device)
        self.policy.load_state_dict(torch.load(ckpt_path, map_location=device))
        self.policy.eval()

        self.normalizer = torch.load(normalizer_path)
        self.action_space = action_space
        assert self.action_space in input_key_list, f"action_space {action_space} should in input_key_list {input_key_list}"

        assert isinstance(self.policy, ActionVAE), "model should be DiffusionUnetImagePolicy in DpModelWrapper"
        self.input_key_list = input_key_list

    def predict_action_chunk(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        action_chunk: shape is [t, action_dim]
        """
        # return obs_dict[self.action_space] # to debug
        obs_dict = self.preprocess_obs(obs_dict)
        action = obs_dict[self.action_space]
        action = self.normalizer[self.action_space].normalize(action)
        action = action.to(self.device)

        action = self.policy.encode_then_decode(action)

        action = action.to('cpu')
        action = self.normalizer[self.action_space].unnormalize(action)

        action_chunk = action.detach().numpy()[0]

        action_chunk = self.postprocess_action(action_chunk)

        return action_chunk
    
    def preprocess_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        # Stack multi-frame observations into a batch
        if isinstance(obs, list):
            if self.compress_obs:
                for obs_item in obs:# Real robot env returns a list so images can be compressed conveniently
                    for key in obs_item.keys():
                        if 'img' in key or 'image' in key:
                            obs_item[key] = decompress_image(obs_item[key])

            obs_processed = {
                key: torch.from_numpy(np.stack([o[key] for o in obs])).unsqueeze(0).to(self.device)
                for key in obs[0].keys()
            }
        else:
            if self.compress_obs:
                for key in obs.keys():
                    if 'img' in key or 'image' in key:
                        obs[key] = decompress_video_frames(obs[key])

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

    # def postprocess_action(self, raw_action: np.ndarray) -> np.ndarray:
    #     return raw_action

    def postprocess_action(self, raw_action_chunk: np.ndarray) -> np.ndarray:
        assert len(raw_action_chunk.shape) == 2, "input in postprocess_action should be action chunk"  # (action_steps, d_a)
        assert (raw_action_chunk.shape[1] == 10) or (raw_action_chunk.shape[1] == 9), ""
        if self.return_raw_action:
            return raw_action_chunk
        else:

            left_rot_mat_batch = ortho6d_to_rotation_matrix(raw_action_chunk[:, 3:9])  # (action_steps, 3, 3)
            left_euler_batch = np.array([t3d.euler.mat2euler(rot_mat) for rot_mat in left_rot_mat_batch])  # (action_steps, 3)
            left_trans_batch = raw_action_chunk[:, :3]  # (action_steps, 3)
            left_action_6d = np.concatenate([left_trans_batch, left_euler_batch], axis=1) # (action_steps, 6)

            # if add gripper control
            # left_action_8d = np.concatenate([left_action_6d, raw_action_chunk[:, 9][:, np.newaxis],
            #                               np.zeros((raw_action_chunk.shape[0], 1))], axis=1)

            return left_action_6d
    
    def refine_action(self, action:np.ndarray, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Input: \\
        1. action to refine \\
        2. obs_dict 
        """
        return action # for debugging
    


if __name__ == '__main__':
    cfg = OmegaConf.load("src/rtr_async_sys/configs/user/model_wrapper/vae_act_wrapper.yaml")
    dp_model_wrapper = hydra.utils.instantiate(cfg)
    print(dp_model_wrapper)