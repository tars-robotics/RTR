from typing import Dict, Any, Union, Optional
from typing import Dict, Any, Union, Optional
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
import hydra
import transforms3d as t3d

from rtr_async_sys.core.model_wrapper_base import AbsModelWrapper
from rtr_async_sys.models.reactive_diffusion_policy.model.vae.model import VAE
from rtr_async_sys.utils.action_utils import ortho6d_to_rotation_matrix
from rtr_async_sys.utils.image_utils import decompress_image, decompress_video_frames
import time

class RdpVaeModelWrapper(AbsModelWrapper):
    def __init__(
            self, 
            model:Union[str, DictConfig], 
            device:Union[str, torch.device], 
            ckpt_path: str,
            return_raw_action: bool = False,
            need_refine:bool = False,
            compress_obs:bool = False,
            build_policy:bool = True,
        ):
        super().__init__(model, device, ckpt_path, return_raw_action, need_refine, compress_obs=compress_obs, build_policy=build_policy)
        
        # if isinstance(model, str):
        #     model = OmegaConf.load(model)
        # if isinstance(model, DictConfig):
        #     model = hydra.utils.instantiate(model)
        # self.policy = model
        # self.policy.load_state_dict(ckpt_path)

        assert isinstance(self.policy, VAE), "model should be VAE in DpModelWrapper"
        self.policy.to(device)
        self.input_key_list = ['action']
        print("rdp vae sleep 1s waiting for other components")
        time.sleep(1)
        # self.input_key_list = ['left_wrist_img', 'left_robot_tcp_pose', 'left_robot_gripper_width', 'left_gripper1_marker_offset_emb']

    def reset(self):
        pass

    def predict_action_chunk(self, obs_dict: Dict[str, torch.Tensor]) -> np.ndarray:
        """
        action_chunk: shape is [t, action_dim]
        """
        obs_dict = self.preprocess_obs(obs_dict)

        # action_dict = self.policy.predict_action(obs_dict)
        action_chunk = self.policy.encode_then_decode(obs_dict)
        # action_chunk = self.policy.encode_to_latent(obs_dict)
        # action_chunk = self.policy.decode_from_latent(action_chunk)
        action_chunk = action_chunk.detach().cpu().numpy()[0]

        action_chunk = self.postprocess_action(action_chunk)

        return action_chunk
    
    def preprocess_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        # Stack multi-frame observations into a batch
        if isinstance(obs, list):
            if self.compress_obs:
                for obs_item in obs:# Real robot env returns a list so images can be compressed conveniently
                    for key in obs_item.keys():
                        if 'img' in key:
                            obs_item[key] = decompress_image(obs_item[key])

            obs_processed = {
                key: torch.from_numpy(np.stack([o[key] for o in obs])).unsqueeze(0).to(self.device)
                for key in obs[0].keys()
            }
        else:
            if self.compress_obs:
                for key in obs.keys():
                    if 'img' in key:
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
        print("+"*100)
        print(raw_action_chunk.shape)
        assert raw_action_chunk.shape[1] == 10, ""
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
    

@hydra.main(config_path="../configs/user/model_wrapper", config_name="rdp_vae_wrapper", version_base=None)
def cli_main(cfg: Union[DictConfig, str]):
    if isinstance(cfg, str):
        cfg = OmegaConf.load(cfg)
    vae_model_wrapper = hydra.utils.instantiate(cfg)
    print(vae_model_wrapper)
    return vae_model_wrapper

if __name__ == '__main__':
    # cfg = OmegaConf.load("src/rtr_async_sys/configs/user/model_wrapper/vae_unet_wrapper.yaml")
    # vae_model_wrapper = hydra.utils.instantiate(cfg)
    vae_model_wrapper = cli_main()
    print(vae_model_wrapper)