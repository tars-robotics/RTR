from typing import Dict, Any, Union, Optional, List
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from rtr_async_sys.core.model_wrapper_base import AbsModelWrapper
from rtr_async_sys.utils.action_utils import ortho6d_to_rotation_matrix
import transforms3d as t3d

from rtr_async_sys.utils.image_utils import decompress_image, decompress_video_frames
from rtr_async_sys.utils.action_utils import ortho6d_to_rotation_matrix
import os

from dataclasses import dataclass
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors

from rtr_async_sys.models.vla_vae.pi0_5_vae import Pi0_5_VAE
from omegaconf import OmegaConf
import hydra

import PIL
from PIL import Image
import time



class PI05LatentModelWrapper(AbsModelWrapper):
    """
    map_dict keys are the keys of the new observation, i.e. the VLA inputs.
    """
    def __init__(
            self, 
            device, 
            return_raw_action = False, need_refine = False, build_policy=False, prompt = "wipe the vase.",
            map_dict = None,
            compress_obs:bool = False,
            interpolate:bool = False,
            interpolate_ratio:int = 1,
            cfg:TrainPipelineConfig = None,
            vae_config_path:str|None = None,
            vae_load_path:str|None = None,
            latent_dataset_statistics:str|None = None
        ):
        super().__init__(None, device, ckpt_path=None, return_raw_action=return_raw_action, need_refine=need_refine, build_policy=build_policy, compress_obs=compress_obs)
        
        vae_config = OmegaConf.load(vae_config_path)
        vae:Pi0_5_VAE = hydra.utils.instantiate(vae_config)
        payload = torch.load(vae_load_path)
        # Load the model weights
        vae.load_state_dict(payload['state_dicts']['model'])
        vae.to(device)
        vae.eval()
        print(vae)
        vae._load_dataset_statistics(latent_dataset_statistics)
        self.vae = vae

        if map_dict == None:
            map_dict = {
                'observation.images.image': 'left_wrist_img',
                'observation.state': 'left_robot_tcp_pose',
            }
        self.map_dict = map_dict

        self.language_instruction = prompt

        self.interpolate = interpolate
        self.interpolate_ratio = interpolate_ratio
        cfg.validate()
        dataset = make_dataset(cfg)
        print(f"before make and load policy")
        start_time = time.time()
        self.policy = make_policy(
            cfg=cfg.policy,
            ds_meta=dataset.meta,
            rename_map=cfg.rename_map,
        )
        print(f"load policy done, consumes {time.time()-start_time}s")
        self.policy.eval()
        del dataset

        print(f"before make and load processor")
        start_time = time.time()
        preprocessor_overrides = {
            "device_processor": {"device": str(device)},
        }
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            preprocessor_overrides=preprocessor_overrides,
        )
        print(f"load processor done, consumes {time.time()-start_time}s")



    
    def predict_action_chunk(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Call preprocess_obs and postprocess_action inside this function.
        """
        observation = self.preprocess_obs(obs_dict)
        observation = self.preprocessor(observation)

        with torch.no_grad():
            latent_action = self.policy.predict_action_chunk(observation)
        # vae process
        latent_action = self.vae.denormalize_from_dataset(latent_action,is_latent=True)
        with torch.no_grad():
            naction = self.vae.decode_from_latent(latent_action).detach()
        de_predict_actions = self.vae.denormalize_from_dataset(naction, is_latent=False).cpu().numpy()

        # de_predict_actions = self.postprocessor(predict_actions).cpu().numpy()
        print(f"de_predict_actions.shape is {de_predict_actions.shape}")
        action_chunk = de_predict_actions[0]
        
        if self.interpolate:
            T = action_chunk.shape[0]*self.interpolate_ratio# original_T
            D = action_chunk.shape[1]
            xp = np.arange(0, T, self.interpolate_ratio, dtype=np.float32)
            x = np.arange(T, dtype=np.float32)

            action_chunk_original = np.empty((T, D), dtype=action_chunk.dtype)
            for dim in range(D):
                action_chunk_original[:, dim] = np.interp(x, xp, action_chunk[:, dim])
            action_chunk = action_chunk_original

        action_chunk = self.postprocess_action(action_chunk)

        return action_chunk

    def preprocess_obs(self, obs: Dict[str, Any]) -> PIL.Image:
        # Stack multi-frame observations into a batch
        if isinstance(obs, list):
            if self.compress_obs:
                for obs_item in obs:  # Real-robot env returns a list, so images can be compressed conveniently
                    for key in obs_item.keys():
                        if 'img' in key:
                            obs_item[key] = decompress_image(obs_item[key])

            obs = obs[-1]
        else:
            if self.compress_obs:
                for key in obs.keys():
                    if 'img' in key:
                        obs[key] = decompress_video_frames(obs[key])

            for key in obs.keys():
                if len(obs[key].shape) == 4:# image [t,h,w,c]
                    obs[key] = obs[key][-1]
                if len(obs[key].shape) == 2:# state [t,dim]
                    obs[key] = obs[key][-1]
        for key in obs.keys():
            if 'img' in key or 'image' in key:
                # np.array: transpose from [h,w,c] to [c,h,w]
                obs[key] = obs[key].transpose(2,0,1)
                if obs[key].dtype == np.uint8:
                    obs[key] = obs[key] / 255.0

        new_obs = {}
        for key in self.map_dict:
            map_key = self.map_dict[key]
            item = obs[map_key]
            new_obs[key] = item
        
        new_obs = {
            key: torch.from_numpy(new_obs[key]).unsqueeze(0).to(self.device)
            for key in new_obs.keys()
        }
        
        new_obs['task'] = self.language_instruction
        
        return new_obs

    def postprocess_action(self, raw_action_chunk: Union[np.ndarray, List]) -> np.ndarray:
        """
        Currently only single-arm is supported. \\
        Input is a (10,)-shaped action: xyz + 6d_rotation + gripper. \\
        Outputs: 1. raw model output for dataset_env, used to compute L1 loss and other open-loop metrics; 2. executable action for real_env [x,y,z,yaw,pitch,roll] or [x,y,z,yaw,pitch,roll,gripper_width,gripper_force]. \\
        Angles are in radians; convert to degrees inside the env as needed.
        """
        if isinstance(raw_action_chunk, List):
            raw_action_chunk = np.stack(raw_action_chunk)
        assert len(raw_action_chunk.shape) == 2, "input in postprocess_action should be action chunk"  # (action_steps, d_a)
        if raw_action_chunk.shape[1] == 10:
            left_rot_mat_batch = ortho6d_to_rotation_matrix(raw_action_chunk[:, 3:9])  # (action_steps, 3, 3)
            left_rot_mat_batch = np.asarray(left_rot_mat_batch, dtype=np.float64)
            left_euler_batch = np.array([t3d.euler.mat2euler(rot_mat) for rot_mat in left_rot_mat_batch])  # (action_steps, 3)
            left_trans_batch = raw_action_chunk[:, :3]  # (action_steps, 3)
            left_action_7d = np.concatenate([left_trans_batch, left_euler_batch, raw_action_chunk[:,9:]], axis=1) # (action_steps, 7)
            return left_action_7d
        else: 
            assert raw_action_chunk.shape[1] == 7, ""
            return raw_action_chunk

@hydra.main(
    config_path="configs/model_wrapper",
    config_name="pi0_5_model_wrapper",
    version_base=None
)
def main(cfg: DictConfig):
    if isinstance(cfg, str):
        cfg = OmegaConf.load(cfg)
    # cfg.device = "cuda:1"  # openvla-oft does not seem to support setting the device via cfg; use CUDA_VISIBLE_DEVICES=xxx instead
    model_wrapper:PI05ModelWrapper = hydra.utils.instantiate(cfg)
    print(model_wrapper)

    image_path = os.environ.get("LEROBOT_SAMPLE_IMAGE", "data/sample_images/wipe.png")
    image = Image.open(image_path)
    image = image.convert("RGB")
    image_np = np.array(image)
    state = np.zeros((9,), dtype=np.float32)

    obs_dict = {"left_wrist_img": image_np, "left_robot_tcp_pose": state}
    # import pdb; pdb.set_trace()
    action_chunk = model_wrapper.predict_action_chunk(obs_dict)
    print(action_chunk)

if __name__ == '__main__':
    main()
