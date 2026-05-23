from typing import Dict, Any, Union, Optional, Tuple
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
import hydra
from loguru import logger

from rtr_async_sys.core.model_wrapper_base import AbsModelWrapper
from rtr_async_sys.models.reactive_diffusion_policy.policy.latent_diffusion_unet_image_policy import LatentDiffusionUnetImagePolicy
from rtr_async_sys.models.reactive_diffusion_policy.common.action_utils import (
    interpolate_actions_with_ratio,
    relative_actions_to_absolute_actions,
    absolute_actions_to_relative_actions,
    get_inter_gripper_actions
)
import time
from copy import deepcopy

from rtr_async_sys.models.reactive_diffusion_policy.real_world.real_inference_util import get_real_obs_dict
from rtr_async_sys.utils.action_utils import ortho6d_to_rotation_matrix
from rtr_async_sys.models.reactive_diffusion_policy.common.pytorch_util import dict_apply
import transforms3d as t3d
from rtr_async_sys.utils.image_utils import decompress_image, decompress_video_frames



class RdpModelWrapper(AbsModelWrapper):
    def __init__(
            self, 
            model:Union[str, DictConfig], 
            device:Union[str, torch.device], 
            ckpt_path: str,
            return_raw_action: False,
            shape_meta: DictConfig,
            dataset_obs_temporal_downsample_ratio:int=2,
            use_latent_action_with_rnn_decoder:bool = True,
            use_relative_action:bool = True,
            action_interpolation_ratio:int = 1,
            need_refine=True,
            compress_obs:bool=False,
            return_latent_action:bool=True,
            block_reuse:bool=False,
            reuse_action_num:int=24,
            openloop_eval:bool=False

        ):
        super().__init__(model, device, ckpt_path, return_raw_action=return_raw_action, need_refine=need_refine, compress_obs=compress_obs)
        # assert isinstance(self.policy, LatentDiffusionUnetImagePolicy), "model should be LatentDiffusionUnetImagePolicy in DpModelWrapper"

        self.policy.at.set_normalizer(self.policy.normalizer)
        # TODO: This setting should match eval_real_robot_flexiv
        # print(f"self.policy.num_inference_steps is {self.policy.num_inference_steps}")
        # exit()
        # self.policy.num_inference_steps = 8  # DDIM inference iterations
        # self.policy.n_action_steps = self.policy.horizon - self.policy.n_obs_steps + 1 # not used in latent diffusion
        
        # self.policy:LatentDiffusionUnetImagePolicy = self.policy.eval().to(device)

        # ==========================================================
        # copy from real_runner_sync.py __init__, to init rdp(LatentDiffusionUnetImagePolicy)
        self.shape_meta = dict(shape_meta)

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys

        extended_rgb_keys = list()
        extended_lowdim_keys = list()
        extended_obs_shape_meta = shape_meta.get('extended_obs', dict())
        for key, attr in extended_obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                extended_rgb_keys.append(key)
            elif type == 'low_dim':
                extended_lowdim_keys.append(key)
        self.extended_rgb_keys = extended_rgb_keys
        self.extended_lowdim_keys = extended_lowdim_keys

        self.latency_step = 0
        self.gripper_latency_step = 0
        self.n_obs_steps = self.policy.n_obs_steps
        self.obs_temporal_downsample_ratio = 1 # obs_temporal_downsample_ratio, actual downsample_ratio used when sampling from the environment. Real env uses 1, while training uses 2 because dataset observations are higher frequency than inference-time observations
        self.dataset_obs_temporal_downsample_ratio = dataset_obs_temporal_downsample_ratio# During training, n_obs_steps * dataset_obs_temporal_downsample_ratio actions correspond to current and past states, so those actions are removed from future execution actions
        self.downsample_extended_obs = (self.obs_temporal_downsample_ratio != self.dataset_obs_temporal_downsample_ratio)
        self.use_latent_action_with_rnn_decoder = use_latent_action_with_rnn_decoder
        self.use_relative_action = use_relative_action
        self.action_interpolation_ratio = action_interpolation_ratio
        self.return_latent_action = return_latent_action

        self.block_reuse = block_reuse
        self.reuse_action_num = reuse_action_num
        self.last_action_chunk = None
        self.openloop_eval = openloop_eval


    def reset(self):
        self.last_action_chunk = None

    def predict_action_chunk(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        action_chunk: shape is (t, action_dim+1) or (t, action_dim+action_dim+1)
        """
        obs_dict,absolute_obs_dict = self.preprocess_obs(obs_dict)
        
        if not self.block_reuse:
            action_dict = self.policy.predict_action(
                obs_dict,
                dataset_obs_temporal_downsample_ratio=self.dataset_obs_temporal_downsample_ratio,
                return_latent_action=self.return_latent_action
            )
            action_all = action_dict['action'].detach().cpu().numpy()[0] # (t, action_dim)
        else:
            if self.last_action_chunk is None:
                action_dict = self.policy.predict_action(
                    obs_dict,
                    dataset_obs_temporal_downsample_ratio=self.dataset_obs_temporal_downsample_ratio,
                    return_latent_action=self.return_latent_action
                )
                self.last_action_chunk = action_dict['action'].detach()
                action_all = action_dict['action'].detach().cpu().numpy()[0] # (t, action_dim)
            else:
                if not self.openloop_eval:
                    action_dict = self.policy.predict_action(
                        obs_dict,
                        dataset_obs_temporal_downsample_ratio=self.dataset_obs_temporal_downsample_ratio,
                        return_latent_action=self.return_latent_action
                    )
                    now_action_chunk = action_dict['action'].detach()
                    reuse_action_chunk = self.last_action_chunk[:, -self.reuse_action_num:, :] # (b, reuse_action_num, action_dim)
                    concate_action_chunk = torch.concatenate([reuse_action_chunk, now_action_chunk[:, self.reuse_action_num:, :]], dim=1) # (b, horizon, action_dim)
                    concate_action_latent_chunk = self.policy.at.encode_to_latent(concate_action_chunk)
                    concate_action_chunk = self.policy.at.decode_from_latent(concate_action_latent_chunk).to(self.device)

                    action_all = concate_action_chunk.detach().cpu().numpy()[0]
                    self.last_action_chunk = concate_action_chunk.detach()
                else:# when switch episode, not reuse
                    action_dict = self.policy.predict_action(
                        obs_dict,
                        dataset_obs_temporal_downsample_ratio=self.dataset_obs_temporal_downsample_ratio,
                        return_latent_action=self.return_latent_action
                    )
                    now_action_chunk = action_dict['action'].detach()
                    this_x = now_action_chunk[0,0,0]
                    if self.last_x < 0 and this_x > 0: # switch episode, NOT reuse
                        action_all = now_action_chunk.cpu().numpy()[0]
                        self.last_action_chunk = now_action_chunk
                    else:
                        reuse_action_chunk = self.last_action_chunk[:, -self.reuse_action_num:, :] # (b, reuse_action_num, action_dim)
                        concate_action_chunk = torch.concatenate([reuse_action_chunk, now_action_chunk[:, self.reuse_action_num:, :]], dim=1) # (b, horizon, action_dim)
                        concate_action_latent_chunk = self.policy.at.encode_to_latent(concate_action_chunk)
                        concate_action_chunk = self.policy.at.decode_from_latent(concate_action_latent_chunk).to(self.device)

                        action_all = concate_action_chunk.detach().cpu().numpy()[0]
                        self.last_action_chunk = concate_action_chunk.detach()
                
            self.last_x = action_all[0][0]


        # logger.debug(f"[predict action chunk] action_all[0][0:5] is {action_all[0][0:5]}")
        
        if self.use_relative_action:
            base_absolute_action = np.concatenate([
                absolute_obs_dict['left_robot_tcp_pose'][-1] if 'left_robot_tcp_pose' in absolute_obs_dict else np.array([]),
                absolute_obs_dict['right_robot_tcp_pose'][-1] if 'right_robot_tcp_pose' in absolute_obs_dict else np.array([])
            ], axis=-1)
            # print('base:', base_absolute_action)
            # action_all = np.concatenate([
            #     action_all,
            #     base_absolute_action[np.newaxis, :].repeat(action_all.shape[0], axis=0)
            # ], axis=-1)
        
        if self.return_latent_action == False:
            if self.use_relative_action:
                action_all = relative_actions_to_absolute_actions(action_all, base_absolute_action)
            action_all = self.postprocess_action(action_all)
            return action_all
        else:
            if self.use_relative_action:
                action_all = np.concatenate([
                    action_all,
                    base_absolute_action[np.newaxis, :].repeat(action_all.shape[0], axis=0)
                ], axis=-1)

        # add action step to get corresponding observation, (t, action_dim+1) or (t, action_dim+action_dim+1)
        action_all = np.concatenate([
            action_all,
            np.arange(self.n_obs_steps * self.dataset_obs_temporal_downsample_ratio, action_all.shape[0] + self.n_obs_steps * self.dataset_obs_temporal_downsample_ratio)[:, np.newaxis]
        ], axis=-1)

        # action_chunk = self.postprocess_action(action_chunk)

        return action_all
    
    def refine_action(self, action_latent:np.ndarray, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Input: \\
        1. action to refine. shape: (action_dim+1,) or (action_dim+action_dim+1), 1 is tcp_extended_obs_step, another action_dim is base_action \\
        2. obs_dict \\
        Output: action (action_dim, )
        """
        # logger.info(f"action_latent.shape is {action_latent.shape}") # (74,)??
        # logger.info(f"tcp_step_action[0:5] is {action_latent[0:5]}")
        # ===== modify from real_runner_sync.RealRunner.action_command_thread =====
        extended_obs = obs_dict
        tcp_extended_obs_step = int(action_latent[-1])
        tcp_step_action = action_latent[:-1]

        if self.use_relative_action:
            action_dim = self.shape_meta['obs']['left_robot_tcp_pose']['shape'][0]
            if 'right_robot_tcp_pose' in self.shape_meta['obs']:
                action_dim += self.shape_meta['obs']['right_robot_tcp_pose']['shape'][0]
            tcp_base_absolute_action = tcp_step_action[-action_dim:]
            tcp_step_action = tcp_step_action[:-action_dim]

        # process obs
        # np_extended_obs_dict = dict(extended_obs)
        if isinstance(extended_obs, list):
            np_extended_obs_dict = {
                key: np.stack([o[key] for o in extended_obs])
                for key in extended_obs[0].keys()
            }
        else:
            np_extended_obs_dict = dict(extended_obs)

        np_extended_obs_dict = get_real_obs_dict(
            env_obs=np_extended_obs_dict, shape_meta=self.shape_meta, is_extended_obs=True)
        # np_extended_obs_dict, _ = self.pre_process_extended_obs(np_extended_obs_dict)

        # if len of extended_obs_dict[key] don't match tcp_extended_obs_step
        for key in np_extended_obs_dict.keys():
            item = np_extended_obs_dict[key]
            if item.shape[0] != tcp_extended_obs_step:
                # logger.debug("-"*100)
                # logger.debug(f"extended_obs_dict[{key}].shape[0] {item.shape[0]} don't match tcp_extended_obs_step {tcp_extended_obs_step}. This situation should only happen on dataset_env")
                # This method likely only supports low-dimensional observations
                item = item[-1,:]
                item = item[None,:]
                item = np.repeat(item, repeats=tcp_extended_obs_step, axis=0)
                np_extended_obs_dict[key] = item
        
        extended_obs_dict = dict_apply(np_extended_obs_dict, lambda x: torch.from_numpy(x).unsqueeze(0)) # (b,t,xxx)


        # process action
        tcp_step_latent_action = torch.from_numpy(tcp_step_action.astype(np.float32)).unsqueeze(0)

        # predict action, (t,action_dim)
        tcp_step_action = self.policy.predict_from_latent_action(tcp_step_latent_action, extended_obs_dict, tcp_extended_obs_step, self.dataset_obs_temporal_downsample_ratio)['action'][0].detach().cpu().numpy()
        
        # post process
        if self.use_relative_action:
            # logger.debug(f"tcp_extended_obs_step is {tcp_extended_obs_step}")

            # logger.debug(f"before relative_to_abs, tcp_step_action[-1][0:3] is {tcp_step_action[-1][0:3]}, tcp_base_absolute_action[0:3] is {tcp_base_absolute_action[0:3]}")
            
            tcp_step_action = relative_actions_to_absolute_actions(tcp_step_action, tcp_base_absolute_action)

            # logger.debug(f"after relative_to_abs, tcp_step_action[-1][0:3] is {tcp_step_action[-1][0:3]}")
        if self.policy.at.use_rnn_decoder:
            tcp_step_action = tcp_step_action[-1]# (action_dim)
        else:
            tcp_extended_obs_step = tcp_extended_obs_step - self.n_obs_steps * self.dataset_obs_temporal_downsample_ratio
            # print(f"tcp_extended_obs_step is {tcp_extended_obs_step}")
            tcp_step_action = tcp_step_action[tcp_extended_obs_step]
        step_action = self.postprocess_action(tcp_step_action[None,:])
        step_action = step_action.squeeze(0) # (action_dim)

        return step_action



    
    def preprocess_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        # obs = dict(obs)# (t,xxx)
        if isinstance(obs, list):
            if self.compress_obs:
                for obs_item in obs:# Real robot env returns a list so images can be compressed conveniently
                    for key in obs_item.keys():
                        if 'img' in key:
                            obs_item[key] = decompress_image(obs_item[key])

            obs = {
                key: np.stack([o[key] for o in obs])
                for key in obs[0].keys()
            }
        else:
            if self.compress_obs:
                for key in obs.keys():
                    if 'img' in key:
                        obs[key] = decompress_video_frames(obs[key])
                        
            obs = dict(obs)

        obs = get_real_obs_dict(
                    env_obs=obs, shape_meta=self.shape_meta)
        
        # Slice
        for key in self.lowdim_keys:
            obs[key] = obs[key][:, :self.shape_meta['obs'][key]['shape'][0]]
        
        absolute_obs_dict = dict()
        for key in self.lowdim_keys:
            absolute_obs_dict[key] = obs[key].copy()

        # b,t,xxx
        obs_processed = {
            key: torch.from_numpy(obs[key]).unsqueeze(0).to(self.device)
            for key in obs.keys()
        }
        
        # # Slice
        # for key in self.lowdim_keys:
        #     obs_processed[key] = obs_processed[key][:, :, :self.shape_meta['obs'][key]['shape'][0]]

        # Data Processing, have done in get_real_obs_dict
        # for key in obs_processed.keys():
        #     if 'img' in key:
        #         # print(obs_processed[key].shape)
        #         # exit()
        #         obs_processed[key] = obs_processed[key].permute(0, 1, 4, 2, 3)  # BNHWC -> BNCHW
        #         obs_processed[key] = obs_processed[key].float() / 255.0 # real image in env is not normalized

        # input_dict = dict()
        # for key in self.input_key_list:
        #     input_dict[key] = obs_processed[key]
        # return input_dict
        return obs_processed, absolute_obs_dict


    def postprocess_action(self, raw_action_chunk: np.ndarray) -> np.ndarray:
        assert len(raw_action_chunk.shape) == 2, "input in postprocess_action should be action chunk"  # (action_steps, d_a)
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
    
    def _pre_process_obs(self, obs_dict: Dict) -> Tuple[Dict, Dict]:
        """
        obs_dict['key'].shape is [t,xxx] \\
        With the current config, this function only slices shapes and copies obs_dict to absolute_obs_dict, so this function and _pre_process_extended_obs may not be needed.
        """
        obs_dict = deepcopy(obs_dict)

        for key in self.lowdim_keys:
            if "wrt" not in key:
                obs_dict[key] = obs_dict[key][:, :self.shape_meta['obs'][key]['shape'][0]]

        # inter-gripper relative action. With the current config, the operations below do not affect obs_dict unless obs contains left_robot_wrt_right_robot_tcp_pose.
        obs_dict.update(get_inter_gripper_actions(obs_dict, self.lowdim_keys, self.transforms))
        for key in self.lowdim_keys:
            obs_dict[key] = obs_dict[key][:, :self.shape_meta['obs'][key]['shape'][0]]

        absolute_obs_dict = dict()
        for key in self.lowdim_keys:
            absolute_obs_dict[key] = obs_dict[key].copy()

        return obs_dict, absolute_obs_dict
    
    def _pre_process_extended_obs(self, extended_obs_dict: Dict) -> Tuple[Dict, Dict]:
        """
        extended_obs_dict.shape is [t, xxx]
        """
        extended_obs_dict = deepcopy(extended_obs_dict)

        absolute_extended_obs_dict = dict()
        for key in self.extended_lowdim_keys:
            extended_obs_dict[key] = extended_obs_dict[key][:, :self.shape_meta['extended_obs'][key]['shape'][0]]
            absolute_extended_obs_dict[key] = extended_obs_dict[key].copy()

        return extended_obs_dict, absolute_extended_obs_dict
    


if __name__ == '__main__':
    cfg = OmegaConf.load("src/rtr_async_sys/configs/model_wrapper/dp_wrapper.yaml")
    dp_model_wrapper = hydra.utils.instantiate(cfg)
    print(dp_model_wrapper)