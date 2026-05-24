"""
Using dataset to behave like an env, it's used to perform evalution or test code behavior
"""

import threading
import torch
import numpy as np
from loguru import logger
import os
import pickle
import collections
import transforms3d as t3d
from typing import Dict,Union

import hydra
from omegaconf import OmegaConf
from omegaconf import DictConfig
from hydra.utils import instantiate

from rtr_async_sys.models.reactive_diffusion_policy.dataset.base_dataset import BaseImageDataset
from rtr_async_sys.core.env_base import AbsEnv

from rtr_async_sys.models.reactive_diffusion_policy.common.action_utils import (
    relative_actions_to_absolute_actions,
)

from rtr_async_sys.utils.image_utils import compress_video_frames

from rtr_async_sys.utils.action_utils import ortho6d_to_rotation_matrix
import transforms3d as t3d
import time

OmegaConf.register_new_resolver("eval", eval, replace=True)


def pose_6d_to_4x4matrix(pose: np.ndarray) -> np.ndarray:
    # convert 6D pose (x, y, z, r, p, y) to 4x4 transformation matrix
    mat = np.eye(4)
    quat = t3d.euler.euler2quat(pose[3], pose[4], pose[5])
    mat[:3, :3] = t3d.quaternions.quat2mat(quat)
    mat[:3, 3] = pose[:3]
    return mat

def pose_6d_to_pose_9d(pose: np.ndarray) -> np.ndarray:
    """
    Convert 6D state to 9D state
    :param pose: np.ndarray (6,), (x, y, z, rx, ry, rz)
    :return: np.ndarray (9,), (x, y, z, rx1, rx2, rx3, ry1, ry2, ry3)
    """
    rot_6d = pose_6d_to_4x4matrix(pose)[:3, :2].T.flatten()
    return np.concatenate((pose[:3], rot_6d), axis=0)

def normalize_vector(v: np.ndarray) -> np.ndarray:
    """
    Normalize a vector (batch * 3)
    """
    v_mag = np.linalg.norm(v, axis=1, keepdims=True)  # batch * 1
    v_mag = np.maximum(v_mag, 1e-8)
    v = v / v_mag
    return v

def ortho6d_to_rotation_matrix(ortho6d: np.ndarray) -> np.ndarray:
    """
    Compute rotation matrix from ortho6d representation
    """
    x_raw = ortho6d[:, 0:3]  # batch * 3
    y_raw = ortho6d[:, 3:6]  # batch * 3
    x = normalize_vector(x_raw)  # batch * 3
    z = np.cross(x, y_raw)  # batch * 3
    z = normalize_vector(z)  # batch * 3
    y = np.cross(z, x)  # batch * 3

    x = x[:, :, np.newaxis]
    y = y[:, :, np.newaxis]
    z = z[:, :, np.newaxis]

    matrix = np.concatenate((x, y, z), axis=2)  # batch * 3 * 3
    return matrix


class ObservationBuffer:

    def __init__(self, maxlen: int = 8):
        self._buf = collections.deque(maxlen=maxlen)

    def append_obs(self, obs):
        self._buf.append(obs)

    def get_new_obs(self, n_obs_steps):
        if len(self._buf) < n_obs_steps:
            return None
        else:
            obs_list = list(self._buf)[-n_obs_steps:]
            return obs_list





class DpDatasetEnv(AbsEnv):
    def __init__(self,
                 dataset:Union[DictConfig,BaseImageDataset],
                 n_obs_steps: int = 2,
                 pca_load_dir: str = "data/processed/vase2_new_A/rdp_pca",
                 relative_action: bool = False,
                 downsample_ratio: int = 1, # useful for rdp
                 log_tactile:bool=False,
                 log_dir:str = "outputs/log_dir",
                 compress_obs:bool = False,
                 visualize_image:bool = False,
                 obs_need_action:bool = False,
                 eval_length:int = -1,
                 save_plot_action_path:str = "outputs/vis_outputs/dp_plot_actions.pkl",
                 **kwargs
                #  robo_ip='192.168.1.239'
                 ):
        logger.info("init DatasetRobotEnv")
        self.relative_action = relative_action
        self.max_steps = 100
        self.left_tac_transform_matrix = np.load(os.path.join(pca_load_dir, 'pca_matrix1.npy'))
        self.left_tac_mean_matrix = np.load(os.path.join(pca_load_dir, 'pca_mean1.npy'))

        self.n_obs_steps = n_obs_steps

        self.dataset = dataset
        if isinstance(self.dataset, DictConfig):
            self.dataset = hydra.utils.instantiate(dataset)
        self.timestep = 0
        self.executed_actions = []
        self.plot_actions = []
        self.downsample_ratio = downsample_ratio
        self.log_tactile = log_tactile
        self.log_dir = log_dir
        self.log_step = 0
        self.tactile_dict = {}
        self.compress_obs = compress_obs
        self.visualize_image = visualize_image
        self.obs_need_action = obs_need_action

        self.last_xyz = None
        debug_action = False
        if debug_action:
            self.debug_actions()
        self.eval_length = eval_length
        self.save_plot_action_path = save_plot_action_path
        logger.info(f"dataset len is {len(self.dataset)}")
    
    def start(self):
        logger.info("[RealRobotEnv] start")

    def debug_actions(self):
        """
        Print the first 100 action chunks.
        """
        output_file = "outputs/output_action_chunk.txt"

        # Ensure the directory exists.
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w") as f:
            for timestep in range(100):
                # Bounds check.
                if timestep >= len(self.dataset):
                    f.write(f"[t={timestep}] Trajectory is too short; stopping early.\n")
                    break

                # Get action.
                action_chunk = self.dataset[timestep]["action"].numpy()[:,0:3]

                # Write to file.
                f.write(f"----- timestep {timestep} -----\n")
                f.write(str(action_chunk) + "\n")

        print(f"[debug_actions] Wrote the first 100 action chunks to {output_file}")

    def get_obs(self, n_obs_steps=None) -> Dict[str, np.ndarray]:
        """
        return \\
        obs['left_wrist_img']: shape  [t, h, w, c]
        """
        if self.eval_length != -1 and self.timestep > self.eval_length:# exceed max length
            logger.info("timestep exceeds eval_length, exit eval system")
            raise SystemExit(0)
        if self.timestep >= len(self.dataset):
            logger.info("dataset exhausted, exit eval system")
            raise SystemExit(0)
        obs = self.dataset[self.timestep]['obs']
        if self.obs_need_action:
            obs['action'] = self.dataset[self.timestep]['action']
        for key in obs.keys():
            if 'img' in key:
                obs[key] = obs[key].permute(0,2,3,1).float() * 255.0 # match with real env
            obs[key] = obs[key].numpy()
            if 'img' in key:
                # # convert dtype from float32 to np.uint8
                obs[key] = obs[key].astype(np.uint8)
                # print(obs[key].shape)
                if self.visualize_image:
                    vis_img = obs[key][-1]
                    episode_dir = "outputs/vis_outputs/vis_imgs"
                    os.makedirs(episode_dir, exist_ok=True)

                    # Save image files as step_000.png
                    img_path = os.path.join(episode_dir, f"step_{self.timestep:03d}.png")
                    import cv2
                    # convert RGB images to BGR for visualization
                    vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
                    # cv2.imwrite(img_path, vis_img)
                    cv2.imwrite(img_path, vis_img_bgr)

                if self.compress_obs:
                    obs[key] = compress_video_frames(obs[key])

        if self.log_tactile:
            left_gripper1_marker_offset_emb = obs['left_gripper1_marker_offset_emb'][-1]
            self.tactile_dict[self.log_step] = left_gripper1_marker_offset_emb
            if self.log_step % 5 == 0:
                log_path = os.path.join(self.log_dir, "tactile_dataset.pkl")
                with open(log_path, 'wb') as f:
                    pickle.dump(self.tactile_dict, f)
            self.log_step += 1

        return obs


    
    def execute_action(self, action: np.ndarray): 
        # if self.last_xyz is not None:
        #     stride = action[:3]*1000 - self.last_xyz[:3]*1000
        #     print(f"stride is {stride}")
        #     time.sleep(0.5)
        self.executed_actions.append(action)
        self.last_xyz = action

 
    def clear(self):
        self.end_of_chunk()
    
    def stop(self):
        pass

    def restart(self):
        pass

    def set_mode(self):
        pass
    
    def reset(self):
        self.timestep = 0
        self.executed_actions = []
        self.plot_actions = []

    
    def end_of_chunk(self):
        """
        Each item of dataset if a chunk. call this function after each end of a chunk.
        """
        action_dim = self.executed_actions[0].shape[0]
        action_chunk = np.stack(self.executed_actions)# shape: [action_dim] -> [execute_horizon, action_dim]
        plot_action = {
            'fact':[],
            'predict':[] ,
            'fact_angle':[],
            'predict_angle':[],
        }
        fact_action_chunk = self.dataset[self.timestep]['action'].numpy()
        if self.relative_action:
            base_position = self.dataset[self.timestep]['obs']['left_robot_tcp_pose'][-1]
            fact_action_chunk = relative_actions_to_absolute_actions(fact_action_chunk, base_position.numpy())

        if action_dim == 7 or action_dim ==  6:# transform the 10 dim action of fact_action_chunk into 7
            left_rot_mat_batch = ortho6d_to_rotation_matrix(fact_action_chunk[:, 3:9])
            left_rot_mat_batch = np.asarray(left_rot_mat_batch, dtype=np.float64)
            left_euler_batch = np.array([t3d.euler.mat2euler(rot_mat) for rot_mat in left_rot_mat_batch])
            left_trans_batch = fact_action_chunk[:, :3]
            left_action_6d = np.concatenate([left_trans_batch, left_euler_batch], axis=1) # (action_steps, 6)
            if action_dim == 7: # the gripper is not trained, so remove gripper dimension when computing L1 loss
                action_chunk = action_chunk[:, 0:6] # (action_steps, 6)
            # if action_dim == 7:
            #     left_action_6d = np.concatenate([left_trans_batch, left_euler_batch, fact_action_chunk[:, 9:]], axis=1) # (action_steps, 7)
            # else:
            #     left_action_6d = np.concatenate([left_trans_batch, left_euler_batch], axis=1) # (action_steps, 6)
            fact_action_chunk = left_action_6d

            


        execute_horizon = action_chunk.shape[0]
        # logger.info(f"fact_action_chunk.shape is {fact_action_chunk.shape}, action_chunk.shape is {action_chunk.shape}")
        start = self.dataset.image_downsample_ratio*self.downsample_ratio*(self.n_obs_steps-1) # Keep this aligned with model behavior; some models drop the first obs_steps - 1 actions and some do not.
        if fact_action_chunk.shape[0] - start < execute_horizon:
            start = 0
        logger.debug(f"start is {start}, execute_horizon is {execute_horizon}, ")

        fact_action_chunk = fact_action_chunk[start:start+execute_horizon:, ]

        l1 = torch.mean(torch.abs(torch.Tensor(action_chunk[None,:,:9]) - torch.Tensor(fact_action_chunk[None,:,:9])))
        logger.info(f"l1 loss is {l1.item()}, timestep is {self.timestep}")
        
        for i in range(action_chunk.shape[0]):
            fact_action = fact_action_chunk[i]
            predict_action = action_chunk[i]
            # fact_action = fact_action_chunk[i][0:3]*1000
            # predict_action = action_chunk[i][0:3]*1000
            # fact_angle = fact_action_chunk[i][3:6]
            # predict_angle = action_chunk[i][3:6]
            plot_action['fact'].append(fact_action)
            plot_action['predict'].append(predict_action)
            # plot_action['fact_angle'].append(fact_angle)
            # plot_action['predict_angle'].append(predict_angle)
        self.plot_actions.append(plot_action)

        self.timestep += 1
        self.executed_actions = []
        # manual save
        if self.timestep % 10 == 0:
            self.save_plot_actions(self.save_plot_action_path)
    
    def save_plot_actions(self, save_path="outputs/vis_outputs/dp_plot_actions.pkl"):
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        
        with open(save_path, "wb") as f:
            pickle.dump(self.plot_actions, f)



def build_dp_dataset_env(cfg: Union[DictConfig,str]) -> DpDatasetEnv:
    logger.info("Instantiating DpDatasetEnv...")
    if isinstance(cfg, str):
        cfg = OmegaConf.load(cfg)
    env = instantiate(cfg)      # or instantiate(cfg.env), depending on the config structure
    return env


@hydra.main(config_path="../configs/env", config_name="dp_dataset_env", version_base=None)
def cli_main(cfg: Union[DictConfig, str]):
    if isinstance(cfg, str):
        cfg = OmegaConf.load(cfg)
    env = build_dp_dataset_env(cfg)
    print(env)
    obs = env.get_obs()
    # for key in obs.keys():
    #     if 'img' not in key:
    #         print(obs[key])



if __name__ == "__main__":
    # cli_main()
    cfg = OmegaConf.load(
        "src/rtr_async_sys/configs/executor/env/dp_dataset_env_test_obs_keys.yaml"
    )
    env:DpDatasetEnv = hydra.utils.instantiate(cfg) #build_dp_dataset_env(cfg)
    env.log_tactile = False
    env.log_dir = "outputs/log_dir"
    obs = env.get_obs()
    print(len(env.dataset))
    for key in obs.keys():
        print(f"obs[{key}].shape is {obs[key].shape}, type is {obs[key].dtype}")
    # for _ in range(70):
    #     env.get_obs()
    # obs = env.get_obs()
    # for key in obs.keys():
    #     if 'img' not in key:
    #         print(f"obs[key] is {obs[key]}, key is {key}, shape is {obs[key].shape}")
