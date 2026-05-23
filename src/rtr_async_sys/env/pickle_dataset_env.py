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
import tqdm
from torch.utils.data import Dataset, DataLoader

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




class PickleDataset(Dataset):
    def __init__(self, data_dir, prefetch=True):

        self.data = [os.path.join(data_dir, i) for i in sorted(os.listdir(data_dir))]
        if prefetch:
            print(f"prefetch data ing")
            self.data = [pickle.load(open(data_path, 'rb')) for data_path in tqdm(self.data)]
        self.prefetch = prefetch
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        dict_keys(['state', 'abs_action', 'delta_action', 'relative_action', 'visual_action', 'tactile', 'camera1_image'])
        data[state].shape is (16, 9)
        data[abs_action].shape is (16, 9)
        data[delta_action].shape is (16, 9)
        data[relative_action].shape is (16, 9)
        data[visual_action].shape is (16, 2)
        data[tactile].shape is (16, 700, 6)
        data[camera1_image].shape is (4, 480, 640, 3)
        """
        if not self.prefetch:
            sample_data = pickle.load(open(self.data[idx], 'rb'))
        else:
            sample_data = self.data[idx]
        return sample_data


class PickleDatasetEnv(AbsEnv):
    def __init__(self,
                data_dir:str,
                prefetch:bool=True,
                compress_obs:bool = False,
                visualize_image:bool = False,
                map_dict = {
                    'abs_action': 'abs_action'
                },
                action_space:str = 'abs_action',
                relative_action:bool = False
                #  robo_ip='192.168.1.239'
                 ):
        """
        :param map_dict: keys are requested return items and values are dataset items.
        """
        logger.info("init PickleDatasetRobotEnv")

        self.dataset = PickleDataset(data_dir=data_dir, prefetch=prefetch)

        self.prefetch = prefetch

        self.timestep = 0
        self.executed_actions = []
        self.plot_actions = []
        self.compress_obs = compress_obs
        self.visualize_image = visualize_image
        self.map_dict = map_dict
        self.relative_action = relative_action
        self.action_space = action_space
    
    def start(self):
        logger.info("[RealRobotEnv] start")



    def get_obs(self, n_obs_steps=None) -> Dict[str, np.ndarray]:
        """
        return \\
        obs['left_wrist_img']: shape  [t, h, w, c]
        """
        obs = self.dataset[self.timestep]
        for key in obs.keys():
            if not isinstance(obs[key], np.ndarray):
                obs[key] = obs[key].numpy()

            if 'img' in key or 'image' in key:
                if obs[key].shape[-1] != 3:
                    obs[key] = np.transpose(obs[key], (0, 2, 3, 1))
                if obs[key].dtype != np.uint8:
                    obs[key] = obs[key] * 255.0
                # # convert dtype from float32 to np.uint8
                obs[key] = obs[key].astype(np.uint8)
                print(f"obs[{key}].shape = {obs[key].shape}")
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

        if self.map_dict != None:
            new_obs = {}
            for key in self.map_dict.keys():
                v = self.map_dict[key]
                new_obs[key] = obs[v]
            obs = new_obs

        return obs


    
    def execute_action(self, action: np.ndarray):
        self.executed_actions.append(action)

 
    def clear(self):
        self.end_of_chunk()
    
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
            'predict':[],
            'fact_angle':[],
            'predict_angle':[],
        }
        fact_action_chunk = self.dataset[self.timestep][self.action_space]
        if self.relative_action:
            raise NotImplementedError("have not implemented relative action for pickle_dataset_env")
            # base_position = self.dataset[self.timestep]['obs']['left_robot_tcp_pose'][-1]
            # fact_action_chunk = relative_actions_to_absolute_actions(fact_action_chunk, base_position.numpy())

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
        # start = self.downsample_ratio*self.n_obs_steps-1 # Keep this aligned with model behavior; automate this later.
        start = 0
        fact_action_chunk = fact_action_chunk[start:start+execute_horizon:, ]

        l1 = torch.mean(torch.abs(torch.Tensor(action_chunk[None,:,:9]) - torch.Tensor(fact_action_chunk[None,:,:9])))
        logger.info(f"l1 loss is {l1.item()}, timestep is {self.timestep}")
        
        for i in range(action_chunk.shape[0]):
            fact_action = fact_action_chunk[i][0:3]*1000
            predict_action = action_chunk[i][0:3]*1000
            fact_angle = fact_action_chunk[i][3:6]
            predict_angle = action_chunk[i][3:6]
            plot_action['fact'].append(fact_action)
            plot_action['predict'].append(predict_action)
            plot_action['fact_angle'].append(fact_angle)
            plot_action['predict_angle'].append(predict_angle)
        self.plot_actions.append(plot_action)

        self.timestep += 1
        self.executed_actions = []
        # manual save
        self.save_plot_actions()
    
    def save_plot_actions(self, save_path="outputs/vis_outputs/dp_plot_actions.pkl"):
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        
        with open(save_path, "wb") as f:
            pickle.dump(self.plot_actions, f)



def build_dp_dataset_env(cfg: Union[DictConfig,str]) -> PickleDatasetEnv:
    logger.info("Instantiating DpDatasetEnv...")
    if isinstance(cfg, str):
        cfg = OmegaConf.load(cfg)
    env = instantiate(cfg)      # or instantiate(cfg.env), depending on the config structure
    return env


@hydra.main(config_path="../configs/env", config_name="pickle_dataset_env", version_base=None)
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
        "src/rtr_async_sys/configs/executor/env/pickle_dataset_env.yaml"
    )
    env:PickleDatasetEnv = hydra.utils.instantiate(cfg) #build_dp_dataset_env(cfg)
    obs = env.get_obs()
    print(len(env.dataset))
    for key in obs.keys():
        print(f"obs[{key}].shape is {obs[key].shape}, type is {obs[key].dtype}")
    for _ in range(20):
        env.get_obs()
        env.timestep += 1
    # obs = env.get_obs()
    # for key in obs.keys():
    #     if 'img' not in key:
    #         print(f"obs[key] is {obs[key]}, key is {key}, shape is {obs[key].shape}")
