import threading
import time
import os.path as osp
import numpy as np
import torch
import tqdm
from loguru import logger
from typing import Dict, Tuple, Union, Optional
import transforms3d as t3d
import py_cli_interaction
from omegaconf import DictConfig, ListConfig
from reactive_diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from reactive_diffusion_policy.common.pytorch_util import dict_apply
from reactive_diffusion_policy.common.precise_sleep import precise_sleep
# from reactive_diffusion_policy.env.real_bimanual.real_env import RealRobotEnvironment
from reactive_diffusion_policy.env.ours.sensors import RealRobotEnv
from reactive_diffusion_policy.real_world.real_inference_util import (
    get_real_obs_dict)
from reactive_diffusion_policy.real_world.real_world_transforms import RealWorldTransforms
from reactive_diffusion_policy.common.space_utils import ortho6d_to_rotation_matrix
from reactive_diffusion_policy.common.ensemble import EnsembleBuffer
from reactive_diffusion_policy.common.action_utils import (
    interpolate_actions_with_ratio,
    relative_actions_to_absolute_actions,
    absolute_actions_to_relative_actions,
    get_inter_gripper_actions
)
import requests
# /home/robotics/Prometheus/Robotiq-Gripper/gripper.py

import os
import psutil
from copy import deepcopy

# add this to prevent assigning too may threads when using numpy
os.environ["OPENBLAS_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["NUMEXPR_NUM_THREADS"] = "12"
os.environ["OMP_NUM_THREADS"] = "12"

import cv2
# add this to prevent assigning too may threads when using open-cv
cv2.setNumThreads(12)

# Get the total number of CPU cores
total_cores = psutil.cpu_count()
# Define the number of cores you want to bind to
num_cores_to_bind = 10
# Calculate the indices of the first ten cores
# Ensure the number of cores to bind does not exceed the total number of cores
cores_to_bind = set(range(min(num_cores_to_bind, total_cores)))
# Set CPU affinity for the current process to the first ten cores
os.sched_setaffinity(0, cores_to_bind)

class RealRunner:
    def __init__(self,
                 output_dir: str,
                 transform_params: DictConfig,
                 env_params: DictConfig,
                 shape_meta: DictConfig,
                 tcp_ensemble_buffer_params: DictConfig,
                 gripper_ensemble_buffer_params: DictConfig,
                 latent_tcp_ensemble_buffer_params: DictConfig = None,
                 latent_gripper_ensemble_buffer_params: DictConfig = None,
                 use_latent_action_with_rnn_decoder: bool = False,
                 use_relative_action: bool = False,
                 use_relative_tcp_obs_for_relative_action: bool = False,
                 action_interpolation_ratio: int = 1,
                 eval_episodes=10,
                 max_duration_time: float = 30,
                 tcp_action_update_interval: int = 6,
                 gripper_action_update_interval: int = 10,
                 tcp_pos_clip_range: ListConfig = ListConfig([[0.6, -0.4, 0.03], [1.0, 0.45, 0.4]]),
                 tcp_rot_clip_range: ListConfig = ListConfig([[-np.pi, 0., np.pi], [-np.pi, 0., np.pi]]),
                 tqdm_interval_sec = 5.0,
                 control_fps: float = 12,
                 inference_fps: float = 6,
                 latency_step: int = 0,
                 gripper_latency_step: Optional[int] = None,
                 n_obs_steps: int = 2,
                 obs_temporal_downsample_ratio: int = 2,
                 dataset_obs_temporal_downsample_ratio: int = 1,
                 downsample_extended_obs: bool = True,
                 enable_video_recording: bool = False,
                 vcamera_server_ip: Optional[Union[str, ListConfig]] = None,
                 vcamera_server_port: Optional[Union[int, ListConfig]] = None,
                 task_name=None,
                 ):
        self.task_name = task_name
        self.transforms = RealWorldTransforms(option=transform_params)
        self.shape_meta = dict(shape_meta)
        self.eval_episodes = eval_episodes

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

        # self.env = RealRobotEnvironment(transforms=self.transforms, **env_params)
        self.env = RealRobotEnv(transforms=self.transforms, **env_params)
        # set gripper to max width
        # self.env.send_gripper_command_direct(self.env.max_gripper_width, self.env.max_gripper_width)
        # self.env.gripper.open_gripper()
        time.sleep(2)

        self.max_duration_time = max_duration_time
        self.tcp_action_update_interval = tcp_action_update_interval
        self.gripper_action_update_interval = gripper_action_update_interval
        self.tcp_pos_clip_range = tcp_pos_clip_range
        self.tcp_rot_clip_range = tcp_rot_clip_range
        self.tqdm_interval_sec = tqdm_interval_sec
        self.control_fps = control_fps
        self.control_interval_time = 1.0 / control_fps
        self.inference_fps = inference_fps
        self.inference_interval_time = 1.0 / inference_fps
        assert self.control_fps % self.inference_fps == 0
        self.latency_step = latency_step
        self.gripper_latency_step = gripper_latency_step if gripper_latency_step is not None else latency_step
        self.n_obs_steps = n_obs_steps
        self.obs_temporal_downsample_ratio = obs_temporal_downsample_ratio
        self.dataset_obs_temporal_downsample_ratio = dataset_obs_temporal_downsample_ratio
        self.downsample_extended_obs = downsample_extended_obs
        self.use_latent_action_with_rnn_decoder = use_latent_action_with_rnn_decoder
        if self.use_latent_action_with_rnn_decoder:
            assert latent_tcp_ensemble_buffer_params.ensemble_mode == 'new', "Only support new ensemble mode for latent action."
            assert latent_gripper_ensemble_buffer_params.ensemble_mode == 'new', "Only support new ensemble mode for latent action."
            self.tcp_ensemble_buffer = EnsembleBuffer(**latent_tcp_ensemble_buffer_params)
            self.gripper_ensemble_buffer = EnsembleBuffer(**latent_gripper_ensemble_buffer_params)
        else:
            self.tcp_ensemble_buffer = EnsembleBuffer(**tcp_ensemble_buffer_params)
            self.gripper_ensemble_buffer = EnsembleBuffer(**gripper_ensemble_buffer_params)
        self.use_relative_action = use_relative_action
        self.use_relative_tcp_obs_for_relative_action = use_relative_tcp_obs_for_relative_action
        self.action_interpolation_ratio = action_interpolation_ratio

        self.enable_video_recording = enable_video_recording
        if enable_video_recording:
            assert isinstance(vcamera_server_ip, str) and isinstance(vcamera_server_port, int) or \
                     isinstance(vcamera_server_ip, ListConfig) and isinstance(vcamera_server_port, ListConfig), \
                "vcamera_server_ip and vcamera_server_port should be a string or ListConfig."
        if isinstance(vcamera_server_ip, str):
            vcamera_server_ip_list = [vcamera_server_ip]
            vcamera_server_port_list = [vcamera_server_port]
        elif isinstance(vcamera_server_ip, ListConfig):
            vcamera_server_ip_list = list(vcamera_server_ip)
            vcamera_server_port_list = list(vcamera_server_port)
        self.vcamera_server_ip_list = vcamera_server_ip_list
        self.vcamera_server_port_list = vcamera_server_port_list
        self.video_dir = osp.join(output_dir, 'videos')

        self.stop_event = threading.Event()
        self.session = requests.Session()
        self.latent_cnt = 0
        self.fast_cnt = 0
        self.step_cnt = 0

    def pre_process_obs(self, obs_dict: Dict) -> Tuple[Dict, Dict]:
        obs_dict = deepcopy(obs_dict)

        for key in self.lowdim_keys:
            if "wrt" not in key:
                obs_dict[key] = obs_dict[key][:, :self.shape_meta['obs'][key]['shape'][0]]

        # inter-gripper relative action
        obs_dict.update(get_inter_gripper_actions(obs_dict, self.lowdim_keys, self.transforms))
        for key in self.lowdim_keys:
            obs_dict[key] = obs_dict[key][:, :self.shape_meta['obs'][key]['shape'][0]]

        absolute_obs_dict = dict()
        for key in self.lowdim_keys:
            absolute_obs_dict[key] = obs_dict[key].copy()

        # convert absolute action to relative action
        if self.use_relative_action and self.use_relative_tcp_obs_for_relative_action:
            for key in self.lowdim_keys:
                if 'robot_tcp_pose' in key and 'wrt' not in key:
                    base_absolute_action = obs_dict[key][-1].copy()
                    obs_dict[key] = absolute_actions_to_relative_actions(obs_dict[key], base_absolute_action=base_absolute_action)

        return obs_dict, absolute_obs_dict

    def pre_process_extended_obs(self, extended_obs_dict: Dict) -> Tuple[Dict, Dict]:
        extended_obs_dict = deepcopy(extended_obs_dict)

        absolute_extended_obs_dict = dict()
        for key in self.extended_lowdim_keys:
            extended_obs_dict[key] = extended_obs_dict[key][:, :self.shape_meta['extended_obs'][key]['shape'][0]]
            absolute_extended_obs_dict[key] = extended_obs_dict[key].copy()

        # convert absolute action to relative action
        if self.use_relative_action and self.use_relative_tcp_obs_for_relative_action:
            for key in self.extended_lowdim_keys:
                if 'robot_tcp_pose' in key and 'wrt' not in key:
                    base_absolute_action = extended_obs_dict[key][-1].copy()
                    extended_obs_dict[key] = absolute_actions_to_relative_actions(extended_obs_dict[key], base_absolute_action=base_absolute_action)

        return extended_obs_dict, absolute_extended_obs_dict

    def post_process_action(self, action: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Post-process the action before sending to the robot
        """
        assert len(action.shape) == 2  # (action_steps, d_a)
        if self.env.data_processing_manager.use_6d_rotation:
            if action.shape[-1] == 4 or action.shape[-1] == 8:
                # convert to 6D pose
                left_trans_batch = action[:, :3]  # (action_steps, 3)
                # we use default euler angles as 0
                left_euler_batch = np.zeros_like(left_trans_batch)
                left_action_6d = np.concatenate([left_trans_batch, left_euler_batch], axis=1)  # (action_steps, 6)
                if action.shape[-1] == 8:
                    right_trans_batch = action[:, 3:6]  # (action_steps, 3)
                    right_euler_batch = np.zeros_like(right_trans_batch)
                    right_action_6d = np.concatenate([right_trans_batch, right_euler_batch], axis=1)
                else:
                    right_action_6d = None
            elif action.shape[-1] == 10 or action.shape[-1] == 20:
                # convert to 6D pose
                left_rot_mat_batch = ortho6d_to_rotation_matrix(action[:, 3:9])  # (action_steps, 3, 3)
                left_euler_batch = np.array([t3d.euler.mat2euler(rot_mat) for rot_mat in left_rot_mat_batch])  # (action_steps, 3)
                left_trans_batch = action[:, :3]  # (action_steps, 3)
                left_action_6d = np.concatenate([left_trans_batch, left_euler_batch], axis=1)  # (action_steps, 6)
                if action.shape[-1] == 20:
                    right_rot_mat_batch = ortho6d_to_rotation_matrix(action[:, 12:18])
                    right_euler_batch = np.array([t3d.euler.mat2euler(rot_mat) for rot_mat in right_rot_mat_batch])
                    right_trans_batch = action[:, 9:12]
                    right_action_6d = np.concatenate([right_trans_batch, right_euler_batch], axis=1)
                else:
                    right_action_6d = None
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        # clip action (x, y, z)
        left_action_6d[:, :3] = np.clip(left_action_6d[:, :3], np.array(self.tcp_pos_clip_range[0]), np.array(self.tcp_pos_clip_range[1]))
        if right_action_6d is not None:
            right_action_6d[:, :3] = np.clip(right_action_6d[:, :3], np.array(self.tcp_pos_clip_range[2]), np.array(self.tcp_pos_clip_range[3]))
        # clip action (r, p, y)
        left_action_6d[:, 3:] = np.clip(left_action_6d[:, 3:], np.array(self.tcp_rot_clip_range[0]), np.array(self.tcp_rot_clip_range[1]))
        if right_action_6d is not None:
            right_action_6d[:, 3:] = np.clip(right_action_6d[:, 3:], np.array(self.tcp_rot_clip_range[2]), np.array(self.tcp_rot_clip_range[3]))
        # add gripper action
        if action.shape[-1] == 4:
            left_action = np.concatenate([left_action_6d, action[:, 3][:, np.newaxis],
                                          np.zeros((action.shape[0], 1))], axis=1)
            right_action = None
        elif action.shape[-1] == 8:
            left_action = np.concatenate([left_action_6d, action[:, 6][:, np.newaxis],
                                          np.zeros((action.shape[0], 1))], axis=1)
            right_action = np.concatenate([right_action_6d, action[:, 7][:, np.newaxis],
                                           np.zeros((action.shape[0], 1))], axis=1)
        elif action.shape[-1] == 10:
            left_action = np.concatenate([left_action_6d, action[:, 9][:, np.newaxis],
                                          np.zeros((action.shape[0], 1))], axis=1)
            right_action = None
        elif action.shape[-1] == 20:
            left_action = np.concatenate([left_action_6d, action[:, 18][:, np.newaxis],
                                          np.zeros((action.shape[0], 1))], axis=1)
            right_action = np.concatenate([right_action_6d, action[:, 19][:, np.newaxis],
                                          np.zeros((action.shape[0], 1))], axis=1)

        else:
            raise NotImplementedError

        if right_action is None:
            right_action = left_action.copy()
            is_bimanual = False
        else:
            is_bimanual = True
        action_all = np.concatenate([left_action, right_action], axis=-1)
        return (action_all, is_bimanual)

    def action_command_thread(self, policy: Union[DiffusionUnetImagePolicy], stop_event):
        while not stop_event.is_set():
            start_time = time.time()
            # get step action from ensemble buffer
            tcp_step_action = self.tcp_ensemble_buffer.get_action()
            gripper_step_action = self.gripper_ensemble_buffer.get_action()
            if tcp_step_action is None or gripper_step_action is None:  # no action in the buffer => no movement.
                cur_time = time.time()
                precise_sleep(max(0., self.control_interval_time - (cur_time - start_time)))
                logger.debug(f"Step: {self.action_step_count}, control_interval_time: {self.control_interval_time}, "
                             f"cur_time-start_time: {cur_time - start_time}")
                self.action_step_count += 1
                continue

            if self.use_latent_action_with_rnn_decoder:
                tcp_extended_obs_step = int(tcp_step_action[-1])
                gripper_extended_obs_step = int(gripper_step_action[-1])
                tcp_step_action = tcp_step_action[:-1]
                gripper_step_action = gripper_step_action[:-1]

                longer_extended_obs_step = max(tcp_extended_obs_step, gripper_extended_obs_step)
                obs_temporal_downsample_ratio = self.obs_temporal_downsample_ratio if self.downsample_extended_obs else 1
                extended_obs = self.env.get_obs(longer_extended_obs_step,
                                                    temporal_downsample_ratio= obs_temporal_downsample_ratio)

                if self.use_relative_action:
                    action_dim = self.shape_meta['obs']['left_robot_tcp_pose']['shape'][0]
                    if 'right_robot_tcp_pose' in self.shape_meta['obs']:
                        action_dim += self.shape_meta['obs']['right_robot_tcp_pose']['shape'][0]
                    tcp_base_absolute_action = tcp_step_action[-action_dim:]
                    gripper_base_absolute_action = gripper_step_action[-action_dim:]
                    tcp_step_action = tcp_step_action[:-action_dim]
                    gripper_step_action = gripper_step_action[:-action_dim]

                np_extended_obs_dict = dict(extended_obs)
                np_extended_obs_dict = get_real_obs_dict(
                    env_obs=np_extended_obs_dict, shape_meta=self.shape_meta, is_extended_obs=True)
                np_extended_obs_dict, _ = self.pre_process_extended_obs(np_extended_obs_dict)
                extended_obs_dict = dict_apply(np_extended_obs_dict, lambda x: torch.from_numpy(x).unsqueeze(0))

                tcp_step_latent_action = torch.from_numpy(tcp_step_action.astype(np.float32)).unsqueeze(0)
                gripper_step_latent_action = torch.from_numpy(gripper_step_action.astype(np.float32)).unsqueeze(0)

                dataset_obs_temporal_downsample_ratio = self.dataset_obs_temporal_downsample_ratio
                tcp_step_action = policy.predict_from_latent_action(tcp_step_latent_action, extended_obs_dict, tcp_extended_obs_step, dataset_obs_temporal_downsample_ratio)['action'][0].detach().cpu().numpy()
                gripper_step_action = policy.predict_from_latent_action(gripper_step_latent_action, extended_obs_dict, gripper_extended_obs_step, dataset_obs_temporal_downsample_ratio)['action'][0].detach().cpu().numpy()
                self.fast_cnt += 1
                logger.info(f"Slow cnt: {self.latent_cnt}, Fast cnt: {self.fast_cnt}, Step cnt: {self.step_cnt}")
                if self.use_relative_action:
                    tcp_step_action = relative_actions_to_absolute_actions(tcp_step_action, tcp_base_absolute_action)
                    gripper_step_action = relative_actions_to_absolute_actions(gripper_step_action, gripper_base_absolute_action)

                if tcp_step_action.shape[-1] == 4: # (x, y, z, gripper_width)
                    tcp_len = 3
                elif tcp_step_action.shape[-1] == 8: # (x_l, y_l, z_l, x_r, y_r, z_r, gripper_width_l, gripper_width_r)
                    tcp_len = 6
                elif tcp_step_action.shape[-1] == 10: # (x, y, z, rx1, rx2, rx3, ry1, ry2, ry3)
                    tcp_len = 9
                elif tcp_step_action.shape[-1] == 20: # (x_l, y_l, z_l, rotation_l, x_r, y_r, z_r, rotation_r, gripper_width_l, gripper_width_r)
                    tcp_len = 18
                else:
                    raise NotImplementedError

                if self.env.enable_exp_recording:
                    self.env.get_predicted_action(tcp_step_action[:, :tcp_len], type='partial_tcp')
                    self.env.get_predicted_action(gripper_step_action[:, tcp_len:], type='partial_gripper')

                    full_tcp_step_action = policy.predict_from_latent_action(tcp_step_latent_action, extended_obs_dict, tcp_extended_obs_step, dataset_obs_temporal_downsample_ratio, extend_obs_pad_after=True)['action'][0].detach().cpu().numpy()
                    full_gripper_step_action = policy.predict_from_latent_action(gripper_step_latent_action, extended_obs_dict, gripper_extended_obs_step, dataset_obs_temporal_downsample_ratio, extend_obs_pad_after=True)['action'][0].detach().cpu().numpy()
                    if self.use_relative_action:
                        full_tcp_step_action = relative_actions_to_absolute_actions(full_tcp_step_action, tcp_base_absolute_action)
                        full_gripper_step_action = relative_actions_to_absolute_actions(full_gripper_step_action, gripper_base_absolute_action)
                    self.env.get_predicted_action(full_tcp_step_action[:, :tcp_len], type='full_tcp')
                    self.env.get_predicted_action(full_gripper_step_action[:, tcp_len:], type='full_gripper')

                tcp_step_action = tcp_step_action[-1]
                gripper_step_action = gripper_step_action[-1]

                tcp_step_action = tcp_step_action[:tcp_len]
                gripper_step_action = gripper_step_action[tcp_len:]

            combined_action = np.concatenate([tcp_step_action, gripper_step_action], axis=-1)
            # convert to 16-D robot action (TCP + gripper of both arms)
            # TODO: handle rotation in temporal ensemble buffer!
            step_action, is_bimanual = self.post_process_action(combined_action[np.newaxis, :])
            step_action = step_action.squeeze(0)

            # send action to the robot
            self.env.execute_action(step_action, use_relative_action=False, is_bimanual=is_bimanual)

            cur_time = time.time()
            precise_sleep(max(0., self.control_interval_time - (cur_time - start_time)))
            self.action_step_count += 1

    def start_record_video(self, video_path):
        for vcamera_server_ip, vcamera_server_port in zip(self.vcamera_server_ip_list, self.vcamera_server_port_list):
            response = self.session.post(f'http://{vcamera_server_ip}:{vcamera_server_port}/start_recording/{video_path}')
            if response.status_code == 200:
                logger.info(f"Start recording video to {video_path}")
            else:
                logger.error(f"Failed to start recording video to {video_path}")

    def stop_record_video(self):
        for vcamera_server_ip, vcamera_server_port in zip(self.vcamera_server_ip_list, self.vcamera_server_port_list):
            response = self.session.post(f'http://{vcamera_server_ip}:{vcamera_server_port}/stop_recording')
            if response.status_code == 200:
                logger.info(f"Stop recording video")
            else:
                logger.error(f"Failed to stop recording video")

    def run(self, policy: Union[DiffusionUnetImagePolicy]):
        if self.use_latent_action_with_rnn_decoder:
            assert policy.at.use_rnn_decoder, "Policy should use rnn decoder for latent action."
        else:
            assert not hasattr(policy, 'at') or not policy.at.use_rnn_decoder, "Policy should not use rnn decoder for action."

        device = policy.device
        episode_idx = 0

        logger.info(f"Start evaluation episode")
        self.env.reset()
        # set gripper to max width
        # self.env.send_gripper_command_direct(self.env.max_gripper_width, self.env.max_gripper_width)
        # self.env.gripper.open_gripper()
        time.sleep(1)

        policy.reset()
        self.tcp_ensemble_buffer.clear()
        self.gripper_ensemble_buffer.clear()
        logger.debug("Reset environment and policy.")

        if self.enable_video_recording:
            video_path = os.path.join(self.video_dir, f'episode.mp4')
            self.start_record_video(video_path)
            logger.info(f"Start recording video to {video_path}")

        self.stop_event.clear()
        time.sleep(0.5)
        # start a new thread for action command

        self.action_step_count = 0
        step_count = 0

        action_thread = threading.Thread(target=self.action_command_thread, args=(policy, self.stop_event,),
                                            daemon=True)
        rossub_thread = threading.Thread(target=self.env.ros_thread, daemon=True)
        action_thread.start()
        rossub_thread.start()
        steps_per_inference = int(self.control_fps / self.inference_fps)
        start_timestamp = time.time()
        try:
            time.sleep(10)
            while True:
                self.step_cnt = step_count
                # profiler = Profiler()
                # profiler.start()
                start_time = time.time()
                # get obs
                obs = self.env.get_obs(
                    obs_steps=self.n_obs_steps,
                    temporal_downsample_ratio=self.obs_temporal_downsample_ratio)
                # obs = dict()

                if len(obs) == 0:
                    logger.warning("No observation received! Skip this step.")
                    cur_time = time.time()
                    precise_sleep(max(0., self.inference_interval_time - (cur_time - start_time)))
                    step_count += steps_per_inference
                    continue

                # create obs dict
                np_obs_dict = dict(obs)
                # get transformed real obs dict
                np_obs_dict = get_real_obs_dict(
                    env_obs=np_obs_dict, shape_meta=self.shape_meta)
                np_obs_dict, np_absolute_obs_dict = self.pre_process_obs(np_obs_dict)

                # device transfer
                obs_dict = dict_apply(np_obs_dict,
                                        lambda x: torch.from_numpy(x).unsqueeze(0).to(
                                            device=device))

                policy_time = time.time()
                # run policy
                with torch.no_grad():
                    if self.use_latent_action_with_rnn_decoder:
                        action_dict = policy.predict_action(obs_dict,
                                                            dataset_obs_temporal_downsample_ratio=self.dataset_obs_temporal_downsample_ratio,
                                                            return_latent_action=True)
                    else:
                        action_dict = policy.predict_action(obs_dict)
                self.latent_cnt += 1
                logger.debug(f"Policy inference time: {time.time() - policy_time:.3f}s")

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())

                action_all = np_action_dict['action'].squeeze(0)
                if self.use_latent_action_with_rnn_decoder:
                    # add first absolute action to get absolute action
                    if self.use_relative_action:
                        base_absolute_action = np.concatenate([
                            np_absolute_obs_dict['left_robot_tcp_pose'][-1] if 'left_robot_tcp_pose' in np_absolute_obs_dict else np.array([]),
                            np_absolute_obs_dict['right_robot_tcp_pose'][-1] if 'right_robot_tcp_pose' in np_absolute_obs_dict else np.array([])
                        ], axis=-1)
                        # print('base:', base_absolute_action)
                        action_all = np.concatenate([
                            action_all,
                            base_absolute_action[np.newaxis, :].repeat(action_all.shape[0], axis=0)
                        ], axis=-1)
                    # add action step to get corresponding observation
                    action_all = np.concatenate([
                        action_all,
                        np.arange(self.n_obs_steps * self.dataset_obs_temporal_downsample_ratio, action_all.shape[0] + self.n_obs_steps * self.dataset_obs_temporal_downsample_ratio)[:, np.newaxis]
                    ], axis=-1)
                else:
                    if self.use_relative_action:
                        base_absolute_action = np.concatenate([
                            np_absolute_obs_dict['left_robot_tcp_pose'][-1] if 'left_robot_tcp_pose' in np_absolute_obs_dict else np.array([]),
                            np_absolute_obs_dict['right_robot_tcp_pose'][-1] if 'right_robot_tcp_pose' in np_absolute_obs_dict else np.array([])
                        ], axis=-1)
                        action_all = relative_actions_to_absolute_actions(action_all, base_absolute_action)

                if self.action_interpolation_ratio > 1:
                    if self.use_latent_action_with_rnn_decoder:
                        action_all = action_all.repeat(self.action_interpolation_ratio, axis=0)
                    else:
                        action_all = interpolate_actions_with_ratio(action_all, self.action_interpolation_ratio)

                # TODO: only takes the first n_action_steps and add to the ensemble buffer
                if step_count % self.tcp_action_update_interval == 0:
                    if self.use_latent_action_with_rnn_decoder:
                        tcp_action = action_all[self.latency_step:, ...]
                    else:
                        if action_all.shape[-1] == 4:
                            tcp_action = action_all[self.latency_step:, :3]
                        elif action_all.shape[-1] == 8:
                            tcp_action = action_all[self.latency_step:, :6]
                        elif action_all.shape[-1] == 10:
                            tcp_action = action_all[self.latency_step:, :9]
                        elif action_all.shape[-1] == 20:
                            tcp_action = action_all[self.latency_step:, :18]
                        else:
                            raise NotImplementedError
                    # add to ensemble buffer
                    # logger.debug(f"Step: {step_count}, Add TCP action to ensemble buffer: {tcp_action}")
                    self.tcp_ensemble_buffer.add_action(tcp_action, step_count)

                    if self.env.enable_exp_recording and not self.use_latent_action_with_rnn_decoder:
                        self.env.get_predicted_action(tcp_action, type='full_tcp')

                if step_count % self.gripper_action_update_interval == 0:
                    if self.use_latent_action_with_rnn_decoder:
                        gripper_action = action_all[self.gripper_latency_step:, ...]
                    else:
                        if action_all.shape[-1] == 4:
                            gripper_action = action_all[self.gripper_latency_step:, 3:]
                        elif action_all.shape[-1] == 8:
                            gripper_action = action_all[self.gripper_latency_step:, 6:]
                        elif action_all.shape[-1] == 10:
                            gripper_action = action_all[self.gripper_latency_step:, 9:]
                        elif action_all.shape[-1] == 20:
                            gripper_action = action_all[self.gripper_latency_step:, 18:]
                        else:
                            raise NotImplementedError
                    # add to ensemble buffer
                    # logger.debug(f"Step: {step_count}, Add gripper action to ensemble buffer: {gripper_action}")
                    self.gripper_ensemble_buffer.add_action(gripper_action, step_count)

                    if self.env.enable_exp_recording and not self.use_latent_action_with_rnn_decoder:
                        self.env.get_predicted_action(gripper_action, type='full_gripper')

                cur_time = time.time()
                precise_sleep(max(0., self.inference_interval_time - (cur_time - start_time)))
                if cur_time - start_timestamp >= self.max_duration_time:
                    logger.info(f"Episode reaches max duration time {self.max_duration_time} seconds.")
                    break
                step_count += steps_per_inference
                # profiler.stop()
                # profiler.print()

        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt! Terminate the episode now!")
        finally:
            self.stop_event.set()
            action_thread.join()
            if self.enable_video_recording:
                self.stop_record_video()
            self.env.save_exp(episode_idx)
