"""
Fake env: provides observations by indexing into the dataset.
"""

import serial
import time
import binascii
import threading
import torch
import requests
import numpy as np
from omegaconf import DictConfig
# from xarm.wrapper import XArmAPI
from typing import Union, List, Dict, Optional
from loguru import logger
import os
import cv2
# import rospy
import pickle
# from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped, PointStamped
# import sensor_msgs.point_cloud2 as pc2
# import message_filters
# from cv_bridge import CvBridge
import transforms3d as t3d

import hydra
from omegaconf import OmegaConf
from reactive_diffusion_policy.dataset.base_dataset import BaseImageDataset
from torch.utils.data import DataLoader


from reactive_diffusion_policy.common.ring_buffer import RingBuffer
from reactive_diffusion_policy.real_world.post_process_utils import DataPostProcessingManager
from reactive_diffusion_policy.real_world.real_world_transforms import RealWorldTransforms

from reactive_diffusion_policy.common.action_utils import relative_actions_to_absolute_actions

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

def stack_last_n_obs(all_obs, n_steps: int) -> Union[np.ndarray, torch.Tensor]:
    assert(len(all_obs) > 0)
    all_obs = list(all_obs)
    if isinstance(all_obs[0], np.ndarray):
        result = np.zeros((n_steps,) + all_obs[-1].shape,
            dtype=all_obs[-1].dtype)
        start_idx = -min(n_steps, len(all_obs))
        result[start_idx:] = np.array(all_obs[start_idx:])
        if n_steps > len(all_obs):
            # pad
            result[:start_idx] = result[start_idx]
    elif isinstance(all_obs[0], torch.Tensor):
        result = torch.zeros((n_steps,) + all_obs[-1].shape,
            dtype=all_obs[-1].dtype)
        start_idx = -min(n_steps, len(all_obs))
        result[start_idx:] = torch.stack(all_obs[start_idx:])
        if n_steps > len(all_obs):
            # pad
            result[:start_idx] = result[start_idx]
    else:
        raise RuntimeError(f'Unsupported obs type {type(all_obs[0])}')
    return result

# class RobotiqGripper:
#     def __init__(self, port='/dev/ttyUSB0', baudrate=115200, timeout=1):
#         """
#         Open the serial port and set default communication parameters.
#         """
#         self.ser = serial.Serial(
#             port=port,
#             baudrate=baudrate,
#             timeout=timeout,
#             parity=serial.PARITY_NONE,
#             stopbits=serial.STOPBITS_ONE,
#             bytesize=serial.EIGHTBITS
#         )
    
#     def send_command(self, command):
#         """
#         Send a command to the gripper and read back the response.
#         """
#         self.ser.write(command)
#         time.sleep(0.05)
#         response = self.ser.read_all()
#         return response

#     def receive(self, command):
#         """
#         Read response data from the gripper.
#         """
#         self.ser.write(command)
#         time.sleep(0.05)
#         response = self.ser.read_all()
#         return response

#     def activate_gripper(self):
#         """
#         Activate the gripper.
#         """
#         command = b'\x09\x10\x03\xE8\x00\x03\x06\x00\x00\x00\x00\x00\x00\x73\x30'
#         response = self.send_command(command)
#         print(f"Activate Response: {binascii.hexlify(response)}")
#         return response

#     def deactivate_gripper(self):
#         """
#         Reset the gripper.
#         """
#         command = b'\x09\x10\x03\xE8\x00\x03\x06\x00\x00\x00\x00\x00\x00\x73\x30'
#         response = self.send_command(command)
#         print(f"Deactivate Response: {binascii.hexlify(response)}")
#         return response

#     def close_gripper(self):
#         """
#         Close the gripper.
#         """
#         command = b'\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00\xFF\xFF\xFF\x42\x29'
#         response = self.send_command(command)
#         # print(f"Close Gripper Response: {binascii.hexlify(response)}")
#         return response

#     def open_gripper(self):
#         """
#         Open the gripper.
#         """
#         command = b'\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00\x00\xFF\xFF\x72\x19'
#         response = self.send_command(command)
#         # print(f"Open Gripper Response: {binascii.hexlify(response)}")
#         return response

#     def move(self, position, speed=255, force=255):
#         """
#         Move the gripper to an arbitrary target position (non-blocking).
#         """
#         if not (0 <= position <= 255):
#             raise ValueError("target position must be in [0, 255]")
#         if not (0 <= speed <= 255 or 0 <= force <= 255):
#             raise ValueError("speed or force is out of range")

#         # build command
#         command = (
#             b'\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00' +
#             bytes([position, speed, force])
#         )
#         crc = self._calculate_crc(command)
#         command += crc

#         # send command asynchronously
#         self.ser.write(command)

#     def _calculate_crc(self, data):
#         """
#         Compute the CRC.
#         :param data: bytes to be checked
#         :return: CRC16 little-endian (low byte first, high byte second)
#         """
#         crc = 0xFFFF
#         for pos in data:
#             crc ^= pos
#             for _ in range(8):
#                 if crc & 0x0001:
#                     crc >>= 1
#                     crc ^= 0xA001
#                 else:
#                     crc >>= 1
#         return crc.to_bytes(2, byteorder='little')

#     def get_gripper_status(self):
#         """
#         Get the gripper status.
#         Returns a dict with:
#         - gripper_status: gripper state (gACT, gGTO, gSTA)
#         - object_status: object detection status (gOBJ)
#         - fault_status: fault status (gFLT)
#         - position_request_echo: position request echo (gPR)
#         - position: current position (gPO)
#         - current: current draw (gCU)
#         """
#         # read input registers (FC04) -- starting at address 0x07D0, 3 registers (6 bytes)
#         command = b'\x09\x04\x07\xD0\x00\x03\xB1\xCE'
#         # read response (11 bytes: 1 address + 1 function code + 1 byte count + 6 data + 2 CRC)
#         response = self.receive(command)

#         if len(response) != 11:
#             print("Error: Invalid response length")
#             return None
        
#         # parse response data
#         data = response[3:-2]  # strip address, function code, byte count, and CRC
#         # parse fields per the register map in the documentation
#         status = {
#             'gripper_status': {
#                 'gACT': (data[0] >> 0) & 0x01,  # activation status
#                 'gGTO': (data[0] >> 3) & 0x01,  # motion status
#                 'gSTA': (data[0] >> 4) & 0x03,  # gripper status
#                 'gOBJ': (data[0] >> 6) & 0x03   # object detection status
#             },
#             'fault_status': data[2],             # fault status
#             'position_request_echo': data[3],    # position request echo
#             'position': data[4],                 # current position
#             'current': data[5]                   # current draw (value*10 ~= mA)
#         }
#         return status
    
#     def get_gripper_extended_status(self):
#         """
#         Get extended gripper status (including human-readable descriptions).
#         """
#         status = self.get_gripper_status()
#         if status is None:
#             return None
        
#         # detailed status descriptions
#         gSTA_desc = {
#             0x00: "Gripper is in reset (or automatic release) state",
#             0x01: "Activation in progress",
#             0x03: "Activation is completed"
#         }
#         gOBJ_desc = {
#             0x00: "Fingers are in motion towards requested position. No object detected",
#             0x01: "Fingers have stopped due to a contact while opening before requested position. Object detected opening",
#             0x02: "Fingers have stopped due to a contact while closing before requested position. Object detected closing",
#             0x03: "Fingers are at requested position. No object detected or object has been lost/dropped"
#         }
#         # fault status descriptions
#         fault_desc = {
#             0x00: "No fault (solid blue LED)",
#             0x05: "Action delayed, the activation must be completed prior to performing the action",
#             0x07: "The activation bit must be set prior to performing the action",
#             0x08: "Maximum operating temperature exceeded",
#             0x09: "No communication during at least 1 second",
#             0x0A: "Under minimum operating voltage",
#             0x0B: "Automatic release in progress",
#             0x0C: "Internal fault",
#             0x0D: "Activation fault",
#             0x0E: "Overcurrent triggered",
#             0x0F: "Automatic release completed"
#         }
        
#         # attach descriptions
#         gripper_status = status['gripper_status']
#         gripper_status['gSTA_desc'] = gSTA_desc.get(gripper_status['gSTA'], "Unknown")
#         gripper_status['gOBJ_desc'] = gOBJ_desc.get(gripper_status['gOBJ'], "Unknown")
#         status['fault_desc'] = fault_desc.get(status['fault_status'], "Unknown fault")
        
#         # convert current to mA and compute approximate torque
#         status['current_mA'] = status['current'] * 10
#         torque_constant = 0.02  # assumed motor torque constant: 0.02 N*m / A
#         status['motor_torque_Nm'] = (status['current_mA'] / 1000) * torque_constant
        
#         return status

#     def disconnect(self):
#         """
#         Close the serial connection.
#         """
#         self.ser.close()

# class XArmController:
#     def __init__(self, ip='192.168.1.239'):
#         self.arm = XArmAPI(ip)
#         time.sleep(0.5)
#         self.clean_errors()
#         self.arm.motion_enable(enable=True)
#         self.arm.set_mode(0)
#         self.arm.set_state(state=0)
#         # self.actions = np.load(action_file)[:128]

#     def clean_errors(self):
#         if self.arm.warn_code != 0:
#             self.arm.clean_warn()
#         if self.arm.error_code != 0:
#             self.arm.clean_error()

#     def move_to_pose(self, action):
#         """
#         parameters:
#         input:
#         set_position x,y,z unit in mm, roll,pitch,yaw unit in degree

#         return: x,y,z unit in mm, roll,pitch,yaw unit in degree
#         """
#         # print(f"Executing action: {action}")
#         # x,y,z unit in mm, roll,pitch,yaw unit in degree
#         self.arm.set_position(x=action[0], y=action[1], z=action[2], roll=action[3], pitch=action[4], yaw=action[5], speed=100, is_radian=False, wait=True)
        
    
#     def get_pose(self):
#         pose = self.arm.get_position()[1]

#         return pose

class DatasetRobotEnv():
    def __init__(self,
                cfg:OmegaConf,
                #  robot_server_ip: str,
                #  robot_server_port: int,
                 transforms: RealWorldTransforms,
                #  device_mapping_server_ip: str,
                #  device_mapping_server_port: int,
                 data_processing_params: DictConfig,
                #  max_fps: int = 30,
                #  pca_load_dir: str = "/home/robotics/Prometheus/reactive_diffusion_policy/tactile_pca",
                #  # gripper control parameters
                #  use_force_control_for_gripper: bool = True,
                #  max_gripper_width: float = 0.05,
                #  min_gripper_width: float = 0.,
                #  grasp_force: float = 5.0,
                #  enable_gripper_interval_control: bool = False,
                #  gripper_control_time_interval: float = 60,
                #  gripper_control_width_precision: float = 0.02,
                #  gripper_width_threshold: float = 0.04,
                #  enable_gripper_width_clipping: bool = True,
                #  enable_exp_recording: bool = False,
                #  output_dir: Optional[str] = None,
                #  vcamera_server_ip: Optional[str] = None,
                #  vcamera_server_port: Optional[int] = None,
                #  time_check: bool = False,
                #  debug: bool = False
                 ):
        print("init DatasetRobotEnv")
        dataset: BaseImageDataset
        self.dataset = hydra.utils.instantiate(cfg.task.dataset)
        self.timestep = 1
        self.executed_actions = []
        self.raw_predict_actions = []
        self.plot_actions = []  # per-timestep plot_actions[0] is {'fact': [[x,y,z], [x,y,z], ...], 'predict': [[x,y,z], [x,y,z], ...]}
        # dict_keys(['obs', 'action', 'extended_obs'])

 
        self.data_processing_manager = DataPostProcessingManager(transforms,
                                                                 **data_processing_params)
        
      
        self.enable_exp_recording = False


    def ros_thread(self):
        print("nothing happened in ros_thread")
        # self.sub_camera_1_image = message_filters.Subscriber('/camera_1_image', Image)
        # self.sub_camera_2_image = message_filters.Subscriber('/camera_2_image', Image)
        # self.sub_tac1_data = message_filters.Subscriber('/tac1_data', PointCloud2)
        # self.sub_tac2_data = message_filters.Subscriber('/tac2_data', PointCloud2)
        # self.sub_xarm_eef = message_filters.Subscriber('/xarm_eef_pose', PoseStamped)
        # self.sub_gripper_pos = message_filters.Subscriber('/gripper_tele_pos', PointStamped)  
        # self.ts = message_filters.ApproximateTimeSynchronizer(
        #         [self.sub_camera_1_image,
        #          self.sub_camera_2_image,
        #          self.sub_tac1_data,
        #          self.sub_tac2_data,
        #          self.sub_xarm_eef,
        #          ],
        #         queue_size=40,
        #         slop=0.06,
        #         allow_headerless=False
        #     )
        # self.ts.registerCallback(self.synced_callback)
        # rospy.spin()

    def synced_callback(self, camera_1_image, camera_2_image, tac1_data, tac2_data, xarm_eef):
        print("nothing happedned in synced_callback")
        

    def check_sync(self):
        print("nothing happened in check_sync")
       

    def check_timestamp(self):
        print("nothing happened in check_timestep")
        


    def get_predicted_action(self, action: np.ndarray, type):
        pass
    
    def execute_raw_action(self, action):
        self.raw_predict_actions.append(action[-1][0:3]*1000)

    def execute_action(self, action: np.ndarray, use_relative_action: bool = False, is_bimanual: bool = False) -> None:
        """
        Send action (in robot coordinate system) to robot
        :param action: np.ndarray, shape (16,) (left+right) (x, y, z, r, p, y, gripper_width, gripper_force)
        """
        start = time.time()
        left_action = action[:8]

        # calculate target gripper width
        if use_relative_action:
            raise NotImplementedError
        else:
            left_gripper_width_target = float(left_action[-2])
            left_gripper_width_target = np.clip(left_gripper_width_target, 0.0, 1)
        # self.gripper.move(int(left_gripper_width_target), 255, 20)

        if use_relative_action:
            raise NotImplementedError
        else:
            left_action[:3] *= 1000.0
            left_action[2] = left_action[2]
            left_action[3:6] = np.rad2deg(left_action[3:6])
            left_tcp_target_6d_in_robot = left_action[:6]
            # logger.debug(f"robot_action: {left_tcp_target_6d_in_robot}")
            # print(f"robot_action: {left_tcp_target_6d_in_robot}")
        # self.motor.move_to_pose(left_tcp_target_6d_in_robot)#you can add comment out this line to disable realy control. for debug
        # six_action = left_tcp_target_6d_in_robot
        self.executed_actions.append(left_tcp_target_6d_in_robot[0:3])
        end = time.time()
        logger.debug(f"[RealRobotEnv] execute_action latency is {end - start}")


    def reset(self) -> None:
        self.start_gripper_interval_control = False
        self.timestep = 1
        self.raw_predict_actions = []
        self.executed_actions = []
 

    def get_obs(self,obs_steps: int = 2, temporal_downsample_ratio: int = 2, ) -> Dict[str, np.ndarray]:
        """
        For convenience, downsampling is not supported here.

        Get observations with temporal downsampling support.

        Args:
            obs_steps: The number of observations to stack.
            temporal_downsample_ratio: The ratio for temporal downsampling.
                For example, if ratio=2, it will sample every other observation.
        Returns:
            A dictionary containing stacked observations
        """


        result = dict()

        obs_list = []
        # This may not be appropriate for the fast model: it returns past tactile signals, but the fast model predicts future actions.
        # Possibly change this to self.timestep-1, self.timestep, ..., self.timestep.
        # for i in reversed(range(obs_steps)):
        #     idx = self.timestep - i
        #     if idx >= 0:
        #         obs_list.append(self.dataset[idx]['obs'])
        #     else:
        #         obs_list.append(self.dataset[0]['obs'])

        for i in range(obs_steps):
            if i == 0:
                obs_list.append(self.dataset[self.timestep-1]['obs'])
            else:
                obs_list.append(self.dataset[self.timestep]['obs'])

        # obs1 = self.dataset[self.timestep-1]['obs']
        # obs2 = self.dataset[self.timestep]['obs']

        # Stack observations for each key
        for key in obs_list[0].keys():
            cat_list = [obs[key][1].unsqueeze(0) for obs in obs_list]
            result[key] = torch.cat(cat_list, dim=0)
            # result[key] = torch.cat((obs1[key][1].unsqueeze(0), obs2[key][1].unsqueeze(0)), dim=0)
            #  need t,hi,wi,ci, but original is t, c, h, w
            # print(f"key is {key}")
            if key == 'left_wrist_img':
                # print("permute")
                result[key] = result[key].permute(0,2,3,1)
            # convert to numpy
            result[key] = result[key].numpy()



        return result
    
    def update_obs(self):
        # raw == under the relative-mode setting, the not-yet-converted-to-absolute action. In absolute mode raw is undefined.
        plot_action = {
            'fact':[],
            'predict':[],
            'raw_fact':[],
            'raw_predict':[]
        }
        fact_action_chunk = self.dataset[self.timestep]['action']  # [32, 10]
        raw_fact_action_chunk = fact_action_chunk.numpy()
        # TODO: convert relative_action to absolute
        base_position = self.dataset[self.timestep]['obs']['left_robot_tcp_pose'][1]
        fact_action_chunk = relative_actions_to_absolute_actions(fact_action_chunk.numpy(), base_position.numpy())

        # record
        start = fact_action_chunk.shape[0] - len(self.executed_actions)
        # is `start` correct?
        # start = 0
        start = 4-1  # the first obs_steps*temporal_downsample_ratio actions are past states; the slow system already drops the past actions from the chunk, so we drop them from the GT here too for alignment (some misalignment may still remain)
        fact_action_chunk = fact_action_chunk[start:]  # leading entries are past positions
        raw_fact_action_chunk = raw_fact_action_chunk[start:]
        for i in range(len(self.executed_actions)):
            fact_action = fact_action_chunk[i][0:3] * 1000  # x, y, z
            raw_fact_action = raw_fact_action_chunk[i][0:3]*1000
            # numpy array
            plot_action['fact'].append(fact_action)
            plot_action['predict'].append(self.executed_actions[i])
            plot_action['raw_fact'].append(raw_fact_action)
            plot_action['raw_predict'].append(self.raw_predict_actions[i])


        self.plot_actions.append(plot_action)

        self.timestep += 1
        if self.timestep >= len(self.dataset):
            self.timestep = 0
        self.executed_actions = []
        self.raw_predict_actions = []

    def save_plot_actions(self, save_path="data/outputs/vis_outputs/plot_actions.pkl"):
        with open(save_path, "wb") as f:
            pickle.dump(self.plot_actions, f)

    def save_exp(self, episode_idx):
        pass

        