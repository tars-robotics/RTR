import serial
import time
import binascii
import threading
import torch
import requests
import numpy as np
from omegaconf import DictConfig
from xarm.wrapper import XArmAPI
from typing import Union, List, Dict, Optional
from loguru import logger
import os
import cv2
import rospy
import pickle
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped, PointStamped
import sensor_msgs.point_cloud2 as pc2
import message_filters
from cv_bridge import CvBridge
import transforms3d as t3d

from reactive_diffusion_policy.common.ring_buffer import RingBuffer
from reactive_diffusion_policy.real_world.post_process_utils import DataPostProcessingManager
from reactive_diffusion_policy.real_world.real_world_transforms import RealWorldTransforms

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

class RobotiqGripper:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200, timeout=1):
        """
        Open the serial port and set default communication parameters.
        """
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=timeout,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS
        )
    
    def send_command(self, command):
        """
        Send a command to the gripper and read back the response.
        """
        self.ser.write(command)
        time.sleep(0.05)
        response = self.ser.read_all()
        return response

    def receive(self, command):
        """
        Read response data from the gripper.
        """
        self.ser.write(command)
        time.sleep(0.05)
        response = self.ser.read_all()
        return response

    def activate_gripper(self):
        """
        Activate the gripper.
        """
        command = b'\x09\x10\x03\xE8\x00\x03\x06\x00\x00\x00\x00\x00\x00\x73\x30'
        response = self.send_command(command)
        print(f"Activate Response: {binascii.hexlify(response)}")
        return response

    def deactivate_gripper(self):
        """
        Reset the gripper.
        """
        command = b'\x09\x10\x03\xE8\x00\x03\x06\x00\x00\x00\x00\x00\x00\x73\x30'
        response = self.send_command(command)
        print(f"Deactivate Response: {binascii.hexlify(response)}")
        return response

    def close_gripper(self):
        """
        Close the gripper.
        """
        command = b'\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00\xFF\xFF\xFF\x42\x29'
        response = self.send_command(command)
        # print(f"Close Gripper Response: {binascii.hexlify(response)}")
        return response

    def open_gripper(self):
        """
        Open the gripper.
        """
        command = b'\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00\x00\xFF\xFF\x72\x19'
        response = self.send_command(command)
        # print(f"Open Gripper Response: {binascii.hexlify(response)}")
        return response

    def move(self, position, speed=255, force=255):
        """
        Move the gripper to an arbitrary target position (non-blocking).
        """
        if not (0 <= position <= 255):
            raise ValueError("target position must be in [0, 255]")
        if not (0 <= speed <= 255 or 0 <= force <= 255):
            raise ValueError("speed or force is out of range")

        # build command
        command = (
            b'\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00' +
            bytes([position, speed, force])
        )
        crc = self._calculate_crc(command)
        command += crc

        # send command asynchronously
        self.ser.write(command)

    def _calculate_crc(self, data):
        """
        Compute the CRC.
        :param data: bytes to be checked
        :return: CRC16 little-endian (low byte first, high byte second)
        """
        crc = 0xFFFF
        for pos in data:
            crc ^= pos
            for _ in range(8):
                if crc & 0x0001:
                    crc >>= 1
                    crc ^= 0xA001
                else:
                    crc >>= 1
        return crc.to_bytes(2, byteorder='little')

    def get_gripper_status(self):
        """
        Get the gripper status.
        Returns a dict with:
        - gripper_status: gripper state (gACT, gGTO, gSTA)
        - object_status: object detection status (gOBJ)
        - fault_status: fault status (gFLT)
        - position_request_echo: position request echo (gPR)
        - position: current position (gPO)
        - current: current draw (gCU)
        """
        # read input registers (FC04) -- starting at address 0x07D0, 3 registers (6 bytes)
        command = b'\x09\x04\x07\xD0\x00\x03\xB1\xCE'
        # read response (11 bytes: 1 address + 1 function code + 1 byte count + 6 data + 2 CRC)
        response = self.receive(command)

        if len(response) != 11:
            print("Error: Invalid response length")
            return None
        
        # parse response data
        data = response[3:-2]  # strip address, function code, byte count, and CRC
        # parse fields per the register map in the documentation
        status = {
            'gripper_status': {
                'gACT': (data[0] >> 0) & 0x01,  # activation status
                'gGTO': (data[0] >> 3) & 0x01,  # motion status
                'gSTA': (data[0] >> 4) & 0x03,  # gripper status
                'gOBJ': (data[0] >> 6) & 0x03   # object detection status
            },
            'fault_status': data[2],             # fault status
            'position_request_echo': data[3],    # position request echo
            'position': data[4],                 # current position
            'current': data[5]                   # current draw (value*10 ~= mA)
        }
        return status
    
    def get_gripper_extended_status(self):
        """
        Get extended gripper status (including human-readable descriptions).
        """
        status = self.get_gripper_status()
        if status is None:
            return None
        
        # detailed status descriptions
        gSTA_desc = {
            0x00: "Gripper is in reset (or automatic release) state",
            0x01: "Activation in progress",
            0x03: "Activation is completed"
        }
        gOBJ_desc = {
            0x00: "Fingers are in motion towards requested position. No object detected",
            0x01: "Fingers have stopped due to a contact while opening before requested position. Object detected opening",
            0x02: "Fingers have stopped due to a contact while closing before requested position. Object detected closing",
            0x03: "Fingers are at requested position. No object detected or object has been lost/dropped"
        }
        # fault status descriptions
        fault_desc = {
            0x00: "No fault (solid blue LED)",
            0x05: "Action delayed, the activation must be completed prior to performing the action",
            0x07: "The activation bit must be set prior to performing the action",
            0x08: "Maximum operating temperature exceeded",
            0x09: "No communication during at least 1 second",
            0x0A: "Under minimum operating voltage",
            0x0B: "Automatic release in progress",
            0x0C: "Internal fault",
            0x0D: "Activation fault",
            0x0E: "Overcurrent triggered",
            0x0F: "Automatic release completed"
        }
        
        # attach descriptions
        gripper_status = status['gripper_status']
        gripper_status['gSTA_desc'] = gSTA_desc.get(gripper_status['gSTA'], "Unknown")
        gripper_status['gOBJ_desc'] = gOBJ_desc.get(gripper_status['gOBJ'], "Unknown")
        status['fault_desc'] = fault_desc.get(status['fault_status'], "Unknown fault")
        
        # convert current to mA and compute approximate torque
        status['current_mA'] = status['current'] * 10
        torque_constant = 0.02  # assumed motor torque constant: 0.02 N*m / A
        status['motor_torque_Nm'] = (status['current_mA'] / 1000) * torque_constant
        
        return status

    def disconnect(self):
        """
        Close the serial connection.
        """
        self.ser.close()

class XArmController:
    def __init__(self, ip='192.168.1.239'):
        self.arm = XArmAPI(ip)
        time.sleep(0.5)
        self.clean_errors()
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        # self.actions = np.load(action_file)[:128]

    def clean_errors(self):
        if self.arm.warn_code != 0:
            self.arm.clean_warn()
        if self.arm.error_code != 0:
            self.arm.clean_error()

    def move_to_pose(self, action):
        """
        parameters:
        input:
        set_position x,y,z unit in mm, roll,pitch,yaw unit in degree

        return: x,y,z unit in mm, roll,pitch,yaw unit in degree
        """
        # print(f"Executing action: {action}")
        # x,y,z unit in mm, roll,pitch,yaw unit in degree
        self.arm.set_position(x=action[0], y=action[1], z=action[2], roll=action[3], pitch=action[4], yaw=action[5], speed=100, is_radian=False, wait=True)
        
    
    def get_pose(self):
        pose = self.arm.get_position()[1]

        return pose

class RealRobotEnv():
    def __init__(self,
                 robot_server_ip: str,
                 robot_server_port: int,
                 transforms: RealWorldTransforms,
                 device_mapping_server_ip: str,
                 device_mapping_server_port: int,
                 data_processing_params: DictConfig,
                 max_fps: int = 30,
                 pca_load_dir: str = "./tactile_pca",
                 # gripper control parameters
                 use_force_control_for_gripper: bool = True,
                 max_gripper_width: float = 0.05,
                 min_gripper_width: float = 0.,
                 grasp_force: float = 5.0,
                 enable_gripper_interval_control: bool = False,
                 gripper_control_time_interval: float = 60,
                 gripper_control_width_precision: float = 0.02,
                 gripper_width_threshold: float = 0.04,
                 enable_gripper_width_clipping: bool = True,
                 enable_exp_recording: bool = False,
                 output_dir: Optional[str] = None,
                 vcamera_server_ip: Optional[str] = None,
                 vcamera_server_port: Optional[int] = None,
                 time_check: bool = False,
                 debug: bool = False,
                 add_z_mm: int = 0,
                 max_stride = 5
                 ):
        self.gripper = RobotiqGripper()
        self.motor = XArmController()
        rospy.init_node('all_topics_subscriber', anonymous=True)
        self.subscribers = []
        self.obs_buffer = RingBuffer(size=1024, fps=max_fps)
        self.data_processing_manager = DataPostProcessingManager(transforms,
                                                                 **data_processing_params)
        
        self.left_tac_transform_matrix = np.load(os.path.join(pca_load_dir, 'pca_matrix1.npy'))
        self.left_tac_mean_matrix = np.load(os.path.join(pca_load_dir, 'pca_mean1.npy'))

        self.mutex = threading.Lock()
        self.bridge = CvBridge()

        # self.time_check = time_check
        self.time_check = True
        self.timestamps = {'cam1': [], 'tac1': [], 'xarm': [], 'gripper': []}
        self.prev_time = time.time()
        self.frame_count = 0

        self.enable_exp_recording = enable_exp_recording
        if self.enable_exp_recording:
            assert output_dir is not None, "output_dir must be provided for experiment recording"
        self.predicted_full_tcp_action_buffer = RingBuffer(size=1024, fps=max_fps)
        self.predicted_full_gripper_action_buffer = RingBuffer(size=1024, fps=max_fps)
        self.predicted_partial_tcp_action_buffer = RingBuffer(size=1024, fps=max_fps)
        self.predicted_partial_gripper_action_buffer = RingBuffer(size=1024, fps=max_fps)

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
        #         slop=0.05,
        #         allow_headerless=False
        #     )
        # self.ts.registerCallback(self.synced_callback)
        self.session = requests.session()
        self.time = 0
        self.add_z_mm = add_z_mm
        self.last_xyz = None
        self.max_stride = max_stride

    def ros_thread(self):
        self.sub_camera_1_image = message_filters.Subscriber('/camera_1_image', Image)
        self.sub_camera_2_image = message_filters.Subscriber('/camera_2_image', Image)
        self.sub_tac1_data = message_filters.Subscriber('/tac1_data', PointCloud2)
        self.sub_tac2_data = message_filters.Subscriber('/tac2_data', PointCloud2)
        self.sub_xarm_eef = message_filters.Subscriber('/xarm_eef_pose', PoseStamped)
        self.sub_gripper_pos = message_filters.Subscriber('/gripper_tele_pos', PointStamped)  
        self.ts = message_filters.ApproximateTimeSynchronizer(
                [self.sub_camera_1_image,
                 self.sub_camera_2_image,
                 self.sub_tac1_data,
                 self.sub_tac2_data,
                 self.sub_xarm_eef,
                 ],
                queue_size=40,
                slop=0.06,
                allow_headerless=False
            )
        self.ts.registerCallback(self.synced_callback)
        rospy.spin()

    def synced_callback(self, camera_1_image, camera_2_image, tac1_data, tac2_data, xarm_eef):
        raw_obs_dict = {}
        points = []
        for p in pc2.read_points(tac1_data, field_names=("x", "y", "z", "dx", "dy", "dz"), skip_nans=True):
            points.append(p)
        tac1_data_array = np.array(points)
        mesh = tac1_data_array[:,:3]
        deform = tac1_data_array[:,3:-1]
        deform = deform.reshape(-1, 1)[0]
        deform_emb = (deform - self.left_tac_mean_matrix) @ self.left_tac_transform_matrix
        tac1_ts = int(1000 * (tac1_data.header.stamp.secs + tac1_data.header.stamp.nsecs * 1e-9))

        cam1_img = self.bridge.imgmsg_to_cv2(camera_1_image, desired_encoding='bgr8')
        cam1_ts = int(1000 * (camera_1_image.header.stamp.secs + camera_1_image.header.stamp.nsecs * 1e-9))

        tx = xarm_eef.pose.position.x,
        ty = xarm_eef.pose.position.y,
        tz = xarm_eef.pose.position.z,
        rr = xarm_eef.pose.orientation.x,
        rp = xarm_eef.pose.orientation.y,
        ry = xarm_eef.pose.orientation.z,
        xarm_eef_array = np.array([tx, ty, tz, rr, rp, ry])[:, 0]
        xarm_eef_array[:3] *= 0.001
        xarm_eef_array[3:] = np.radians(xarm_eef_array[3:])
        xarm_eef_array = pose_6d_to_pose_9d(xarm_eef_array)
        xarm_ts = (int(1000 * (xarm_eef.header.stamp.secs + xarm_eef.header.stamp.nsecs * 1e-9)))
        
        # gripper_pos_array = gripper_pos.point.x
        # gripper_ts = int(1000 * (gripper_pos.header.stamp.secs + gripper_pos.header.stamp.nsecs * 1e-9))
        gripper_pos_array = 100.0
        
        raw_obs_dict['left_robot_tcp_pose'] = xarm_eef_array
        raw_obs_dict['left_robot_gripper_width'] = np.array([gripper_pos_array / 255.0])
        raw_obs_dict['left_wrist_img'] = cv2.resize(cam1_img, (320, 240))[...,::-1]
        raw_obs_dict['left_gripper1_marker_offset_emb'] = deform_emb
        raw_obs_dict['timestamp'] = np.array([cam1_ts, xarm_ts, tac1_ts]).mean(keepdims=True)

        self.obs_buffer.push(raw_obs_dict)
        # print('test subscriber time:', raw_obs_dict['timestamp'] - self.time)
        self.time = raw_obs_dict['timestamp']

        if self.time_check:
            self.timestamps['cam1'].append(cam1_ts)
            self.timestamps['tac1'].append(tac1_ts)
            self.timestamps['xarm'].append(xarm_ts)
            # self.timestamps['gripper'].append(gripper_ts)

        # calculate fps
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.prev_time
        if elapsed_time >= 1.0:
            frame_rate = self.frame_count / elapsed_time
            self.prev_time = current_time
            self.frame_count = 0
            if self.time_check:
                logger.debug(f"Frame rate: {frame_rate:.2f} FPS")
                self.check_sync()
                self.check_timestamp()

    def check_sync(self):
        # Check and log timestamp differences across topics
        all_times = list(self.timestamps.values())
        if not all(all_times):
            return

        # Calculate time differences for each frame across topics
        for i in range(len(all_times[0])):
            max_diff = 0
            for j in range(len(all_times)):
                for k in range(j + 1, len(all_times)):
                    if i < len(all_times[j]) and i < len(all_times[k]):
                        time_diff = abs(all_times[j][i] - all_times[k][i])
                        max_diff = max(max_diff, time_diff)
            logger.info(f"Frame {i}: Maximum time difference across topics: {max_diff:.6f} seconds")

    def check_timestamp(self):
        # check the interval between different time stampss
        all_times = list(self.timestamps.values())
        if not all(all_times):
            return

        time_stamps = []
        for i in range(len(all_times[0])):
            timestamps_for_frame = []
            for j in range(len(all_times)):
                if i < len(all_times[j]):
                    timestamp = all_times[j][i]
                    timestamps_for_frame.append(timestamp)

            if timestamps_for_frame:
                mean_time_stamp = sum(timestamps_for_frame) / len(timestamps_for_frame)
                time_stamps.append(mean_time_stamp)
                logger.info(f"Frame {i}: Mean timestamp: {mean_time_stamp:.6f} seconds")


    def get_predicted_action(self, action: np.ndarray, type):
        if self.enable_exp_recording:
            if type == 'full_tcp':
                self.predicted_full_tcp_action_buffer.push(action)
            elif type == "full_gripper":
                self.predicted_full_gripper_action_buffer.push(action)
            elif type == "partial_tcp":
                self.predicted_partial_tcp_action_buffer.push(action)
            elif type == "partial_gripper":
                self.predicted_partial_gripper_action_buffer.push(action)
            else:
                raise ValueError(f"Unknown action type: {type}")

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
            left_action[2] = left_action[2] + self.add_z_mm
            logger.debug(f"[execute_action] raise z by {self.add_z_mm}mm to avoid edge cases")

            if self.last_xyz is None:
                # first action: just record it
                self.last_xyz = (left_action[0], left_action[1], left_action[2])
            else:
                now_xyz = (left_action[0], left_action[1], left_action[2])

                # -------------------------------
                # compute the stride and clip it
                # -------------------------------
                last = np.array(self.last_xyz, dtype=float)
                now = np.array(now_xyz, dtype=float)

                diff = now - last
                dist = np.linalg.norm(diff)

                if dist > self.max_stride:
                    # over max stride -> scale the direction vector to length max_stride
                    scale = self.max_stride / dist
                    now = last + diff * scale

                    # keep left_action in sync
                    left_action[0], left_action[1], left_action[2] = now.tolist()

                logger.debug(f"[execute_action] stride {dist:.2f}, max {self.max_stride}")

                # update last_xyz
                self.last_xyz = (left_action[0], left_action[1], left_action[2])


            left_action[3:6] = np.rad2deg(left_action[3:6])
            left_tcp_target_6d_in_robot = left_action[:6]
            # logger.debug(f"robot_action: {left_tcp_target_6d_in_robot}")
            # print(f"robot_action: {left_tcp_target_6d_in_robot}")
        self.motor.move_to_pose(left_tcp_target_6d_in_robot)#you can add comment out this line to disable realy control. for debug
        end = time.time()
        logger.debug(f"[RealRobotEnv] execute_action latency is {end - start}")


    def reset(self) -> None:
        self.start_gripper_interval_control = False
        self.obs_buffer.reset()
        if self.enable_exp_recording:
            self.predicted_full_tcp_action_buffer.reset()
            self.predicted_full_gripper_action_buffer.reset()
            self.predicted_partial_tcp_action_buffer.reset()
            self.predicted_partial_gripper_action_buffer.reset()

    def get_obs(self,obs_steps: int = 2, temporal_downsample_ratio: int = 2, ) -> Dict[str, np.ndarray]:
        """
        Get observations with temporal downsampling support.

        Args:
            obs_steps: The number of observations to stack.
            temporal_downsample_ratio: The ratio for temporal downsampling.
                For example, if ratio=2, it will sample every other observation.
        Returns:
            A dictionary containing stacked observations
        """
        # Get last n*ratio observations to ensure we have enough samples after downsampling
        start = time.time()
        last_n_obs_list, _ = self.obs_buffer.peek_last_n(
            obs_steps * temporal_downsample_ratio)  # newest to oldest

        result = dict()
        # Filter out None observations
        last_n_obs_list = [obs for obs in last_n_obs_list if obs is not None]
        if len(last_n_obs_list) == 0:
            return result

        # Apply temporal downsampling
        # If ratio=2, it will take every other observation: [0, 2, 4, ...]
        # If ratio=3, it will take every third observation: [0, 3, 6, ...]
        downsampled_obs_list = last_n_obs_list[::temporal_downsample_ratio]
        # Take only the last n_obs_steps observations after downsampling
        downsampled_obs_list = downsampled_obs_list[:obs_steps]

        # reverse the order to oldest to newest
        downsampled_obs_list = downsampled_obs_list[::-1]

        # Stack observations for each key
        for key in downsampled_obs_list[0].keys():
            result[key] = stack_last_n_obs(
                [obs[key] for obs in downsampled_obs_list], obs_steps)

        # convert current time (ROS time) to python time
        current_timestamp = 1000 * (rospy.get_rostime().secs + rospy.get_rostime().nsecs * 1e-9) 
        # find out the latency compared to current time
        latency = current_timestamp - downsampled_obs_list[-1]['timestamp'][0]
        # the overall latency is approximately 20ms - 70ms (max 110ms) (~5000 points per pcd)
        end = time.time()
        # logger.debug(f"[old] Overall latency for get_obs() : {latency:.4f} ms")  # latency is the time from sampling the observation to the policy receiving it, not the execution time
        # logger.debug(f"[RealRobotEnv] get_obs() latency is {end - start}")

        return result

    def save_exp(self, episode_idx):
        if self.enable_exp_recording:
            logger.debug('Trying to save sensor messages...')
            if not os.path.exists(self.exp_dir):
                os.makedirs(self.exp_dir)
            record_path = os.path.join(self.exp_dir, f'episode_{episode_idx}.pkl')
            if os.path.exists(record_path):
                record_path = ".".join(record_path.split('.')[:-1]) + f'{time.strftime("_%Y%m%d_%H%M%S")}.pkl'
                logger.warning(f'Experiment path already exists, save to {record_path}')
            with open(record_path, 'wb') as f:
                with self.mutex:
                    pickle.dump(self.sensor_msg_list, f)
            logger.debug(f'Saved experiment record to {record_path}')
        