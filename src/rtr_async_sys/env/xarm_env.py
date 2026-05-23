from rtr_async_sys.core.env_base import AbsEnv
from typing import Any, Dict
import numpy as np

# import library of xarm_env
import serial
import time
import binascii
import threading
from xarm.wrapper import XArmAPI
from loguru import logger
import os
import cv2
import rospy
import collections
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped, PointStamped
import sensor_msgs.point_cloud2 as pc2
import message_filters
from cv_bridge import CvBridge
import transforms3d as t3d

# instantiate
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

import pickle

from rtr_async_sys.utils.image_utils import compress_image


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


class RobotiqGripper:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200, timeout=1):
        """
        Initialize the serial connection with default communication parameters.
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
        Send a command to the gripper and read the response.
        """
        self.ser.write(command)
        time.sleep(0.05)
        response = self.ser.read_all()
        return response

    def receive(self, command):
        """
        Receive response data from the gripper.
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
        Move the gripper to a target position without blocking.
        """
        if not (0 <= position <= 255):
            raise ValueError("Target position must be in the range [0, 255].")
        if not (0 <= speed <= 255 or 0 <= force <= 255):
            raise ValueError("Speed or force is out of range.")

        # Build command
        command = (
            b'\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00' +
            bytes([position, speed, force])
        )
        crc = self._calculate_crc(command)
        command += crc

        # Send command asynchronously
        self.ser.write(command)

    def _calculate_crc(self, data):
        """
        Compute CRC checksum
        :param data: bytes to validate
        :return: CRC16 checksum, returned low byte first and high byte second
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
        Get gripper status information
        Return a dictionary with the following fields::
        - gripper_status: gripper status (gACT, gGTO, gSTA)
        - object_status: object detection status (gOBJ)
        - fault_status: fault status (gFLT)
        - position_request_echo: position request echo (gPR)
        - position: current position (gPO)
        - current: current current (gCU)
        """
        # Read input registers (FC04) - address0x07D0start, read3registers(6bytes)
        command = b'\x09\x04\x07\xD0\x00\x03\xB1\xCE'
        # Read response (11bytes: address1 + function code1 + byte count1 + data6 + CRC2)
        response = self.receive(command)

        if len(response) != 11:
            print("Error: Invalid response length")
            return None
        
        # Parse response data
        data = response[3:-2]  # Remove address, function code, byte count, and CRC
        # Parse data according to the register map in the documentation
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
            'current': data[5]                   # current current (value x 10 ~= mA)
        }
        return status
    
    def get_gripper_extended_status(self):
        """
        Get detailed gripper status information with human-readable descriptions.
        """
        status = self.get_gripper_status()
        if status is None:
            return None
        
        # Detailed status descriptions
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
        # Fault status description.
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
        
        # Add description fields
        gripper_status = status['gripper_status']
        gripper_status['gSTA_desc'] = gSTA_desc.get(gripper_status['gSTA'], "Unknown")
        gripper_status['gOBJ_desc'] = gOBJ_desc.get(gripper_status['gOBJ'], "Unknown")
        status['fault_desc'] = fault_desc.get(status['fault_status'], "Unknown fault")
        
        # Convert current to mA and compute torque
        status['current_mA'] = status['current'] * 10
        torque_constant = 0.02  # Assume motor torque constant is 0.02 N·m / A
        status['motor_torque_Nm'] = (status['current_mA'] / 1000) * torque_constant
        
        return status

    def disconnect(self):
        """
        Close the serial connection.
        """
        self.ser.close()


class XArmController:
    def __init__(self, ip='192.168.1.239', servo_mode=False):
        self.servo_mode = servo_mode
        self.arm = XArmAPI(ip)
        time.sleep(0.5)
        self.clean_errors()
        self.arm.motion_enable(enable=True)
        if self.servo_mode:
            self.arm.set_mode(1)
        else:
            self.arm.set_mode(0)
        self.arm.set_state(state=0)
        # self.actions = np.load(action_file)[:128]

    def stop(self):
        self.arm.set_state(4)
    
    def restart(self,servo_mode=False):
        self.clean_errors()
        self.arm.motion_enable(enable=True)
        if servo_mode:
            self.arm.set_mode(1)
        else:
            self.arm.set_mode(0)
        self.servo_mode = servo_mode
        self.arm.set_state(state=0)
    
    def set_mode(self,servo_mode=False):
        if servo_mode:
            self.arm.set_mode(1)
        else:
            self.arm.set_mode(0)
        self.servo_mode = servo_mode
        self.arm.set_state(state=0)

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
        if self.servo_mode:
            self.arm.set_servo_cartesian(mvpose=action, is_radian=False)
        else:
            self.arm.set_position(x=action[0], y=action[1], z=action[2], roll=action[3], pitch=action[4], yaw=action[5], speed=100, is_radian=False, wait=True)
        
    
    def get_pose(self):
        pose = self.arm.get_position()[1]

        return pose




class XarmEnv(AbsEnv):
    """
    Please start() before you use this env.
    """
    def __init__(self,
                 n_obs_steps: int = 2,
                 pca_load_dir: str = "",
                 robo_ip='192.168.1.239',
                 add_z_mm:float=0,
                 log_tactile:bool=False,
                 log_dir:str = "outputs/log_dir",
                 compress_obs:bool = False,
                 visualize_image:bool = False,
                 servo_mode:bool = False,
                 log_traj_dir:str = None,
                 ):
        super().__init__()
        logger.info(f"[XarmEnv] init XarmEnv")
        self.servo_mode = servo_mode
        self.gripper = RobotiqGripper()
        self.motor = XArmController(ip=robo_ip,servo_mode=servo_mode)
        rospy.init_node('all_topics_subscriber', anonymous=True)
        self.obs_buffer = ObservationBuffer(maxlen=36)
        # self.motor = XArmController()
        self.max_steps = 100
        self.left_tac_transform_matrix = np.load(os.path.join(pca_load_dir, 'pca_matrix1.npy'))
        self.left_tac_mean_matrix = np.load(os.path.join(pca_load_dir, 'pca_mean1.npy'))

        self.mutex = threading.Lock()
        self.bridge = CvBridge()
        self.n_obs_steps = n_obs_steps
        self.add_z_mm = add_z_mm
        self.log_dir = log_dir
        self.log_tactile = log_tactile
        self._logstep = 0
        self.tactile_dict = {}
        self.input_key_list = ['left_wrist_img', 'left_robot_tcp_pose', 'left_robot_gripper_width', 'left_gripper1_marker_offset_emb']
        self.compress_obs = compress_obs
        self.visualize_image = visualize_image

        self.log_traj_dir = log_traj_dir
        self.log_print_list = [] # Used to store printed log messages
        self.log_switch_list = []
        self.log_traj_list = []
        if self.log_traj_dir is not None:
            os.makedirs(self.log_traj_dir, exist_ok=True)
            # Scan existing traj_XXX.pkl files and find the largest index
            existing_files = [
                fname for fname in os.listdir(self.log_traj_dir)
                if fname.startswith("traj_") and fname.endswith(".pkl")
            ]

            if existing_files:
                # Extract the numeric part, e.g. "traj_002.pkl" -> "002" -> 2
                indices = []
                for fname in existing_files:
                    num_str = fname[len("traj_"):-len(".pkl")]
                    if num_str.isdigit():
                        indices.append(int(num_str))
                max_idx = max(indices) if indices else -1
            else:
                max_idx = -1

            next_idx = max_idx + 1
            # Log trajectory
            self.log_traj_path = os.path.join(self.log_traj_dir, f"traj_{next_idx:03d}.pkl")
            # Log executor print output
            self.log_print_path = os.path.join(self.log_traj_dir, f"print_{next_idx:03d}.txt")
            # Log last_xyz and now_xyz across chunk switches
            self.log_switch_path = os.path.join(self.log_traj_dir, f"switch_chunk_xyz_{next_idx:03d}.pkl")

    def ros_thread(self):
        self.sub_camera_1_image = message_filters.Subscriber('/camera_1_image', Image)
        self.sub_camera_2_image = message_filters.Subscriber('/camera_2_image', Image)
        self.sub_camera_1_depth = message_filters.Subscriber('/camera_1_depth', Image)
        self.sub_camera_2_depth = message_filters.Subscriber('/camera_2_depth', Image)
        self.sub_tac1_data = message_filters.Subscriber('/tac1_data', PointCloud2)
        self.sub_tac2_data = message_filters.Subscriber('/tac2_data', PointCloud2)
        self.sub_xarm_eef = message_filters.Subscriber('/xarm_eef_pose', PoseStamped)
        self.sub_gripper_pos = message_filters.Subscriber('/gripper_tele_pos', PointStamped)  
        self.ts = message_filters.ApproximateTimeSynchronizer(
                [self.sub_camera_1_image,
                 self.sub_camera_2_image,
                 self.sub_camera_1_depth,
                 self.sub_camera_2_depth,
                 self.sub_tac1_data,
                 self.sub_tac2_data,
                 self.sub_xarm_eef,
                 ],
                queue_size=100,
                slop=0.8,
                allow_headerless=False
            )
        self.ts.registerCallback(self.synced_callback)
        rospy.spin()

    def process_tac_data_msg(self, tac_data):
        points = []
        for p in pc2.read_points(tac_data, field_names=("x", "y", "z", "dx", "dy", "dz"), skip_nans=True):
            points.append(p)
        tac_data_array = np.array(points)
        mesh = tac_data_array[:,:3]
        deform = tac_data_array[:,3:]
        
        return mesh, deform

    def synced_callback(self, camera_1_image, camera_2_image, camera_1_depth, camera_2_depth, tac1_data, tac2_data, xarm_eef):
        raw_obs_dict = {}
        mesh1, deform1 = self.process_tac_data_msg(tac1_data)
        mesh2, deform2 = self.process_tac_data_msg(tac2_data)

        deform_emb = (deform1[:,:-1].reshape(-1, 1)[0] - self.left_tac_mean_matrix) @ self.left_tac_transform_matrix

        cam1_img = self.bridge.imgmsg_to_cv2(camera_1_image, desired_encoding='bgr8')

        cam2_depth = self.bridge.imgmsg_to_cv2(camera_2_depth, desired_encoding='32FC1')

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
        
        gripper_pos_array = 112
        
        raw_obs_dict['left_robot_tcp_pose'] = xarm_eef_array
        raw_obs_dict['left_robot_gripper_width'] = np.array([gripper_pos_array / 255.0])
        if not self.compress_obs:
            # Convert to RGB
            raw_obs_dict['left_wrist_img'] = cv2.resize(cam1_img, (320, 240))[...,::-1]
        else:
            # Convert to RGB and compress
            raw_obs_dict['left_wrist_img'] = compress_image(cv2.resize(cam1_img, (320, 240))[...,::-1])
        raw_obs_dict['left_gripper1_marker_offset_emb'] = deform_emb
        raw_obs_dict['left_gripper1_tactile'] = np.concatenate((mesh1, deform1), axis=-1)
        raw_obs_dict['left_gripper2_tactile'] = np.concatenate((mesh2, deform2), axis=-1)
        raw_obs_dict['global_depth'] = cam2_depth
        
        # Only transmit required data to reduce communication
        real_obs_dict = {}
        for key in raw_obs_dict.keys():
            if key in self.input_key_list:
                real_obs_dict[key] = raw_obs_dict[key]
        raw_obs_dict = real_obs_dict

        self.obs_buffer.append_obs(raw_obs_dict)

    def start(self):
        rossub_thread = threading.Thread(target=self.ros_thread, daemon=True)
        rossub_thread.start()
        time.sleep(1)
        logger.info("[XarmEnv] ros_thread is running in daemon")
    
    def stop(self):
        self.motor.stop()
    
    def restart(self, servo_mode=False):
        self.motor.restart(servo_mode)
    
    def set_mode(self, servo_mode=False):
        self.motor.set_mode(servo_mode)

    # def post_process_action(self, action: np.ndarray) -> np.ndarray:
    #     left_rot_mat_batch = ortho6d_to_rotation_matrix(action[None, 3:9])  # (action_steps, 3, 3)
    #     left_euler_batch = np.array([t3d.euler.mat2euler(rot_mat) for rot_mat in left_rot_mat_batch])  # (action_steps, 3)
    #     left_trans_batch = action[None, :3]  # (action_steps, 3)
    #     left_action_6d = np.concatenate([left_trans_batch, left_euler_batch], axis=1)

    #     return left_action_6d[0]
    
    def post_process_action(self, action: np.ndarray) -> np.ndarray:
        """
        Post-processing should live in model_wrapper because different models output different shapes. Convert actions to a unified format before sending them to env, e.g. one-arm actions as (x, y, z, r, p, y, gripper_width, gripper_force), two-arm actions as two such vectors, or a dict with left/right entries.
        """

        return action[0:6] # Currently supports one arm without gripper control only, (x, y, z, r, p, y)

    
    def get_obs(self,n_obs_steps=None) -> Dict[str, np.ndarray]:
        """
        obs['left_wrist_img']: shape  [t, h, w, c]
        """
        if n_obs_steps == None:
            obs = self.obs_buffer.get_new_obs(self.n_obs_steps)
        else:
            # TODO: only send useful obs
            obs = self.obs_buffer.get_new_obs(n_obs_steps)

        if obs is None:
            return None
        
        if self.log_tactile:
            left_gripper1_marker_offset_emb = obs[-1]['left_gripper1_marker_offset_emb']
            logger.debug("="*100)
            logger.debug(left_gripper1_marker_offset_emb)
            self.tactile_dict[self._logstep] = left_gripper1_marker_offset_emb.copy()
            if self._logstep % 5 == 0:
                log_path = os.path.join(self.log_dir, "tactile_realenv.pkl")
                with open(log_path, 'wb') as f:
                    pickle.dump(self.tactile_dict, f)

        if self.visualize_image:
            vis_img = obs[-1]["left_wrist_img"]
            episode_dir = "outputs/vis_outputs/vis_imgs"
            os.makedirs(episode_dir, exist_ok=True)

            # Save image files as step_000.png
            img_path = os.path.join(episode_dir, f"step_{self._logstep:03d}.png")
            import cv2
            cv2.imwrite(img_path, vis_img)


        return obs
    
    def end_of_chunk(self):
        pass
    
    def execute_action(self, action: np.ndarray, not_append_traj = False):
        """
        action: shape[10]
        """
        if self._logstep >= self.max_steps:
            logger.info("env's step beyond env's max_steps. Env will stop!!")

        action = self.post_process_action(action)

        # log action: xyz is m, rpy is radian
        if self.log_traj_dir is not None:
            if not not_append_traj:
                now_time = time.time()
                self.log_traj_list.append((action.copy(),now_time))
            with open(self.log_traj_path, "wb") as f:
                pickle.dump(self.log_traj_list, f)
            with open(self.log_print_path, "w", encoding="utf-8") as f:
                for line in self.log_print_list:
                    f.write(line + "\n")
            with open(self.log_switch_path, "wb") as f:
                pickle.dump(self.log_switch_list, f)

        action[:3] *= 1000.0
        if self.add_z_mm != 0:
            raise ValueError('add_z_mm should be zero for real experiments')
        action[2] += self.add_z_mm # DEBUG: add add_z_mm millimeters of offset
        action[3:6] = np.rad2deg(action[3:6])
        left_tcp_target_6d_in_robot = action[:6]
        logger.debug(f"[XarmEnv] robot_action: {left_tcp_target_6d_in_robot}")
        self.motor.move_to_pose(left_tcp_target_6d_in_robot) 
        
        self._logstep += 1

    def clear(self) -> None:
        # TODO: implement xarm hardware buffer clear
        logger.info("clear XarmEnv. TODO: implement xarm hardware buffer clear")

    def reset(self) -> None:
        logger.info("reset XarmEnv")

@hydra.main(config_path="../configs/executor/env", config_name="xarm_env", version_base=None)
def instantiate_xarmenv(env: DictConfig):
    if isinstance(env, str):
        env = OmegaConf.load(env)
    if isinstance(env, OmegaConf):
        env = hydra.utils.instantiate(env)
    # env = build_dp_dataset_env(cfg)
    print(env)
    obs = env.get_obs()
    env.start()
    time.sleep(20)
    print(f"obs.keys is {obs.keys()}")

if __name__ == '__main__':
    # instantiate_xarmenv()
    env = "src/rtr_async_sys/configs/executor/env/xarm_env.yaml"
    env = OmegaConf.load(env)
    env = hydra.utils.instantiate(env)
    env.start()
    i=0
    while True:
        time.sleep(0.1)
        env.get_obs()
        print(i)
        i+=1
    time.sleep(20)
    # print(f"obs.keys is {obs.keys()}")


