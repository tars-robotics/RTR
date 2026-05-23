import socket
import rclpy
import time
import numpy as np
import bson
import json
import requests
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Point, TwistStamped, WrenchStamped
from std_msgs.msg import Header
from reactive_diffusion_policy.real_world.real_world_transforms import RealWorldTransforms
from reactive_diffusion_policy.common.data_models import BimanualRobotStates, ForceSensorMessage, Arrow
from reactive_diffusion_policy.common.space_utils import pose_7d_to_4x4matrix, matrix4x4_to_pose_6d
from loguru import logger

class BimanualRobotPublisher(Node):
    """
    ROS 2 node that publishes gripper states and TCP poses, velocities, and wrenches of a bimanual robot.
    """
    def __init__(self,
                 robot_server_ip: str,
                 robot_server_port: int,
                 transforms: RealWorldTransforms,
                 vr_server_ip: str = '127.0.0.1',
                 vr_server_tcp_port: int = 10001,
                 vr_server_force_port: int = 10005,
                 fps: int = 120,
                 debug: bool = False):
        super().__init__('bimanual_robot_publisher')
        self.robot_server_ip = robot_server_ip
        self.robot_server_port = robot_server_port
        # Initialize the real world transforms
        self.transforms = transforms
        self.vr_server_ip = vr_server_ip
        self.vr_server_tcp_port = vr_server_tcp_port
        self.vr_server_force_port = vr_server_force_port
        self.fps = fps
        self.time_interval = 1 / fps

        # Publishers for TCP poses
        self.tcp_pose_left_publisher = self.create_publisher(PoseStamped, 'left_tcp_pose', 10)
        self.tcp_pose_right_publisher = self.create_publisher(PoseStamped, 'right_tcp_pose', 10)

        # Publishers for gripper states
        self.left_gripper_publisher = self.create_publisher(JointState, 'left_gripper_state', 10)
        self.right_gripper_publisher = self.create_publisher(JointState, 'right_gripper_state', 10)

        # Publishers for TCP velocities and wrenches
        self.left_tcp_vel_publisher = self.create_publisher(TwistStamped, 'left_tcp_vel', 10)
        self.right_tcp_vel_publisher = self.create_publisher(TwistStamped, 'right_tcp_vel', 10)
        self.left_tcp_wrench_publisher = self.create_publisher(WrenchStamped, 'left_tcp_wrench', 10)
        self.right_tcp_wrench_publisher = self.create_publisher(WrenchStamped, 'right_tcp_wrench', 10)

        self.timer = self.create_timer(1 / fps, self.timer_callback)
        # Create a session with robot server
        self.session = requests.session()
        # Create a socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # FPS counter
        self.prev_time = time.time()
        self.frame_count = 0
        logger.info("BimanualRobotPublisher node initialized")
        self.debug = debug

    def send_robot_msg(self, robot_states: BimanualRobotStates):

        left_tcp_pose = np.array(robot_states.leftRobotTCP)
        left_tcp_pose_7d_in_unity = self.transforms.robot_frame2unity(left_tcp_pose, left=True)
        right_tcp_pose = np.array(robot_states.rightRobotTCP)
        right_tcp_pose_7d_in_unity = self.transforms.robot_frame2unity(right_tcp_pose, left=False)

        robot_states_in_unity_dict = BimanualRobotStates(
            leftGripperState=robot_states.leftGripperState,
            rightGripperState=robot_states.rightGripperState,
            leftRobotTCP=left_tcp_pose_7d_in_unity,
            rightRobotTCP=right_tcp_pose_7d_in_unity
        ).model_dump()
        if self.debug:
            with open(f'robot_states.json', 'w') as json_file:
                json.dump(robot_states_in_unity_dict, json_file)

        packed_data = bson.dumps(robot_states_in_unity_dict)
        if self.debug:
            logger.debug(f"Sending robot states to VR server: {robot_states_in_unity_dict}")
        self.socket.sendto(packed_data, (self.vr_server_ip, self.vr_server_tcp_port))

        # send force sensor message
        force_scale_factor = 0.025
        left_start_point_7d_in_robot_frame = np.concatenate([np.array(robot_states.leftRobotTCP[:3]),
                                                             np.array([1., 0., 0., 0.])]) # (x, y, z, qw, qx, qy, qz)
        left_tcp_to_robot_base_transform_matrix = pose_7d_to_4x4matrix(np.array(robot_states.leftRobotTCP))
        # convert tcp force in TCP frame into robot base frame
        left_tcp_force_vector_7d_in_tcp_frame = np.concatenate([np.array(robot_states.leftRobotTCPWrench[:3]) * force_scale_factor,
                                                           np.array([1., 0., 0., 0.])]) # (x, y, z, qw, qx, qy, qz)
        left_tcp_force_vector_7d_in_robot_frame = matrix4x4_to_pose_6d(left_tcp_to_robot_base_transform_matrix @
                                                                    pose_7d_to_4x4matrix(left_tcp_force_vector_7d_in_tcp_frame))  # (x, y, z, qw, qx, qy, qz)
        left_end_point_7d_in_robot_frame = np.concatenate([left_tcp_force_vector_7d_in_robot_frame[:3],
                                                           np.array([1., 0., 0., 0.])]) # (x, y, z, qw, qx, qy, qz)
        # convert to Unity coordinate system
        left_arrow = Arrow(start=self.transforms.robot_frame2unity(left_start_point_7d_in_robot_frame, left=True)[:3],
                           end=self.transforms.robot_frame2unity(left_end_point_7d_in_robot_frame, left=True)[:3])

        right_start_point_7d_in_robot_frame = np.concatenate([np.array(robot_states.rightRobotTCP[:3]),
                                                                np.array([1., 0., 0., 0.])])
        right_tcp_to_robot_base_transform_matrix = pose_7d_to_4x4matrix(np.array(robot_states.rightRobotTCP))
        right_tcp_force_vector_7d_in_tcp_frame = np.concatenate([np.array(robot_states.rightRobotTCPWrench[:3]) * force_scale_factor,
                                                              np.array([1., 0., 0., 0.])])
        right_tcp_force_vector_7d_in_robot_frame = matrix4x4_to_pose_6d(right_tcp_to_robot_base_transform_matrix @
                                                                        pose_7d_to_4x4matrix(right_tcp_force_vector_7d_in_tcp_frame))
        right_end_point_7d_in_robot_frame = np.concatenate([right_tcp_force_vector_7d_in_robot_frame[:3],
                                                            np.array([1., 0., 0., 0.])])
        right_arrow = Arrow(start=self.transforms.robot_frame2unity(right_start_point_7d_in_robot_frame, left=False)[:3],
                            end=self.transforms.robot_frame2unity(right_end_point_7d_in_robot_frame, left=False)[:3])
        left_force_sensor_msg_dict = ForceSensorMessage(device_id='left', arrow=left_arrow).model_dump()
        right_force_sensor_msg_dict = ForceSensorMessage(device_id='right', arrow=right_arrow).model_dump()
        if self.debug:
            logger.debug(f"Sending left force sensor message to VR server: {left_force_sensor_msg_dict}")
            logger.debug(f"Sending right force sensor message to VR server: {right_force_sensor_msg_dict}")
        left_packed_data = bson.dumps(left_force_sensor_msg_dict)
        right_packed_data = bson.dumps(right_force_sensor_msg_dict)
        self.socket.sendto(left_packed_data, (self.vr_server_ip, self.vr_server_force_port))
        self.socket.sendto(right_packed_data, (self.vr_server_ip, self.vr_server_force_port))


    def send_command(self, endpoint: str, data: dict = None):
        url = f"http://{self.robot_server_ip}:{self.robot_server_port}{endpoint}"
        if 'get' in endpoint:
            response = self.session.get(url)
        else:
            response = self.session.post(url, json=data)
        response.raise_for_status()  # Raise an error for bad responses
        return response.json()

    def timer_callback(self):
        # this step has 0.5ms - 1ms latency
        robot_states = BimanualRobotStates.model_validate(self.send_command(f'/get_current_robot_states'))
        timestamp = self.get_clock().now().to_msg()

        self.send_robot_msg(robot_states)

        # Create and publish left gripper state
        left_gripper_state = JointState()
        left_gripper_state.header = Header()
        left_gripper_state.header.stamp = timestamp
        left_gripper_state.name = ['left_gripper']
        left_gripper_state.position = [robot_states.leftGripperState[0]]  # Example width in meters
        left_gripper_state.effort = [robot_states.leftGripperState[1]]  # Example force in Newtons
        self.left_gripper_publisher.publish(left_gripper_state)

        # Create and publish right gripper state
        right_gripper_state = JointState()
        right_gripper_state.header = Header()
        right_gripper_state.header.stamp = timestamp
        right_gripper_state.name = ['right_gripper']
        right_gripper_state.position = [robot_states.rightGripperState[0]]  # Example width in meters
        right_gripper_state.effort = [robot_states.rightGripperState[1]]  # Example force in Newtons
        self.right_gripper_publisher.publish(right_gripper_state)

        # Create and publish TCP pose messages for left arm
        tcp_pose_left_msg = PoseStamped()
        tcp_pose_left_msg.header = Header()
        tcp_pose_left_msg.header.stamp = timestamp
        tcp_pose_left_msg.header.frame_id = 'tcp_left'
        # robot_states.leftRobotTCP (x, y, z, qw, qx, qy, qz)
        tcp_pose_left_msg.pose.position = Point(x=robot_states.leftRobotTCP[0],
                                                y=robot_states.leftRobotTCP[1],
                                                z=robot_states.leftRobotTCP[2])
        tcp_pose_left_msg.pose.orientation.w = robot_states.leftRobotTCP[3]
        tcp_pose_left_msg.pose.orientation.x = robot_states.leftRobotTCP[4]
        tcp_pose_left_msg.pose.orientation.y = robot_states.leftRobotTCP[5]
        tcp_pose_left_msg.pose.orientation.z = robot_states.leftRobotTCP[6]
        self.tcp_pose_left_publisher.publish(tcp_pose_left_msg)

        # Create and publish TCP pose messages for right arm
        tcp_pose_right_msg = PoseStamped()
        tcp_pose_right_msg.header = Header()
        tcp_pose_right_msg.header.stamp = timestamp
        tcp_pose_right_msg.header.frame_id = 'tcp_right'

        # robot_states.rightRobotTCP (x, y, z, qw, qx, qy, qz)
        tcp_pose_right_msg.pose.position = Point(x=robot_states.rightRobotTCP[0],
                                                 y=robot_states.rightRobotTCP[1],
                                                 z=robot_states.rightRobotTCP[2])
        tcp_pose_right_msg.pose.orientation.w = robot_states.rightRobotTCP[3]
        tcp_pose_right_msg.pose.orientation.x = robot_states.rightRobotTCP[4]
        tcp_pose_right_msg.pose.orientation.y = robot_states.rightRobotTCP[5]
        tcp_pose_right_msg.pose.orientation.z = robot_states.rightRobotTCP[6]
        self.tcp_pose_right_publisher.publish(tcp_pose_right_msg)

        # Create and publish TCP velocity and wrench messages for left and right arms
        left_tcp_vel_msg = TwistStamped()
        left_tcp_vel_msg.header = Header()
        left_tcp_vel_msg.header.stamp = timestamp
        left_tcp_vel_msg.twist.linear.x = robot_states.leftRobotTCPVel[0]
        left_tcp_vel_msg.twist.linear.y = robot_states.leftRobotTCPVel[1]
        left_tcp_vel_msg.twist.linear.z = robot_states.leftRobotTCPVel[2]
        left_tcp_vel_msg.twist.angular.x = robot_states.leftRobotTCPVel[3]
        left_tcp_vel_msg.twist.angular.y = robot_states.leftRobotTCPVel[4]
        left_tcp_vel_msg.twist.angular.z = robot_states.leftRobotTCPVel[5]
        self.left_tcp_vel_publisher.publish(left_tcp_vel_msg)

        right_tcp_vel_msg = TwistStamped()
        right_tcp_vel_msg.header = Header()
        right_tcp_vel_msg.header.stamp = timestamp
        right_tcp_vel_msg.twist.linear.x = robot_states.rightRobotTCPVel[0]
        right_tcp_vel_msg.twist.linear.y = robot_states.rightRobotTCPVel[1]
        right_tcp_vel_msg.twist.linear.z = robot_states.rightRobotTCPVel[2]
        right_tcp_vel_msg.twist.angular.x = robot_states.rightRobotTCPVel[3]
        right_tcp_vel_msg.twist.angular.y = robot_states.rightRobotTCPVel[4]
        right_tcp_vel_msg.twist.angular.z = robot_states.rightRobotTCPVel[5]
        self.right_tcp_vel_publisher.publish(right_tcp_vel_msg)

        left_tcp_wrench_msg = WrenchStamped()
        left_tcp_wrench_msg.header = Header()
        left_tcp_wrench_msg.header.stamp = timestamp
        left_tcp_wrench_msg.wrench.force.x = robot_states.leftRobotTCPWrench[0]
        left_tcp_wrench_msg.wrench.force.y = robot_states.leftRobotTCPWrench[1]
        left_tcp_wrench_msg.wrench.force.z = robot_states.leftRobotTCPWrench[2]
        left_tcp_wrench_msg.wrench.torque.x = robot_states.leftRobotTCPWrench[3]
        left_tcp_wrench_msg.wrench.torque.y = robot_states.leftRobotTCPWrench[4]
        left_tcp_wrench_msg.wrench.torque.z = robot_states.leftRobotTCPWrench[5]
        self.left_tcp_wrench_publisher.publish(left_tcp_wrench_msg)

        right_tcp_wrench_msg = WrenchStamped()
        right_tcp_wrench_msg.header = Header()
        right_tcp_wrench_msg.header.stamp = timestamp
        right_tcp_wrench_msg.wrench.force.x = robot_states.rightRobotTCPWrench[0]
        right_tcp_wrench_msg.wrench.force.y = robot_states.rightRobotTCPWrench[1]
        right_tcp_wrench_msg.wrench.force.z = robot_states.rightRobotTCPWrench[2]
        right_tcp_wrench_msg.wrench.torque.x = robot_states.rightRobotTCPWrench[3]
        right_tcp_wrench_msg.wrench.torque.y = robot_states.rightRobotTCPWrench[4]
        right_tcp_wrench_msg.wrench.torque.z = robot_states.rightRobotTCPWrench[5]
        self.right_tcp_wrench_publisher.publish(right_tcp_wrench_msg)

        # Calculate fps
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.prev_time
        if elapsed_time >= 1.0:
            frame_rate = self.frame_count / elapsed_time
            logger.debug(f"Frame rate: {frame_rate:.2f} FPS")
            self.prev_time = current_time
            self.frame_count = 0


def main(args=None):
    rclpy.init(args=args)

    from hydra import initialize, compose
    import threading

    with initialize(config_path='../../config', version_base="1.3"):
        # config is relative to a module
        cfg = compose(config_name="real_world_env", overrides=["task=wipe_vase_two_realsense_one_gelsight_24fps"])

    from reactive_diffusion_policy.real_world.robot.bimanual_flexiv_server import BimanualFlexivServer

    # create robot server
    robot_server = BimanualFlexivServer(**cfg.task.robot_server)
    robot_server_thread = threading.Thread(target=robot_server.run, daemon=True)
    # start the robot server
    robot_server_thread.start()
    # wait for the robot server to start
    time.sleep(1)
    transforms = RealWorldTransforms(option=cfg.task.transforms)

    node = BimanualRobotPublisher(transforms=transforms, **cfg.task.publisher.robot_publisher)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()