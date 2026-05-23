import pickle
import os
import os.path as osp
import time
import requests
from rclpy.node import Node
from typing import List, Optional, Tuple
from message_filters import ApproximateTimeSynchronizer, Subscriber
from reactive_diffusion_policy.real_world.real_world_transforms import RealWorldTransforms
from reactive_diffusion_policy.real_world.device_mapping.device_mapping_server import DeviceToTopic
from reactive_diffusion_policy.real_world.device_mapping.device_mapping_utils import get_topic_and_type
from loguru import logger
from reactive_diffusion_policy.real_world.ros_data_converter import ROS2DataConverter
from reactive_diffusion_policy.common.data_models import (SensorMessage, SensorMessageList)
from reactive_diffusion_policy.common.time_utils import convert_ros_time_to_float
import pyinstrument

class DataRecorder(Node):
    last_sensor_msg: SensorMessage = None
    def __init__(self,
                 transforms: RealWorldTransforms,
                 device_mapping_server_ip: str,
                 device_mapping_server_port: int,
                 save_path: str = 'data/test.pkl',
                 tactile_camera_marker_dimension: int = 2,
                 debug: bool = True,
                 time_check: bool = False,
                 ):
        super().__init__('sync_listener')
        self.transforms = transforms
        self.debug = debug
        self.time_check = time_check
        self.subscribers = []
        self.sensor_msg_list: SensorMessageList = SensorMessageList(sensorMessages=[])
        self.save_path = save_path

        # Get device to topic mapping
        response = requests.get(
            f"http://{device_mapping_server_ip}:{device_mapping_server_port}/get_mapping")
        self.device_to_topic_mapping = DeviceToTopic.model_validate(response.json())

        self.timestamps = {name: [] for name, _ in get_topic_and_type(self.device_to_topic_mapping)}
        self.cnt = 0

        logger.debug("Initializing SyncListener node...")

        subs_name_type = get_topic_and_type(self.device_to_topic_mapping)
        depth_camera_point_cloud_topic_names: List[Optional[str]] = [None, None, None]  # external, left wrist, right wrist
        depth_camera_rgb_topic_names: List[Optional[str]] = [None, None, None]  # external, left wrist, right wrist
        tactile_camera_rgb_topic_names: List[Optional[str]] = [None, None, None, None]  # left gripper1, left gripper2, right gripper1, right gripper2
        tactile_camera_marker_topic_names: List[Optional[str]] = [None, None, None, None]  # left gripper1, left gripper2, right gripper1, right gripper2

        for topic, msg_type in subs_name_type:
            if "depth/points" in topic:
                if "external_camera" in topic:
                    depth_camera_point_cloud_topic_names[0] = topic
                elif "left_wrist_camera" in topic:
                    depth_camera_point_cloud_topic_names[1] = topic
                elif "right_wrist_camera" in topic:
                    depth_camera_point_cloud_topic_names[2] = topic
            elif "color/image_raw" in topic:
                if "gripper_camera" in topic:
                    if "left_gripper_camera_1" in topic:
                        tactile_camera_rgb_topic_names[0] = topic
                    elif "left_gripper_camera_2" in topic:
                        tactile_camera_rgb_topic_names[1] = topic
                    elif "right_gripper_camera_1" in topic:
                        tactile_camera_rgb_topic_names[2] = topic
                    elif "right_gripper_camera_2" in topic:
                        tactile_camera_rgb_topic_names[3] = topic
                else:
                    if "external_camera" in topic:
                        depth_camera_rgb_topic_names[0] = topic
                    elif "left_wrist_camera" in topic:
                        depth_camera_rgb_topic_names[1] = topic
                    elif "right_wrist_camera" in topic:
                        depth_camera_rgb_topic_names[2] = topic
            elif "marker_offset/information" in topic:
                if "left_gripper_camera_1" in topic:
                    tactile_camera_marker_topic_names[0] = topic
                elif "left_gripper_camera_2" in topic:
                    tactile_camera_marker_topic_names[1] = topic
                elif "right_gripper_camera_1" in topic:
                    tactile_camera_marker_topic_names[2] = topic
                elif "right_gripper_camera_2" in topic:
                    tactile_camera_marker_topic_names[3] = topic
        

        if self.debug:
            logger.debug(f"Depth camera point cloud topic names: {depth_camera_point_cloud_topic_names}")
            logger.debug(f"Depth camera rgb topic names: {depth_camera_rgb_topic_names}")
            logger.debug(f"Tactile camera rgb topic names: {tactile_camera_rgb_topic_names}")
            logger.debug(f"Tactile camera marker topic names: {tactile_camera_marker_topic_names}")

        self.data_converter = ROS2DataConverter(self.transforms,
                                                depth_camera_point_cloud_topic_names,
                                                depth_camera_rgb_topic_names,
                                                tactile_camera_rgb_topic_names,
                                                tactile_camera_marker_topic_names,
                                                tactile_camera_marker_dimension = tactile_camera_marker_dimension,
                                                debug=self.debug)
        if self.data_converter is None:
            logger.warning("no calling data converter")
        
        for name, msg_type in subs_name_type:
            self.subscribers.append(Subscriber(self, msg_type, name))
            logger.debug(f"Subscribed to topic: {name} with type: {msg_type}")

        # ApproximateTimeSynchronizer is used to synchronize multiple topics
        self.ts = ApproximateTimeSynchronizer(self.subscribers, queue_size=10, slop=0.1,
                                              allow_headerless=False)
            
        self.ts.registerCallback(self.callback)

        # for calculating FPS
        self.prev_time = time.time()
        self.frame_count = 0

    def callback(self, *msgs):
        topic_dict = dict()
        
        self.cnt += 1        
        for i, msg in enumerate(msgs):
            topic_name = self.subscribers[i].topic
            topic_dict[topic_name] = msg

        if self.debug:
            # calculate the lastest timestamp in the topic_dict
            latest_timestamp = max([msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 for msg in msgs])
            # convert current time (ROS time) to python time
            current_timestamp = convert_ros_time_to_float(self.get_clock().now())
            # find out the latency compared to current time
            latency = current_timestamp - latest_timestamp
            # the latency is approximately 12ms - 24ms (RGB only)
            logger.debug(f"Latency: {latency:.4f} seconds")
            
        if self.time_check:
            # check the time differences across topics and interval between time stamps
            for i, msg in enumerate(msgs):
                topic_name = self.subscribers[i].topic
                self.timestamps[topic_name].append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
                
        # this part takes about 25ms - 100ms (with point cloud) for now
        # this part takes about 3ms (with RGB only) for now
        sensor_msg = self.data_converter.convert_all_data(topic_dict)
        if self.last_sensor_msg is None:
            # we need at least two sensor messages to calculate the robot action
            self.last_sensor_msg = sensor_msg
            return

        # TODO: record actual action
        # save the sensor message to list in memory
        self.sensor_msg_list.sensorMessages.append(sensor_msg)
        self.last_sensor_msg = sensor_msg

        # calculate fps
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.prev_time
        if elapsed_time >= 1.0:
            frame_rate = self.frame_count / elapsed_time
            logger.debug(f"Frame rate: {frame_rate:.2f} FPS")
            self.prev_time = current_time
            self.frame_count = 0
            if self.time_check:
                self.check_sync()
                self.check_timestamp()


    def save(self):
        # save sensor_msg_list to pickle file
        logger.debug('Trying to save sensor messages...')
        if not osp.exists(osp.dirname(self.save_path)):
            os.makedirs(osp.dirname(self.save_path))
        with open(self.save_path, 'wb') as f:
            pickle.dump(self.sensor_msg_list, f)
        logger.info(f"Saved sensor messages to {self.save_path}")
        
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



