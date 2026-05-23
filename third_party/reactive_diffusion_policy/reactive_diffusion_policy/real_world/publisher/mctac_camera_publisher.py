import os.path
import numpy as np
import rclpy
import bson
import socket
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image, PointCloud2, PointField
import json
from loguru import logger
import math
import uuid
import time
import cv2
import time as tm
import copy
import requests
import struct
import open3d as o3d

from reactive_diffusion_policy.common.data_models import TactileSensorMessage, Arrow
from reactive_diffusion_policy.common.tactile_marker_utils import marker_normalization
from reactive_diffusion_policy.real_world.publisher.lib import find_marker
from reactive_diffusion_policy.real_world.publisher.mctac_utility import MCTacUtility
from scipy.ndimage import gaussian_filter

class MCTacCameraPublisher(Node):
    '''
    MC-Tac Camera publisher Class
    '''

    def __init__(self,
                 camera_index: int = 0,
                 camera_type: str = 'MCTac',
                 fps: int = 30,
                 exposure: int = -6,
                 contrast: int = 100,
                 image_rotation: int = 0,
                 camera_name: str = 'left_gripper_camera_1',
                 vr_server_ip: str = '127.0.0.1',
                 vr_server_port: int = 10002,
                 teleop_server_ip: str = '192.168.2.187',
                 teleop_server_port: int = 8082,
                 dimension=3,
                 marker_vis_rotation_angle: float = 0.,  # in degrees
                 debug=False,
                 video_path="../../../data/tactile_video/video_001.mp4",
                 recorded=False,
                 enable_streaming: bool = False,
                 streaming_server_ip: str = '127.0.0.1',
                 streaming_server_port: int = 10004,
                 streaming_quality: int = 10,
                 streaming_chunk_size: int = 1024,
                 streaming_display_params_list: list = None,
                 # for visualization of resetting pattern
                 vis_latency_steps: int = 5,
                 ):
        node_name = f'{camera_name}_publisher_{camera_index}'
        super().__init__(node_name)
        self.camera_index = camera_index
        self.camera_name = camera_name
        self.vr_server_ip = vr_server_ip
        self.vr_server_port = vr_server_port
        self.cap = None
        self.img = None
        self.marker_img = None
        self.fps = fps
        self.contrast = contrast
        self.exposure = exposure
        self.width = 640
        self.height = 480
        self.image_rotation = image_rotation
        self.debug = debug
        self.dimension = dimension
        self.marker_vis_rotation_angle = np.deg2rad(marker_vis_rotation_angle)
        self.recorded = recorded
        self.camera_type = camera_type

        self.color_publisher_ = self.create_publisher(Image, f'/{camera_name}/color/image_raw', 10)
        self.marker_publisher = self.create_publisher(PointCloud2, f'/{camera_name}/marker_offset/information', 10)
        self.timer = self.create_timer(1 / fps, self.timer_callback)
        self.timestamp_offset = None

        self.last_print_time = tm.time()  # Add a variable to keep track of the last print time
        self.fps_list = []
        self.frame_intervals = []
        self.last_frame_time = None

        # track the markers
        self.initial_markers = None
        self.cur_markers = None
        self.prev_markers = None
        self.initial_markers_3d = None
        self.vertical_scale = 0.02

        self.prev_time = time.time()
        self.frame_count = 0

        self.video_path = video_path
        if recorded:
            assert os.path.exists(self.video_path), f"Video path {self.video_path} does not exist!"

        # streaming configuration
        self.enable_streaming = enable_streaming
        if self.enable_streaming:
            self.id = uuid.uuid4()
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.streaming_server_ip = streaming_server_ip
            self.streaming_server_port = streaming_server_port
            self.streaming_quality = streaming_quality
            self.streaming_chunk_size = streaming_chunk_size
            streaming_display_params_list = [{k: list(v) for k, v in d.items()} for d in
                                             streaming_display_params_list]
            self.streaming_display_params_list = streaming_display_params_list

        self.vis_latency_steps = vis_latency_steps
        self.latency_counter = 0

        # start the camera
        self.start()

        # Create a socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # create matching for MC-Tac camera
        self.MCTacHandler = MCTacUtility(RESCALE=1)
        self.RESCALE = self.MCTacHandler.RESCALE
        self.m = find_marker.Matching(
            N_=self.MCTacHandler.N,
            M_=self.MCTacHandler.M,
            fps_=self.MCTacHandler.fps,
            x0_=self.MCTacHandler.x0,
            y0_=self.MCTacHandler.y0,
            dx_=self.MCTacHandler.dx,
            dy_=self.MCTacHandler.dy)
        """
        N_, M_: the row and column of the marker array
        x0_, y0_: the coordinate of upper-left marker
        dx_, dy_: the horizontal and vertical interval between adjacent markers
        """

        self.teleop_server_ip = teleop_server_ip
        self.teleop_server_port = teleop_server_port
        self.cur_base_marker_motion = None
        self.prev_gripper_state = {
            "left_gripper_stable_closed": False,
            "right_gripper_stable_closed": False,
            "left_gripper_stable_open": True,
            "right_gripper_stable_open": True
        }

    def start(self):
        '''
        Start the usb camera
        Usb camera has no internal time,
        so we use the time we get the frame as the initial time of the topic
        '''
        if not self.recorded:
            if self.cap is None:
                self.cap = cv2.VideoCapture(self.camera_index)

                if not self.cap.isOpened():
                    self.cap.open(self.camera_index)
                    if not self.cap.isOpened():
                        logger.error("Could not open video device")
                        raise Exception("Could not open video device")

                logger.info(f"{self.camera_name} started")
            else:
                logger.warning("Camera is already running")
        else:
            self.cap = cv2.VideoCapture(self.video_path)

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info(f"Camera {self.camera_index} stopped")
        else:
            logger.warning("Camera is not running")

    def get_rgb_frame(self):
        if self.cap is not None:
            frame = None
            num_trial = 0
            while frame is None:
                if num_trial > 0:
                    logger.warning(f"No frame when reading from MCTac camera in trial {num_trial}")
                ret, frame = self.cap.read()
                tm.sleep(0.001)
                num_trial += 1
            # rotated_frame = self.MCTacHandler.rotate_image(frame, self.image_rotation)
            
            timestamp = self.get_clock().now()
            # logger.info(f"The shape of the frame is {frame.shape}")
            if not ret:
                logger.error(f"Failed to capture image from camera {self.camera_index}")
                raise Exception("Failed to capture image")
            else:
                self.img = frame

            # resized_frame = cv2.resize(rotated_frame, (self.width, self.height))
            resized_frame = cv2.resize(frame, (self.width, self.height))
            # logger.info(f'the shape of the resized frame is {resized_frame.shape}')
            frame = self.MCTacHandler.img_initiation(resized_frame)
            # logger.info(f'The size of the frame after initiation is {frame.shape}')
            return frame, timestamp
        else:
            logger.error("Camera is not running")
            raise Exception("Camera is not running")

    def get_marker_image(self, img):
        #  get marker images and motion vectors of the frame

        mask = self.MCTacHandler.find_marker(img)
        markers_detected = self.MCTacHandler.marker_center(mask)
        
        # guarantee for sudden marker vanishing
        if len(markers_detected) == 0:
            markers_detected = self.prev_markers
        self.prev_markers = markers_detected
        
        # img_display = img.copy()
        # for i in range(len(markers_detected)):
        #     x, y = int(markers_detected[i][0]), int(markers_detected[i][1])
        #     cv2.drawMarker(img_display, (x, y), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

        # cv2.imshow("Marker Tracking", img_display)
        # cv2.waitKey(1)

        # initial_markers: (num of markers, 3), the third dimension is 0 if 2-dimension and vertical offset if 3-dimension
        # marker_motion: (num of markers, 2)
        # rotate the marker for marker tracking
        rotated_marker = self.MCTacHandler.rotate_points(markers_detected, self.image_rotation)
        self.initial_markers, self.marekr_motion = self.track_marker(rotated_marker)

        initial_markers = self.initial_markers
        marker_motions = self.marekr_motion

        if self.debug:
            img_show = img.copy()
            showScale = 5
            self.marker_img = self.display_motion(img_show, initial_markers, marker_motions, showScale, self.dimension)
            return self.marker_img
        else:
            return initial_markers, marker_motions

    def track_marker(self, marker_center):
        self.m.init(marker_center)
        self.m.run()

        """
        output: (Ox, Oy, Cx, Cy, Occupied) = flow
            Ox, Oy: N*M matrix, the x and y coordinate of each marker at frame 0
            Cx, Cy: N*M matrix, the x and y coordinate of each marker at current frame
            Occupied: N*M matrix, the index of the marker at each position, -1 means inferred. 
                e.g. Occupied[i][j] = k, meaning the marker mc[k] lies in row i, column j.
        """
        flow = self.m.get_flow()

        '''
        Turn flow into np.ndarray into self.initial_markers, self.cur_markers and self.marker_motion
        '''
        
        Ox, Oy, Cx, Cy, _ = flow
        M, N = len(Ox), len(Ox[0])

        if self.initial_markers_3d is None:
            self.initial_markers_3d = self.MCTacHandler.ComputesurroundingArea(Ox, Oy)
                    
        initial_marker = np.zeros((M * N, 3))
        # marker_motion = np.zeros((M * N, self.dimension))
        marker_motion = np.zeros((M * N, 2))
        if self.dimension == 3:
            current_marker_3d = self.MCTacHandler.ComputesurroundingArea(Cx, Cy)

        k = 0
        for i in range(M):
            for j in range(N):
                if self.dimension == 2:
                    initial_marker[k] = [Ox[i][j], Oy[i][j], 0]
                elif self.dimension == 3:
                    initial_marker[k] = [Ox[i][j], Oy[i][j], max((current_marker_3d[i][j] - self.initial_markers_3d[i][j]) * self.vertical_scale, 0)] # type: ignore
                    
                # rotate initial markers to coordinate with the camera 
                marker_xy = initial_marker[:,:2]
                rotated_xy = self.MCTacHandler.rotate_points(marker_xy, -1 * self.image_rotation)
                rotated_marker = np.copy(initial_marker)
                rotated_marker[:, :2] = rotated_xy 
                marker_motion[k] = [Cx[i][j] - Ox[i][j], Cy[i][j] - Oy[i][j]]
                k += 1

        return rotated_marker, marker_motion

    @staticmethod
    def display_motion(img_show, initial_markers, marker_motions, showScale=1, dimension=2):
        if dimension == 2:
            '''
            display the motion vectors as arrows
            return the frame with arrows
            Motion in 2d
            '''
            # Just rounding markerCenters location
            markerCenter = np.around(initial_markers[:, 0:2]).astype(np.int16)
            for i in range(initial_markers.shape[0]):
                if marker_motions[i, 0] != 0 or marker_motions[i, 1] != 0:
                    end_point = (
                        int(initial_markers[i, 0] + marker_motions[i, 0] * showScale),
                        int(initial_markers[i, 1] + marker_motions[i, 1] * showScale)
                    )
                    end_point = (
                        np.clip(end_point[0], 0, img_show.shape[1] - 1),
                        np.clip(end_point[1], 0, img_show.shape[0] - 1)
                    )
                    cv2.arrowedLine(img_show, (markerCenter[i, 0], markerCenter[i, 1]), end_point, (255, 0, 0), 2)
            return img_show
        elif dimension == 3:
            """
            Visualizes the initial and moved marker positions with spheres.
            Returns:
                geometries: The visualizer object with added components.
            """           
            geometries = []
            arrow_radius = 1
            arrow_length_ratio = 0.8
            
            # # This code is only used for  clearly visualize vertical offset
            # for point, vector in zip(initial_markers, marker_motions):
            #     point_init = point.copy()
            #     point_init[2] = 0
            #     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1, resolution = 3)
            #     sphere.translate(point_init)
            #     sphere.paint_uniform_color([0, 1, 0])
            #     geometries.append(sphere)
            #     vertical_vector = np.zeros(3)
            #     vertical_vector[2] = point[2]
                                
            #     length = np.linalg.norm(vertical_vector)
                    
            #     if length > 0:
            #         arrow = o3d.geometry.TriangleMesh.create_arrow(
            #             cylinder_radius=arrow_radius, cone_radius=arrow_radius * 2,
            #             cylinder_height=length * arrow_length_ratio, cone_height=length * (1 - arrow_length_ratio)
            #         )
            #         arrow_direction = vertical_vector / length
            #         arrow_start = point + arrow_direction * 1
            #         default_arrow_direction = np.array([0, 0, 1])
            #         rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz(np.cross(default_arrow_direction, arrow_direction))
                    
            #         arrow.rotate(rotation_matrix)
            #         arrow.translate(arrow_start)
            #         arrow.paint_uniform_color([1, 0, 0])
            
            #         geometries.append(arrow)
            
            for point, vector in zip(initial_markers, marker_motions):
                # new position of spheres
                new_point = point.copy()  # point 3d, the third dimension is the vertical offset

                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1, resolution=3)
                sphere.translate(new_point)
                sphere.paint_uniform_color([0, 1, 0])
                geometries.append(sphere)

                # the first two dimensions of marker motions represent x,y offset)
                offset = vector
                arrow_length = np.linalg.norm(offset)
                
                if arrow_length > 0:
                    # the default arrow direction is z
                    arrow = o3d.geometry.TriangleMesh.create_arrow(
                        cylinder_radius=arrow_radius, 
                        cone_radius=arrow_radius * 2,
                        cylinder_height=arrow_length * arrow_length_ratio, 
                        cone_height=arrow_length * (1 - arrow_length_ratio)
                    )

                    arrow_start = new_point
                    arrow_direction = np.zeros(3,)
                    arrow_direction[:2] = offset / arrow_length
                    
                    default_arrow_direction = [0, 0, 1]
                    rotation_axis = np.cross(default_arrow_direction, arrow_direction)
                    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis) 
                    cos_theta = np.dot(default_arrow_direction, arrow_direction)
                    angle = np.arccos(cos_theta)
                    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)

                    arrow.rotate(rotation_matrix, center=[0, 0, 0])
                    arrow.translate(arrow_start)
                    arrow.paint_uniform_color([1, 0, 0])

                    geometries.append(arrow)
                                
            return geometries

    def marker_track_visualization(self):
        '''
        Visualize the image with arrows
        get rgb frame from usb or from recorded videos based on param 'recorded'
        the marker motion is either 2d or 3d based on motion dimension
        '''
        if self.dimension == 2:
            while self.cap.isOpened(): # type: ignore
                color_frame, _ = self.get_rgb_frame()
                marker_image = self.get_marker_image(color_frame)
                if marker_image is not None:
                    cv2.imshow("Marker Image", marker_image) # type: ignore
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            self.cap.release() # type: ignore
            cv2.destroyAllWindows()
        elif self.dimension == 3:
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            while self.cap.isOpened(): # type: ignore
                color_frame, _ = self.get_rgb_frame()
                marker_image_3d = self.get_marker_image(color_frame)

                if marker_image_3d is not None:
                    vis.clear_geometries()
                    for geom in marker_image_3d:
                        vis.add_geometry(geom)
                        
                    # change the viewpoint
                    view_ctl = vis.get_view_control()
                    view_ctl.set_lookat([300, 200, 0])  # Set look-at point to the center
                    view_ctl.set_up([0, 0, 1])     # Set up vector
                    view_ctl.set_front([1, 1, 1])   # Set front vector to get a side view
                    
                    vis.poll_events()
                    vis.update_renderer()

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            vis.run()
            vis.destroy_window()
            self.cap.release() # type: ignore

    '''
    publish the offset and locations of markers
    '''

    def publish_marker_offset(self, marker_loc, marker_offset, camera_timestamp: Time):
        '''
        Combine marker locations and marker offsets in a single array
        marker_information:(Area_num,4) or (Area_num,6)
        normalize cur_marker and marker offset based on the size of the images
        '''
        cur_marker = copy.deepcopy(marker_loc)
        cur_marker = cur_marker[:, :2]
        marker_information = np.hstack((cur_marker, marker_offset)).astype(np.float32)

        # Fill the message
        msg = PointCloud2()
        msg.header.stamp = camera_timestamp.to_msg()
        msg.header.frame_id = f'camera_marker_offset_{self.camera_name}'

        msg.is_bigendian = False
        msg.point_step = 16  # 4 fields * 4 bytes/field
        msg.is_dense = True

        # Define the field
        msg.fields = [
            PointField(name='marker_location_x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='marker_location_y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='marker_offset_x', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='marker_offset_y', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        # Fill the data
        pointcloud_data = b''.join(
            map(lambda row: struct.pack('ffff', row[0], row[1], row[2], row[3]), marker_information)
        )
        msg.data = pointcloud_data # type: ignore

        self.marker_publisher.publish(msg)

    def publish_color_image(self, color_image, camera_timestamp: Time):
        '''
        publish the color image and track markers
        '''
        success, encoded_image = cv2.imencode('.jpg', color_image)

        # Fill the message
        msg = Image()
        msg.header.stamp = camera_timestamp.to_msg()
        msg.header.frame_id = f"camera_color_frame_{self.camera_index}"
        msg.height, msg.width, _ = color_image.shape
        msg.encoding = "bgr8"
        msg.step = msg.width * 3
        if success:
            image_bytes = encoded_image.tobytes()
            msg.data = image_bytes
        else:
            logger.warning('fail to image encoding!')
            msg.data = color_image.tobytes()
        self.color_publisher_.publish(msg)

    def send_tactile_sensor_msg(self, initial_markers: np.ndarray, marker_offsets: np.ndarray):
        '''
        Send tactile sensor message to VR server
        '''
        # Send HTTP request to TeleopServer
        url = f"http://{self.teleop_server_ip}:{self.teleop_server_port}/get_current_gripper_state"
        response = requests.get(url)
        # check if the response is successful
        try:
            gripper_state = response.json()
            # Check if the gripper state has changed to stable closed or stable open
            if "left" in self.camera_name:
                if (not self.prev_gripper_state["left_gripper_stable_open"] and gripper_state["left_gripper_stable_open"]):
                    self.cur_base_marker_motion = None
                elif (not self.prev_gripper_state["left_gripper_stable_closed"] and gripper_state["left_gripper_stable_closed"]):
                    self.latency_counter = self.vis_latency_steps
            elif "right" in self.camera_name:
                if (not self.prev_gripper_state["right_gripper_stable_open"] and gripper_state["right_gripper_stable_open"]):
                    self.cur_base_marker_motion = None
                elif (not self.prev_gripper_state["right_gripper_stable_closed"] and gripper_state["right_gripper_stable_closed"]):
                    self.latency_counter = self.vis_latency_steps

            # latency matching
            if self.latency_counter > 0:
                self.latency_counter -= 1
                if self.latency_counter == 0:
                    logger.warning(f"Resetting marker motion for {self.camera_name}")
                    self.cur_base_marker_motion = copy.deepcopy(marker_offsets)

            self.prev_gripper_state = gripper_state

        except Exception as e:
            logger.warning(f"Error occurred while getting gripper state: {e}")
            self.cur_base_marker_motion = None

        # Calculate new marker offsets based on the recorded marker motion
        # TODO: fix bug here
        if self.cur_base_marker_motion is not None:
            marker_offsets = marker_offsets - self.cur_base_marker_motion


        arrow = []
        initial_markers[:, :2] = initial_markers[:, :2] - 0.5  # [-0.5, 0.5]
        initial_markers *= 1 / 5
        initial_markers[:, 2] = initial_markers[:, 2] * 0.1
        marker_offsets *= 2
        z_offset = 0.1
        if self.dimension == 2:
            marker_offsets = np.concatenate((marker_offsets, np.zeros((marker_offsets.shape[0], 1))), axis=1)
        # rotate the marker motion
        rotation_matrix = np.array([[np.cos(self.marker_vis_rotation_angle), -np.sin(self.marker_vis_rotation_angle)],
                                        [np.sin(self.marker_vis_rotation_angle), np.cos(self.marker_vis_rotation_angle)]])
        initial_markers[:, :2] = np.dot(initial_markers[:, :2], rotation_matrix.T)
        marker_offsets[:, :2] = np.dot(marker_offsets[:, :2], rotation_matrix.T)
        for initial_marker, marker_offset in zip(initial_markers, marker_offsets):
            start = [initial_marker[0], initial_marker[1], z_offset + initial_marker[2]]
            end = [initial_marker[0] + marker_offset[0], initial_marker[1] + marker_offset[1],
                   z_offset + initial_marker[2]]
            arrow.append(Arrow(start=start, end=end))

        tactile_sensor_msg_dict = TactileSensorMessage(device_id=self.camera_name, arrows=arrow).model_dump()
        if self.debug:
            logger.debug(f"Sending tactile sensor message: {tactile_sensor_msg_dict}")
            with open(f'{self.camera_name}.json', 'w') as json_file:
                json.dump(tactile_sensor_msg_dict, json_file)
        packed_data = bson.dumps(tactile_sensor_msg_dict)
        self.socket.sendto(packed_data, (self.vr_server_ip, self.vr_server_port))

    def send_streaming_msg(self, color_image):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.streaming_quality]
        ret, color_image_encoded = cv2.imencode('.jpg', color_image, encode_param)
        color_image_bytes = color_image_encoded.tobytes()
        packed_data_dict = {"images": [{"id": self.id,
                                        "inHeadSpace": False,
                                        **display_params,
                                        **{"image": color_image_bytes}}
                                        for display_params in self.streaming_display_params_list]}
        packed_data = bson.dumps(packed_data_dict)

        arrow_address = (self.streaming_server_ip, self.streaming_server_port)
        chunk_size = self.streaming_chunk_size

        self.socket.sendto(len(packed_data).to_bytes(length=4, byteorder='little', signed=False), arrow_address)
        if self.debug:
            logger.debug(f"Sending streaming image to VR server with size {len(packed_data)}")

        self.socket.sendto(chunk_size.to_bytes(length=4, byteorder='little', signed=False), arrow_address)
        count = math.ceil(len(packed_data) / chunk_size)
        if self.debug:
            logger.debug(f"Sending streaming image to VR server with {count} chunks of size {chunk_size}")

        for i in range(count):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            if end > len(packed_data):
                end = len(packed_data)
            self.socket.sendto(packed_data[start:end], arrow_address)
        if self.debug:
            logger.debug(f"Sent streaming image to VR server")

    def timer_callback(self):
        '''
        Publish the color frames
        '''
        while True:
            # get color frames
            # this part takes about 13ms
            color_frame, initial_time = self.get_rgb_frame()

            # get marker and marker offsets
            # this part takes about 11ms
            initial_markers, marker_motion = self.get_marker_image(color_frame) # type: ignore

            initial_markers_copy = copy.deepcopy(initial_markers)
            marker_motion_copy = copy.deepcopy(marker_motion)
            # normalization
            initial_markers, marker_motion = marker_normalization(initial_markers_copy, marker_motion_copy,
                                                                  self.dimension,
                                                                  width=self.width, height=self.height)

            # If rgb frame or marker information not availble, continue
            if (color_frame is None) or (initial_markers is None) or (marker_motion is None):
                continue

            # get the internal camera timestamp of the color frame
            camera_timestamp = initial_time

            initial_markers_copy = copy.deepcopy(initial_markers)
            marker_motion_copy = copy.deepcopy(marker_motion)

            # send tactile sensor message to VR server
            self.send_tactile_sensor_msg(initial_markers_copy, marker_motion_copy)

            # publish the marker offset
            self.publish_marker_offset(initial_markers, marker_motion, camera_timestamp)

            # publish the color image
            # this part takes about 13ms
            self.publish_color_image(color_frame, camera_timestamp)

            # send streaming image
            if self.enable_streaming:
                color_image = color_frame.copy()
                self.send_streaming_msg(color_image)

            # calculate fps
            self.frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - self.prev_time
            if elapsed_time >= 1.0:
                frame_rate = self.frame_count / elapsed_time
                self.fps_list.append(frame_rate)
                logger.debug(f"Frame rate: {frame_rate:.2f} FPS")
                self.prev_time = current_time
                self.frame_count = 0

            # calculate the interval between two frames
            if self.last_frame_time is not None:
                frame_interval = (current_time - self.last_frame_time) * 1000
                self.frame_intervals.append(frame_interval)
            self.last_frame_time = current_time

            # Print info and make plot every 5 seconds
            if current_time - self.last_print_time >= 5:
                logger.info(f"Publishing image from {self.camera_name} at timestamp (s): {initial_time.nanoseconds / 1e9}")
                self.last_print_time = current_time

            break


def main(args=None):
    rclpy.init(args=args)
    node = MCTacCameraPublisher(camera_index=20, camera_name='left_gripper_camera_1',
                              debug=True,
                              recorded=False,
                              image_rotation=0,
                              dimension=2,
                              camera_type="MCTac",
                              video_path="../../../data/tactile_video_mctac_v1/video_002.mp4")
    if node.debug:
        node.marker_track_visualization()
    else:
        try:
            rclpy.spin(node)
        except IndentationError as e:
            logger.exception(e)
        finally:
            node.stop()
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()