import numpy as np
import rclpy
import bson
import socket
from rclpy.node import Node
from rclpy.time import Time
import open3d as o3d
from sensor_msgs.msg import Image, PointCloud2, PointField
import json
from loguru import logger
import time
import cv2
import time as tm
import copy
import struct

from reactive_diffusion_policy.common.time_utils import convert_float_to_ros_time
from reactive_diffusion_policy.common.data_models import TactileSensorMessage, Arrow

import os


class UsbCameraPublisher(Node):
    '''
    Usb Camera publisher Class
    '''

    def __init__(self,
                 camera_index: int = 0,
                 camera_type: str = 'USB',
                 fps: int = 30,
                 exposure: int = -6,
                 contrast: int = 100,
                 camera_name: str = 'left_gripper_camera_1',
                 vr_server_ip: str = '127.0.0.1',
                 vr_server_port: int = 10002,
                 dimension=3,
                 marker_reset_interval: float = 60.0,
                 debug=False,
                 image_folder='data/realworld',
                 recorded=False
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
        self.width = 352
        self.height = 288
        self.debug = debug
        self.dimension = dimension
        self.marker_reset_interval = marker_reset_interval
        self.recorded = recorded

        self.color_publisher_ = self.create_publisher(Image, f'/{camera_name}/color/image_raw', 10)
        self.marker_publisher = self.create_publisher(PointCloud2, f'/{camera_name}/marker_offset/information', 10)
        self.timer = self.create_timer(1 / fps, self.timer_callback)
        self.timestamp_offset = None

        self.last_print_time = tm.time()  # Add a variable to keep track of the last print time
        self.fps_list = []
        self.frame_intervals = []
        self.last_frame_time = None
        self.last_reset_time = tm.time()

        # track the markers
        self.initial_markers = None
        self.cur_markers = None
        self.prev_markers = None

        self.prev_time = time.time()
        self.frame_count = 0

        # start the camera
        self.start()

        # Create a socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        if self.debug:
            # Used for test on recorded videos
            self.image_folder = image_folder
            if os.path.exists(image_folder):
                self.image_files = sorted(
                    [os.path.join(self.image_folder, f) for f in os.listdir(self.image_folder) if f.endswith('.jpg')])
            else:
                self.image_files = []
                logger.warning(f"Image folder {self.image_folder} does not exist!")
            self.total_images = len(self.image_files)
            self.current_image_index = 0

    def start(self):
        '''
        Start the usb camera
        Usb camera has no internal time,
        so we use the time we get the frame as the initial time of the topic
        '''
        if self.recorded == False:
            if self.cap is None:
                self.cap = cv2.VideoCapture(self.camera_index)
                self.set_camera_intrisics(self.cap, self.width, self.height, self.contrast, self.exposure)

                if not self.cap.isOpened():
                    self.cap.open(self.camera_index)
                    self.set_camera_intrisics(self.cap, self.width, self.height, self.contrast, self.exposure)
                    if not self.cap.isOpened():
                        logger.error("Could not open video device")
                        raise Exception("Could not open video device")

                logger.info(f"{self.camera_name} started")
            else:
                logger.warning("Camera is already running")

    def set_camera_intrisics(self, camera, width, height, contrast, exposure):
        '''
        set the resolution, contarst and resolution of the camera
        '''
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        camera.set(cv2.CAP_PROP_CONTRAST, contrast)  # contrast
        camera.set(cv2.CAP_PROP_EXPOSURE, exposure)  # exposure

        actual_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        logger.debug(f"Requested resolution: ({width}, {height}), Actual resolution: ({actual_width}, {actual_height})")

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info(f"Camera {self.camera_index} stopped")
        else:
            logger.warning("Camera is not running")

    def get_rgb_frame(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            timestamp = self.get_clock().now()
            if not ret:
                logger.error(f"Failed to capture image from camera {self.camera_index}")
                raise Exception("Failed to capture image")
            else:
                self.img = frame
            return frame, timestamp
        else:
            logger.error("Camera is not running")
            raise Exception("Camera is not running")

    '''
    get rgb_frame from recorded videos
    '''

    def get_rgb_frame_record(self):
        if self.current_image_index >= self.total_images:
            return None, None

        image_path = self.image_files[self.current_image_index]
        color_frame = cv2.imread(image_path)
        initial_time = time.time()

        self.current_image_index += 1

        return color_frame, initial_time

    '''
    get marker images and motion vectors of the frame
    '''

    def get_marker_image(self, img):
        # track markers
        if self.initial_markers is None:
            self.initial_markers = self.extractMarker(img)
            # TODO: check which deepcopy can be replaced by copy
            self.prev_markers = copy.deepcopy(self.initial_markers)
            self.cur_markers = copy.deepcopy(self.initial_markers)
            self.cur_markers, marker_motion = self.trackMarker(self.cur_markers, self.prev_markers,
                                                               self.initial_markers, self.dimension)
        else:
            self.cur_markers = self.extractMarker(img)
            self.cur_markers, marker_motion = self.trackMarker(self.cur_markers, self.prev_markers,
                                                               self.initial_markers, self.dimension)

        self.prev_markers = self.cur_markers

        if self.debug:
            img_show = img.copy()
            showScale = 5
            self.marker_img = self.displayMotion(img_show, self.initial_markers, marker_motion, showScale,
                                                 self.dimension)
            return self.marker_img
        else:
            initial_markers = self.initial_markers
            return initial_markers, marker_motion

    def extractMarker(self, img):
        '''
        get the initial marker of the frame
        '''
        markerThresh = -17  # -12 gelsight mini#-17
        areaThresh1 = 0  # 500 gelsight mini #40
        areaThresh2 = 400  # 2000 gelsight mini #400
        '''
        # Use CLAHE to enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        img_clahe = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        '''
        img_gaussian = np.int16(cv2.GaussianBlur(img, (15, 15), 0))
        I = img.astype(np.double) - img_gaussian.astype(np.double)

        # dynamically adjust the brightness threshold
        mean_intensity = np.mean(img)
        marker_thresh = markerThresh - (mean_intensity / 255.0) * 10

        # this is the mask of the markers
        markerMask = ((np.max(I, 2)) < markerThresh).astype(np.uint8)
        # lessen the noise to smoothen the marker regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        markerMask = cv2.morphologyEx(markerMask, cv2.MORPH_CLOSE, kernel)

        MarkerCenter = np.empty([0, 3])

        # TODO: support different OpenCV version (>= 4.0)
        cnts, _ = cv2.findContours(markerMask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in cnts:
            AreaCount = cv2.contourArea(contour)

            if AreaCount > areaThresh1 and AreaCount < areaThresh2:
                t = cv2.moments(contour)
                if t['m00'] != 0:
                    MarkerCenter = np.append(MarkerCenter, [[t['m10'] / t['m00'], t['m01'] / t['m00'], AreaCount]],
                                             axis=0)
                else:
                    continue
        # 0:x 1:y


        # constriant all the contours in given regions
        # update the marker center accordingly
        center_coordinates = np.array([self.width / 2, self.height / 2])
        w_length = self.width / 2
        h_length = self.height / 2

        offset = np.abs(MarkerCenter[:, 0:2] - center_coordinates)
        valid_marker_mask = np.logical_and(offset[:, 0] < w_length, offset[:, 1] < h_length)
        MarkerCenter = MarkerCenter[valid_marker_mask]

        return MarkerCenter

    '''
    obtain the motion of the markers
    '''

    def trackMarker(self, marker_present, marker_prev, marker_init, motion_dimension):
        markerCount = len(marker_init)
        Nt = len(marker_present)
        vertical_scale = 0.01

        # trace of markers
        marker_motion = np.zeros((markerCount, motion_dimension))
        # indexes of markers in current frame closet to prevoius frames
        no_seq2 = np.zeros(Nt)
        # locations of markers in current frame
        center_now = np.zeros([markerCount, 3])

        # find the locations of the nearest markers
        for i in range(Nt):
            dif = np.abs(marker_present[i, 0] - marker_prev[:, 0]) + np.abs(marker_present[i, 1] - marker_prev[:, 1])
            # ignore the effect of area change for we split vertical force and sheer force in 3d marker track
            # no_seq2[i]=np.argmin(dif*(50+np.abs(marker_present[i,2]-marker_init[:,2])))
            no_seq2[i] = np.argmin(dif)

        # calculate the difference between current and previous markers
        for i in range(markerCount):
            dif = np.abs(marker_present[:, 0] - marker_prev[i, 0]) + np.abs(marker_present[:, 1] - marker_prev[i, 1])
            # t=dif*(50+np.abs(marker_present[:,2]-marker_init[i,2]))
            t = dif

            if t.size == 0:
                continue

            a = np.amin(t) / 100
            b = np.argmin(t)

            if marker_init[i, 2] < a and a > 1:  # for small area
                center_now[i] = marker_prev[i]
                marker_motion[i] = np.zeros(motion_dimension)
            elif i == no_seq2[b]:
                if motion_dimension == 2:
                    marker_motion[i] = marker_present[b, 0:2] - marker_init[i, 0:2]
                elif motion_dimension == 3:
                    # x,y
                    marker_motion[i, 0:2] = marker_present[b, 0:2] - marker_init[i, 0:2]
                    # z, based on areacount
                    marker_motion[i, 2] = (marker_init[i, 2] - marker_present[b, 2]) * vertical_scale
                center_now[i] = marker_present[b]
            else:
                center_now[i] = marker_prev[i]
                marker_motion[i] = np.zeros(motion_dimension)

        return center_now, marker_motion

    def displayMotion(self, img, marker_init, marker_motion, showScale, motion_dimension):
        if motion_dimension == 2:
            '''
            display the motion vectors as arrows
            return the frame with arrows
            Motion in 2d
            '''
            # Just rounding markerCenters location
            markerCenter = np.around(marker_init[:, 0:2]).astype(np.int16)
            for i in range(marker_init.shape[0]):
                if marker_motion[i, 0] != 0 or marker_motion[i, 1] != 0:
                    end_point = (
                        int(marker_init[i, 0] + marker_motion[i, 0] * showScale),
                        int(marker_init[i, 1] + marker_motion[i, 1] * showScale)
                    )
                    end_point = (
                        np.clip(end_point[0], 0, img.shape[1] - 1),
                        np.clip(end_point[1], 0, img.shape[0] - 1)
                    )
                    cv2.arrowedLine(img, (markerCenter[i, 0], markerCenter[i, 1]), end_point, (0, 255, 255), 2)
            return img
        elif motion_dimension == 3:
            """
            Visualizes the initial and moved marker positions with spheres.
            Returns:
                geometries: The visualizer object with added components.
            """
            geometries = []
            arrow_radius = 1
            arrow_length_ratio = 0.8

            # Copy initial marker positions
            initial_positions = copy.deepcopy(marker_init)
            initial_positions[:, 2] = 0

            for point, vector in zip(initial_positions, marker_motion):
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1, resolution=3)
                sphere.translate(point)
                sphere.paint_uniform_color([0, 1, 0])
                geometries.append(sphere)

                length = np.linalg.norm(vector)

                if length > 0:
                    arrow = o3d.geometry.TriangleMesh.create_arrow(
                        cylinder_radius=arrow_radius, cone_radius=arrow_radius * 2,
                        cylinder_height=length * arrow_length_ratio, cone_height=length * (1 - arrow_length_ratio)
                    )
                    arrow_direction = vector / length
                    arrow_start = point + arrow_direction * 1
                    default_arrow_direction = np.array([0, 0, 1])
                    rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz(
                        np.cross(default_arrow_direction, arrow_direction))

                    arrow.rotate(rotation_matrix)
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
        # TODO: check those arrows unchanged, try to solve this problem
        recorded = self.recorded
        if recorded:
            if self.dimension == 2:
                while True:
                    color_frame, _ = self.get_rgb_frame_record()
                    if color_frame is None:
                        logger.info("All images have been played. Exiting.")
                        break

                    marker_image = self.get_marker_image(color_frame)

                    if marker_image is not None:
                        cv2.imshow("Marker Image", marker_image)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                cv2.destroyAllWindows()
            elif self.dimension == 3:
                vis = o3d.visualization.Visualizer()
                vis.create_window()
                while True:

                    color_frame, _ = self.get_rgb_frame_record()
                    if color_frame is None:
                        logger.info("All images have been played. Exiting.")
                        break

                    marker_image_3d = self.get_marker_image(color_frame)
                    if marker_image_3d is not None:
                        vis.clear_geometries()
                        for geom in marker_image_3d:
                            vis.add_geometry(geom)

                        # change the viewpoint
                        view_ctl = vis.get_view_control()
                        view_ctl.set_lookat([100, 100, 0])  # Set look-at point to the center
                        view_ctl.set_up([0, 0, 1])  # Set up vector
                        view_ctl.set_front([1, 1, 1])  # Set front vector to get a side view

                        vis.poll_events()
                        vis.update_renderer()

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                vis.run()
                vis.destroy_window()
        else:
            if self.dimension == 2:
                while self.cap.isOpened():
                    color_frame, _ = self.get_rgb_frame()
                    marker_image = self.get_marker_image(color_frame)
                    if marker_image is not None:
                        cv2.imshow("Marker Image", marker_image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                self.cap.release()
                cv2.destroyAllWindows()
            elif self.dimension == 3:
                vis = o3d.visualization.Visualizer()
                vis.create_window()
                while self.cap.isOpened():
                    color_frame, _ = self.get_rgb_frame()
                    marker_image_3d = self.get_marker_image(color_frame)

                    if marker_image_3d is not None:
                        vis.clear_geometries()
                        for geom in marker_image_3d:
                            vis.add_geometry(geom)

                        # change the viewpoint
                        view_ctl = vis.get_view_control()
                        view_ctl.set_lookat([100, 100, 0])  # Set look-at point to the center
                        view_ctl.set_up([0, 0, 1])  # Set up vector
                        view_ctl.set_front([1, 1, 1])  # Set front vector to get a side view

                        vis.poll_events()
                        vis.update_renderer()

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                vis.run()
                vis.destroy_window()
                self.cap.release()

    def marker_normalization(self, marker_loc, marker_offset, dimension):
        # normalization
        marker_loc[:, 0] /= self.width
        marker_loc[:, 1] /= self.height
        marker_offset[:, 0] /= self.width
        marker_offset[:, 1] /= self.height
        if dimension == 3:
            marker_offset[:, 2] /= (self.width * self.height)

        return marker_loc, marker_offset

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
        if self.dimension == 2:
            cur_marker = cur_marker[:, :2]
        elif self.dimension == 3:
            cur_marker[:, 2] = 0

        marker_information = np.hstack((cur_marker, marker_offset)).astype(np.float32)

        # Fill the message
        msg = PointCloud2()
        msg.header.stamp = camera_timestamp.to_msg()
        msg.header.frame_id = f'camera_marker_offset_{self.camera_name}'

        msg.is_bigendian = False
        if self.dimension == 2:
            msg.point_step = 16  # 4 fields * 4 bytes/field
        elif self.dimension == 3:
            msg.point_step = 24  # 6 fields * 4 bytes/field
        else:
            logger.warning("Invalid marker motion dimension!")
        msg.is_dense = True

        # Define the field
        if self.dimension == 2:
            msg.fields = [
                PointField(name='marker_location_x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='marker_location_y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='marker_offset_x', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='marker_offset_y', offset=12, datatype=PointField.FLOAT32, count=1),
            ]
        elif self.dimension == 3:
            msg.fields = [
                PointField(name='marker_location_x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='marker_location_y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='marker_location_z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='marker_offset_x', offset=12, datatype=PointField.FLOAT32, count=1),
                PointField(name='marker_offset_y', offset=16, datatype=PointField.FLOAT32, count=1),
                PointField(name='marker_offset_z', offset=20, datatype=PointField.FLOAT32, count=1),
            ]

        # Fill the data
        if self.dimension == 2:
            pointcloud_data = b''.join(
                map(lambda row: struct.pack('ffff', row[0], row[1], row[2], row[3]), marker_information)
            )
        elif self.dimension == 3:
            pointcloud_data = b''.join(
                map(lambda row: struct.pack('ffffff', row[0], row[1], row[2], row[3], row[4], row[5]),
                    marker_information)
            )
        msg.data = pointcloud_data

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
        arrow = []
        initial_markers[:, :2] = initial_markers[:, :2] - 0.5  # [0.5, 1.5]
        initial_markers *= 1 / 4
        marker_offsets *= 4
        z_offset = 0.1
        if self.dimension == 2:
            marker_offsets = np.concatenate((marker_offsets, np.zeros((marker_offsets.shape[0], 1))), axis=1)
        marker_offsets[:, 2] = marker_offsets[:, 2] * 1000
        for initial_marker, marker_offset in zip(initial_markers, marker_offsets):
            start = [initial_marker[0], initial_marker[1], z_offset]
            end = [initial_marker[0] + marker_offset[0], initial_marker[1] + marker_offset[1],
                   z_offset + marker_offset[2]]
            arrow.append(Arrow(start=start, end=end))

        tactile_sensor_msg_dict = TactileSensorMessage(device_id=self.camera_name, arrows=arrow).model_dump()
        if self.debug:
            logger.debug(f"Sending tactile sensor message: {tactile_sensor_msg_dict}")
            with open(f'{self.camera_name}.json', 'w') as json_file:
                json.dump(tactile_sensor_msg_dict, json_file)
        packed_data = bson.dumps(tactile_sensor_msg_dict)
        self.socket.sendto(packed_data, (self.vr_server_ip, self.vr_server_port))

    def timer_callback(self):
        '''
        Publish the color frames
        '''
        while True:
            if tm.time() - self.last_reset_time >= self.marker_reset_interval:
                # reset the markers after a certain interval
                self.initial_markers = None
                self.last_reset_time = tm.time()

            # get color frames
            color_frame, initial_time = self.get_rgb_frame()

            # get marker and marker offsets
            initial_markers, marker_motion = self.get_marker_image(color_frame)

            initial_markers_copy = copy.deepcopy(initial_markers)
            marker_motion_copy = copy.deepcopy(marker_motion)
            # normalization
            initial_markers, marker_motion = self.marker_normalization(initial_markers_copy, marker_motion_copy,
                                                                       self.dimension)

            # If rgb frame or marker information not availble, continue
            if (color_frame is None) or (initial_markers is None) or (marker_motion is None):
                continue

            # get the internal camera timestamp of the color frame
            camera_timestamp = initial_time

            initial_markers_copy = copy.deepcopy(initial_markers)
            marker_motion_copy = copy.deepcopy(marker_motion)

            # send tactile sensor message to VR server
            self.send_tactile_sensor_msg(initial_markers_copy, marker_motion_copy)

            # publish
            self.publish_marker_offset(initial_markers, marker_motion, camera_timestamp)

            # publish the color image
            self.publish_color_image(color_frame, camera_timestamp)

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
    node = UsbCameraPublisher(camera_index=14, camera_name='left_gripper_camera_1',
                              debug=False,
                              recorded=False,
                              image_folder='data/tactile_video/seq01')
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

