import rclpy
import os
import os.path as osp
import copy
import numpy as np
import tqdm
import zarr
import py_cli_interaction
from loguru import logger
from reactive_diffusion_policy.real_world.publisher.gelsight_camera_publisher import GelsightCameraPublisher

VIDEO_DIR = '../../data/tactile_video_gelsight'
SAVE_DATA_DIR = '../../data/marker_motion_gelsight'
TAG = 'gelsight_v1'

def main():
    rclpy.init()
    assert osp.exists(VIDEO_DIR), f'Video directory {VIDEO_DIR} does not exist.'
    # iterate over all the files in the directory
    video_path_list = []
    for f in os.listdir(VIDEO_DIR):
        if f.endswith('.mp4'):
            video_path_list.append(osp.join(VIDEO_DIR, f))
    video_path_list = sorted(video_path_list)
    initial_marker_list = []
    marker_motion_list = []
    color_img_list = []
    for video_path in tqdm.tqdm(video_path_list):
        logger.debug(f'Processing video {video_path}')
        node = GelsightCameraPublisher(debug=False,
                                       recorded=True,
                                       dimension=2,
                                       camera_type="gelsight",
                                       video_path=video_path)
        try:
            while True:
                # get color frames
                color_frame, initial_time = node.get_rgb_frame()

                # get marker and marker offsets
                initial_markers, marker_motion = node.get_marker_image(color_frame)

                initial_markers_copy = copy.deepcopy(initial_markers)
                marker_motion_copy = copy.deepcopy(marker_motion)
                # normalization
                initial_markers, marker_motion = marker_normalization(initial_markers_copy, marker_motion_copy,
                                                                      node.dimension, width=node.width,
                                                                      height=node.height)
                # append to the list
                color_img_list.append(color_frame)
                initial_marker_list.append(initial_markers)
                marker_motion_list.append(marker_motion)
        except Exception as e:
            logger.debug(f'Finished processing video {video_path} due to {e}')

        node.stop()
        node.destroy_node()

    # stack the list
    color_image_array = np.stack(color_img_list, axis=0)
    initial_marker_array = np.stack(initial_marker_list, axis=0)
    marker_motion_array = np.stack(marker_motion_list, axis=0)
    # save arrays with zarr
    save_data_path = osp.join(osp.join(osp.abspath(os.getcwd()), SAVE_DATA_DIR, f'{TAG}.zarr'))
    os.makedirs(SAVE_DATA_DIR, exist_ok=True)
    if os.path.exists(save_data_path):
        logger.info('Data already exists at {}'.format(save_data_path))
        # use py_cli_interaction to ask user if they want to overwrite the data
        if py_cli_interaction.parse_cli_bool('Do you want to overwrite the data?', default_value=True):
            logger.warning('Overwriting {}'.format(save_data_path))
            os.system('rm -rf {}'.format(save_data_path))
    # create zarr group
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    zarr_root = zarr.group(save_data_path)
    color_image_chunk_size = (100, color_image_array.shape[1], color_image_array.shape[2], color_image_array.shape[3])
    initial_marker_chunk_size = (10000, initial_marker_array.shape[1], initial_marker_array.shape[2])
    marker_motion_chunk_size = (10000, marker_motion_array.shape[1], marker_motion_array.shape[2])
    zarr_root.create_dataset('color_image', data=color_image_array,
                             chunks=color_image_chunk_size, dtype='uint8',
                             overwrite=True,
                             compressor=compressor)
    zarr_root.create_dataset('initial_marker', data=initial_marker_array,
                                chunks=initial_marker_chunk_size, dtype='float32',
                                overwrite=True,
                                compressor=compressor)
    zarr_root.create_dataset('marker_motion', data=marker_motion_array,
                                chunks=marker_motion_chunk_size, dtype='float32',
                                overwrite=True,
                                compressor=compressor)
    # print zarr data structure
    logger.info('Zarr data structure')
    logger.info(zarr_root.tree())
    rclpy.shutdown()

if __name__ == '__main__':
    main()