import pickle
import os
from loguru import logger
import zarr
import cv2
import numpy as np
import os.path as osp
import py_cli_interaction
import matplotlib.pyplot as plt
from hydra import initialize, compose
from omegaconf import DictConfig

from reactive_diffusion_policy.real_world.real_world_transforms import RealWorldTransforms
from reactive_diffusion_policy.common.visualization_utils import visualize_rgb_image
from reactive_diffusion_policy.real_world.post_process_utils import DataPostProcessingManager

DEBUG = False
DEBUG_TACTILE_LATENCY = False  # for debugging tactile latency
DRAW_FORCE = False  # for debugging force
USE_ABSOLUTE_ACTION = True

TAG = 'peel_data'
ACTION_DIM = 4  # (4 + 15)
TEMPORAL_DOWNSAMPLE_RATIO = 1  # the ratio for temporal down-sampling
SENSOR_MODE = 'single_arm_two_realsense_two_tactile'
GELSIGHT_PCA_PATH = 'data/PCA_Transform_GelSight'
MCTAC_PCA_PATH = 'data/PCA_Transform_McTAC_v1'
PCA_DIM = 15

if __name__ == '__main__':
    data_dir = f'/home/pc/workspace/zyh/rdp_data/{TAG}'
    save_data_dir = f'/home/pc/workspace/zyh/rdp_data/{TAG}_downsample{TEMPORAL_DOWNSAMPLE_RATIO}{"_debug" if DEBUG else ""}_zarr'
    save_data_path = osp.join(osp.join(osp.abspath(os.getcwd()), save_data_dir, f'replay_buffer.zarr'))
    os.makedirs(save_data_dir, exist_ok=True)
    if os.path.exists(save_data_path):
        logger.info('Data already exists at {}'.format(save_data_path))
        # use py_cli_interaction to ask user if they want to overwrite the data
        if py_cli_interaction.parse_cli_bool('Do you want to overwrite the data?', default_value=True):
            logger.warning('Overwriting {}'.format(save_data_path))
            os.system('rm -rf {}'.format(save_data_path))

    # loading config for transforms
    with initialize(config_path='reactive_diffusion_policy/config', version_base="1.3"):
        # config is relative to a module
        cfg = compose(config_name="real_world_env")
    transforms = RealWorldTransforms(option=cfg.task.transforms)

    # pca params
    gelsight_pca_transform_matrix_path = os.path.join(GELSIGHT_PCA_PATH, 'pca_transform_matrix.npy')
    gelsight_pca_mean_matrix_path = os.path.join(GELSIGHT_PCA_PATH, 'pca_mean_matrix.npy')
    gelsight_pca_param = DictConfig({
        'n_components': PCA_DIM,
        'normalize': False,
        'mode': 'Eval',
        'store': False,
        'transformation_matrix_path': gelsight_pca_transform_matrix_path,
        'mean_matrix_path': gelsight_pca_mean_matrix_path
    })
    mctac_pca_transform_matrix_path = os.path.join(MCTAC_PCA_PATH, 'pca_transform_matrix.npy')
    mctac_pca_mean_matrix_path = os.path.join(MCTAC_PCA_PATH, 'pca_mean_matrix.npy')
    mctac_gelsight_pca_param = DictConfig({
        'n_components': PCA_DIM,
        'normalize': False,
        'mode': 'Eval',
        'store': False,
        'transformation_matrix_path': mctac_pca_transform_matrix_path,
        'mean_matrix_path': mctac_pca_mean_matrix_path
    })
    pca_param_dict = DictConfig({
        'GelSight': gelsight_pca_param,
        'McTac': mctac_gelsight_pca_param
    })
    # create data processing manager
    data_processing_manager = DataPostProcessingManager(transforms=transforms,
                                                        pca_param_dict=pca_param_dict,
                                                        mode=SENSOR_MODE,
                                                        use_6d_rotation=True,
                                                        marker_dimension=2)

    # sensor data arrays
    timestamp_arrays = []
    external_img_arrays = []
    left_wrist_img_arrays = []
    right_wrist_img_arrays = []
    left_gripper1_img_arrays = []
    left_gripper1_initial_marker_arrays = []
    left_gripper1_marker_offset_arrays = []
    left_gripper1_marker_offset_emb_arrays = []
    left_gripper2_img_arrays = []
    left_gripper2_initial_marker_arrays = []
    left_gripper2_marker_offset_arrays = []
    left_gripper2_marker_offset_emb_arrays = []
    right_gripper1_img_arrays = []
    right_gripper1_initial_marker_arrays = []
    right_gripper1_marker_offset_arrays = []
    right_gripper1_marker_offset_emb_arrays = []
    right_gripper2_img_arrays = []
    right_gripper2_initial_marker_arrays = []
    right_gripper2_marker_offset_arrays = []
    right_gripper2_marker_offset_emb_arrays = []
    # robot state arrays
    left_robot_tcp_pose_arrays = []
    left_robot_tcp_vel_arrays = []
    left_robot_tcp_wrench_arrays = []
    left_robot_gripper_width_arrays = []
    left_robot_gripper_force_arrays = []
    right_robot_tcp_pose_arrays = []
    right_robot_tcp_vel_arrays = []
    right_robot_tcp_wrench_arrays = []
    right_robot_gripper_width_arrays = []
    right_robot_gripper_force_arrays = []

    episode_ends_arrays = []
    total_count = 0
    # find all the files in the data directory
    data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pkl')])
    for seq_idx, data_file in enumerate(data_files):
        if DEBUG and seq_idx <= 58:
            continue
        data_path = osp.join(data_dir, data_file)

        abs_path = os.path.abspath(data_path)
        save_data_path = os.path.abspath(save_data_path)
        logger.info(f'Loading data from {abs_path}')

        # Load the data
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        for step_idx, sensor_msg in enumerate(data.sensorMessages):
            if DEBUG and step_idx <= 60:
                continue
            total_count += 1
            logger.info(f'Processing {step_idx}th sensor message in sequence {seq_idx}')

            # TODO: add timestamp
            obs_dict = data_processing_manager.convert_sensor_msg_to_obs_dict(sensor_msg)
            timestamp_arrays.append(sensor_msg.timestamp)

            left_robot_tcp_pose_arrays.append(obs_dict['left_robot_tcp_pose'])
            left_robot_tcp_vel_arrays.append(obs_dict['left_robot_tcp_vel'])
            left_robot_tcp_wrench_arrays.append(obs_dict['left_robot_tcp_wrench'])
            left_robot_gripper_width_arrays.append(obs_dict['left_robot_gripper_width'])
            left_robot_gripper_force_arrays.append(obs_dict['left_robot_gripper_force'])
            right_robot_tcp_pose_arrays.append(obs_dict['right_robot_tcp_pose'])
            right_robot_tcp_vel_arrays.append(obs_dict['right_robot_tcp_vel'])
            right_robot_tcp_wrench_arrays.append(obs_dict['right_robot_tcp_wrench'])
            right_robot_gripper_width_arrays.append(obs_dict['right_robot_gripper_width'])
            right_robot_gripper_force_arrays.append(obs_dict['right_robot_gripper_force'])

            external_img_arrays.append(obs_dict['external_img'])
            left_wrist_img_arrays.append(obs_dict['left_wrist_img'])

            if 'two_tactile' in SENSOR_MODE or 'four_tactile' in SENSOR_MODE:
                left_gripper1_img_arrays.append(obs_dict['left_gripper1_img'])
                left_gripper2_img_arrays.append(obs_dict['left_gripper2_img'])
                left_gripper1_initial_marker_arrays.append(obs_dict['left_gripper1_initial_marker'])
                left_gripper2_initial_marker_arrays.append(obs_dict['left_gripper2_initial_marker'])
                left_gripper1_marker_offset_arrays.append(obs_dict['left_gripper1_marker_offset'])
                left_gripper2_marker_offset_arrays.append(obs_dict['left_gripper2_marker_offset'])
                left_gripper1_marker_offset_emb_arrays.append(obs_dict['left_gripper1_marker_offset_emb'])
                left_gripper2_marker_offset_emb_arrays.append(obs_dict['left_gripper2_marker_offset_emb'])

            if SENSOR_MODE == 'dual_arm_two_realsense_four_tactile':
                right_wrist_img_arrays.append(obs_dict['right_wrist_img'])
                right_gripper1_img_arrays.append(obs_dict['right_gripper1_img'])
                right_gripper2_img_arrays.append(obs_dict['right_gripper2_img'])
                right_gripper1_initial_marker_arrays.append(obs_dict['right_gripper1_initial_marker'])
                right_gripper2_initial_marker_arrays.append(obs_dict['right_gripper2_initial_marker'])
                right_gripper1_marker_offset_arrays.append(obs_dict['right_gripper1_marker_offset'])
                right_gripper2_marker_offset_arrays.append(obs_dict['right_gripper2_marker_offset'])
                right_gripper1_marker_offset_emb_arrays.append(obs_dict['right_gripper1_marker_offset_emb'])
                right_gripper2_marker_offset_emb_arrays.append(obs_dict['right_gripper2_marker_offset_emb'])

            if DEBUG:
                if 'two_tactile' in SENSOR_MODE or 'four_tactile' in SENSOR_MODE:
                    # visualize gripper camera images
                    visualize_rgb_image(sensor_msg.leftGripperCameraRGB1, 'Left Gripper Camera RGB1')
                    visualize_rgb_image(sensor_msg.leftGripperCameraRGB2, 'Left Gripper Camera RGB2')
                    visualize_rgb_image(sensor_msg.rightGripperCameraRGB1, 'Right Gripper Camera RGB1')
                    visualize_rgb_image(sensor_msg.rightGripperCameraRGB2, 'Right Gripper Camera RGB2')

                # visualize external camera image
                visualize_rgb_image(sensor_msg.externalCameraRGB, 'External Camera RGB')
                # visualize left wrist camera image
                visualize_rgb_image(sensor_msg.leftWristCameraRGB, 'Left Wrist Camera RGB')
                # visualize right wrist camera image
                visualize_rgb_image(sensor_msg.rightWristCameraRGB, 'Right Wrist Camera RGB')

                logger.debug(f'left_robot_tcp_pose: {obs_dict["left_robot_tcp_pose"]}, '
                             f'right_robot_tcp_pose: {obs_dict["right_robot_tcp_pose"]}')

            del sensor_msg, obs_dict

        episode_ends_arrays.append(total_count)

    # Convert lists to arrays
    external_img_arrays = np.stack(external_img_arrays, axis=0)
    left_wrist_img_arrays = np.stack(left_wrist_img_arrays, axis=0)
    if 'two_tactile' in SENSOR_MODE or 'four_tactile' in SENSOR_MODE:
        left_gripper1_img_arrays = np.stack(left_gripper1_img_arrays, axis=0)
        left_gripper2_img_arrays = np.stack(left_gripper2_img_arrays, axis=0)
        left_gripper1_initial_marker_arrays = np.stack(left_gripper1_initial_marker_arrays, axis=0)
        left_gripper2_initial_marker_arrays = np.stack(left_gripper2_initial_marker_arrays, axis=0)
        left_gripper1_marker_offset_arrays = np.stack(left_gripper1_marker_offset_arrays, axis=0)
        left_gripper2_marker_offset_arrays = np.stack(left_gripper2_marker_offset_arrays, axis=0)
        left_gripper1_marker_offset_emb_arrays = np.stack(left_gripper1_marker_offset_emb_arrays, axis=0)
        left_gripper2_marker_offset_emb_arrays = np.stack(left_gripper2_marker_offset_emb_arrays, axis=0)

    episode_ends_arrays = np.array(episode_ends_arrays)
    if SENSOR_MODE == 'dual_arm_two_realsense_four_tactile':
        right_gripper1_img_arrays = np.stack(right_gripper1_img_arrays, axis=0)
        right_gripper2_img_arrays = np.stack(right_gripper2_img_arrays, axis=0)
        right_gripper1_initial_marker_arrays = np.stack(right_gripper1_initial_marker_arrays, axis=0)
        right_gripper2_initial_marker_arrays = np.stack(right_gripper2_initial_marker_arrays, axis=0)
        right_gripper1_marker_offset_arrays = np.stack(right_gripper1_marker_offset_arrays, axis=0)
        right_gripper2_marker_offset_arrays = np.stack(right_gripper2_marker_offset_arrays, axis=0)
        right_gripper1_marker_offset_emb_arrays = np.stack(right_gripper1_marker_offset_emb_arrays, axis=0)
        right_gripper2_marker_offset_emb_arrays = np.stack(right_gripper2_marker_offset_emb_arrays, axis=0)
        right_wrist_img_arrays = np.stack(right_wrist_img_arrays, axis=0)

    timestamp_arrays = np.array(timestamp_arrays)
    left_robot_tcp_pose_arrays = np.stack(left_robot_tcp_pose_arrays, axis=0)
    left_robot_tcp_vel_arrays = np.stack(left_robot_tcp_vel_arrays, axis=0)
    left_robot_tcp_wrench_arrays = np.stack(left_robot_tcp_wrench_arrays, axis=0)
    left_robot_gripper_width_arrays = np.stack(left_robot_gripper_width_arrays, axis=0)
    left_robot_gripper_force_arrays = np.stack(left_robot_gripper_force_arrays, axis=0)
    right_robot_tcp_pose_arrays = np.stack(right_robot_tcp_pose_arrays, axis=0)
    right_robot_tcp_vel_arrays = np.stack(right_robot_tcp_vel_arrays, axis=0)
    right_robot_tcp_wrench_arrays = np.stack(right_robot_tcp_wrench_arrays, axis=0)
    right_robot_gripper_width_arrays = np.stack(right_robot_gripper_width_arrays, axis=0)
    right_robot_gripper_force_arrays = np.stack(right_robot_gripper_force_arrays, axis=0)

    if ACTION_DIM == 4: # (left_tcp_x, left_tcp_y, left_tcp_z, left_gripper_width)
        state_arrays = np.concatenate([left_robot_tcp_pose_arrays[:, :3], left_robot_gripper_width_arrays], axis=-1)
    elif ACTION_DIM == 8: # (left_tcp_x, left_tcp_y, left_tcp_z, right_tcp_x, right_tcp_y, right_tcp_z, left_gripper_width, right_gripper_width)
        state_arrays = np.concatenate([left_robot_tcp_pose_arrays[:, :3], right_robot_tcp_pose_arrays[:, :3],
                                       left_robot_gripper_width_arrays, right_robot_gripper_width_arrays], axis=-1)
    elif ACTION_DIM == 10: # (left_tcp_x, left_tcp_y, left_tcp_z, left_6d_rotation, left_gripper_width)
        state_arrays = np.concatenate([left_robot_tcp_pose_arrays, left_robot_gripper_width_arrays], axis=-1)
    elif ACTION_DIM == 20: # (left_tcp_x, left_tcp_y, left_tcp_z, left_6d_rotation, right_tcp_x, right_tcp_y, right_tcp_z, right_6d_rotation, left_gripper_width, right_gripper_width)
        state_arrays = np.concatenate([left_robot_tcp_pose_arrays, right_robot_tcp_pose_arrays,
                                       left_robot_gripper_width_arrays, right_robot_gripper_width_arrays], axis=-1)
    elif ACTION_DIM == 4 + PCA_DIM: # (left_tcp_x, left_tcp_y, left_tcp_z, left_gripper_width, left_pca_embedding)
        state_arrays = np.concatenate([left_robot_tcp_pose_arrays[:, :3], left_robot_gripper_width_arrays,
                                       left_gripper2_marker_offset_emb_arrays], axis=-1)
    else:
        # TODO: support left_gripper1_marker_offset_emb_arrays
        # TODO: support right_gripper1_marker_offset_emb_arrays
        # TODO: support right_gripper2_marker_offset_emb_arrays
        raise NotImplementedError
    if USE_ABSOLUTE_ACTION:
        # override action to absolute value
        # action is basically next state
        new_action_arrays = state_arrays[1:, ...].copy()
        action_arrays = np.concatenate([new_action_arrays, new_action_arrays[-1][np.newaxis, :]], axis=0)
        # fix the last action of each episode
        for i in range(0, len(episode_ends_arrays)):
            action_arrays[episode_ends_arrays[i] - 1] = action_arrays[episode_ends_arrays[i] - 2]
    else:
        raise NotImplementedError

    # If no filtering, just rename
    valid_mask = np.ones(len(action_arrays), dtype=bool)

    if TEMPORAL_DOWNSAMPLE_RATIO > 1:
        # Calculate indices to keep after downsampling
        keep_indices = []
        current_episode_start = 0

        # Process each episode separately
        for episode_end in episode_ends_arrays:
            # Get indices for current episode
            episode_indices = np.arange(current_episode_start, episode_end)

            # Calculate downsampled indices for this episode
            # Keep first and last frame of each episode, downsample middle frames
            if len(episode_indices) > 2:
                middle_indices = episode_indices[1:-1]
                downsampled_middle_indices = middle_indices[::TEMPORAL_DOWNSAMPLE_RATIO]
                episode_keep_indices = np.concatenate([[episode_indices[0]],
                                                       downsampled_middle_indices,
                                                       [episode_indices[-1]]])
            else:
                # If episode is too short, keep all frames
                episode_keep_indices = episode_indices

            keep_indices.extend(episode_keep_indices)
            current_episode_start = episode_end

        keep_indices = np.array(keep_indices)

        # Downsample all arrays
        action_arrays = action_arrays[keep_indices]
        timestamp_arrays = timestamp_arrays[keep_indices]
        external_img_arrays = external_img_arrays[keep_indices]
        left_wrist_img_arrays = left_wrist_img_arrays[keep_indices]

        if 'two_tactile' in SENSOR_MODE or 'four_tactile' in SENSOR_MODE:
            left_gripper1_img_arrays = left_gripper1_img_arrays[keep_indices]
            left_gripper2_img_arrays = left_gripper2_img_arrays[keep_indices]
            left_gripper1_initial_marker_arrays = left_gripper1_initial_marker_arrays[keep_indices]
            left_gripper2_initial_marker_arrays = left_gripper2_initial_marker_arrays[keep_indices]
            left_gripper1_marker_offset_arrays = left_gripper1_marker_offset_arrays[keep_indices]
            left_gripper2_marker_offset_arrays = left_gripper2_marker_offset_arrays[keep_indices]
            left_gripper1_marker_offset_emb_arrays = left_gripper1_marker_offset_emb_arrays[keep_indices]
            left_gripper2_marker_offset_emb_arrays = left_gripper2_marker_offset_emb_arrays[keep_indices]

        if SENSOR_MODE == 'dual_arm_two_realsense_four_tactile':
            right_wrist_img_arrays = right_wrist_img_arrays[keep_indices]
            right_gripper1_img_arrays = right_gripper1_img_arrays[keep_indices]
            right_gripper2_img_arrays = right_gripper2_img_arrays[keep_indices]
            right_gripper1_initial_marker_arrays = right_gripper1_initial_marker_arrays[keep_indices]
            right_gripper2_initial_marker_arrays = right_gripper2_initial_marker_arrays[keep_indices]
            right_gripper1_marker_offset_arrays = right_gripper1_marker_offset_arrays[keep_indices]
            right_gripper2_marker_offset_arrays = right_gripper2_marker_offset_arrays[keep_indices]
            right_gripper1_marker_offset_emb_arrays = right_gripper1_marker_offset_emb_arrays[keep_indices]
            right_gripper2_marker_offset_emb_arrays = right_gripper2_marker_offset_emb_arrays[keep_indices]

        left_robot_tcp_pose_arrays = left_robot_tcp_pose_arrays[keep_indices]
        left_robot_tcp_vel_arrays = left_robot_tcp_vel_arrays[keep_indices]
        left_robot_tcp_wrench_arrays = left_robot_tcp_wrench_arrays[keep_indices]
        left_robot_gripper_width_arrays = left_robot_gripper_width_arrays[keep_indices]
        left_robot_gripper_force_arrays = left_robot_gripper_force_arrays[keep_indices]
        right_robot_tcp_pose_arrays = right_robot_tcp_pose_arrays[keep_indices]
        right_robot_tcp_vel_arrays = right_robot_tcp_vel_arrays[keep_indices]
        right_robot_tcp_wrench_arrays = right_robot_tcp_wrench_arrays[keep_indices]
        right_robot_gripper_width_arrays = right_robot_gripper_width_arrays[keep_indices]
        right_robot_gripper_force_arrays = right_robot_gripper_force_arrays[keep_indices]

        # Recalculate episode_ends
        new_episode_ends = []
        count = 0
        current_episode_start = 0
        for episode_end in episode_ends_arrays:
            episode_indices = np.arange(current_episode_start, episode_end)
            if len(episode_indices) > 2:
                middle_indices = episode_indices[1:-1]
                downsampled_middle_indices = middle_indices[::TEMPORAL_DOWNSAMPLE_RATIO]
                count += len(downsampled_middle_indices) + 2  # +2 for first and last frame
            else:
                count += len(episode_indices)
            new_episode_ends.append(count)
            current_episode_start = episode_end

        episode_ends_arrays = np.array(new_episode_ends)

    if DEBUG_TACTILE_LATENCY:
        # Extract the third column
        # left_robot_tcp_pose_col = -left_robot_tcp_pose_arrays[:125, 2]
        # left_gripper2_marker_offset_emb_col = left_gripper2_marker_offset_emb_arrays[:125, 2]
        left_robot_tcp_pose_col = -left_robot_tcp_pose_arrays[3000:3200, 2]
        left_gripper2_marker_offset_emb_col = left_gripper2_marker_offset_emb_arrays[3000:3200, 2]

        # Normalize the columns to range 0-1
        left_robot_tcp_pose_col_norm = (left_robot_tcp_pose_col - np.min(left_robot_tcp_pose_col)) / (
                    np.max(left_robot_tcp_pose_col) - np.min(left_robot_tcp_pose_col))
        left_gripper2_marker_offset_emb_col_norm = (left_gripper2_marker_offset_emb_col - np.min(
            left_gripper2_marker_offset_emb_col)) / (np.max(left_gripper2_marker_offset_emb_col) - np.min(
            left_gripper2_marker_offset_emb_col))

        # Shift the indices of left_robot_tcp_pose_col_norm by K
        K = 3
        shifted_left_robot_tcp_pose_col_norm = np.pad(left_robot_tcp_pose_col_norm, (K, 0), mode='constant')[:-K]

        # Plot the normalized columns
        plt.figure(figsize=(10, 6))
        plt.plot(left_robot_tcp_pose_col_norm, label='Left Robot TCP Pose (Normalized)')
        plt.plot(left_gripper2_marker_offset_emb_col_norm, label='Left Gripper2 Marker Offset Emb (Normalized)')
        plt.plot(shifted_left_robot_tcp_pose_col_norm, label=f'Left Robot TCP Pose (Shifted by {K} frame)')
        plt.xlabel('Index')
        plt.ylabel('Normalized Value')
        plt.title('Normalized Columns Visualization')
        plt.legend()
        # Set more dense ticks on the x-axis
        # plt.xticks(np.arange(0, len(left_robot_tcp_pose_col_norm), step=1))
        plt.show()

    if DRAW_FORCE:
        # draw the force
        left_robot_tcp_wrench_force = left_robot_tcp_wrench_arrays[:200, :3]
        # draw the x-axis, y-axis, z-axis force
        plt.plot(left_robot_tcp_wrench_force[:, 0], label='x-axis force')
        plt.plot(left_robot_tcp_wrench_force[:, 1], label='y-axis force')
        plt.plot(left_robot_tcp_wrench_force[:, 2], label='z-axis force')
        plt.xlabel('Index')
        plt.ylabel('Force')
        plt.title('Force Visualization')
        plt.legend()
        plt.show()

    # create zarr file4
    zarr_root = zarr.group(save_data_path)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')

    # Compute chunk sizes
    external_img_chunk_size = (100, external_img_arrays.shape[1], external_img_arrays.shape[2], external_img_arrays.shape[3])
    wrist_img_chunk_size = (100, left_wrist_img_arrays.shape[1], left_wrist_img_arrays.shape[2], left_wrist_img_arrays.shape[3])
    if len(action_arrays.shape) == 2:
        action_chunk_size = (10000, action_arrays.shape[1])
    else:
        raise NotImplementedError

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    zarr_data.create_dataset('timestamp', data=timestamp_arrays, chunks=(10000,), dtype='float32', overwrite=True,
                             compressor=compressor)
    zarr_data.create_dataset('left_robot_tcp_pose', data=left_robot_tcp_pose_arrays, chunks=(10000, 9), dtype='float32', overwrite=True,
                             compressor=compressor)
    zarr_data.create_dataset('left_robot_tcp_vel', data=left_robot_tcp_vel_arrays, chunks=(10000, 6), dtype='float32', overwrite=True,
                                compressor=compressor)
    zarr_data.create_dataset('left_robot_tcp_wrench', data=left_robot_tcp_wrench_arrays, chunks=(10000, 6), dtype='float32', overwrite=True,
                                compressor=compressor)
    zarr_data.create_dataset('left_robot_gripper_width', data=left_robot_gripper_width_arrays, chunks=(10000, 1), dtype='float32', overwrite=True,
                                compressor=compressor)
    zarr_data.create_dataset('left_robot_gripper_force', data=left_robot_gripper_force_arrays, chunks=(10000, 1), dtype='float32', overwrite=True,
                                compressor=compressor)
    zarr_data.create_dataset('right_robot_tcp_pose', data=right_robot_tcp_pose_arrays, chunks=(10000, 9), dtype='float32', overwrite=True,
                                compressor=compressor)
    zarr_data.create_dataset('right_robot_tcp_vel', data=right_robot_tcp_vel_arrays, chunks=(10000, 6), dtype='float32', overwrite=True,
                                compressor=compressor)
    zarr_data.create_dataset('right_robot_tcp_wrench', data=right_robot_tcp_wrench_arrays, chunks=(10000, 6), dtype='float32', overwrite=True,
                                compressor=compressor)
    zarr_data.create_dataset('right_robot_gripper_width', data=right_robot_gripper_width_arrays, chunks=(10000, 1), dtype='float32', overwrite=True,
                                compressor=compressor)
    zarr_data.create_dataset('right_robot_gripper_force', data=right_robot_gripper_force_arrays, chunks=(10000, 1), dtype='float32', overwrite=True,
                                compressor=compressor)

    zarr_data.create_dataset('target', data=state_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True,
                             compressor=compressor)
    zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True,
                             compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, chunks=(10000,), dtype='int64', overwrite=True,
                             compressor=compressor)

    zarr_data.create_dataset('external_img', data=external_img_arrays, chunks=external_img_chunk_size, dtype='uint8', overwrite=True,
                             compressor=compressor)
    zarr_data.create_dataset('left_wrist_img', data=left_wrist_img_arrays, chunks=wrist_img_chunk_size, dtype='uint8')
    if 'two_tactile' in SENSOR_MODE or 'four_tactile' in SENSOR_MODE:
        gripper_img_chunk_size = (
            100, left_gripper1_img_arrays.shape[1], left_gripper1_img_arrays.shape[2], left_gripper1_img_arrays.shape[3])
        zarr_data.create_dataset('left_gripper1_img', data=left_gripper1_img_arrays, chunks=gripper_img_chunk_size, dtype='uint8')
        zarr_data.create_dataset('left_gripper2_img', data=left_gripper2_img_arrays, chunks=gripper_img_chunk_size, dtype='uint8')

        marker_chunk_size = (10000, left_gripper1_initial_marker_arrays.shape[1], left_gripper1_initial_marker_arrays.shape[2])
        marker_emb_chunk_size = (10000, left_gripper2_marker_offset_emb_arrays.shape[1])
        zarr_data.create_dataset('left_gripper1_initial_marker', data=left_gripper1_initial_marker_arrays, chunks=marker_chunk_size, dtype='float32')
        zarr_data.create_dataset('left_gripper2_initial_marker', data=left_gripper2_initial_marker_arrays, chunks=marker_chunk_size, dtype='float32')
        zarr_data.create_dataset('left_gripper1_marker_offset', data=left_gripper1_marker_offset_arrays, chunks=marker_chunk_size, dtype='float32')
        zarr_data.create_dataset('left_gripper2_marker_offset', data=left_gripper2_marker_offset_arrays, chunks=marker_chunk_size, dtype='float32')
        zarr_data.create_dataset('left_gripper1_marker_offset_emb', data=left_gripper1_marker_offset_emb_arrays, chunks=marker_emb_chunk_size, dtype='float32')
        zarr_data.create_dataset('left_gripper2_marker_offset_emb', data=left_gripper2_marker_offset_emb_arrays, chunks=marker_emb_chunk_size, dtype='float32')
        if SENSOR_MODE == 'dual_arm_two_realsense_four_tactile':
            zarr_data.create_dataset('right_wrist_img', data=right_wrist_img_arrays, chunks=wrist_img_chunk_size,
                                     dtype='uint8', overwrite=True)
            zarr_data.create_dataset('right_gripper1_img', data=right_gripper1_img_arrays, chunks=gripper_img_chunk_size,
                                     dtype='uint8', overwrite=True)
            zarr_data.create_dataset('right_gripper2_img', data=right_gripper2_img_arrays, chunks=gripper_img_chunk_size,
                                     dtype='uint8', overwrite=True)
            zarr_data.create_dataset('right_gripper1_initial_marker', data=right_gripper1_initial_marker_arrays,
                                     chunks=marker_chunk_size, dtype='float32', overwrite=True)
            zarr_data.create_dataset('right_gripper2_initial_marker', data=right_gripper2_initial_marker_arrays,
                                     chunks=marker_chunk_size, dtype='float32', overwrite=True)
            zarr_data.create_dataset('right_gripper1_marker_offset', data=right_gripper1_marker_offset_arrays,
                                     chunks=marker_chunk_size, dtype='float32', overwrite=True)
            zarr_data.create_dataset('right_gripper2_marker_offset', data=right_gripper2_marker_offset_arrays,
                                     chunks=marker_chunk_size, dtype='float32', overwrite=True)
            zarr_data.create_dataset('right_gripper1_marker_offset_emb', data=right_gripper1_marker_offset_emb_arrays,
                                     chunks=marker_emb_chunk_size, dtype='float32', overwrite=True)
            zarr_data.create_dataset('right_gripper2_marker_offset_emb', data=right_gripper2_marker_offset_emb_arrays,
                                     chunks=marker_emb_chunk_size, dtype='float32', overwrite=True)

    # print zarr data structure
    logger.info('Zarr data structure')
    logger.info(zarr_data.tree())
    logger.info(f'Total count after filtering: {action_arrays.shape[0]}')
    logger.info(f'Save data at {save_data_path}')
