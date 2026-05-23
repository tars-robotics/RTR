import cv2
import zarr
import copy
import numpy as np
import os
from reactive_diffusion_policy.common.tactile_marker_utils import display_motion
from reactive_diffusion_policy.model.common.pca_embedding import PCAEmbedding
import matplotlib.pyplot as plt
import argparse

def reshape_reconstructed_data(reconstructed_data, original_shape=(63, 2)):
    n_samples = reconstructed_data.shape[0]
    reshaped_data = reconstructed_data.reshape(n_samples, *original_shape)
    return reshaped_data

def flatten_data(data: np.ndarray):
    n_samples = data.shape[0]
    flattened_data = np.empty((n_samples, np.prod(data.shape[1:])), dtype=np.float32)
    for i in range(n_samples):
        flattened_data[i] = data[i].flatten()
    return flattened_data


TRANSFORM_MATRIX_PATH = 'data/PCA_Transform_McTAC_v1/pca_transform_matrix.npy'
MEAN_MATRIX_PATH = 'data/PCA_Transform_McTAC_v1/pca_mean_matrix.npy'
TRAIN_TACTILE_DATA_PATH = 'data/marker_motion/mctac_v1.zarr'
EVAL_TACTILE_DATA_PATH = 'data/peel_v3_zarr/replay_buffer.zarr'
MODE = 'Train'

def main():
    pca_embedding = PCAEmbedding(n_components=15, normalize=False, mode=MODE,
                                 store= (MODE == 'Train'),
                                 transformation_matrix_path=None if MODE == 'Train' else TRANSFORM_MATRIX_PATH,
                                 mean_matrix_path=None if MODE == 'Train' else MEAN_MATRIX_PATH)

    if pca_embedding.mode == 'Train':
        zarr_data= zarr.open(TRAIN_TACTILE_DATA_PATH, mode='r')
        motion_data = zarr_data['marker_motion'][:]
        images = zarr_data['color_image'][:]
        initial_markers = zarr_data['initial_marker'][:]
        original_shape = motion_data.shape[1:]

        flattened_marker_motion = motion_data.reshape(motion_data.shape[0], -1)  # (28165, 63,2), (28165, 126)
        X_pca, X_reconstructed = pca_embedding.reduce_and_reconstruct(flattened_marker_motion)  # (28165, num_component), (28165, 126)
        W = pca_embedding.pca.components_.T  # (126, n_component)
        mean = pca_embedding.pca.mean_  # (126,)
        pca_embedding.W = W
        pca_embedding.mean = mean

        if pca_embedding.store:
            np.save(TRANSFORM_MATRIX_PATH, W)
            np.save(MEAN_MATRIX_PATH, mean)

        motion_reshaped = reshape_reconstructed_data(X_reconstructed, original_shape)
    elif pca_embedding.mode == 'Eval':
        zarr_data = zarr.open(EVAL_TACTILE_DATA_PATH, mode='r')['data']
        motion_data = zarr_data['left_gripper2_marker_offset'][:]
        images = zarr_data['left_gripper2_img'][:]
        initial_markers = zarr_data['left_gripper2_initial_marker'][:]
        original_shape = motion_data.shape[1:]

        flattened_marker_motion = motion_data.reshape(motion_data.shape[0], -1)  # (len(dataset), 63, 2), (len(dataset), 126)
        X_pca = pca_embedding.pca_reduction(flattened_marker_motion)  # (len(dataset), n_components)
        X_reconstructed = pca_embedding.pca_reconstruction(X_pca)  # type: ignore # (len(dataset), 126)

        motion_reshaped = reshape_reconstructed_data(X_reconstructed, original_shape)
    else:
        raise ValueError("mode should be either 'Train' or 'Eval'")

    # visualize both based on original marker motion and reconstructed marker motion
    for i in range(len(images)):
        frame = images[i]
        frame_display = copy.deepcopy(frame)
        h, w = frame.shape[:2]

        marker_motion = motion_data[i]
        marker_motion_PCA = motion_reshaped[i] * np.array([w, h])
        denormalized_marker_motion = marker_motion * np.array([w, h])
        initial_marker = initial_markers[i][:, :2]
        denormalized_initial_markers = initial_marker * np.array([w, h])

        # Notion that initial markers and marker motions here should be denormalized
        marker_img = display_motion(frame, denormalized_initial_markers, denormalized_marker_motion, 5, 2)
        cv2.imshow('frame with marker motion', marker_img)
        PCA_marker_img = display_motion(frame_display, denormalized_initial_markers, marker_motion_PCA, 5, 2)
        cv2.imshow('frame with marker motion reconstructed after PCA', PCA_marker_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
