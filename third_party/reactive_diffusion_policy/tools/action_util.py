import numpy as np
import cv2
from scipy.spatial.transform import Rotation

# def rot6d_to_matrix(rot6d):
#     # rot6d: (N,6)
#     x = rot6d[:, :3]
#     y = rot6d[:, 3:6]
#     x = x / np.linalg.norm(x, axis=1, keepdims=True)
#     z = np.cross(x, y)
#     z = z / np.linalg.norm(z, axis=1, keepdims=True)
#     y = np.cross(z, x)
#     R = np.stack([x, y, z], axis=-1)  # (N,3,3)
#     return R

def rot6d_to_matrix(rot6d):
    x = rot6d[:, :3]
    y = rot6d[:, 3:6]
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    y_proj = np.sum(x * y, axis=1, keepdims=True) * x  # projection component
    y = y - y_proj
    y = y / np.linalg.norm(y, axis=1, keepdims=True)
    z = np.cross(x, y)
    R = np.stack([x, y, z], axis=-1)  # (N, 3, 3)
    return R

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

def pose_3d_9d_to_homo_matrix_batch(pose: np.ndarray) -> np.ndarray:
    """
    Convert 3D / 9D states to 4x4 matrix
    :param pose: np.ndarray (N, 9) or (N, 3)
    :return: np.ndarray (N, 4, 4)
    """
    assert pose.shape[1] in [3, 9], "pose should be (N, 3) or (N, 9)"
    mat = np.eye(4)[None, :, :].repeat(pose.shape[0], axis=0)
    mat[:, :3, 3] = pose[:, :3]
    if pose.shape[1] == 9:
        mat[:, :3, :3] = ortho6d_to_rotation_matrix(pose[:, 3:9])
    return mat

def homo_matrix_to_pose_9d_batch(mat: np.ndarray) -> np.ndarray:
    """
    Convert 4x4 matrix to 9D state
    :param mat: np.ndarray (N, 4, 4)
    :return: np.ndarray (N, 9)
    """
    assert mat.shape[1:] == (4, 4), "mat should be (N, 4, 4)"
    pose = np.zeros((mat.shape[0], 9))
    pose[:, :3] = mat[:, :3, 3]
    pose[:, 3:9] = mat[:, :3, :2].swapaxes(1, 2).reshape(mat.shape[0], -1)
    return pose

def absolute_actions_to_relative_actions(actions: np.ndarray, base_absolute_action=None):
    actions = actions.copy()
    T, D = actions.shape

    if D == 3 or D == 4:  # (x, y, z(, gripper_width))
        tcp_dim_list = [np.arange(3)]
    elif D == 6 or D == 8:  # (x_l, y_l, z_l, x_r, y_r, z_r(, gripper_width_l, gripper_width_r))
        tcp_dim_list = [np.arange(3), np.arange(3, 6)]
    elif D == 9 or D == 10:  # (x, y, z, rx1, rx2, rx3, ry1, ry2, ry3(, gripper_width))
        tcp_dim_list = [np.arange(9)]
    elif D == 18 or D == 20:  # (x_l, y_l, z_l, rotation_l, x_r, y_r, z_r, rotation_r(, gripper_width_l, gripper_width_r))
        tcp_dim_list = [np.arange(9), np.arange(9, 18)]
    else:
        raise NotImplementedError

    if base_absolute_action is None:
        base_absolute_action = actions[0].copy()
    for tcp_dim in tcp_dim_list:
        assert len(tcp_dim) == 3 or len(tcp_dim) == 9, "Only support 3D or 9D tcp pose now"
        base_tcp_pose_mat = pose_3d_9d_to_homo_matrix_batch(base_absolute_action[None, tcp_dim])
        actions[:, tcp_dim] = homo_matrix_to_pose_9d_batch(np.linalg.inv(base_tcp_pose_mat) @ pose_3d_9d_to_homo_matrix_batch(
            actions[:, tcp_dim]))[:, :len(tcp_dim)]

    return actions

def relative_actions_to_absolute_actions(actions: np.ndarray, base_absolute_action: np.ndarray):
    actions = actions.copy()
    T, D = actions.shape

    if D == 3 or D == 4:  # (x, y, z(, gripper_width))
        tcp_dim_list = [np.arange(3)]
    elif D == 6 or D == 8:  # (x_l, y_l, z_l, x_r, y_r, z_r(, gripper_width_l, gripper_width_r))
        tcp_dim_list = [np.arange(3), np.arange(3, 6)]
    elif D == 9 or D == 10:  # (x, y, z, rx1, rx2, rx3, ry1, ry2, ry3(, gripper_width))
        tcp_dim_list = [np.arange(9)]
    elif D == 18 or D == 20:  # (x_l, y_l, z_l, rotation_l, x_r, y_r, z_r, rotation_r(, gripper_width_l, gripper_width_r))
        tcp_dim_list = [np.arange(9), np.arange(9, 18)]
    else:
        raise NotImplementedError

    for tcp_dim in tcp_dim_list:
        assert len(tcp_dim) == 3 or len(tcp_dim) == 9, "Only support 3D or 9D tcp pose now"
        base_tcp_pose_mat = pose_3d_9d_to_homo_matrix_batch(base_absolute_action[None, tcp_dim])
        actions[:, tcp_dim] = homo_matrix_to_pose_9d_batch(base_tcp_pose_mat @ pose_3d_9d_to_homo_matrix_batch(
            actions[:, tcp_dim]))[:, :len(tcp_dim)]

def absolute_actions_to_delta_actions(actions: np.ndarray, base_absolute_action: np.ndarray):
    actions = actions.copy()
    actions = np.concatenate([base_absolute_action[None, :], actions], axis=0)
    new_action = list()
    action_position = actions.copy()[: , :3]
    action_rotation = rot6d_to_matrix(actions.copy()[: , 3:])
    for i in range(actions.shape[0] - 1):
        delta_pos = action_position[i+1] - action_position[i]
        delta_rot = (action_rotation[i+1] @ action_rotation[i].T)
        delta_rot6d = np.concatenate([delta_rot[:, 0], delta_rot[:, 1]])
        new_action.append(np.concatenate([delta_pos, delta_rot6d])[None, :])
    new_action_array = np.concatenate(new_action, axis=0)

    return new_action_array

def delta_actions_to_absolute_actions(action: np.ndarray, base_absolute_action: np.ndarray):
    # action: [9,]
    delta_pos = action[:3]
    delta_rot = rot6d_to_matrix(action[3:][None, :])[0]
    base_pos = base_absolute_action[:3]
    base_rot = rot6d_to_matrix(base_absolute_action[3:][None, :])[0]
    new_pos = base_pos + delta_pos
    new_rot = delta_rot @ base_rot
    new_rot6d = np.concatenate([new_rot[:, 0], new_rot[:, 1]])

    return np.concatenate([new_pos, new_rot6d])

c2h = np.array([[-1, 0, 0, 5],
                [0, -0.9659, -0.2588, 96.678],
                [0, -0.2588, 0.9659, -26.625],
                [0, 0, 0, 1]])

intrinsic = np.array([[604.307, 0, 310.155],
                      [0, 604.662, 251.013],
                      [0, 0, 1]])

def project_points_to_image(points_cam, K):
    X = points_cam[0]
    Y = points_cam[1]
    Z = points_cam[2]
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy
    return np.stack([u, v], axis=-1)

def get_visual_action(action: np.ndarray, base_absolute_action: np.ndarray):
    base_translation = base_absolute_action[:3] * 1000.0
    base_rotation = rot6d_to_matrix(base_absolute_action[None, 3:])[0]
    translation = action[: ,:3] * 1000.0
    rotation = rot6d_to_matrix(action[:, 3:])
    
    base_h2b = np.eye(4)
    base_h2b[:3, :3] = base_rotation
    base_h2b[:3, 3] = base_translation
    base_c2b = base_h2b @ c2h

    visual_actions = list()

    for i in range(translation.shape[0]):
        h2b = np.eye(4)
        h2b[:3, :3] = rotation[i]
        h2b[:3, 3] = translation[i]
        g0 = h2b @ np.array([-10, 0, 230, 1])
        g0c = np.linalg.inv(base_c2b) @ g0
        p0 = project_points_to_image(g0c, intrinsic)
        visual_actions.append(p0[None, :])

    return np.concatenate(visual_actions)

def vis_visual_action(img, uv_coords, radius=2, color=(0, 0, 255)):
    img_draw = img.copy()
    for u, v in uv_coords:
        cv2.circle(img_draw, (int(u), int(v)), radius, color, -1)
    return img_draw