from typing import Dict, Any, Union, Optional
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
import hydra
import transforms3d as t3d
import zarr

from rtr_async_sys.core.model_wrapper_base import AbsModelWrapper
from rtr_async_sys.utils.action_utils import ortho6d_to_rotation_matrix
from rtr_async_sys.utils.image_utils import decompress_image, decompress_video_frames


class ReplayModelWrapper(AbsModelWrapper):
    def __init__(
            self,
            model: Union[str, DictConfig]=None,
            device: Union[str, torch.device]=None,
            ckpt_path: str=None,
            return_raw_action: bool = False,
            need_refine: bool = False,
            compress_obs: bool = False,
            horizon: int = 48,
            zarr_path: str = "data/ckpts/vase_sponge_test1_60hz/rdp_zarr/replay_buffer.zarr"
        ):
        super().__init__(
            model, device, ckpt_path,
            return_raw_action, need_refine,
            compress_obs=compress_obs,
            build_policy=False
        )
        self.horizon = int(horizon)
        self.zarr_path = zarr_path

        # ---- Load first trajectory actions from zarr ----
        zarr_root = zarr.open(self.zarr_path, mode="r")
        actions = zarr_root["data/action"]            # (N, 10)
        episode_ends = zarr_root["meta/episode_ends"] # (num_episodes,)

        if len(episode_ends) < 1:
            raise ValueError(f"No episodes found in zarr: {self.zarr_path}")

        first_end = int(np.asarray(episode_ends[0]))
        if first_end <= 0:
            raise ValueError(f"First episode seems empty: episode_ends[0]={first_end}")

        # cache first-episode actions into memory (float32)
        self._traj_actions = np.asarray(actions[:first_end], dtype=np.float32)  # (T0, 10)
        if self._traj_actions.ndim != 2 or self._traj_actions.shape[1] != 10:
            raise ValueError(f"Expected first-episode actions shape (T,10), got {self._traj_actions.shape}")

        self._traj_len = int(self._traj_actions.shape[0])
        self._step = 0

    def predict_action_chunk(self, obs_dict: Dict[str, torch.Tensor]) -> Optional[np.ndarray]:
        """
        action_chunk: shape is [t, action_dim]
        Return None if the first trajectory has been fully traversed.
        """
        # trajectory finished
        if self._step >= self._traj_len:
            return None

        start = self._step
        # end = min(self._step + self.horizon, self._traj_len)
        end = start + self.horizon
        if end > self._traj_len:
            print(f"end of chunk")
            exit()
            return None

        # slice (may be shorter than horizon near the end)
        action_chunk = self._traj_actions[start:end]  # (t, 10)

        # step += horizon (as you requested)
        self._step += self.horizon

        action_chunk = self.postprocess_action(action_chunk)
        return action_chunk

    def preprocess_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        # replay wrapper doesn't need obs; keep passthrough
        return obs

    def postprocess_action(self, raw_action_chunk: np.ndarray) -> np.ndarray:
        assert len(raw_action_chunk.shape) == 2, "input in postprocess_action should be action chunk"
        assert raw_action_chunk.shape[1] == 10, f"Expected action dim 10, got {raw_action_chunk.shape[1]}"
        if self.return_raw_action:
            return raw_action_chunk
        else:
            left_rot_mat_batch = ortho6d_to_rotation_matrix(raw_action_chunk[:, 3:9])  # (t, 3, 3)
            left_euler_batch = np.array([t3d.euler.mat2euler(rot_mat) for rot_mat in left_rot_mat_batch])  # (t, 3)
            left_trans_batch = raw_action_chunk[:, :3]  # (t, 3)
            left_action_6d = np.concatenate([left_trans_batch, left_euler_batch], axis=1)  # (t, 6)
            return left_action_6d

    def refine_action(self, action: np.ndarray, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return action


if __name__ == '__main__':
    cfg = OmegaConf.load("src/rtr_async_sys/configs/model_wrapper/dp_wrapper.yaml")
    dp_model_wrapper = hydra.utils.instantiate(cfg)
    print(dp_model_wrapper)
