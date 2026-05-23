from typing import Dict
import torch
import numpy as np
import os
from threadpoolctl import threadpool_limits
import copy
import tqdm
from reactive_diffusion_policy.common.pytorch_util import dict_apply
from reactive_diffusion_policy.dataset.base_dataset import BaseImageDataset
from reactive_diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from reactive_diffusion_policy.common.replay_buffer import ReplayBuffer
from reactive_diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from reactive_diffusion_policy.common.normalize_util import (
    get_image_range_normalizer,
    get_action_normalizer
)
from reactive_diffusion_policy.common.action_utils import absolute_actions_to_relative_actions, get_inter_gripper_actions
from reactive_diffusion_policy.real_world.real_world_transforms import RealWorldTransforms

class RealImageTactileDataset(BaseImageDataset):
    def __init__(self,
                 shape_meta: dict,
                 dataset_path: str,
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 n_obs_steps=None,
                 obs_temporal_downsample_ratio=1, # for latent diffusion
                 n_latency_steps=0,
                 seed=42,
                 val_ratio=0.0,
                 max_train_episodes=None,
                 delta_action=False,
                 relative_action=False,
                 relative_tcp_obs_for_relative_action=False,
                 transform_params=None,
                 action_quant_bits: int = 0,
                 action_noise_std: float = 0.0,
                 ):
        assert os.path.isdir(dataset_path)

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)

        extended_rgb_keys = list()
        extended_lowdim_keys = list()
        extended_obs_shape_meta = shape_meta.get('extended_obs', dict())
        for key, attr in extended_obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                extended_rgb_keys.append(key)
            elif type == 'low_dim':
                extended_lowdim_keys.append(key)

        print(f"[DEBUG] RealImageTactileDataset.__init__: dataset_path = '{dataset_path}'")
        print(f"[DEBUG] rgb_keys: {rgb_keys}")
        print(f"[DEBUG] lowdim_keys: {lowdim_keys}")
        zarr_path = os.path.join(dataset_path, 'replay_buffer.zarr')
        print(f"[DEBUG] zarr_path = '{zarr_path}'")
        zarr_load_keys = set(rgb_keys + lowdim_keys + extended_rgb_keys + extended_lowdim_keys + ['action'])
        zarr_load_keys = list(filter(lambda key: "wrt" not in key, zarr_load_keys))
        print(f"[DEBUG] zarr_load_keys: {zarr_load_keys}")
        replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=zarr_load_keys)

        # compute dataset-level action min/max for quantization (if requested)
        action_dim = shape_meta['action']['shape'][0]
        # try:
        #     action_vals = replay_buffer['action'][:, :action_dim]
        #     # ensure float
        # Note: action now includes tactile embedding (10 + 15 = 25 dims)
        action_shape_cfg = shape_meta['action']['shape'][0]
        
        # Get base action (10 dims)
        base_action_vals = replay_buffer['action'][:].astype(np.float32)
        
        # Check if we need to append tactile embeddings to action
        tactile_keys = [k for k in shape_meta.get('obs', {}).keys() if 'marker_offset_emb' in k]
        if tactile_keys and action_shape_cfg > base_action_vals.shape[1]:
            # Need to concatenate tactile data to action for min/max computation
            tactile_key = tactile_keys[0]  # use first tactile key
            tactile_vals = replay_buffer[tactile_key][:].astype(np.float32)
            # Concatenate along feature dimension
            full_action_vals = np.concatenate([base_action_vals, tactile_vals], axis=-1)
            action_dim = full_action_vals.shape[1]
        else:
            full_action_vals = base_action_vals
            action_dim = action_shape_cfg
        
        try:
            action_vals = full_action_vals[:, :action_dim]
        # avoid zero range
            action_vals = action_vals.astype(np.float32)
            action_min = np.min(action_vals, axis=0)
            action_max = np.max(action_vals, axis=0)
        except Exception:
            # fallback to zeros/range 1 if unexpected shape
            action_min = np.zeros((action_dim,), dtype=np.float32)
            action_max = np.ones((action_dim,), dtype=np.float32)


        action_range = action_max - action_min
        action_range[action_range == 0.0] = 1.0

        self._dataset_action_min = action_min
        self._dataset_action_max = action_max
        self._dataset_action_range = action_range
        # store quantization / noise parameters
        self.action_quant_bits = action_quant_bits
        self.action_noise_std = action_noise_std
        print(f"[DEBUG] Quantization/Noise config: quant_bits={action_quant_bits}, noise_std={action_noise_std}")

        if delta_action:
            # replace action as relative to previous frame
            actions = replay_buffer['action'][:]
            # support positions only at this time
            assert actions.shape[1] <= 3
            actions_diff = np.zeros_like(actions)
            episode_ends = replay_buffer.episode_ends[:]
            for i in range(len(episode_ends)):
                start = 0
                if i > 0:
                    start = episode_ends[i-1]
                end = episode_ends[i]
                # delta action is the difference between previous desired position and the current
                # it should be scheduled at the previous timestep for the current timestep
                # to ensure consistency with positional mode
                actions_diff[start+1:end] = np.diff(actions[start:end], axis=0)
            replay_buffer['action'][:] = actions_diff
        
        self.relative_action = relative_action
        self.relative_tcp_obs_for_relative_action = relative_tcp_obs_for_relative_action
        self.transforms = RealWorldTransforms(option=transform_params)

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                if key not in extended_rgb_keys + extended_lowdim_keys:
                    key_first_k[key] = n_obs_steps * obs_temporal_downsample_ratio
        self.key_first_k = key_first_k

        self.seed = seed
        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon+n_latency_steps,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.extended_rgb_keys = extended_rgb_keys
        self.extended_lowdim_keys = extended_lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.obs_downsample_ratio = obs_temporal_downsample_ratio
        self.val_mask = val_mask
        self.horizon = horizon
        self.n_latency_steps = n_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon+self.n_latency_steps,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask
            )
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # calculate relative action / obs
        if "left_robot_wrt_right_robot_tcp_pose" in self.lowdim_keys or "right_robot_wrt_left_robot_tcp_pose" in self.lowdim_keys:
            inter_gripper_data_dict = {key: list() for key in self.lowdim_keys if 'robot_tcp_pose' in key and 'wrt' in key}
            for data in tqdm.tqdm(self, leave=False, desc='Calculating inter-gripper relative obs for normalizer'):
                for key in inter_gripper_data_dict.keys():
                    inter_gripper_data_dict[key].append(data['obs'][key])
            inter_gripper_data_dict = dict_apply(inter_gripper_data_dict, np.stack)

        if self.relative_action:
            relative_data_dict = {key: list() for key in (self.lowdim_keys + ['action']) if ('robot_tcp_pose' in key and 'wrt' not in key) or 'action' in key}
            for data in tqdm.tqdm(self, leave=False, desc='Calculating relative action/obs for normalizer'):
                for key in relative_data_dict.keys():
                    if key == 'action':
                        relative_data_dict[key].append(data[key])
                    else:
                        relative_data_dict[key].append(data['obs'][key])
            relative_data_dict = dict_apply(relative_data_dict, np.stack)

        # action
        if self.relative_action:
            action_all = relative_data_dict['action']
        else:
            # Get base action from replay buffer
            base_action = self.replay_buffer['action'][:]
            
            # Check if we need to append tactile embeddings to action
            action_target_dim = self.shape_meta['action']['shape'][0]
            if base_action.shape[-1] < action_target_dim:
                # Need to append tactile data to action
                tactile_keys = [k for k in self.shape_meta.get('obs', {}).keys() if 'marker_offset_emb' in k]
                if tactile_keys:
                    tactile_key = tactile_keys[0]
                    tactile_data = self.replay_buffer[tactile_key][:]
                    print(f"[DEBUG] Concatenating tactile: base_action.shape={base_action.shape}, tactile_data.shape={tactile_data.shape}")
                    # Concatenate along feature dimension
                    action_all = np.concatenate([base_action, tactile_data], axis=-1)
                    print(f"[DEBUG] After concat: action_all.shape={action_all.shape}")
                    print(f"[DEBUG] base_action stats: min={np.nanmin(base_action):.4f}, max={np.nanmax(base_action):.4f}, has_nan={np.isnan(base_action).any()}")
                    print(f"[DEBUG] tactile_data stats: min={np.nanmin(tactile_data):.4f}, max={np.nanmax(tactile_data):.4f}, has_nan={np.isnan(tactile_data).any()}")
                else:
                    action_all = base_action[:, :action_target_dim]
            else:
                action_all = base_action[:, :action_target_dim]

        normalizer['action'] = get_action_normalizer(action_all)

        # obs
        for key in list(set(self.lowdim_keys)):
            if self.relative_action and key in relative_data_dict:
                normalizer[key] = get_action_normalizer(relative_data_dict[key])
            elif 'robot_tcp_pose' in key and 'wrt' in key:
                normalizer[key] = get_action_normalizer(inter_gripper_data_dict[key])
            elif 'robot_tcp_pose' in key and 'wrt' not in key:
                normalizer[key] = get_action_normalizer(self.replay_buffer[key][:, :self.shape_meta['obs'][key]['shape'][0]])
            else:
                normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                    self.replay_buffer[key][:, :self.shape_meta['obs'][key]['shape'][0]])

        for key in list(set(self.extended_lowdim_keys)):
            if key in self.lowdim_keys:
                assert self.shape_meta['extended_obs'][key]['shape'][0] == self.shape_meta['obs'][key]['shape'][0], \
                    f"Extended obs {key} has different shape from obs {key}"
            else:
                if self.relative_action and key in relative_data_dict:
                    normalizer[key] = get_action_normalizer(relative_data_dict[key])
                elif 'robot_tcp_pose' in key and 'wrt' in key:
                    normalizer[key] = get_action_normalizer(inter_gripper_data_dict[key])
                elif 'robot_tcp_pose' in key and 'wrt' not in key: # not used now
                    normalizer[key] = get_action_normalizer(self.replay_buffer[key][:, :self.shape_meta['extended_obs'][key]['shape'][0]])
                else:
                    normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                        self.replay_buffer[key][:, :self.shape_meta['extended_obs'][key]['shape'][0]])

        # image
        for key in list(set(self.rgb_keys + self.extended_rgb_keys)):
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'][:, :self.shape_meta['action']['shape'][0]])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)
        obs_downsample_ratio = self.obs_downsample_ratio

        obs_dict = dict()
        # Save tactile data before it gets deleted
        tactile_data_for_action = None
        tactile_keys = [k for k in self.shape_meta.get('obs', {}).keys() if 'marker_offset_emb' in k]
        if tactile_keys:
            tactile_key = tactile_keys[0]
            # Get raw data slice from replay buffer using sampler's indices for this item
            buffer_start_idx, buffer_end_idx, _, _ = self.sampler.indices[idx]
            indices = np.arange(buffer_start_idx, buffer_end_idx)
            tactile_data_for_action = self.replay_buffer[tactile_key][indices].astype(np.float32)
        
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[key][T_slice][::-obs_downsample_ratio][::-1],-1,1
                ).astype(np.float32) / 255.
            # T,C,H,W
            # save ram
            if key not in self.rgb_keys:
                del data[key]
        for key in self.lowdim_keys:
            if 'wrt' not in key:
                obs_dict[key] = data[key][:, :self.shape_meta['obs'][key]['shape'][0]][T_slice][::-obs_downsample_ratio][::-1].astype(np.float32)
                # save ram
                if key not in self.extended_lowdim_keys:
                    del data[key]

        # inter-gripper relative action
        obs_dict.update(get_inter_gripper_actions(obs_dict, self.lowdim_keys, self.transforms))
        for key in ['left_robot_wrt_right_robot_tcp_pose']:
            if key in obs_dict:
                obs_dict[key] = obs_dict[key][:, :self.shape_meta['obs'][key]['shape'][0]].astype(np.float32)
        
        extended_obs_dict = dict()
        for key in self.extended_rgb_keys:
            extended_obs_dict[key] = np.moveaxis(data[key],-1,1
                ).astype(np.float32) / 255.
            del data[key]
        for key in self.extended_lowdim_keys:
            if 'wrt' not in key:
                extended_obs_dict[key] = data[key][:, :self.shape_meta['extended_obs'][key]['shape'][0]].astype(np.float32)
                del data[key]

        action = data['action'].astype(np.float32)
        # handle latency by dropping first n_latency_steps action
        # observations are already taken care of by T_slice
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps:]
        
        # Concatenate tactile embedding to action if specified in shape_meta
        action_target_dim = self.shape_meta['action']['shape'][0]
        if not hasattr(self, '_debug_logged'):
            self._debug_logged = True
            print(f"[DEBUG] __getitem__(idx={idx}) before concat check: action.shape={action.shape}, target_dim={action_target_dim}")
        if action.shape[-1] < action_target_dim:
            # Need to append tactile data to action
            if tactile_data_for_action is not None:
                # Apply same latency handling as action
                tactile_data = tactile_data_for_action
                if self.n_latency_steps > 0:
                    tactile_data = tactile_data[self.n_latency_steps:]

                # Pad tactile_data if its length is less than action's length
                if tactile_data.shape[0] < action.shape[0]:
                    if not hasattr(self, '_debug_padding_logged'):
                        self._debug_padding_logged = True
                        print(f"[DEBUG] Padding tactile_data: from {tactile_data.shape[0]} to {action.shape[0]}")
                    pad_len = action.shape[0] - tactile_data.shape[0]
                    last_row = tactile_data[-1:]
                    padding = np.repeat(last_row, pad_len, axis=0)
                    tactile_data = np.concatenate([tactile_data, padding], axis=0)

                # Concatenate along feature dimension
                if not hasattr(self, '_debug_concat_logged'):
                    self._debug_concat_logged = True
                    print(f"[DEBUG] Before concat in __getitem__:")
                    print(f"  action.shape={action.shape}, has_nan={np.isnan(action).any()}")
                    print(f"  tactile_data.shape={tactile_data.shape}, has_nan={np.isnan(tactile_data).any()}")
                action = np.concatenate([action, tactile_data], axis=-1)
                if not hasattr(self, '_debug_after_concat_logged'):
                    self._debug_after_concat_logged = True
                    print(f"[DEBUG] After concat in __getitem__:")
                    print(f"  action.shape={action.shape}, has_nan={np.isnan(action).any()}")
            else:
                if not hasattr(self, '_debug_no_tactile_logged'):
                    self._debug_no_tactile_logged = True
                    print(f"[WARNING] No tactile data available for concatenation!")
        
        if self.relative_action:
            base_absolute_action = np.concatenate([
                obs_dict['left_robot_tcp_pose'][-1] if 'left_robot_tcp_pose' in obs_dict else np.array([]),
            ], axis=-1)
            action = absolute_actions_to_relative_actions(action, base_absolute_action=base_absolute_action)

            if self.relative_tcp_obs_for_relative_action:
                for key in self.lowdim_keys:
                    if 'robot_tcp_pose' in key and 'wrt' not in key:
                        obs_dict[key]  = absolute_actions_to_relative_actions(obs_dict[key], base_absolute_action=base_absolute_action)

        if not hasattr(self, '_debug_return_logged'):
            self._debug_return_logged = True
            print(f"[DEBUG] __getitem__ returning: action.shape = {action.shape}")
            print(f"[DEBUG] action has NaN: {np.isnan(action).any()}, has Inf: {np.isinf(action).any()}")

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(action.astype(np.float32)),
            'extended_obs': dict_apply(extended_obs_dict, torch.from_numpy)
        }

        return torch_data

def test():
    import hydra
    from hydra import initialize, compose
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    with initialize('../config'):
        cfg = hydra.compose('train_diffusion_unet_real_image_workspace',
                            overrides=['task=real_peel_image_gelsight_emb_absolute_12fps'])
        OmegaConf.resolve(cfg)
        dataset = hydra.utils.instantiate(cfg.task.dataset)

    from matplotlib import pyplot as plt
    normalizer = dataset.get_normalizer()

    for i in range(len(dataset)):
        data = dataset[i]

if __name__ == '__main__':
    test()
