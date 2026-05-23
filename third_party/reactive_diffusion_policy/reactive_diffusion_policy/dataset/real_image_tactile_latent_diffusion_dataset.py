import einops
import numpy as np
import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from reactive_diffusion_policy.model.vae.model import VAE
from reactive_diffusion_policy.model.vae.block_vae_model import BlockEncodeVAE
from reactive_diffusion_policy.dataset.real_image_tactile_dataset import RealImageTactileDataset
from reactive_diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer

from reactive_diffusion_policy.dataset.real_image_tactile_dataset_reverse import RealImageTactileDatasetReverse


class ActionOnlyDataset(Dataset):
    def __init__(self, dataset):
        self.indices = dataset.sampler.indices
        self.action_arr = dataset.replay_buffer['action']
        self.action_dim = dataset.shape_meta['action']['shape'][0]
        self.sequence_length = dataset.horizon + dataset.n_latency_steps
        self.n_latency_steps = dataset.n_latency_steps

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]
        sample = self.action_arr[buffer_start_idx:buffer_end_idx, :self.action_dim]
        if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
            action = np.zeros((self.sequence_length, self.action_dim), dtype=sample.dtype)
            if sample_start_idx > 0:
                action[:sample_start_idx] = sample[0]
            if sample_end_idx < self.sequence_length:
                action[sample_end_idx:] = sample[-1]
            action[sample_start_idx:sample_end_idx] = sample
        else:
            action = sample
        action = action[self.n_latency_steps:]
        return action.astype(np.float32, copy=False)

class RealImageTactileLatentDiffusionDataset(RealImageTactileDataset):
    def __init__(self,
                 at: VAE,
                 use_latent_action_before_vq: bool,
                 use_block_vae: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.at = at
        self.use_block_vae = use_block_vae
        self.at.eval()
        self.use_latent_action_before_vq = use_latent_action_before_vq

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = super().get_normalizer(**kwargs)

        latent_action_all = []

        batch_size = kwargs.get('latent_action_batch_size', 1024)
        num_workers = kwargs.get('latent_action_num_workers', 8)
        if self.relative_action:
            action_iter = (
                data['action'].unsqueeze(0)
                for data in tqdm.tqdm(self, leave=False, desc='Calculating latent action for normalizer')
            )
        else:
            action_loader = DataLoader(
                ActionOnlyDataset(self),
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
                pin_memory=True,
                persistent_workers=num_workers > 0,
            )
            action_iter = tqdm.tqdm(action_loader, leave=False, desc='Calculating latent action for normalizer')

        with torch.no_grad():
            for action in action_iter:
                action = action.to(self.at.device)
                if not self.use_block_vae:
                    action = normalizer['action'].normalize(action)
                    latent_action = self.at.encoder(
                        self.at.preprocess(action / self.at.act_scale)
                    )
                    if self.at.use_vq:
                        if not self.use_latent_action_before_vq:
                            latent_action, _, _ = self.at.quant_state_with_vq(latent_action)
                    else:
                        latent_action, _ = self.at.quant_state_without_vq(latent_action)
                    if self.at.use_conv_encoder:
                        latent_action = einops.rearrange(latent_action, "N (T A) -> N T A", T=self.at.downsampled_input_h)
                    else:
                        latent_action = einops.rearrange(latent_action, "N (T A) -> N T A", T=1)
                    latent_action_all.append(latent_action.cpu().detach().numpy())
                else:
                    self.at:BlockEncodeVAE = self.at
                    batch_size, T, action_dim = action.shape
                    action = action.reshape(batch_size*self.at.encode_block_num, T//self.at.encode_block_num, action_dim)
                    latent_action = self.at.encode_to_latent(action,reshape_in_vae=False)
                    latent_action = latent_action.reshape(batch_size, self.at.downsampled_input_h*self.at.encode_block_num, -1)
                    latent_action_all.append(latent_action.cpu().detach().numpy())

        latent_action_all = np.concatenate(latent_action_all, axis=0).reshape(-1, latent_action_all[0].shape[-1])

        normalizer['latent_action'] = SingleFieldLinearNormalizer.create_fit(latent_action_all)

        return normalizer


class RealImageTactileLatentDiffusionDatasetReverse(RealImageTactileDatasetReverse):
    def __init__(self,
                 at: VAE,
                 use_latent_action_before_vq: bool,
                 **kwargs):
        super().__init__(**kwargs)
        self.at = at
        self.at.eval()
        self.use_latent_action_before_vq = use_latent_action_before_vq

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = super().get_normalizer(**kwargs)

        latent_action_all = []

        for data in tqdm.tqdm(self, leave=False, desc='Calculating latent action for normalizer'):
            action = data['action'].to(self.at.device).unsqueeze(0)
            action = normalizer['action'].normalize(action)
            latent_action = self.at.encoder(
                self.at.preprocess(action / self.at.act_scale)
            )
            if self.at.use_vq:
                if not self.use_latent_action_before_vq:
                    latent_action, _, _ = self.at.quant_state_with_vq(latent_action)
            else:
                latent_action, _ = self.at.quant_state_without_vq(latent_action)
            if self.at.use_conv_encoder:
                latent_action = einops.rearrange(latent_action, "N (T A) -> N T A", T=self.at.downsampled_input_h)
            else:
                latent_action = einops.rearrange(latent_action, "N (T A) -> N T A", T=1)
            latent_action_all.append(latent_action[0].cpu().detach().numpy())

        latent_action_all = np.concatenate(latent_action_all, axis=0)

        normalizer['latent_action'] = SingleFieldLinearNormalizer.create_fit(latent_action_all)

        return normalizer
