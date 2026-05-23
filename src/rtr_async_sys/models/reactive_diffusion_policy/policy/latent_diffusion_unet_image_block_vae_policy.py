from typing import Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from ...reactive_diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from ...reactive_diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from ...reactive_diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder
from ...reactive_diffusion_policy.model.vae.block_vae_model import BlockEncodeVAE
from ...reactive_diffusion_policy.model.common.normalizer import LinearNormalizer
from ...reactive_diffusion_policy.common.pytorch_util import dict_apply

class LatentDiffusionUnetImageBlockVaePolicy(DiffusionUnetImagePolicy):
    def __init__(self,
                 at: BlockEncodeVAE,
                 use_latent_action_before_vq: bool,
                 shape_meta: dict,
                 noise_scheduler: DDPMScheduler,
                 obs_encoder: Union[MultiImageObsEncoder, TimmObsEncoder],
                 horizon,
                 n_action_steps,
                 n_obs_steps,
                 num_inference_steps=None,
                 obs_as_global_cond=True,
                 diffusion_step_embed_dim=256,
                 down_dims=(256, 512, 1024),
                 kernel_size=5,  # dummy
                 n_groups=8,
                 cond_predict_scale=True,
                 change_kernel_size=True,
                 # parameters passed to step
                 **kwargs):

        if at.use_vq:
            shape_meta['action']['shape'] = [at.n_latent_dims]
        else:
            shape_meta['action']['shape'] = [at.n_embed]

        original_horizon = horizon
        if at.use_conv_encoder:
            horizon = at.downsampled_input_h * at.encode_block_num
            if change_kernel_size:# original behavior
                print("change_kernel_size"+"="*100)
                kernel_size = 3
            else:
                print("NOT change_kernel_size"+"="*100)
        else:
            # hack: latent action can be viewed as a sequence of latent actions with horizon 1
            horizon = at.downsampled_input_h * at.encode_block_num
            kernel_size = 1

        super().__init__(shape_meta,
                         noise_scheduler,
                         obs_encoder,
                         horizon,
                         n_action_steps,
                         n_obs_steps,
                         num_inference_steps,
                         obs_as_global_cond,
                         diffusion_step_embed_dim,
                         down_dims,
                         kernel_size,
                         n_groups,
                         cond_predict_scale,
                         **kwargs)

        self.original_horizon = original_horizon

        if horizon == 1:
            # hack: remove the upsample/downsample layers
            for module_list in self.model.down_modules:
                module_list[-1] = nn.Identity()

            for module_list in self.model.up_modules:
                module_list[-1] = nn.Identity()

        self.at = at
        self.at.eval()
        self.use_latent_action_before_vq = use_latent_action_before_vq

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        self.at.set_normalizer(normalizer)

    def to(self, device):
        super().to(device)
        self.at.to(device)
        return self



    def predict_action(self,
                       obs_dict: Dict[str, torch.Tensor],
                       dataset_obs_temporal_downsample_ratio: int,
                       extended_obs_dict: Dict[str, torch.Tensor] = None,
                       return_latent_action=False
                       ) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict  # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            raise NotImplementedError

        # run sampling
        nlatent_sample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        # unnormalize prediction
        nlatent_sample = self.normalizer['latent_action'].unnormalize(nlatent_sample)

        # decode latent action
        if return_latent_action:
            action_pred = nlatent_sample
        else:
            action_pred = self.at.decode_from_latent(nlatent_sample)

        action_pred = action_pred.to(device)
        action = action_pred.clone()  # torch.Tensor

        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])

        action = batch['action']
        batch_size,T,action_dim = action.shape
        action = action.reshape(batch_size*self.at.encode_block_num, T//self.at.encode_block_num, action_dim)
        nlatent_actions = self.at.encode_to_latent(action,reshape_in_vae=False)
        nlatent_actions = nlatent_actions.reshape(batch_size, self.at.downsampled_input_h*self.at.encode_block_num, -1)
        nlatent_actions = self.normalizer['latent_action'].normalize(nlatent_actions)

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nlatent_actions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs,
                                   lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            raise NotImplementedError

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps,
                          local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
