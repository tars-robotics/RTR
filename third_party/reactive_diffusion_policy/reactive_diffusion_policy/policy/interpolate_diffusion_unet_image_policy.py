from typing import Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from reactive_diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from reactive_diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from reactive_diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder
from reactive_diffusion_policy.model.vae.model import VAE
from reactive_diffusion_policy.model.common.normalizer import LinearNormalizer
from reactive_diffusion_policy.common.pytorch_util import dict_apply


import torch
import torch.nn.functional as F

def torch_upsample_with_anchor(action_chunk: torch.Tensor, T: int, r: int) -> torch.Tensor:
    """
    action_chunk: [B, N, D]  (downsampled points)
    returns: [B, T, D]  (interpolated back to the original length); guarantees out[:, xp[k], :] == action_chunk[:, k, :]
    """
    device = action_chunk.device
    dtype = action_chunk.dtype

    B, N, D = action_chunk.shape

    # equivalent to np.arange(0, T, r)
    xp = torch.arange(0, T, r, device=device)  # [N] (should normally equal N)
    # optional: align xp length with N (guard against non-divisible boundaries)
    if xp.numel() != N:
        # truncate / align to action_chunk's N
        xp = xp[:N]

    if N == 1:
        # [B, 1, D] -> [B, T, D]
        return action_chunk.expand(B, T, D).clone()

    # 1) F.interpolate expects [batch, channels, length]
    #    treat D as channels and N as length
    x = action_chunk.permute(0, 2, 1).to(torch.float32)          # [B, D, N]
    y = F.interpolate(x, size=T, mode="linear", align_corners=True)  # [B, D, T]
    out = y.permute(0, 2, 1).to(dtype)                           # [B, T, D]

    # 2) overwrite the anchor positions so out[:, xp, :] exactly matches the input
    #    out[:, xp, :] has shape [B, N, D] and can be assigned directly
    out[:, xp, :] = action_chunk

    return out


# Train with the DP train_cfg. Key point: do NOT use the latent dataset.
class InterpolateDiffusionUnetImagePolicy(DiffusionUnetImagePolicy):
    def __init__(self,
                #  at: VAE,
                #  use_latent_action_before_vq: bool,
                 interpolate_ratio:int,
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

        # if at.use_vq:
        #     shape_meta['action']['shape'] = [at.n_latent_dims]
        # else:
        #     shape_meta['action']['shape'] = [at.n_embed]

        # original_horizon = horizon
        # if at.use_conv_encoder:
        #     horizon = at.downsampled_input_h
        #     if change_kernel_size:  # original behavior
        #         print("change_kernel_size"+"="*100)
        #         kernel_size = 3
        #     else:
        #         print("NOT change_kernel_size"+"="*100)
        # else:
        #     # hack: latent action can be viewed as a sequence of latent actions with horizon 1
        #     horizon = at.downsampled_input_h
        #     kernel_size = 1
        
        self.interpolate_ratio = interpolate_ratio
        assert horizon % interpolate_ratio == 0, "horizon must be multiple of interpolate_ratio"
        original_horizon = horizon
        horizon = original_horizon // self.interpolate_ratio
        print(f"original_horizon is {original_horizon}, horizon is {horizon}, interpolate_ratio is {interpolate_ratio}")

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
        self.horizon = horizon


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
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
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # interpolate back to original_horizon
        print(f"before interpolation action_pred.shape is {action_pred.shape}")
        action_pred = torch_upsample_with_anchor(action_pred, self.original_horizon, self.interpolate_ratio)
        print(f"after interpolation action_pred.shape is {action_pred.shape}")

        # get action
        # start = To - 1 is a hack: naively this would drop n_obs_steps actions, but we only want to drop n_obs_steps-1, because in the dataset action[0] for obs[0] is actually the next-step state.
        start = (To-1)*self.image_downsample_ratio
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,         # actions to be executed in the future
            'action_pred': action_pred,  # full predicted action sequence
        }
        return result



    # ========= training  ============
    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        
        # Debug: check action shape before normalization
        if hasattr(self, '_debug_count'):
            self._debug_count += 1
        else:
            self._debug_count = 0
        if self._debug_count == 0:
            print(f"[DEBUG] compute_loss: batch['action'].shape = {batch['action'].shape}")
            print(f"[DEBUG] compute_loss: action has NaN = {torch.isnan(batch['action']).any()}")
        
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        nactions = nactions[:,::self.interpolate_ratio]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

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
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
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
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
