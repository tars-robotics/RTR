import math

import einops
import torch
import torch.nn as nn

from reactive_diffusion_policy.model.common.normalizer import LinearNormalizer
from reactive_diffusion_policy.model.vae.distributions import DiagonalGaussianDistribution


def _group_count(channels):
    for group_count in (8, 4, 2, 1):
        if channels % group_count == 0:
            return group_count
    return 1


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=5, dilation=1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        groups = _group_count(channels)
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(groups, channels),
            nn.Mish(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(groups, channels),
        )
        self.activation = nn.Mish()

    def forward(self, x):
        return self.activation(x + self.net(x))


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        action_dim,
        hidden_dim,
        latent_dim,
        downsample_factor,
        blocks_per_level,
        kernel_size,
    ):
        super().__init__()
        downsample_levels = int(math.log2(downsample_factor))
        layers = [
            nn.Conv1d(action_dim, hidden_dim, 1),
            nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
            nn.Mish(),
        ]
        for _ in range(downsample_levels):
            for block_idx in range(blocks_per_level):
                layers.append(
                    TemporalResidualBlock(
                        hidden_dim,
                        kernel_size=kernel_size,
                        dilation=2 ** block_idx,
                    )
                )
            layers.extend([
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
                nn.Mish(),
            ])
        for block_idx in range(blocks_per_level):
            layers.append(
                TemporalResidualBlock(
                    hidden_dim,
                    kernel_size=kernel_size,
                    dilation=2 ** block_idx,
                )
            )
        layers.append(nn.Conv1d(hidden_dim, latent_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, T, A]
        x = einops.rearrange(x, "b t a -> b a t")
        x = self.net(x)
        return einops.rearrange(x, "b c t -> b t c")


class TemporalDecoder(nn.Module):
    def __init__(
        self,
        action_dim,
        hidden_dim,
        latent_dim,
        downsample_factor,
        blocks_per_level,
        kernel_size,
    ):
        super().__init__()
        upsample_levels = int(math.log2(downsample_factor))
        layers = [
            nn.Conv1d(latent_dim, hidden_dim, 1),
            nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
            nn.Mish(),
        ]
        for _ in range(upsample_levels):
            for block_idx in range(blocks_per_level):
                layers.append(
                    TemporalResidualBlock(
                        hidden_dim,
                        kernel_size=kernel_size,
                        dilation=2 ** block_idx,
                    )
                )
            layers.extend([
                nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
                nn.Mish(),
            ])
        for block_idx in range(blocks_per_level):
            layers.append(
                TemporalResidualBlock(
                    hidden_dim,
                    kernel_size=kernel_size,
                    dilation=2 ** block_idx,
                )
            )
        layers.append(nn.Conv1d(hidden_dim, action_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, horizon):
        # x: [B, T_latent, C]
        x = einops.rearrange(x, "b t c -> b c t")
        x = self.net(x)
        if x.shape[-1] > horizon:
            x = x[..., :horizon]
        elif x.shape[-1] < horizon:
            pad = horizon - x.shape[-1]
            x = torch.nn.functional.pad(x, (0, pad))
        return einops.rearrange(x, "b a t -> b t a")


class TemporalConvVAE:
    def __init__(
        self,
        horizon=48,
        shape_meta=None,
        n_latent_dims=32,
        hidden_dim=128,
        downsample_factor=4,
        blocks_per_level=3,
        kernel_size=5,
        kl_multiplier=1e-6,
        n_embed=32,
        eval=True,
        device="cuda",
        load_dir=None,
        encoder_loss_multiplier=1.0,
        act_scale=1.0,
        **kwargs,
    ):
        if shape_meta is None:
            shape_meta = {}
        if downsample_factor < 1 or downsample_factor & (downsample_factor - 1) != 0:
            raise ValueError("downsample_factor must be a power of two")
        if horizon % downsample_factor != 0:
            raise ValueError("horizon must be divisible by downsample_factor")

        self.input_dim_h = horizon
        self.input_dim_w = shape_meta["action"]["shape"][0]
        self.n_latent_dims = n_latent_dims
        self.n_embed = n_embed
        self.downsample_factor = downsample_factor
        self.downsampled_input_h = horizon // downsample_factor
        self.use_vq = False
        self.use_conv_encoder = True
        self.use_rnn_decoder = False
        self.kl_multiplier = kl_multiplier
        self.encoder_loss_multiplier = encoder_loss_multiplier
        self.act_scale = act_scale
        self.device = device
        self.embedding_dim = n_latent_dims

        self.normalizer = LinearNormalizer()
        self.encoder = TemporalEncoder(
            action_dim=self.input_dim_w,
            hidden_dim=hidden_dim,
            latent_dim=n_latent_dims,
            downsample_factor=downsample_factor,
            blocks_per_level=blocks_per_level,
            kernel_size=kernel_size,
        ).to(device)
        self.quant = nn.Conv1d(n_latent_dims, 2 * n_embed, 1).to(device)
        self.post_quant = nn.Conv1d(n_embed, n_latent_dims, 1).to(device)
        self.decoder = TemporalDecoder(
            action_dim=self.input_dim_w,
            hidden_dim=hidden_dim,
            latent_dim=n_latent_dims,
            downsample_factor=downsample_factor,
            blocks_per_level=blocks_per_level,
            kernel_size=kernel_size,
        ).to(device)

        self.optim_params = (
            list(self.encoder.parameters())
            + list(self.quant.parameters())
            + list(self.post_quant.parameters())
            + list(self.decoder.parameters())
        )

        if load_dir is not None:
            try:
                state_dict = torch.load(load_dir)
            except RuntimeError:
                state_dict = torch.load(load_dir, map_location=torch.device("cpu"))
            if "state_dicts" in state_dict:
                state_dict = state_dict["state_dicts"]["model"]
            self.load_state_dict(state_dict)

        if eval:
            self.eval()
        else:
            self.train()

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _normalize_action(self, action):
        action = self.normalizer["action"].normalize(action)
        action = action / self.act_scale
        return action.to(self.device)

    def _encode_distribution(self, action):
        state_rep = self.encoder(action)
        state_rep = einops.rearrange(state_rep, "b t c -> b c t")
        moments = self.quant(state_rep)
        return DiagonalGaussianDistribution(moments)

    def _posterior_latent(self, posterior, sample):
        latent = posterior.sample() if sample else posterior.mode()
        return einops.rearrange(latent, "b c t -> b t c")

    def _postprocess_latent(self, latent):
        latent = einops.rearrange(latent, "b t c -> b c t")
        latent = self.post_quant(latent)
        return einops.rearrange(latent, "b c t -> b t c")

    def _as_temporal_latent(self, latent, channels):
        if latent.ndim == 2:
            return einops.rearrange(latent, "b (t c) -> b t c", t=self.downsampled_input_h, c=channels)
        return latent

    def _decode_normalized(self, latent):
        latent = self._as_temporal_latent(latent, self.n_embed)
        latent = self._postprocess_latent(latent)
        return self._decode_processed_normalized(latent)

    def _decode_processed_normalized(self, latent):
        latent = self._as_temporal_latent(latent, self.n_latent_dims)
        return self.decoder(latent, self.input_dim_h)

    def preprocess(self, state):
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, device=self.device)
        return state.to(self.device)

    def quant_state_without_vq(self, state):
        state = self._as_temporal_latent(state, self.n_latent_dims)
        state = einops.rearrange(state, "b t c -> b c t")
        moments = self.quant(state)
        posterior = DiagonalGaussianDistribution(moments)
        state_vq = posterior.sample()
        state_vq = einops.rearrange(state_vq, "b c t -> b (t c)")
        return state_vq, posterior

    def postprocess_quant_state_without_vq(self, state_vq):
        state_vq = self._as_temporal_latent(state_vq, self.n_embed)
        state_vq = self._postprocess_latent(state_vq)
        return einops.rearrange(state_vq, "b t c -> b (t c)")

    def compute_loss_and_metric(self, batch):
        state = self._normalize_action(batch["action"])
        posterior = self._encode_distribution(state)
        latent = self._posterior_latent(posterior, sample=True)
        dec_out = self._decode_normalized(latent)

        encoder_loss = (state - dec_out).abs().mean()
        vae_recon_loss = torch.nn.functional.mse_loss(state, dec_out)
        kl_loss = posterior.kl().mean()
        rep_loss = encoder_loss * self.encoder_loss_multiplier + kl_loss * self.kl_multiplier

        return {
            "loss": rep_loss,
            "encoder_loss": encoder_loss.item(),
            "vae_recon_loss": vae_recon_loss.item(),
            "kl_loss": kl_loss.item(),
            "rep_loss": rep_loss.item(),
        }

    def encode_to_latent(self, batch):
        if isinstance(batch, dict):
            action = batch["action"]
        else:
            action = batch
        action = self._normalize_action(action)
        posterior = self._encode_distribution(action)
        return self._posterior_latent(posterior, sample=False)

    def decode_from_latent(self, action):
        dec_out = self._decode_normalized(action.to(self.device))
        dec_out = dec_out * self.act_scale
        return self.normalizer["action"].unnormalize(dec_out)

    def get_action_from_latent(self, latent):
        dec_out = self._decode_processed_normalized(latent.to(self.device))
        return dec_out * self.act_scale

    def encode_then_decode(self, batch):
        latent = self.encode_to_latent(batch)
        return self.decode_from_latent(latent)

    def eval(self):
        self.encoder.eval()
        self.quant.eval()
        self.post_quant.eval()
        self.decoder.eval()

    def train(self):
        self.encoder.train()
        self.quant.train()
        self.post_quant.train()
        self.decoder.train()

    def to(self, device):
        self.encoder.to(device)
        self.quant.to(device)
        self.post_quant.to(device)
        self.decoder.to(device)
        self.device = device

    def state_dict(self):
        return {
            "encoder": self.encoder.state_dict(),
            "quant": self.quant.state_dict(),
            "post_quant": self.post_quant.state_dict(),
            "decoder": self.decoder.state_dict(),
            "normalizer": self.normalizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict["encoder"])
        self.quant.load_state_dict(state_dict["quant"])
        self.post_quant.load_state_dict(state_dict["post_quant"])
        self.decoder.load_state_dict(state_dict["decoder"])
        self.normalizer.load_state_dict(state_dict["normalizer"])
