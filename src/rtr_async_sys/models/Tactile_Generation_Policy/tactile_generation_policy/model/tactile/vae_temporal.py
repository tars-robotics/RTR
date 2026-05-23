# -*- coding: utf-8 -*-
# Tactile Encoder (Learned Local Only, No Gating)
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from einops import rearrange
tactile_generation_policy_root = os.environ.get("TACTILE_GENERATION_POLICY_ROOT")
if tactile_generation_policy_root:
    sys.path.append(tactile_generation_policy_root)
from typing import Literal, List
from tactile_generation_policy.model.tactile.causal_decoder import TactileDecoder, ImplicitTactileDecoder
from tactile_generation_policy.model.tactile.temporal_encoder import TactileEncoder


def make_coord(shape):
    yy, xx = torch.meshgrid(
        torch.arange(shape[0], dtype=torch.float32),
        torch.arange(shape[1], dtype=torch.float32),
        indexing='ij'
    )
    norm_x = 2.0 * xx / (shape[1] - 1) - 1.0  # [H, W]
    norm_y = 2.0 * yy / (shape[0] - 1) - 1.0  # [H, W]
    grid = torch.stack((norm_x, norm_y), dim=-1)  # [H, W, 2]
    vgrid = grid.unsqueeze(0)  # [1, H, W, 2]

    return vgrid

class TemporalTactileVAE(nn.Module):
    def __init__(
        self,
        input_dim: int=3, 
        latent_dim: int=16,
        hidden_state: int=64,
        time_compression_ratio: int=4,
        spatial_compression_ratio: int=4,
        decoder_mode: Literal["recon", "implicit"] = "implicit",
        sample_points: List=[128, 74],
    ):
        super().__init__()
        self.decoder_mode = decoder_mode
        self.sample_points = sample_points

        self.encoder = TactileEncoder(
                in_channels=input_dim,
                latent_dim=latent_dim,
                hidden_channels=hidden_state,
        )

        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)

        if self.decoder_mode == 'recon':
            self.decoder = TactileDecoder(
                    in_channels=latent_dim,
                    out_channels=input_dim,
                    hidden_channels=hidden_state,
                    time_compression_ratio=time_compression_ratio,
                    spatial_compression_ratio=spatial_compression_ratio,
            )
        elif self.decoder_mode == 'implicit':
            self.decoder = ImplicitTactileDecoder(
                    in_channels=latent_dim,
                    out_channels=input_dim,
                    hidden_channels=hidden_state,
                    time_compression_ratio=time_compression_ratio,
                    spatial_compression_ratio=spatial_compression_ratio,
            )
        for m in [self.fc_mu, self.fc_logvar]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def encode(self, batch):
        mu, logvar = self.encoder(batch)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, vgrid=None):
        if vgrid is not None:
            out = self.decoder(z, vgrid)
        else:
            out = self.decoder(z)
        return out

    def calculate_loss(self, batch, val=False):
        mu, logvar = self.encode(batch)
        z = self.reparameterize(mu, logvar)
        if self.decoder_mode == 'recon':
            recon = self.decode(z)
            loss, recon_loss, kl = self.loss_function(recon, batch, mu, logvar)
        elif self.decoder_mode == 'implicit':
            gt = list()
            if not val:
                vgrid = make_coord(shape=self.sample_points)
            else:
                vgrid = make_coord(shape=[36, 20])
            vgrid = vgrid.repeat(batch.shape[0], 1, 1, 1).to(z.device)
            recon = self.decode(z, vgrid)
            
            batch = rearrange(batch, 'b c t h w -> t b c h w')
            for i in range(batch.shape[0]):
                batch_i = batch[i, ...]
                gt_i = F.grid_sample(batch_i, vgrid, padding_mode='border', align_corners=True, mode='bilinear').permute(0, 2, 3, 1)
                gt.append(gt_i.reshape(gt_i.shape[0], -1, gt_i.shape[-1]).unsqueeze(1))
            gt = torch.cat(gt, dim=1).reshape(-1, recon.shape[1], recon.shape[2])
        
            loss, recon_loss, kl = self.loss_function(recon, gt, mu, logvar)
        return loss, recon_loss, kl
    
    def get_latent(self, batch):
        mu, logvar = self.encode(batch)
        z = self.reparameterize(mu, logvar)
        return z
    
    def get_recon(self, batch, implicit=False):
        mu, logvar = self.encode(batch)
        z = self.reparameterize(mu, logvar)
        if not implicit:
            recon = self.decode(z)
        else:
            vgrid = make_coord(shape=[36, 20])
            vgrid = vgrid.repeat(batch.shape[0], 1, 1, 1).to(z.device)
            recon = self.decode(z, vgrid).unsqueeze(0).permute(0, 3, 1, 2)
            B, C, T, L = recon.shape
            recon = recon.reshape(B, C, T, 36, 20)
        return recon
    
    @staticmethod
    def kl_divergence(mu, logvar):
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def loss_function(self, recon_x, x, mu, logvar, beta=1e-6):
        # x: [N, 3, H, W]
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kl = self.kl_divergence(mu, logvar)
        loss = recon_loss + beta * kl
        return loss, recon_loss, kl

if __name__ == "__main__":
    # self-test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    Nbatch = 2
    raw_tactile = torch.ones((2, 3, 8, 36, 20)).cuda()
    vae = TemporalTactileVAE(
        input_dim=3, 
        latent_dim=16,
        hidden_state=64,
        time_compression_ratio=4,
        spatial_compression_ratio=4,
        decoder_mode="condition",
    ).to(device)

    raw_tactile = vae.calculate_loss(raw_tactile)

    print('end')
