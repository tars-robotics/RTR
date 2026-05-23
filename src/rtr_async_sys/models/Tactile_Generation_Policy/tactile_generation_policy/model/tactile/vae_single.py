# -*- coding: utf-8 -*-
# Tactile Encoder (Learned Local Only, No Gating)
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from typing import Tuple, Literal, Optional
    

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True, pool_type='max', k=3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=1, padding=k//2, bias=True)
        self.act = nn.SiLU(inplace=True)
        if pool:
            if pool_type == 'max':
                self.pool = nn.MaxPool2d(2, 2)
            else:
                self.pool = nn.AvgPool2d(2, 2)
        else:
            self.pool = nn.Identity()

    def forward(self, x):
        x = self.act(self.conv(x))
        x = self.pool(x)
        return x

class FeatureExtractor(nn.Module):
    def __init__(self, in_ch: int, base_ch: int = 16, conv_layers: int = 1):
        super().__init__()
        assert conv_layers in (0, 1, 2)
        layers = []
        c_in = in_ch
        c_out = base_ch
        if conv_layers >= 1:
            layers.append(ConvBlock(c_in, c_out, k=3))  # ~ H/2, W/2
            c_in = c_out
        if conv_layers == 2:
            c_out = base_ch * 2
            layers.append(ConvBlock(c_in, c_out, k=3))  # ~ H/4, W/4
            c_in = c_out
        self.net = nn.Sequential(*layers)
        self.out_ch = c_in
        self.conv_layers = conv_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv_layers == 0:
            return x  # not recommended conv_layers=0,unless subsequent learning should run directly on raw images(usually not recommended)
        return self.net(x)

class TactileEncoder(nn.Module):
    def __init__(
        self,
        H: int = 36,
        W: int = 20,
        base_ch: int = 16,
        conv_layers_normal: int = 1,
        conv_layers_tangent: int = 1,
        tangent_input_mode: str = "dir+mag",  # "dir+mag" use [Ux,Uy,Mt];"raw" use [Tx,Ty]
        eps: float = 1e-8,
    ):
        super().__init__()
        assert tangent_input_mode in ("dir+mag", "raw")
        self.H, self.W = H, W
        self.eps = eps
        self.tangent_input_mode = tangent_input_mode
        self.add_global = False

        # Backbones(only produces learned local features)
        n_in_ch = 1
        t_in_ch = 3 if tangent_input_mode == "dir+mag" else 2
        self.extractor_n = FeatureExtractor(in_ch=n_in_ch, base_ch=base_ch, conv_layers=conv_layers_normal)
        self.extractor_t = FeatureExtractor(in_ch=t_in_ch, base_ch=base_ch, conv_layers=conv_layers_tangent)

    @staticmethod
    def _grid_pool_learned(feat: torch.Tensor, g: int) -> torch.Tensor:
        pooled = F.adaptive_max_pool2d(feat, (g, g))
        return pooled.flatten(1)

    def _prep_tangent(self, tangent: torch.Tensor) -> torch.Tensor:
        if self.tangent_input_mode == "raw":
            return tangent  # (N,2,H,W) -> [Tx, Ty]
        # "dir+mag": [Ux, Uy, Mt]
        Tx, Ty = tangent[:, 0:1], tangent[:, 1:2]
        Mt = torch.sqrt(Tx * Tx + Ty * Ty + self.eps)
        Ux = Tx / (Mt + self.eps)
        Uy = Ty / (Mt + self.eps)
        return torch.cat([Ux, Uy, Mt], dim=1)

    def forward(self, normal: torch.Tensor, tangent: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Input:
          - normal:  (N,1,H,W)
          - tangent: (N,2,H,W) -> (Tx, Ty)
        Output:
          - features: (N, branch_dim_n + branch_dim_t),= concat([normal_local_feat, tangent_local_feat])
          - aux: intermediate features, including learned feature maps and local grids
        Note:
          - Local features are not averaged from raw data; they come from grid pooling over convolutional feature maps
        """
        assert normal.dim() == 4 and normal.size(1) == 1, "normal expects (N,1,H,W)"
        assert tangent.dim() == 4 and tangent.size(1) == 2, "tangent expects (N,2,H,W)"
        N, _, H, W = normal.shape
        assert (H, W) == (self.H, self.W), f"Expect input size {(self.H, self.W)}, got {(H, W)}"
        
        # learned feature maps
        Fn = self.extractor_n(normal)         # (N, Cn, h, w)
        T_in = self._prep_tangent(tangent)   # (N, Ct_in, H, W)
        Ft = self.extractor_t(T_in)           # (N, Ct, h, w)
        B, C, h, w = Fn.shape

        # grid-pool to g×g(learned features only)
        Fn_g = self._grid_pool_learned(Fn, 1).squeeze(-1)  # (N, Cn*g*g)
        Ft_g = self._grid_pool_learned(Ft, 1).squeeze(-1)  # (N, Ct*g*g)

        if self.add_global:
            Fn_c = Fn.view(B, C, h * w) + Fn_g.view(B, C, 1).expand(-1, -1, h * w)  # [B, C, L]
            Ft_c = Ft.view(B, C, h * w) + Ft_g.view(B, C, 1).expand(-1, -1, h * w)  # [B, C, L]
            Fn_c = Fn_c.permute(0, 2, 1)  # [B, L, C]
            Ft_c = Ft_c.permute(0, 2, 1)  # [B, L, C]
        else:
            Fn_c = Fn.view(B, C, h * w).permute(0, 2, 1)  # [B, L, C]
            Ft_c = Ft.view(B, C, h * w).permute(0, 2, 1)  # [B, L, C]

        return Fn_c, Ft_c

class TacDecoder(nn.Module):
    def __init__(self, h, w, out_H, out_W, hidden_dim=64):
        super().__init__()
        self.h, self.w = h, w
        self.out_H, self.out_W = out_H, out_W
        self.conv1 = nn.Conv2d(64, hidden_dim, 3, padding=1)
        self.up1 = nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, 2, 1)
        self.up2 = nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, 4, 2, 1)
        self.final = nn.Conv2d(hidden_dim // 4, 3, 3, padding=1)
    
    def forward(self, z):
        B = z.size(0)
        # z: [B, 2C, L] -> [B, 2C, h, w]
        x = z.view(B, -1, self.h, self.w)
        x = F.relu(self.conv1(x))
        x = F.relu(self.up1(x))
        x = F.relu(self.up2(x))
        x = self.final(x)  # [B, 3, H, W]
        if x.shape[-2:] != (self.out_H, self.out_W):
            x = F.interpolate(x, size=(self.out_H, self.out_W), mode='bilinear', align_corners=False)
        return x

class NormalDecoder(nn.Module):
    """
    z_n: (B, C, L), C=h*w, L=latent_dim
    Output: (B, 1, H, W)
    """
    def __init__(self, latent_dim: int, h: int, w: int, out_H: int, out_W: int):
        super().__init__()
        self.h, self.w = h, w
        self.out_H, self.out_W = out_H, out_W
        self.init_conv = nn.Conv2d(latent_dim, 64, kernel_size=3, padding=1)
        self.upsample1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.upsample2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.final_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, z_n: torch.Tensor) -> torch.Tensor:
        B = z_n.size(0)
        x = z_n.permute(0, 2, 1).view(B, -1, self.h, self.w)
        x = F.relu(self.init_conv(x))
        x = F.relu(self.upsample1(x))         # (B, 32, h*2, w*2)
        x = F.relu(self.upsample2(x))         # (B, 16, h*4, w*4) ≈ (B, 16, H, W)
        x = self.final_conv(x)                # (B, 1, H, W)
        if x.shape[-2:] != (self.out_H, self.out_W):
                x = F.interpolate(x, size=(self.out_H, self.out_W), mode='bilinear', align_corners=False)

        return x

class TangentDecoderCond(nn.Module):
    """
    z_t, z_n: [B, C, L], C=h*w, L=latent_dim
    Output: (B, 2, H, W)
    """
    def __init__(
        self,
        latent_dim_t: int,
        cond_dim_n: int,
        h: int = 9, 
        w: int = 5,
        out_H: int = 36, 
        out_W: int = 20,
        hidden_dim: int = 64,
        mode: Literal["concat", "film"] = "film"
    ):
        super().__init__()
        self.h, self.w = h, w
        self.out_H, self.out_W = out_H, out_W
        self.mode = mode

        # backbone feature mapping
        self.main_conv = nn.Conv2d(latent_dim_t, hidden_dim, 3, padding=1)
        # upsample
        self.upsample1 = nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, 2, 1)
        self.upsample2 = nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, 4, 2, 1)
        self.final_conv = nn.Conv2d(hidden_dim // 4, 2, 3, padding=1)

        if mode == "film":
            # Conditioning branch: FiLM parameters gamma and beta
            self.film_conv = nn.Conv2d(cond_dim_n, hidden_dim, 1)
            self.film_gamma = nn.Conv2d(hidden_dim, hidden_dim, 1)
            self.film_beta  = nn.Conv2d(hidden_dim, hidden_dim, 1)
        elif mode == "concat":
            # Simple concatenation of z_t and z_n
            self.concat_conv = nn.Conv2d(latent_dim_t + cond_dim_n, hidden_dim, 3, padding=1)
        else:
            raise ValueError("mode must be 'concat' or 'film'")

    def forward(self, z_t: torch.Tensor, z_n: torch.Tensor):
        B = z_t.shape[0]
        h, w = self.h, self.w

        # [B, L, C] -> [B, C, h, w]
        zt = z_t.permute(0,2,1).contiguous().view(B, -1, h, w)  # [B, latent_dim_t, h, w]
        zn = z_n.permute(0,2,1).contiguous().view(B, -1, h, w)  # [B, cond_dim_n, h, w]

        if self.mode == "film":
            x_main = F.relu(self.main_conv(zt))    # [B, hidden_dim, h, w]
            cond   = F.relu(self.film_conv(zn))    # [B, hidden_dim, h, w]
            gamma  = torch.tanh(self.film_gamma(cond))
            beta   = torch.tanh(self.film_beta(cond))
            x = x_main * (1 + gamma) + beta        # FiLM modulation
        else:
            x = torch.cat([zt, zn], dim=1)         # [B, latent_dim_t + cond_dim_n, h, w]
            x = F.relu(self.concat_conv(x))        # [B, hidden_dim, h, w]

        # upsample to original size
        x = F.relu(self.upsample1(x))              # (B, hidden_dim//2, h*2, w*2)
        x = F.relu(self.upsample2(x))              # (B, hidden_dim//4, h*4, w*4) ≈ (B, *, H, W)
        x = self.final_conv(x)                     # (B, 2, H, W)

        # interpolate if there is a small size mismatch
        if x.shape[-2:] != (self.out_H, self.out_W):
            x = F.interpolate(x, size=(self.out_H, self.out_W), mode='bilinear', align_corners=False)
        return x


class TactileVAE(nn.Module):
    def __init__(
        self, 
        base_ch: int = 16,
        conv_layers_normal: int = 2,
        conv_layers_tangent: int = 2,
        latent_dim: int = 64,
        out_H: int = 36,
        out_W: int = 20,
        decoder_mode: Literal["one", "condition"] = "condition",
    ):
        super().__init__()
        self.encoder = TactileEncoder(
                            H=out_H, 
                            W=out_W,
                            base_ch=base_ch,
                            conv_layers_normal=conv_layers_normal,
                            conv_layers_tangent=conv_layers_tangent,
                            tangent_input_mode="dir+mag",
                        )
        self.latent_dim = latent_dim
        self.output_H = out_H
        self.output_W = out_W
        self.decoder_mode = decoder_mode

        feat_dim_n = base_ch * 2 ** (conv_layers_normal - 1)
        feat_dim_t = base_ch * 2 ** (conv_layers_tangent - 1)

        self.fuse = nn.Sequential(
                nn.Linear(feat_dim_n + feat_dim_t, feat_dim_n + feat_dim_t),
                nn.SiLU(),
            )

        self.fc_mu = nn.Linear(feat_dim_n + feat_dim_t, latent_dim)
        self.fc_logvar = nn.Linear(feat_dim_n + feat_dim_t, latent_dim)

        # decoder
        hidden_dim = max(256, latent_dim*4)
        self.h = out_H // 2 ** conv_layers_normal
        self.w = out_W // 2 ** conv_layers_normal

        if decoder_mode == "one":
            self.decoder = TacDecoder(self.h, self.w, out_H, out_W)
        elif decoder_mode == "condition":
            self.normal_decoder = NormalDecoder(latent_dim // 2, self.h, self.w, out_H, out_W)
            self.tangent_decoder = TangentDecoderCond(
                latent_dim_t=latent_dim // 2,
                cond_dim_n=latent_dim // 2,
                h=self.h,
                w=self.w,
                hidden_dim=hidden_dim,
                out_H=out_H,
                out_W=out_W,
                mode="film",
            )

        for m in [self.fc_mu, self.fc_logvar]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def encode(self, normal, tangent):
        normal = torch.cat([normal, normal[:, :, -1:, :]], dim=-2)
        tangent = torch.cat([tangent, tangent[:, :, -1:, :]], dim=-2)
        fnc, ftc = self.encoder(normal, tangent)
        fc = torch.cat([fnc, ftc], dim=-1)
        fc = self.fuse(fc)

        mu = self.fc_mu(fc)
        logvar = self.fc_logvar(fc)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        if self.decoder_mode == "one":
            out = self.decoder(z)
        elif self.decoder_mode == "condition":
            z_n, z_t = z.split([self.latent_dim // 2, self.latent_dim // 2], dim=1)
            z_n = z_n.permute(0, 2, 1)
            z_t = z_t.permute(0, 2, 1)
            Fn = self.normal_decoder(z_n)
            Ft = self.tangent_decoder(z_t, z_n)
            out = torch.cat([Ft, Fn], dim=1)
        out = out[:, :, :-1, :]

        return out

    def calculate_loss(self, batch):
        tangent = batch[:, 0:2, :, :]
        normal = batch[:, 2:3, :, :]
        mu, logvar = self.encode(normal, tangent)
        z = self.reparameterize(mu, logvar).permute(0, 2, 1)
        recon = self.decode(z)
        loss, recon_loss, kl = self.loss_function(recon, torch.cat([tangent, normal], dim=1), mu, logvar)
        
        return loss, recon_loss, kl
    
    def get_latent(self, batch):
        tangent = batch[:, :2, :, :]
        normal = batch[:, 2:, :, :]
        mu, logvar = self.encode(normal, tangent)
        z = self.reparameterize(mu, logvar).permute(0, 2, 1)
        z = z.view(z.shape[0], z.shape[1], self.h, self.w)

        return z
    
    @staticmethod
    def kl_divergence(mu, logvar):
        # KL(N(mu, sigma) || N(0, 1))
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
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
    normal = torch.rand(Nbatch, 1, 35, 20, device=device)
    tangent = torch.randn(Nbatch, 2, 35, 20, device=device)

    # Direction-sensitive version (default): tangential force input is [Ux, Uy, Mt]
    model = TactileVAE(
        out_H=36,
        out_W=20,
        base_ch=16,
        conv_layers_normal=2,
        conv_layers_tangent=2,
        decoder_mode="one",
    ).to(device)

    model.eval()
    with torch.no_grad():
        Fn, Ft = model(normal, tangent)