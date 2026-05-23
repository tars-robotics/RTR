import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


class CausalConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        
        self.spatial_padding = (
            (kernel_size[1] - 1) * dilation[1] // 2,
            (kernel_size[2] - 1) * dilation[2] // 2
        )
        
        self.temporal_padding = (kernel_size[0] - 1) * dilation[0]
        
        self.conv = nn.Conv3d(
            in_channels, out_channels, 
            kernel_size, stride, 
            padding=0,
            dilation=dilation
        )
    
    def forward(self, x):
        x = F.pad(x, (
            self.spatial_padding[1], self.spatial_padding[1],  # W
            self.spatial_padding[0], self.spatial_padding[0],  # H
            self.temporal_padding, 0                           # T
        ))
        return self.conv(x)


class CausalResBlock3d(nn.Module):
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=channels, eps=1e-6)
        self.conv1 = CausalConv3d(channels, channels, kernel_size=3)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=channels, eps=1e-6)
        self.conv2 = CausalConv3d(channels, channels, kernel_size=3)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.SiLU()
    
    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = self.act(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return x + h


class CausalDownsample3d(nn.Module):
    def __init__(self, channels, scale_factor=(2, 2, 2)):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = CausalConv3d(
            channels, channels, 
            kernel_size=3, 
            stride=scale_factor
        )
    
    def forward(self, x):
        return self.conv(x)


class TactileEncoder(nn.Module):
    """
    Input: [B, 3, T, H, W]
    Output: [B, 4, T//4, H//4, W//4] (mean), 
          [B, 4, T//4, H//4, W//4] (logvar)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 16,
        hidden_channels: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.conv_in = CausalConv3d(in_channels, hidden_channels, kernel_size=3)
        
        self.down1_res = CausalResBlock3d(hidden_channels, dropout)
        self.down1 = CausalDownsample3d(hidden_channels, scale_factor=(2, 2, 2))
        
        self.down2_res = CausalResBlock3d(hidden_channels, dropout)
        self.down2 = CausalDownsample3d(hidden_channels, scale_factor=(2, 2, 2))
        
        self.mid_res = CausalResBlock3d(hidden_channels, dropout)
        
        self.norm_out = nn.GroupNorm(8, hidden_channels, eps=1e-6)
        self.conv_out = CausalConv3d(hidden_channels, latent_dim * 2, kernel_size=3)
    
    def forward(self, x):
        # Input: [B, 3, T, H, W]
        h = self.conv_in(x)          # [B, 64, T, H, W]
        h = self.down1_res(h)        # [B, 64, T, H, W]
        h = self.down1(h)            # [B, 64, T/2, H/2, W/2]
        h = self.down2_res(h)        # [B, 64, T/2, H/2, W/2]
        h = self.down2(h)            # [B, 64, T/4, H/4, W/4]
        h = self.mid_res(h)          # [B, 64, T/4, H/4, W/4]
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)         # [B, 32, T/4, H/4, W/4]
        mean, logvar = torch.chunk(h, 2, dim=1)  # [B, 16, T/4, H/4, W/4]
        
        return mean, logvar


if __name__ == "__main__":
    encoder = TactileEncoder(
        in_channels=3,
        latent_dim=16,
        hidden_channels=64,
    )

    x = torch.randn(2, 3, 8, 36, 20)
    
    # Encode
    mean, logvar = encoder(x)
    
    # Compute parameter count
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nTotal parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    
    with torch.no_grad():
        mean1, logvar1 = encoder(x)
        
        x_modified = x.clone()
        x_modified[:, :, -1:] += 10  # modify the latter half
        
        mean2, logvar2 = encoder(x_modified)
        
        mean_diff = (mean1[:, :, :4] - mean2[:, :, :4]).abs().max()
        logvar_diff = (logvar1[:, :, :4] - logvar2[:, :, :4]).abs().max()
        
        is_causal_mean = mean_diff.item() < 1e-5
        is_causal_logvar = logvar_diff.item() < 1e-5
        
        print(f"Mean causality: {'passed' if is_causal_mean else 'failed'}")
        print(f"  maximum difference: {mean_diff.item():.6e}")
        print(f"Logvar causality: {'passed' if is_causal_logvar else 'failed'}")
        print(f"  maximum difference: {logvar_diff.item():.6e}")