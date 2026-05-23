import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx

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
            padding=0,  # manual padding
            dilation=dilation
        )
    
    def forward(self, x):
        x = F.pad(x, (
            self.spatial_padding[1], self.spatial_padding[1],  # W
            self.spatial_padding[0], self.spatial_padding[0],  # H
            self.temporal_padding, 0                           # T (causal)
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
        
        return x + h  # residual connection

class CausalUpsample3d(nn.Module):
    def __init__(self, channels, scale_factor=(2, 2, 2)):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = CausalConv3d(channels, channels, kernel_size=3)
    
    def forward(self, x):
        B, C, T, H, W = x.shape
        x = F.interpolate(
            x, 
            size=(
                T * self.scale_factor[0],
                H * self.scale_factor[1],
                W * self.scale_factor[2]
            ),
            mode='trilinear',
            align_corners=False
        )
        # convolutional smoothing
        x = self.conv(x)
        return x

class TransposeUpsample3d(nn.Module):
    def __init__(self, channels, scale_factor=(2, 2, 2)):
        super().__init__()
        self.scale_factor = scale_factor
        self.deconv = nn.ConvTranspose3d(
            channels, channels, 
            kernel_size=scale_factor, 
            stride=scale_factor, 
            padding=0,
            output_padding=0,
            bias=False
        )
        self.conv = CausalConv3d(channels, channels, kernel_size=3)
    
    def forward(self, x):
        # x: [B, C, T, H, W]
        x = self.deconv(x)
        x = self.conv(x)
        return x

class TactileDecoder(nn.Module):
    """
    Minimal version: each layer has one residual block.
    Parameter count is less than 10% of the original version.
    """
    
    def __init__(
        self,
        in_channels=4,
        out_channels=3,
        hidden_channels=64,
        time_compression_ratio=4,
        spatial_compression_ratio=4,
    ):
        super().__init__()
        
        self.conv_in = CausalConv3d(in_channels, hidden_channels, kernel_size=3)
        self.mid_res = CausalResBlock3d(hidden_channels)
        num_spatial_up = int(math.log2(spatial_compression_ratio))
        num_time_up = int(math.log2(time_compression_ratio))
        
        self.up_layers = nn.ModuleList()
        
        for i in range(num_spatial_up):
            do_time_up = (i >= num_spatial_up - num_time_up)
            scale_factor = (2 if do_time_up else 1, 2, 2)
            
            self.up_layers.append(nn.ModuleList([
                CausalResBlock3d(hidden_channels),
                CausalUpsample3d(hidden_channels, scale_factor=scale_factor)
            ]))
        
        self.norm_out = nn.GroupNorm(8, hidden_channels, eps=1e-6)
        self.conv_out = CausalConv3d(hidden_channels, out_channels, kernel_size=3)
    
    def forward(self, x):
        h = self.conv_in(x)
        h = self.mid_res(h)
        
        for res_block, upsample in self.up_layers:
            h = res_block(h)
            h = upsample(h)
        
        h = self.norm_out(h)
        h = F.silu(h)
        out = self.conv_out(h)
        return out

class ImplicitTactileDecoder(nn.Module):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''
    def __init__(self,
                 in_channels=16,
                 out_channels=3,
                 hidden_channels=64,
                 time_compression_ratio=4,
                 spatial_compression_ratio=4,
                 n_blocks=3):
        super().__init__()
        self.in_channels = in_channels
        self.n_blocks = n_blocks

        self.conv_in = CausalConv3d(in_channels, hidden_channels, kernel_size=3)
        self.mid_res = CausalResBlock3d(hidden_channels)
        num_spatial_up = int(math.log2(spatial_compression_ratio))
        num_time_up = int(math.log2(time_compression_ratio))
        
        self.up_layers = nn.ModuleList()
        
        for i in range(num_spatial_up):
            do_time_up = (i >= num_spatial_up - num_time_up)
            scale_factor = (2 if do_time_up else 1, 2, 2)
            
            self.up_layers.append(nn.ModuleList([
                CausalResBlock3d(hidden_channels),
                TransposeUpsample3d(hidden_channels, scale_factor=scale_factor)
            ]))
        
        self.norm_out = nn.GroupNorm(8, hidden_channels, eps=1e-6)

        if in_channels != 0:
            self.fc_c = nn.Linear(hidden_channels, hidden_channels)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_channels) for i in range(n_blocks)
        ])
        self.fc_out = nn.Linear(hidden_channels, out_channels)
        self.fc_p = nn.Linear(2, hidden_channels)

    def sample_plane_feature(self, vgrid, c):
        c_up = list()
        for i in range(c.shape[2]):
            c_i = c[:, :, i, :, :]
            c_i = F.grid_sample(c_i, vgrid, padding_mode='border', align_corners=True, mode='bilinear').squeeze(-1)
            B, C, H, W = c_i.shape
            c_up.append(c_i.reshape(B, C, H*W).permute(0, 2, 1).unsqueeze(1))
        return c_up

    def forward(self, x, queries):
        h = self.conv_in(x)
        h = self.mid_res(h)
        
        for res_block, upsample in self.up_layers:
            h = res_block(h)
            h = upsample(h)
        
        h = self.norm_out(h)
        h = F.silu(h)

        if self.in_channels != 0:
            c = self.sample_plane_feature(queries, h)
            t = len(c)
            c = torch.cat(c, dim=1)
            c = c.reshape(-1, c.shape[2], c.shape[3])

        queries = queries.reshape(queries.shape[0], -1, 2).float()
        net = self.fc_p(queries).unsqueeze(1).repeat(1, t, 1, 1)
        B, T, L, C = net.shape
        net = net.reshape(B*T, L, C)

        if self.in_channels != 0:
            net = net + self.fc_c(c)
        for i in range(self.n_blocks):
            net = self.blocks[i](net)

        out = self.fc_out(F.relu(net))

        return out    
    

if __name__ == "__main__":
    # decoder = TactileDecoder(
    #     in_channels=16,
    #     out_channels=3,
    #     hidden_channels=64,
    #     time_compression_ratio=4,
    #     spatial_compression_ratio=4,
    # )
    # decoder.eval()
    
    # latent = torch.randn(1, 16, 2, 9, 5)  # compressed latent representation
    
    # output1, h1 = decoder(latent)
    # print(f"Input: {latent.shape}")
    # print(f"Output: {output1.shape}")
    # print(f"parameter count: {sum(p.numel() for p in decoder.parameters()) / 1e6:.2f}M")
    
    # print("=== Validate causality ===")
    # latent_modified = latent.clone()
    # latent_modified[:, :, -1:, :, :] += 100  # modify the last frame
    # output_modified, h2 = decoder(latent_modified)
    # diff = (output1[:, :, :-1] - output_modified[:, :, :-1]).abs().max()
    # diff2 = (h1[:, :, :-1] - h2[:, :, :-1]).abs().max()

    # print(f"maximum difference in earlier frames: {diff.item():.6f}, {diff2.item():.6f}")
    # print(f"Causality check: {'passed' if diff < 1e-5 else 'failed'}")

    decoder = ImplicitTactileDecoder(
                 dim=2, 
                 in_channels=16,
                 out_channels=3,
                 hidden_channels=64)
    latent = torch.randn(2, 16, 2, 9, 5)
    output = decoder(latent)