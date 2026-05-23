"""
PointNet-style encoder for point cloud observations.
Processes point cloud data [N, 3] -> 1024-dim feature vector.
"""

import torch
import torch.nn as nn
from typing import Dict


class PointNetEncoder(nn.Module):
    """
    PointNet-style encoder for point cloud data.
    
    Takes point cloud input [B, N, 3] and outputs global feature [B, output_dim].
    Uses MLP followed by max pooling aggregation.
    
    Args:
        input_dim: Input point dimension (default: 3 for xyz)
        output_dim: Output feature dimension (default: 1024)
        hidden_dims: Hidden layer dimensions (default: [64, 128, 1024])
    """
    
    def __init__(
        self, 
        input_dim: int = 3,
        output_dim: int = 1024,
        hidden_dims: list = None
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 128, 1024]
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build MLP layers
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ])
            in_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # If final MLP output != desired output_dim, add projection
        if hidden_dims[-1] != output_dim:
            self.projection = nn.Linear(hidden_dims[-1], output_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through PointNet encoder.
        
        Args:
            x: Point cloud tensor of shape (B, N, 3) where
               B = batch size
               N = number of points (e.g., 8192)
               3 = xyz coordinates
        
        Returns:
            Global feature tensor of shape (B, output_dim)
        """
        # Input shape: (B, N, 3)
        B, N, C = x.shape
        
        # Apply MLP to each point
        # Shape: (B, N, 3) -> (B, N, hidden_dims[-1])
        point_features = self.mlp(x)
        
        # Max pooling over all points to get global feature
        # Shape: (B, N, hidden_dims[-1]) -> (B, hidden_dims[-1])
        global_feature = torch.max(point_features, dim=1)[0]
        
        # Project to desired output dimension if needed
        # Shape: (B, hidden_dims[-1]) -> (B, output_dim)
        global_feature = self.projection(global_feature)
        
        return global_feature


class PointNetObsEncoder(nn.Module):
    """
    Observation encoder wrapper for PointNet.
    Compatible with the policy interface that expects shape_meta.
    """
    
    def __init__(
        self,
        shape_meta: Dict,
        output_dim: int = 1024,
        hidden_dims: list = None
    ):
        super().__init__()
        
        # Extract point cloud shape from shape_meta
        # Expecting shape_meta['obs']['point_cloud']['shape'] = [8192, 3]
        if 'point_cloud' in shape_meta['obs']:
            point_cloud_shape = shape_meta['obs']['point_cloud']['shape']
            n_points, input_dim = point_cloud_shape
        else:
            # Default fallback
            n_points = 8192
            input_dim = 3
        
        self.n_points = n_points
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Create PointNet encoder
        self.encoder = PointNetEncoder(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims
        )
    
    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode point cloud observation.
        
        Args:
            obs_dict: Dictionary containing 'point_cloud' key with tensor
                     of shape (B, N, 3) or (B, T, N, 3) where T is temporal dim
        
        Returns:
            Encoded features of shape (B, output_dim) or (B, T, output_dim)
        """
        global_pts = obs_dict['global_pts']
        
        # Handle temporal dimension if present
        # Shape: (B, T, N, 3) -> need to flatten B and T for processing
        if global_pts.ndim == 4:
            B, T, N, C = global_pts.shape
            # Reshape to (B*T, N, C)
            global_pts = global_pts.view(B * T, N, C)
            # Encode
            features = self.encoder(global_pts)
            # Reshape back to (B, T, output_dim)
            features = features.view(B, T, self.output_dim)
        else:
            # Shape: (B, N, 3)
            features = self.encoder(global_pts)
        
        return features
    
    @property
    def feature_dim(self) -> int:
        """Return output feature dimension."""
        return self.output_dim

    def output_shape(self):
        """
        Return the output shape as a tuple/list, compatible with other encoders.
        """
        return [self.output_dim]


if __name__ == "__main__":
    # Test the encoder
    batch_size = 4
    n_points = 8192
    
    # Create dummy point cloud data
    point_cloud = torch.randn(batch_size, n_points, 3)
    
    # Test PointNetEncoder
    print("Testing PointNetEncoder...")
    encoder = PointNetEncoder(input_dim=3, output_dim=1024)
    output = encoder(point_cloud)
    print(f"Input shape: {point_cloud.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: ({batch_size}, 1024)")
    assert output.shape == (batch_size, 1024), "Output shape mismatch!"
    
    # Test PointNetObsEncoder
    print("\nTesting PointNetObsEncoder...")
    shape_meta = {
        'obs': {
            'point_cloud': {
                'shape': [8192, 3]
            }
        }
    }
    obs_encoder = PointNetObsEncoder(shape_meta, output_dim=1024)
    obs_dict = {'point_cloud': point_cloud}
    output = obs_encoder(obs_dict)
    print(f"Input shape: {point_cloud.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Feature dim: {obs_encoder.feature_dim}")
    
    # Test with temporal dimension
    print("\nTesting with temporal dimension...")
    point_cloud_temporal = torch.randn(batch_size, 2, n_points, 3)  # (B, T, N, 3)
    obs_dict_temporal = {'point_cloud': point_cloud_temporal}
    output_temporal = obs_encoder(obs_dict_temporal)
    print(f"Input shape: {point_cloud_temporal.shape}")
    print(f"Output shape: {output_temporal.shape}")
    print(f"Expected: ({batch_size}, 2, 1024)")
    assert output_temporal.shape == (batch_size, 2, 1024), "Temporal output shape mismatch!"
    
    print("\nAll tests passed!")
# 1. PointNetEncoder
# Core encoder class:
#   Input: [B, 8192, 3] point-cloud data
#   Pipeline:
#     - MLP layers: 3 -> 64 -> 128 -> 1024
#     - ReLU after each layer
#     - max-pool over points to aggregate features
#   Output: [B, 1024] global feature vector
# 2. PointNetObsEncoder
# Observation-encoder wrapper:
#   - Compatible with the policy network interface
#   - Reads point-cloud config from shape_meta
#   - Supports a temporal dim [B, T, N, 3] -> [B, T, 1024]
#   - Exposes a feature_dim property that returns 1024