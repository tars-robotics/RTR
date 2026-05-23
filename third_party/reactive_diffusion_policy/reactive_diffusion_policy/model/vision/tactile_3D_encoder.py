import torch
import torch.nn as nn
import sys
import os
from typing import Dict
import torch.nn.functional as F

sys.path.append(os.path.dirname(__file__))
from pointnet2_utils.pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG
import pointnet2_utils.pointnet2.pytorch_utils as pt_utils


pointnet2_params = 'light'

MSG_CFG = {
    'NPOINTS': [512, 256, 128, 64],
    'RADIUS': [[0.01, 0.02], [0.02, 0.04], [0.04, 0.08], [0.08, 0.16]],
    'NSAMPLE': [[16, 32], [16, 32], [16, 32], [16, 32]],
    'MLPS': [[[16, 16, 32], [32, 32, 64]], 
             [[64, 64, 128], [64, 96, 128]],
             [[128, 196, 256], [128, 196, 256]], 
             [[256, 256, 512], [256, 384, 512]]],
    'FP_MLPS': [[64, 64], [128, 128], [256, 256], [512, 512]],
    'CLS_FC': [128],
    'DP_RATIO': 0.5,
}

ClsMSG_CFG = {
    'NPOINTS': [512, 256, 128, 64, None],
    'RADIUS': [[0.01, 0.02], [0.02, 0.04], [0.04, 0.08], [0.08, 0.16], [None, None]],
    'NSAMPLE': [[16, 32], [16, 32], [16, 32], [16, 32], [None, None]],
    'MLPS': [[[16, 16, 32], [32, 32, 64]], 
             [[64, 64, 128], [64, 96, 128]],
             [[128, 196, 256], [128, 196, 256]], 
             [[256, 256, 512], [256, 384, 512]],
             [[512, 512], [512, 512]]],
    'DP_RATIO': 0.5,
}

ClsMSG_CFG_Dense = {
    'NPOINTS': [512, 256, 128, None],
    'RADIUS': [[0.02, 0.04], [0.04, 0.08], [0.08, 0.16], [None, None]],
    'NSAMPLE': [[32, 64], [16, 32], [8, 16], [None, None]],
    'MLPS': [[[16, 16, 32], [32, 32, 64]],
             [[64, 64, 128], [64, 96, 128]],
             [[128, 196, 256], [128, 196, 256]], 
             [[256, 256, 512], [256, 384, 512]]],
    'DP_RATIO': 0.5,
}


########## Best before 29th April ###########
ClsMSG_CFG_Light = {
    'NPOINTS': [512, 256, 128, None],
    'RADIUS': [[0.02, 0.04], [0.04, 0.08], [0.08, 0.16], [None, None]],
    'NSAMPLE': [[16, 32], [16, 32], [16, 32], [None, None]],
    'MLPS': [[[16, 16, 32], [32, 32, 64]],
             [[64, 64, 128], [64, 96, 128]],
             [[128, 196, 256], [128, 196, 256]], 
             [[256, 256, 512], [256, 384, 512]]],
    'DP_RATIO': 0.5,
}


ClsMSG_CFG_Lighter= {
    'NPOINTS': [512, 256, 128, 64, None],
    'RADIUS': [[0.01], [0.02], [0.04], [0.08], [None]],
    'NSAMPLE': [[64], [32], [16], [8], [None]],
    'MLPS': [[[32, 32, 64]],
             [[64, 64, 128]],
             [[128, 196, 256]],
             [[256, 256, 512]],
             [[512, 512, 1024]]],
    'DP_RATIO': 0.5,
}

if pointnet2_params == 'light':
    SELECTED_PARAMS = ClsMSG_CFG_Light
elif pointnet2_params == 'lighter':
    SELECTED_PARAMS = ClsMSG_CFG_Lighter
elif pointnet2_params == 'dense':
    SELECTED_PARAMS = ClsMSG_CFG_Dense
else:
    raise NotImplementedError

class Pointnet2ClsMSG(nn.Module):
    def __init__(self, input_channels=5):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        for k in range(SELECTED_PARAMS['NPOINTS'].__len__()):
            mlps = SELECTED_PARAMS['MLPS'][k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                PointnetSAModuleMSG(
                    npoint=SELECTED_PARAMS['NPOINTS'][k],
                    radii=SELECTED_PARAMS['RADIUS'][k],
                    nsamples=SELECTED_PARAMS['NSAMPLE'][k],
                    mlps=mlps,
                    use_xyz=True,
                    bn=True
                )
            )
            channel_in = channel_out

# class Tactile3DEncoder(nn.Module):
#     def __init__(self,
#             shape_meta: dict,
#             input_channels: int = 8,  # input feature dim is 8
#             output_channels: int = 1024  # final PointNet feature dim
#         ):
#         """
#         Assumes cloud_point input: B, N, 3
#         Assumes low_dim input: B,D
#         """
#         super().__init__()

#         # rgb_keys = list()

#         low_dim_keys = list()
#         cloud_keys = list()
#         # key_model_map = nn.ModuleDict()
#         # key_transform_map = nn.ModuleDict()
#         # key_shape_map = dict()

#         pcd = obs_dict['global_pts'] #[b,8192,3]
#         tactile1 = obs_dict['left_gripper1_tactile'] #[b,700,6]
#         tactile2 = obs_dict['left_gripper2_tactile'] #[b,700,6]
#         state = obs_dict['left_robot_tcp_pose'] #[b,9]

#         obs_shape_meta = shape_meta['obs']
#         pcd = torch.nn.functional.pad(obs_dict['global_pts'], (0, 5), 'constant', 0)# shape: [B, 8192, 8]
      
#         B, N, _ = tactile1_orig.shape
#         padding_tensor = torch.tensor([0, 1], dtype=tactile1_orig.dtype, device=tactile1_orig.device)
#         padding_tensor = padding_tensor.unsqueeze(0).unsqueeze(0).expand(B, N, -1) # shape: [B, 700, 2]

#         # concatenate raw data with the padding tensor
#         tactile1 = torch.cat([tactile1_orig, padding_tensor], dim=-1) # shape: [B, 700, 8]
#         tactile2 = torch.cat([tactile2_orig, padding_tensor], dim=-1) # shape: [B, 700, 8]

#         combined_points = torch.cat([pcd, tactile1, tactile2], dim=1) # shape: [B, 9592, 8]
#         self.pointnet = Pointnet3DEncoder(input_channels=input_channels)

#         self.shape_meta = shape_meta



    def _break_up_pc(self, pc: torch.Tensor):
        # pc: [B, N, C_total]. First 3 dims are xyz; remaining are features.
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, pointcloud: torch.Tensor):
        xyz, features = self._break_up_pc(pointcloud)
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        # last layer is a global aggregation (N=1) -> returns [B, C]
        return l_features[-1].squeeze(-1)


class Tactile3DEncoder(nn.Module):
    def __init__(self,
                 shape_meta: dict,
                 input_channels: int = 5,   # total channels = 8 (3 coords + 5 features)
                 output_channels: int = 1024):
        super().__init__()
        self.shape_meta = shape_meta
        self._output_dim = output_channels
        self._input_dim = input_channels
        self.pointnet = Pointnet2ClsMSG(input_channels=input_channels)

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        # read inputs
        pcd_orig: torch.Tensor = obs_dict['global_pts']
        # zero_pad = torch.zeros_like(pcd_orig).to(pcd_orig.device)
        # pcd_orig = torch.cat((pcd_orig, zero_pad), dim=-1)                  # [B, 8192, 3]
        tactile1_orig: torch.Tensor = obs_dict['left_gripper1_tactile']  # [B, 700, 6]
        tactile2_orig: torch.Tensor = obs_dict['left_gripper2_tactile']  # [B, 700, 6]

        # pad pcd with 5 zeros -> [B, 8192, 8]
        pcd = F.pad(pcd_orig, (0, 5), mode='constant', value=0)

        # append [0, 1] to each tactile point -> [B, 700, 8]
        B1, N1, _ = tactile1_orig.shape
        pad01 = torch.tensor([0, 1], dtype=tactile1_orig.dtype, device=tactile1_orig.device).view(1, 1, 2)
        tactile1 = torch.cat([tactile1_orig, pad01.expand(B1, N1, 2)], dim=-1)

        B2, N2, _ = tactile2_orig.shape
        tactile2 = torch.cat([tactile2_orig, pad01.expand(B2, N2, 2)], dim=-1)

        # concat along the point dim -> [B, 8192+700+700, 8] = [B, 9592, 8]
        combined_points = torch.cat([pcd, tactile1, tactile2], dim=1)

        # PointNet encoding -> [B, 1024] (determined by PointNet config)
        feats = self.pointnet(combined_points)
        return feats

    def output_shape(self):
        return (self._output_dim,)


if __name__ == '__main__':
    # simple self test
    B = 2
    mock = {
        'global_pts': torch.randn(B, 8192, 3),
        'left_gripper1_tactile': torch.randn(B, 700, 6),
        'left_gripper2_tactile': torch.randn(B, 700, 6),
    }
    enc = Tactile3DEncoder(shape_meta=None)
    out = enc(mock)
    print('Encoder output:', out.shape)  # expect [B, 1024]