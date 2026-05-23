import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import time


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, output_dim=64, resnet_type='resnet18', pretrained=True):
        super().__init__()
        if resnet_type == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            feat_dim = 512
        elif resnet_type == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            feat_dim = 2048
        else:
            raise ValueError("Unsupported resnet_type")

        modules = list(resnet.children())[:-1]  # Remove the FC layer
        self.backbone = nn.Sequential(*modules)  # Output[batch, feat_dim, 1, 1]

        if output_dim != feat_dim:
            self.proj = nn.Linear(feat_dim, output_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        # x: [batch, 3, H, W]
        feat = self.backbone(x)          # [batch, feat_dim, 1, 1]
        feat = feat.view(feat.size(0), -1)  # [batch, feat_dim]
        out = self.proj(feat)               # [batch, output_dim]
        return out


class TactileFusion(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
            nn.ReLU()
        )

    def forward(self, tactile_curr, tactile_tgt):
        x = torch.cat([tactile_curr, tactile_tgt], dim=-1)
        x = self.net(x)
        x_pooled, _ = x.max(dim=1, keepdim=True)
        return x_pooled

class ConditionEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, robot_state, action):
        x = torch.cat([robot_state, action], dim=-1)
        return self.net(x)

class AdmittanceControllerNN(nn.Module):
    def __init__(self, tactile_dim, state_dim, action_dim, img_dim, hidden_dim=32):
        super().__init__()
        self.tactile_fusion = TactileFusion(tactile_dim, hidden_dim)
        self.condition_encoder = ConditionEncoder(state_dim + action_dim, hidden_dim)
        self.obs_encoder = ResNetFeatureExtractor(output_dim=img_dim, 
                                                  resnet_type='resnet18', 
                                                  pretrained=True)
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim + 1) # Output [next action, speed]
        )

    def forward(self, tactile_curr, tactile_tgt, robot_state, action):
        tactile_feat = self.tactile_fusion(tactile_curr, tactile_tgt)
        condition_feat = self.condition_encoder(robot_state, action)
        fused = torch.cat([tactile_feat, condition_feat], dim=-1)
        out = self.fusion_net(fused)
        action_next = out[..., :-1]
        v_next = out[..., -1:]
        return action_next, v_next
    
if __name__ == "__main__":
    t1 = torch.ones((1, 16, 1, 9, 5)).cuda()
    t2 = torch.zeros((1, 16, 1, 9, 5)).cuda()
    t1 = t1.reshape(1, 16, -1).permute(0, 2, 1)
    t2 = t2.reshape(1, 16, -1).permute(0, 2, 1)
    robot_state = torch.ones((1, 1, 10)).cuda()
    action = torch.ones((1, 1, 10)).cuda()
    controller = AdmittanceControllerNN(tactile_dim=16,
                                        state_dim=10,
                                        action_dim=10,
                                        img_dim=128,
                                        hidden_dim=128)
    controller = controller.cuda()
    times = []
    for i in range(20):
        start_time = time.time()
        action, velocity = controller(t1, t2, robot_state, action)
        end_time = time.time()
        times.append(end_time - start_time)

    print('end')