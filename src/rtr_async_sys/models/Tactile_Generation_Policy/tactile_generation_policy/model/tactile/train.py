import torch.nn.functional as F
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
from typing import List
import random
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


def vis_pred_marker_flow(marker_pos, delta_mf, sample_points, sample_labels, pred_lables, save_path):
    marker_pos[: ,0] = (marker_pos[: ,0] + 1) / 2 * 320
    marker_pos[: ,1] = (marker_pos[: ,1] + 1) / 2 * 240
    sample_points[: ,0] = (sample_points[: ,0] + 1) / 2 * 320
    sample_points[: ,1] = (sample_points[: ,1] + 1) / 2 * 240

    delta_mf = delta_mf * 10
    pred_lables = pred_lables * 10
    sample_labels = sample_labels * 10

    plt.figure(figsize=(8, 8))
    plt.scatter(marker_pos[:, 0], marker_pos[:, 1], c='blue', s=30)
    plt.quiver(
        marker_pos[:, 0], marker_pos[:, 1],
        delta_mf[:, 0], delta_mf[:, 1],
        angles='xy', scale_units='xy', scale=10, color='blue', width=0.005
    )

    plt.scatter(sample_points[:, 0], sample_points[:, 1], c='orange', s=30)
    # plt.quiver(
    #     sample_points[:, 0], sample_points[:, 1],
    #     sample_labels[:, 0], sample_labels[:, 1],
    #     angles='xy', scale_units='xy', scale=10, color='green', width=0.005
    # )
    plt.quiver(
        sample_points[:, 0], sample_points[:, 1],
        pred_lables[:, 0], pred_lables[:, 1],
        angles='xy', scale_units='xy', scale=10, color='red', width=0.005
    )

    plt.axis('equal')
    plt.axis('off')           # Hide coordinate axes
    plt.grid(False)           # Hide grid
    plt.xticks([])            # Hide x ticks
    plt.yticks([])            # Hide y ticks
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


class MarkerFlowDataset(Dataset):
    def __init__(self, root_dir, split='train'):

        data_list = os.listdir(root_dir)
        mf_list = []
        # data_list = data_list[:20]
        for mf_file in data_list:
            mf = np.load(os.path.join(root_dir, mf_file))
            mf_list.append(mf[0])
            mf_list.append(mf[1])
        random.shuffle(mf_list)
        if split == 'train':
            self.mf = mf_list[:int(len(mf_list) * 0.95)]
        else:
            self.mf = mf_list[int(len(mf_list) * 0.95):]
        
        self.npoints = 63
        self.padding = 0.1
        self.split = split

    def __len__(self):
        return len(self.mf)
    
    def __getitem__(self, idx):
        mf = self.mf[idx][:, :self.npoints, :]
        x_pos = 2 * mf[0, :, 0] / 320 - 1
        y_pos = 2 * mf[0, :, 1] / 240 - 1
        delta_mf = (mf[1, :, :] - mf[0, :, :])
        marker_pos = np.stack([x_pos, y_pos], axis=-1)

        if self.split == 'train':
            sample_points = np.column_stack([
                np.random.uniform(x_pos.min()/(1+self.padding), x_pos.max()/(1+self.padding), 100),
                np.random.uniform(y_pos.min()/(1+self.padding), y_pos.max()/(1+self.padding), 100)
            ])
            sample_labels = self.knn_interpolate(marker_pos, delta_mf, sample_points, k=3)
        else:
            x = np.linspace(-0.85, 0.85, 10)
            y = np.linspace(-0.85, 0.85, 10)
            xx, yy = np.meshgrid(x, y)
            sample_points = np.stack([xx.ravel(), yy.ravel()], axis=1)
            sample_labels = self.knn_interpolate(marker_pos, delta_mf, sample_points, k=3)

        marker_pos = torch.tensor(marker_pos, dtype=torch.float32)
        delta_mf = torch.tensor(delta_mf, dtype=torch.float32)
        sample_points = torch.tensor(sample_points, dtype=torch.float32)
        sample_labels = torch.tensor(sample_labels, dtype=torch.float32)
        
        return marker_pos, delta_mf, sample_points, sample_labels
    
    def knn_interpolate(self, xy_known, uv_known, xy_query, k=4, eps=1e-8):
        """
        xy_known: (N,2) known pointscoordinates
        uv_known: (N, 2) UV values for known points
        xy_query: (M,2) query pointscoordinates
        k: number of neighbors
        Return: (M,2) query pointsinterpolateuv
        """
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(xy_known)
        dists, idxs = nbrs.kneighbors(xy_query)  # dists: (M,k), idxs: (M,k)
        
        # inverse-distance weighting
        weights = 1.0 / (dists + eps)   # avoid division by zero
        weights /= weights.sum(axis=1, keepdims=True)   # normalize weights
        
        # interpolate
        uv_interp = np.sum(uv_known[idxs] * weights[..., None], axis=1)
        return uv_interp

class Real_MarkerFlowDataset(Dataset):
    def __init__(self, root_dir):

        self.mf = []
        npy_files = []
    
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith('.npy'):
                    npy_files.append(os.path.join(dirpath, filename))
        
        for npy_file in npy_files:
            mf = np.load(npy_file)
            self.mf.append(mf[0])
            self.mf.append(mf[1])
        
        self.mf = [self.mf[0]]
    
    def __len__(self):
        return len(self.mf)
    
    def __getitem__(self, idx):
        mf = self.mf[idx]
        x_pos = 2 * mf[0, :, 0] / 320 - 1
        y_pos = 2 * mf[0, :, 1] / 240 - 1
        delta_mf = mf[1, :, :] - mf[0, :, :]
        marker_pos = np.stack([x_pos, y_pos], axis=-1)
        
        return torch.tensor(marker_pos[:60, :], dtype=torch.float32), torch.tensor(delta_mf[:60, :], dtype=torch.float32)

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

class PointNetFeaNew(nn.Module):
    def __init__(self, point_dim, net_layers: List, batchnorm=False):
        super(PointNetFeaNew, self).__init__()
        self.layer_num = len(net_layers)
        self.conv0 = nn.Conv1d(point_dim, net_layers[0], 1)
        self.bn0 = nn.BatchNorm1d(net_layers[0]) if batchnorm else nn.Identity()
        for i in range(0, self.layer_num - 1):
            self.__setattr__(
                f"conv{i + 1}", nn.Conv1d(net_layers[i], net_layers[i + 1], 1)
            )
            self.__setattr__(
                f"bn{i + 1}",
                nn.BatchNorm1d(net_layers[i + 1]) if batchnorm else nn.Identity(),
            )

        self.output_dim = net_layers[-1]

    def forward(self, x):
        for i in range(0, self.layer_num - 1):
            x = F.relu(self.__getattr__(f"bn{i}")(self.__getattr__(f"conv{i}")(x)))
        x = self.__getattr__(f"bn{self.layer_num - 1}")(
            self.__getattr__(f"conv{self.layer_num - 1}")(x)
        )
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.output_dim)
        return x

class PointNetFeatureExtractor(nn.Module):
    """
    this is a latent feature extractor for point cloud data
    need to distinguish this from other modules defined in feature_extractors.py
    those modules are only used to extract the corresponding input (e.g. point flow, manual feature, etc.) from original observations
    """

    def __init__(self, dim, out_dim, batchnorm=False):
        super(PointNetFeatureExtractor, self).__init__()
        self.dim = dim
        self.pc_feq = 6

        self.pointnet_local_feature_num = out_dim
        self.pointnet_global_feature_num = 512

        self.pointnet_local_fea = nn.Sequential(
            nn.Conv1d(dim, self.pointnet_local_feature_num, 1),
            (
                nn.BatchNorm1d(self.pointnet_local_feature_num)
                if batchnorm
                else nn.Identity()
            ),
            nn.ReLU(),
            nn.Conv1d(
                self.pointnet_local_feature_num, self.pointnet_local_feature_num, 1
            ),
            (
                nn.BatchNorm1d(self.pointnet_local_feature_num)
                if batchnorm
                else nn.Identity()
            ),
            nn.ReLU(),
        )
        self.pointnet_global_fea = PointNetFeaNew(
            self.pointnet_local_feature_num,
            [64, 128, self.pointnet_global_feature_num],
            batchnorm=batchnorm,
        )

        self.mlp_output = nn.Sequential(
            nn.Linear(self.pointnet_global_feature_num, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

        self.encode_pos = nn.Linear((self.pc_feq * 2 + 1) * 2 , self.pointnet_local_feature_num)

    def frequency_encoding(self, xy_n, n_freqs=6):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12
        xy_n : [-1, 1]
        Inputs:
            x: (b n m)
        Outputs:
            out: (b n o)
        """
        freq_bands = 2 ** torch.linspace(0, n_freqs - 1, n_freqs)
        freq_bands = freq_bands.to(xy_n.device)
        xy_n = xy_n.to(freq_bands.dtype)
        xy_feq = xy_n.unsqueeze(-1) * freq_bands  # (b n m 1)
        sin_xyz, cos_xyz = torch.sin(xy_feq), torch.cos(xy_feq)  # (b n m nf)
        encoding = torch.cat([xy_n.unsqueeze(-1), sin_xyz, cos_xyz], -1).reshape(*xy_n.shape[:2], -1)
        
        return encoding

    def forward(self, points_w_flow):
        """
        :param marker_pos: Tensor, size (batch, num_points, 4)
        :return:
        """
        marker_pos = points_w_flow
        xys = points_w_flow[:, :, :2]

        if marker_pos.ndim == 2:
            marker_pos = torch.unsqueeze(marker_pos, dim=0)

        marker_pos = torch.transpose(marker_pos, 1, 2)
        local_feature = self.pointnet_local_fea(
            marker_pos
        )  # (batch_num, self.pointnet_local_feature_num, point_num)
        # shape: (batch, step * 2, num_points)
        position_embedding = self.frequency_encoding(xys, self.pc_feq)
        position_encoding = self.encode_pos(position_embedding).permute(0,2,1)

        local_feature = local_feature + position_encoding

        return local_feature

        global_feature = self.pointnet_global_fea(local_feature).view(
            -1, self.pointnet_global_feature_num
        )  # (batch_num, self.pointnet_global_feature_num)

        pred = self.mlp_output(global_feature)
        # pred shape: (batch_num, out_dim)
        return pred

class AutoEncoder(nn.Module):
    def __init__(self, dim=4, out_dim=64, mask_ratio=0.4):
        super(AutoEncoder, self).__init__()
        self.pc_feq = 6
        self.mask_ratio = mask_ratio
        self.out_dim = out_dim
        self.ae = PointNetFeatureExtractor(dim=dim, out_dim=out_dim)
        self.ff = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
        )
        self.decoder = LocalDecoder(dim=2, c_dim=out_dim, out_dim=2, hidden_size=128)

    def random_masking(self, x, mask_ratio):
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1,1,D))
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward(self, marker_pos, delta_flow, sampled_points):
        points_w_flow = torch.cat((marker_pos, delta_flow), dim=-1)
        local_fea = self.ae(points_w_flow).permute(0, 2, 1)
        recon_tac = self.decoder(sampled_points, marker_pos, local_fea)

        # B, N, D = points_w_flow.shape
        # x_masked, mask, ids_restore = self.random_masking(points_w_flow, self.mask_ratio)
        # features = self.ae(x_masked).permute(0, 2, 1)
        # mask_token = nn.Parameter(torch.zeros(1, 1, self.out_dim, device=points_w_flow.device))
        # full_fea = torch.cat([
        #     features,
        #     mask_token.expand(B, N - features.shape[1], self.out_dim)
        # ], dim=1)
        # full_fea = torch.gather(full_fea, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, self.out_dim))
        # rec_points = self.decoder(full_fea)
        
        return recon_tac

class LocalDecoder(nn.Module):
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

    def __init__(self, dim=3, c_dim=128, out_dim=1,
                 hidden_size=256, n_blocks=5, leaky=False, 
                 sample_mode='bilinear', padding=0.1): #desc_type: [field, occp]
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_size, out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding


    def normalize_coordinate(self, p, padding=0.15, plane='xz'):
        ''' Normalize coordinate to [0, 1] for unit cube experiments

        Args:
            p (tensor): point
            padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
            plane (str): plane feature type, ['xz', 'xy', 'yz']
        '''
        if plane == 'xz':
            xy = p[:, :, [0, 2]]
        elif plane =='xy':
            xy = p[:, :, [0, 1]]
        else:
            xy = p[:, :, [1, 2]]

        xy_new = xy / (1 + padding + 10e-6) # (-0.5, 0.5)
        xy_new = xy_new + 0.5 # range (0, 1)

        # f there are outliers out of the range
        if xy_new.max() >= 1:
            xy_new[xy_new >= 1] = 1 - 10e-6
        if xy_new.min() < 0:
            xy_new[xy_new < 0] = 0.0
        return xy_new

    def sample_plane_feature(self, p, c, plane='xy'):
        # xy = self.normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        xy = p.clone()
        xy = xy[:, :, None].float()
        # vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        vgrid = xy
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c

    def knn_interpolate(self, query_xyz, src_xyz, src_feat, k=3, eps=1e-6):
        """
        query_xyz: (B, Q, 3)  query pointscoordinates
        src_xyz:   (B, M, 3) source point-cloud coordinates
        src_feat:  (B, M, C) source point-cloud features
        Return:      (B, Q, C) interpolated features for query points
        """
        B, Q, _ = query_xyz.shape
        _, M, _ = src_xyz.shape

        # Compute Euclidean distances (B, Q, M).
        dist = torch.cdist(query_xyz, src_xyz) + eps

        # Select the nearest k points.
        knn_dist, knn_idx = dist.topk(k, largest=False, dim=2)    # (B, Q, k)

        # Gather corresponding features (B, Q, k, C).
        knn_feat = torch.gather(src_feat.unsqueeze(1).expand(-1, Q, -1, -1), 2, knn_idx.unsqueeze(-1).expand(-1, -1, -1, src_feat.shape[-1]))

        # Compute weighted interpolation.
        weight = 1.0 / (knn_dist + eps)   # (B, Q, k)
        weight = weight / weight.sum(dim=2, keepdim=True)  # Normalize
        feat_interp = (knn_feat * weight.unsqueeze(-1)).sum(dim=2)  # (B, Q, C)
        
        # print('feat_nan:', torch.isnan(feat_interp).any())
        
        return feat_interp

    def forward(self, queries, xys, c_plane):
        
        if self.c_dim != 0:
            c = self.knn_interpolate(queries, xys, c_plane)

        queries = queries.float()
        net = self.fc_p(queries)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))

        return out

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, writer, save_dir):
    best_val_acc = 255

    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        step = 0
        
        for marker_pos, delta_flow, sample_points, sample_labels in train_bar:
            marker_pos = marker_pos.to(device)
            delta_flow = delta_flow.to(device)
            sample_points = sample_points.to(device)
            sample_labels = sample_labels.to(device)

            # print('data_nan', torch.isnan(sample_points).any(), torch.isnan(sample_labels).any())
            
            optimizer.zero_grad()
            outputs = model(marker_pos, delta_flow, sample_points)
            loss = criterion(outputs, sample_labels)
            ratio = loss.item() / sample_labels.mean()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_bar.set_postfix({'loss': f'{loss.item():.4f}', 'ratio': f'{ratio:.4f}'})

            writer.add_scalar('Loss/train_step', loss.item(), step)
            step += 1

        epoch_loss = running_loss / len(train_loader)
        tqdm.write(f"[Epoch {epoch+1}/{num_epochs}] Avg Train Loss: {epoch_loss:.4f}")
        writer.add_scalar('Loss/train_epoch', epoch_loss, epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        # validation phase
        model.eval()
        vis = False
        save_vis_num = 0
        val_loss = 0
        val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        if epoch % 10 == 0 and epoch > 5:
            vis = True
        else:
            vis = False
        with torch.no_grad():
            for marker_pos, delta_flow, sample_points, sample_labels in val_bar:
                marker_pos = marker_pos.to(device)
                delta_flow = delta_flow.to(device)
                sample_points = sample_points.to(device)
                sample_labels = sample_labels.to(device)

                outputs = model(marker_pos, delta_flow, sample_points)
                loss = criterion(outputs, sample_labels)
                val_loss += loss.item()
                if vis:
                    marker_pos = marker_pos.cpu().numpy()
                    delta_flow = delta_flow.cpu().numpy()
                    sample_points = sample_points.cpu().numpy()
                    sample_labels = sample_labels.cpu().numpy()
                    outputs = outputs.cpu().numpy()
                    for i in range(marker_pos.shape[0]):
                        save_path = os.path.join('outputs/tac_encoder/save_vis', str("%04d"%save_vis_num) + '.png')
                        vis_pred_marker_flow(marker_pos[i], delta_flow[i], sample_points[i], sample_labels[i], outputs[i], save_path)
                        save_vis_num += 1
        
        avg_val_loss = val_loss / len(val_loader)


        tqdm.write(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}")
        writer.add_scalar('Val Loss/val_epoch', avg_val_loss, epoch)

        scheduler.step()
        
        # save best model
        if avg_val_loss <= best_val_acc:
            best_val_acc = avg_val_loss
            best_save_path = os.path.join(save_dir, 'best_offset.pth')
            torch.save(model.state_dict(), best_save_path)
        
        if epoch % 20 == 0:
            model_save_epoch = os.path.join(save_dir, f"model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), model_save_epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='root', help='root dir')
    parser.add_argument('--save_dir', type=str, default='exp', help='exp dir')
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training and validation (default: 128).")
    parser.add_argument('--epochs', type=int, default=40, help="Number of epochs to train (default: 40).")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for the optimizer (default: 0.001).")
    args = parser.parse_args()
    
    # create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_dataset = MarkerFlowDataset(args.root_dir, split='train')
    val_dataset = MarkerFlowDataset(args.root_dir, split='val')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8)
    
    writer = SummaryWriter(log_dir=args.save_dir)
    model = AutoEncoder(dim=4, out_dim=64).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    def lr_lambda(epoch):
        return max(1e-4 / args.lr, 1 - epoch / args.epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, args.epochs, device, writer, args.save_dir)
    

if __name__ == "__main__":
    main()
