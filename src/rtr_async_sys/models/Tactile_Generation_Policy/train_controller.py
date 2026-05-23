import torch
import os
import pickle
import numpy as np
import argparse
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from einops import rearrange
import yaml
from tactile_generation_policy.common.action_utils import absolute_actions_to_relative_actions, rot6d_to_matrix, project_points_to_image
from tactile_generation_policy.model.tactile.vae_temporal import TemporalTactileVAE
from tactile_generation_policy.model.controller.controller import AdmittanceControllerNN, E2EController
from types import SimpleNamespace

def load_config_as_namespace(config_file):
    with open(config_file, "r") as file:
        config_dict = yaml.safe_load(file)
    return convert_dict_to_namespace(config_dict)

def convert_dict_to_namespace(d):
    """Recursively converts a dictionary into a SimpleNamespace."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: convert_dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [convert_dict_to_namespace(item) for item in d]
    else:
        return d

class ControllerDataset(Dataset):
    def __init__(self, 
                 data_dir, 
                 data_norm_config, 
                 delta_action=False,
                 relative_action=False,
                 mode='train'):
        self.norm = load_config_as_namespace(data_norm_config)
        data_list = [os.path.join(data_dir, i) for i in sorted(os.listdir(data_dir))]
        data_len = len(data_list)
        train_indices, test_indices = self.split_indices(data_len)
        if mode == 'train':
            self.data = [data_list[i] for i in train_indices]
        else:
            self.data = [data_list[i] for i in test_indices]
        self.delta_action = delta_action
        self.relative_action = relative_action
        self.c2h = np.array([[-1, 0, 0, 5],
                             [0, -0.9659, -0.2588, 96.678],
                             [0, -0.2588, 0.9659, -26.625],
                             [0, 0, 0, 1]])

        self.intrinsic = np.array([[604.307, 0, 310.155],
                                   [0, 604.662, 251.013],
                                   [0, 0, 1]])

    @staticmethod
    def split_indices(num_samples, train_ratio=0.9, seed=42):
        np.random.seed(seed)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        train_size = int(num_samples * train_ratio)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        return train_indices, test_indices
    
    def normalize_image(self, image):
        mean = np.array([0.485, 0.456, 0.406])  # RGB
        std = np.array([0.229, 0.224, 0.225])
        image = image.astype(np.float32) / 255.0
        image = (image - mean) / std
        return image
    
    def get_visual_action(self, state):
        translation = state[:3] * 1000.0
        rotation = rot6d_to_matrix(state[None, 3:])[0]
        h2b = np.eye(4)
        h2b[:3, :3] = rotation
        h2b[:3, 3] = translation
        c2b = h2b @ self.c2h
        g1 = h2b @ np.array([28, 0, 230, 1])
        g2 = h2b @ np.array([-45, 0, 230, 1])
        g1c = np.linalg.inv(c2b) @ g1
        g2c = np.linalg.inv(c2b) @ g2
        p1 = project_points_to_image(g1c, self.intrinsic)
        p2 = project_points_to_image(g2c, self.intrinsic)

        return np.concatenate((p1[None, :], p2[None, :]), axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_data = pickle.load(open(self.data[idx], 'rb'))
        tactile_data = sample_data['tactile']
        image = sample_data['camera1_image']
        depth = sample_data['depth']
        state = sample_data['state']
        action = sample_data['action']
        state = np.concatenate((state, action))[:-1]
        
        image = self.normalize_image(image)
        left_tactile_norm = self.normalize_tactile(tactile_data[:,:,:3])
        right_tactile_norm = self.normalize_tactile(tactile_data[:,:,3:])
        tactile_data = np.concatenate((left_tactile_norm, right_tactile_norm), axis=-1)
        tactile_data = tactile_data.reshape(-1, 35, 20, 6)

        points = self.get_visual_action(state)
        points[:, 0] = points[:, 0] / image.shape[0]
        points[:, 0] = points[:, 0] / image.shape[1]

        if self.delta_action:
            new_action = np.concatenate((state, action))
            action = np.diff(new_action)
        if self.relative_action:
            action = absolute_actions_to_relative_actions(action, base_absolute_action=state)

        image = torch.tensor(image, dtype=torch.float)
        tactile_data = torch.tensor(tactile_data, dtype=torch.float)
        points = torch.tensor(points)

        points = points.unsqueeze(0).repeat(tactile_data.shape[0], 1, 1)     # (m, n, 2)
        image = image.unsqueeze(0).repeat(tactile_data.shape[0], 1, 1, 1)

        return {'image': image,
                'tactile': tactile_data,
                'points': points,
                'action': action}

def train(controller_config, vae_cfg):
    train_dataset = ControllerDataset(controller_config.data_path, vae_cfg.data_norm_config, 'train')
    train_loader = DataLoader(train_dataset, batch_size=controller_config.batch_size, shuffle=True, num_workers=4, drop_last=True)

    val_dataset = ControllerDataset(controller_config.data_path, vae_cfg.data_norm_config, 'val')
    val_loader = DataLoader(val_dataset, batch_size=controller_config.val_batch_size, shuffle=False, num_workers=2, drop_last=False)

    tactile_vae = TemporalTactileVAE(
        input_dim=vae_cfg.input_dim, 
        latent_dim=vae_cfg.latent_dim,
        hidden_state=vae_cfg.hidden_state,
        time_compression_ratio=vae_cfg.time_compression_ratio,
        spatial_compression_ratio=vae_cfg.spatial_compression_ratio,
        decoder_mode="condition").to(controller_config.device)

    for param in tactile_vae.parameters():
        param.requires_grad = False
    tactile_vae.eval()

    tac_dim = tactile_vae.get_latent(torch.ones([1, 3, vae_cfg.out_H, vae_cfg.out_W])).shape
    
    model = AdmittanceControllerNN(tac_dim=tac_dim,
                                   hidden_dim=128, 
                                   out_dim=6
                                   ).to(controller_config.device)
    

    optimizer = optim.Adam(model.parameters(), lr=controller_config.lr)
    writer = SummaryWriter(log_dir=controller_config.log_dir)

    best_val_loss = float('inf')

    for epoch in range(controller_config.epochs):
        model.train()
        train_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch} [Train]") as pbar:
            for batch in pbar:
                B = batch['image'].size(0)
                batch = batch
                image = batch['image'].to(controller_config.device)
                tactile = batch['tactile'].to(controller_config.device)
                points = batch['points'].to(controller_config.device)
                action = batch['action'].to(controller_config.device)

                image = rearrange(image, 'b t h w c -> (b t) c h w')
                tactile = rearrange(tactile, 'b t h w c -> (b t) c h w')
                points = rearrange(points, 'b t h w c -> (b t) c h w')
                action = rearrange(action, 'b t h w c -> (b t) c h w')
                
                optimizer.zero_grad()

                tac_fea_l = tactile_vae.get_latent(tactile[:, :3, :, :])
                tac_fea_r = tactile_vae.get_latent(tactile[:, 3:, :, :])
                current_tac_fea = torch.cat((tac_fea_l, tac_fea_r))
                
                
                loss, recon_loss, kl_loss = model(current_tac_fea, current_tac_fea, image)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * B
                train_recon_loss += recon_loss.item() * B
                train_kl_loss += kl_loss.item() * B

                curr_loss = train_loss / ((pbar.n + 1) * B)
                curr_recon = train_recon_loss / ((pbar.n + 1) * B)
                curr_kl = train_kl_loss / ((pbar.n + 1) * B)
                pbar.set_postfix({
                    'loss': f'{curr_loss:.4f}',
                    'recon': f'{curr_recon:.4f}',
                    'kl': f'{curr_kl:.4f}'
                })

            avg_train_loss = train_loss / len(train_loader.dataset)
            avg_train_recon_loss = train_recon_loss / len(train_loader.dataset)
            avg_train_kl_loss = train_kl_loss / len(train_loader.dataset)
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            writer.add_scalar('Loss/recon', avg_train_recon_loss, epoch)
            writer.add_scalar('Loss/kl', avg_train_kl_loss, epoch)

        # Validation
        if epoch % 10 == 0:
            model.eval()
            val_loss = 0.0
            val_recon_loss = 0.0
            val_kl_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                    B = batch.size(0)
                    batch = batch.to(controller_config.device)
                    batch = rearrange(batch, 'b t h w c -> (b t) c h w')
                    loss, recon_loss, kl_loss = model.calculate_loss(batch)
                    val_loss += loss.item() * B
                    val_recon_loss += recon_loss.item() * B
                    val_kl_loss += kl_loss.item() * B
                
            avg_val_loss = val_loss / len(val_loader.dataset)
            avg_val_recon_loss = val_recon_loss / len(val_loader.dataset)
            avg_val_kl_loss = val_kl_loss / len(val_loader.dataset)
            writer.add_scalar('Loss/val', avg_val_loss, epoch)
            writer.add_scalar('Loss/val_recon', avg_val_recon_loss, epoch)
            writer.add_scalar('Loss/val_kl', avg_val_kl_loss, epoch)

            # Save model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), os.path.join(controller_config.ckpt_dir, 'best_model.pth'))
            torch.save(model.state_dict(), os.path.join(controller_config.ckpt_dir, 'latest.pth'))
            print(f"[Epoch {epoch}] train loss: {avg_train_loss:.6f} | val loss: {avg_val_loss:.6f}")

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_config", type=str, default="tactile_generation_policy/config/vae_temporal.yaml", help="Path to the config file")
    parser.add_argument("--controller_config", type=str, default="tactile_generation_policy/config/controller_config.yaml", help="Path to the config file")
    args = parser.parse_args()

    vae_cfg = load_config_as_namespace(args.vae_config)
    controller_config = load_config_as_namespace(args.controller_config)

    if controller_config.ckpt_dir == "":
        controller_config.ckpt_dir = os.path.join(controller_config.log_dir, 'checkpoints')
    os.makedirs(controller_config.ckpt_dir, exist_ok=True)
    os.makedirs(controller_config.log_dir, exist_ok=True)
    
    train(controller_config, vae_cfg)