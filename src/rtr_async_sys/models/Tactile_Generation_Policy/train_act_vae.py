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
from types import SimpleNamespace
from .tactile_generation_policy.model.action.vae import ActionVAE
import sys
import time
sys.path.append('../tools')

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

class ActionDataset(Dataset):
    def __init__(self, root_dir, action_space, training=True, prefetch=True):
        if training:
            data_dir = os.path.join(root_dir, 'train')
        else:
            data_dir = os.path.join(root_dir, 'test')

        self.data = [os.path.join(data_dir, i) for i in sorted(os.listdir(data_dir))]
        if prefetch:
            print(f"prefetch data ing")
            self.data = [pickle.load(open(data_path, 'rb')) for data_path in tqdm(self.data)]
        self.action_space = action_space
        self.prefetch = prefetch
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not self.prefetch:
            sample_data = pickle.load(open(self.data[idx], 'rb'))
        else:
            sample_data = self.data[idx]
        action = sample_data[self.action_space]
        action = torch.tensor(action, dtype=torch.float)
        return action

def train_temporal(cfg):
    data_path = cfg.data_path + f'_{cfg.window_size}' + f'_downsample_{cfg.downsample}'
    train_dataset = ActionDataset(data_path, cfg.action_space, training=True)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_dataset = ActionDataset(data_path, cfg.action_space, training=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg.val_batch_size, shuffle=False, num_workers=2, drop_last=False)
    normalizer = torch.load(os.path.join(data_path, 'normalizer.pth'))

    model = ActionVAE(
        input_dim=cfg.input_dim,
        horizon=cfg.horizon, 
        latent_dim=cfg.latent_dim,
        hidden_state=cfg.hidden_state,
        n_embed=cfg.n_embed,
        mlp_layer_num=cfg.mlp_layer_num,
        time_compression_ratio=cfg.time_compression_ratio,
        act_scale=cfg.act_scale,
    ).to(cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    
    log_dir = os.path.join(cfg.log_dir, f'window_size_{cfg.window_size}' + f'_downsample_{cfg.downsample}')
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    best_val_loss = float('inf')

    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0
        train_recon_loss = 0.0
        train_kl_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch} [Train]") as pbar:
            # start = time.time()
            for batch in pbar:
                # print(f"load time is {time.time()-start}")
                # start = time.time()
                B = batch.size(0)
                batch = normalizer[cfg.action_space].normalize(batch)
                batch = batch.to(cfg.device)               
                optimizer.zero_grad()
                loss, recon_loss, kl_loss = model.calculate_loss(batch)
                # print(f"calculate loss time is {time.time()-start}")
                # start = time.time()
                
                loss.backward()
                # print(f"backward time is {time.time()-start}")
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

        if epoch % 5 == 0:
            model.eval()
            val_loss = 0.0
            val_recon_loss = 0.0
            val_kl_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                    B = batch.size(0)
                    batch = normalizer[cfg.action_space].normalize(batch)
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

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_model.pth'))
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'latest.pth'))
            print(f"[Epoch {epoch}] train loss: {avg_train_loss:.6f} | val loss: {avg_val_loss:.6f}")

    writer.close()

def eval_temporal_vae(cfg):
    data_path = cfg.data_path + f'_{cfg.window_size}' + f'_downsample_{cfg.downsample}'
    val_dataset = ActionDataset(data_path, cfg.action_space, training=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False)
    normalizer = torch.load(os.path.join(data_path, 'normalizer.pth'))
    vae = ActionVAE(
        input_dim=cfg.input_dim,
        horizon=cfg.horizon, 
        latent_dim=cfg.latent_dim,
        hidden_state=cfg.hidden_state,
        n_embed=cfg.n_embed,
        mlp_layer_num=cfg.mlp_layer_num,
        time_compression_ratio=cfg.time_compression_ratio,
        act_scale=cfg.act_scale,
    ).to(cfg.device)
    vae.load_state_dict(torch.load(cfg.load_ckpt_path, map_location=cfg.device))
    vae.eval()
    val_recon_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Eval"):
            B = batch.size(0)
            batch = normalizer['tactile'].normalize(batch)
            batch = batch.to(cfg.device)
            if cfg.data_mode == 'single_hand':
                left_batch = batch[..., :3]
                right_batch = batch[..., 3:]
                batch = torch.cat([left_batch, right_batch], dim=0)
            batch = rearrange(batch, 'b t h w c -> b c t h w')
            _, recon_loss, _ = vae.calculate_loss(batch, val=True)
            val_recon_loss += recon_loss.item() * B
            recon = vae.get_recon(batch, implicit=(cfg.decoder_mode=='implicit'))
            
            recon = normalizer['tactile'].unnormalize((rearrange(recon[0], 'c t h w -> t h w c'))).cpu().numpy()
            gt = normalizer['tactile'].unnormalize((rearrange(batch[0], 'c t h w -> t h w c'))).cpu().numpy()
        
        avg_val_recon_loss = val_recon_loss / len(val_loader.dataset)
        print(avg_val_recon_loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="tactile_generation_policy/config/vae_config.yaml", help="Path to the config file")
    parser.add_argument("--test", action='store_true')
    args = parser.parse_args()

    cfg = load_config_as_namespace(args.config)
    if hasattr(cfg, "lr"):
        cfg.lr = float(cfg.lr)

    if args.test:
        eval_temporal_vae(cfg)
    else:
        train_temporal(cfg)