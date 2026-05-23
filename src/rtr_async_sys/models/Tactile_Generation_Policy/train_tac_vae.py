import torch
import os
import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from einops import rearrange
import yaml
from types import SimpleNamespace
from tactile_generation_policy.model.tactile.vae_single import TactileVAE
from tactile_generation_policy.model.tactile.vae_temporal import TemporalTactileVAE
import sys
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
    
def plot_l1_heatmaps(pred, gt, save_path):
    """Plot L1 distance heatmaps."""  
    l1_distances = np.abs(pred - gt)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    labels = ['x1', 'y1', 'z1', 'x2', 'y2', 'z2']
    for i in range(6):
        row = i // 3
        col = i % 3
        data = l1_distances[:, :, i]
        im = axes[row, col].imshow(
            data,
            cmap='hot',
            vmin=0,
            vmax=0.3,
            origin='lower',
            aspect='auto'
        )
        mean_error = np.mean(data)
        max_error = np.max(data)

        title = f'L1 Distance: {labels[i]}\n'
        title += f'Mean: {mean_error:.4f} | Max: {max_error:.4f}'
        axes[row, col].set_title(title, fontsize=12, fontweight='bold')
        axes[row, col].set_xlabel('Width (20)', fontsize=11)
        axes[row, col].set_ylabel('Height (35)', fontsize=11)
        cbar = plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
        cbar.set_label('L1 Distance', fontsize=10)
    
    # Overall title
    overall_mean = np.mean(l1_distances)
    fig.suptitle(
        f'L1 Distance Heatmaps (Overall Mean: {overall_mean:.4f})',
        fontsize=16,
        fontweight='bold'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_tactile_grids(pred, gt, save_path):
    # x = np.linspace(-8.5, 8.5, 20)
    # y = np.linspace(30, 0, 35)
    x = np.linspace(0.2, 0.8, 8)
    y = np.linspace(0.2, 0.8, 8)
    X, Y = np.meshgrid(x, y)
    grid = np.stack([X, Y], axis=-1)
    left_tactile_pred = pred[:, :, :3]
    right_tactile_pred = pred[:, :, 3:]
    left_tactile_gt = gt[:, :, :3]
    right_tactile_gt = gt[:, :, 3:]

    fig, axs = plt.subplots(2, 2, figsize=(16, 14), sharex=True, sharey=True)

    axs[0, 0].scatter(grid[..., 0], grid[..., 1], color='k', s=10, alpha=0.3)
    axs[0, 0].quiver(grid[..., 0], grid[..., 1], left_tactile_gt[..., 0], left_tactile_gt[..., 1],
                    color='blue', angles='xy', scale_units='xy', scale=1, width=0.005, label='Left GT')
    axs[0, 0].quiver(grid[..., 0], grid[..., 1], left_tactile_pred[..., 0], left_tactile_pred[..., 1],
                    color='orange', angles='xy', scale_units='xy', scale=1, width=0.005, label='Left Pred')
    axs[0, 0].set_title('Left Hand (GT & Pred)')
    axs[0, 0].set_xlabel('X')
    axs[0, 0].set_ylabel('Y')
    handles, labels = axs[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs[0, 0].legend(by_label.values(), by_label.keys())

    axs[0, 1].scatter(grid[..., 0], grid[..., 1], color='k', s=10, alpha=0.3)
    axs[0, 1].quiver(grid[..., 0], grid[..., 1], right_tactile_gt[..., 0], right_tactile_gt[..., 1],
                    color='blue', angles='xy', scale_units='xy', scale=1, width=0.005, label='Right GT')
    axs[0, 1].quiver(grid[..., 0], grid[..., 1], right_tactile_pred[..., 0], right_tactile_pred[..., 1],
                    color='orange', angles='xy', scale_units='xy', scale=1, width=0.005, label='Right Pred')
    axs[0, 1].set_title('Right Hand (GT & Pred)')
    axs[0, 1].set_xlabel('X')
    handles, labels = axs[0, 1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs[0, 1].legend(by_label.values(), by_label.keys())

    axs[1, 0].scatter(grid[..., 0], grid[..., 1], color='k', s=10, alpha=0.3)
    axs[1, 0].quiver(grid[..., 0], grid[..., 1], left_tactile_gt[..., 0], left_tactile_gt[..., 1],
                    color='blue', angles='xy', scale_units='xy', scale=1, width=0.005, label='Left GT')
    axs[1, 0].set_title('Left Hand GT Only')
    axs[1, 0].set_xlabel('X')
    axs[1, 0].set_ylabel('Y')
    axs[1, 0].legend()

    axs[1, 1].scatter(grid[..., 0], grid[..., 1], color='k', s=10, alpha=0.3)
    axs[1, 1].quiver(grid[..., 0], grid[..., 1], right_tactile_gt[..., 0], right_tactile_gt[..., 1],
                    color='blue', angles='xy', scale_units='xy', scale=1, width=0.005, label='Right GT')
    axs[1, 1].set_title('Right Hand GT Only')
    axs[1, 1].set_xlabel('X')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def vis_tactile(pred, gt, save_path='outputs/tactile_vis'):
    os.makedirs(save_path, exist_ok=True)
    dir_name = str("%04d" % len(os.listdir(save_path)))
    save_vis_dir = os.path.join(save_path, dir_name)
    os.makedirs(save_vis_dir, exist_ok=True)
    for i in range(pred.shape[0]):
        save_heat_map_vis = os.path.join(save_vis_dir, str("%02d"%i) + '_heat_.png')
        save_force_map_vis = os.path.join(save_vis_dir, str("%02d"%i) + '_force_.png')
        pred_i = pred[i]
        gt_i = gt[i]
        # plot_l1_heatmaps(pred_i, gt_i, save_heat_map_vis)
        plot_tactile_grids(pred_i, gt_i, save_force_map_vis)

class TactileDataset(Dataset):
    def __init__(self, root_dir, training=True):
        if training:
            data_dir = os.path.join(root_dir, 'train')
        else:
            data_dir = os.path.join(root_dir, 'test')
        self.data = [os.path.join(data_dir, i) for i in sorted(os.listdir(data_dir))]
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_data = pickle.load(open(self.data[idx], 'rb'))
        tactile_data = sample_data['tactile']
        tactile_data = tactile_data.reshape(-1, 35, 20, 6)
        tactile_data = torch.tensor(tactile_data, dtype=torch.float)
        last_row = tactile_data[:, -1:, :, :]  # [t, 1, 20, c]
        tactile_data = torch.cat([tactile_data, last_row], dim=1) # [t, 36, 20, c]
        return tactile_data

def train_temporal(cfg):
    data_path = cfg.data_path + f'_{cfg.window_size}' + f'_downsample_{cfg.downsample}'
    train_dataset = TactileDataset(data_path, training=True)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_dataset = TactileDataset(data_path, training=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg.val_batch_size, shuffle=False, num_workers=2, drop_last=False)
    normalizer = torch.load(os.path.join(data_path, 'normalizer.pth'))

    model = TemporalTactileVAE(
        input_dim=cfg.input_dim, 
        latent_dim=cfg.latent_dim,
        hidden_state=cfg.hidden_state,
        time_compression_ratio=cfg.time_compression_ratio,
        spatial_compression_ratio=cfg.spatial_compression_ratio,
        decoder_mode=cfg.decoder_mode,
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
            for batch in pbar:
                B = batch.size(0)
                batch = normalizer['tactile'].normalize(batch)
                batch = batch.to(cfg.device)
                if cfg.data_mode == 'single_hand':
                    left_batch = batch[..., :3]
                    right_batch = batch[..., 3:]
                    batch = torch.cat([left_batch, right_batch], dim=0)
                batch = rearrange(batch, 'b t h w c -> b c t h w')                
                optimizer.zero_grad()
                loss, recon_loss, kl_loss = model.calculate_loss(batch)
                
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

        if epoch % 5 == 0:
            model.eval()
            val_loss = 0.0
            val_recon_loss = 0.0
            val_kl_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                    B = batch.size(0)
                    batch = normalizer['tactile'].normalize(batch)
                    batch = batch.to(cfg.device)
                    if cfg.data_mode == 'single_hand':
                        left_batch = batch[..., :3]
                        right_batch = batch[..., 3:]
                        batch = torch.cat([left_batch, right_batch], dim=0)
                    batch = rearrange(batch, 'b t h w c -> b c t h w')
                    loss, recon_loss, kl_loss = model.calculate_loss(batch, val=True)
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
    val_dataset = TactileDataset(data_path, training=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False)
    normalizer = torch.load(os.path.join(data_path, 'normalizer.pth'))
    vae = TemporalTactileVAE(
        input_dim=cfg.input_dim, 
        latent_dim=cfg.latent_dim,
        hidden_state=cfg.hidden_state,
        time_compression_ratio=cfg.time_compression_ratio,
        spatial_compression_ratio=cfg.spatial_compression_ratio,
        decoder_mode=cfg.decoder_mode,
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
            vis_tactile(recon, gt)
        
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

    if cfg.data_mode == "single_hand":
        cfg.input_dim = 3
    elif cfg.data_mode == "bi_hand":
        cfg.input_dim = 6

    if args.test:
        eval_temporal_vae(cfg)
    else:
        train_temporal(cfg)
