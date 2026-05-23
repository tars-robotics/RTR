if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np

from reactive_diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from reactive_diffusion_policy.workspace.base_workspace import BaseWorkspace
from reactive_diffusion_policy.model.vae.model import VAE
from reactive_diffusion_policy.dataset.base_dataset import BaseImageDataset
from reactive_diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from reactive_diffusion_policy.common.json_logger import JsonLogger
from reactive_diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

class EvalATWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: VAE
        self.model = hydra.utils.instantiate(cfg.policy)

        payload = torch.load(cfg.vae_load_path)
        # Load the model weights
        self.model.load_state_dict(payload['state_dicts']['model'])
        self.model.eval()

        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)


        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        # normalizer = dataset.get_normalizer()

        # configure validation dataset
        # val_dataset = dataset.get_validation_dataset()
        # val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        # self.model.set_normalizer(normalizer)

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        self.model.eval()

        total_losses = []
        plot_actions = []
        with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
            for batch_idx, batch in enumerate(tepoch):
                # device transfer
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                # l1loss
                gt_action = batch['action'].to(device)
                predict_action = self.model.encode_then_decode(batch).to(device)
                # import pdb;pdb.set_trace()
                # print(f"gt_action.shape is {gt_action.shape}")
                l1loss = torch.mean(torch.abs(predict_action - gt_action))
                total_losses.append(l1loss)

                # delta xyz
                predict_action = predict_action.detach().cpu().numpy()
                gt_action = gt_action.detach().cpu().numpy()
                plot_action = {
                        'fact':[],
                        'predict':[] ,
                }
                for i in range(predict_action.shape[1]):
                    plot_action['fact'].append(gt_action[0][i][0:3]*1000)
                    plot_action['predict'].append(predict_action[0][i][0:3]*1000)

                total_losses.append(l1loss)
                plot_actions.append(plot_action)

                #
        mean_loss = sum(total_losses)/len(total_losses)
        print(f"mean loss is {mean_loss}")
        print(f"total step is {len(total_losses)}")

        from rtr_async_sys.tools.utils import mean_abs_delta_xyz
        stats_xyz = mean_abs_delta_xyz(plot_actions)
        print("mean abs delta (mm):", stats_xyz)

    

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = EvalATWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()