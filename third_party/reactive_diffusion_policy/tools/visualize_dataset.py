"""
Usage:
bash test_dataset.sh

Inspect the dataset contents (prerequisite for the visualization script).

(Pdb) data.keys()
dict_keys(['obs', 'action', 'extended_obs'])
(Pdb) data["obs"].keys()
dict_keys(['left_wrist_img', 'left_robot_tcp_pose', 'left_robot_gripper_width', 'left_gripper1_marker_offset_emb'])
(Pdb) data["action"].shape
torch.Size([1, 32, 10])
(Pdb) data['extended_obs'].keys()
dict_keys(['left_gripper1_marker_offset_emb'])
"""

import sys
import os
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from reactive_diffusion_policy.workspace.base_workspace import BaseWorkspace

from reactive_diffusion_policy.dataset.base_dataset import BaseImageDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch
import numpy as np

from reactive_diffusion_policy.common.action_utils import relative_actions_to_absolute_actions, absolute_actions_to_relative_actions


def vis_obs(cfg: OmegaConf):
    taset: BaseImageDataset
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    assert isinstance(dataset, BaseImageDataset)
    output_dir = "data/outputs/vis_outputs/vis_dataset"
    os.makedirs(output_dir, exist_ok=True)

    vis_len = 20
    for i in range(vis_len):
        data = dataset[i]
        image_1 = data['obs']['left_wrist_img'][0]# torch.Tensor, torch.Size([3, 240, 320])
        image_2 = data['obs']['left_wrist_img'][1]
        image_all = torch.cat((image_1.unsqueeze(0), image_2.unsqueeze(0)), dim=0)
        print(image_1.shape, image_2.shape, image_all.shape)
        # TODO: visualize image_1 and image_2 with filenames i+image_1 / i+image_2
        # cast to float for saving if not already float
        if image_1.dtype != torch.float32:
            image_1 = image_1.float() / 255.0
        if image_2.dtype != torch.float32:
            image_2 = image_2.float() / 255.0

        # build filenames and save
        save_path_1 = os.path.join(output_dir, f"{i:03d}_image_1.png")
        save_path_2 = os.path.join(output_dir, f"{i:03d}_image_2.png")

        save_image(image_1, save_path_1)
        save_image(image_2, save_path_2)

        print(f"[OK] Saved {save_path_1} and {save_path_2}")

        
def print_xyz(cfg:OmegaConf):
    taset: BaseImageDataset
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    assert isinstance(dataset, BaseImageDataset)
    data = dataset[0]
    actions = data['action']
    base_position=data['obs']['left_robot_tcp_pose'][-1]
    # print("base_postion:", base_position)
    print("relative_to_abs_base_position:", relative_actions_to_absolute_actions(data['obs']['left_robot_tcp_pose'].numpy(),base_position.numpy()))
    abs_base_position = relative_actions_to_absolute_actions(data['obs']['left_robot_tcp_pose'].numpy(),base_position.numpy())[-1]
    print("base_postion:", base_position)
    print("abs_base_postiion:", abs_base_position)
    
    abs_actions = relative_actions_to_absolute_actions(actions.numpy(), base_position.numpy())
    abs_abs_actions = relative_actions_to_absolute_actions(actions.numpy(), abs_base_position)

    for i in range(32):
        print(f"action.shape is {actions[i].shape}")
        print("relative: ", actions[i][0:3])
        # abs
        # abs_action = 
        print("abs: ", abs_actions[i][0:3])



# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        '../reactive_diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    vis_obs(cfg)
    # print_xyz(cfg)

if __name__ == "__main__":
    main()
