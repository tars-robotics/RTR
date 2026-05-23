"""
Construct the model and env directly and run without the system interfaces; used only for testing code.
"""
import sys
import os
import pathlib
import hydra
from omegaconf import OmegaConf, DictConfig
import torch
import numpy as np
from termcolor import cprint
import copy
import threading
# from real_sensors import RealRobotEnv
# from real_sensors_from_dataset import RealRobotEnv
from rtr_async_sys.env.dp_dataset_env import DpDatasetEnv, build_dp_dataset_env
# ROOT_DIR = str(pathlib.Path(__file__).parent)
# sys.path.append(ROOT_DIR)
# os.chdir(ROOT_DIR)
# from reactive_diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from rtr_async_sys.models.reactive_diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from rtr_async_sys.models.dp import build_dp_model

from typing import Union
from loguru import logger

OmegaConf.register_new_resolver("eval", eval, replace=True)
input_key_list = ['left_wrist_img', 'left_robot_tcp_pose', 'left_robot_gripper_width', 'left_gripper1_marker_offset_emb']

class FakeDpRunner:
    def __init__(self, ckpt_path:str, env_cfg:Union[DictConfig, DpDatasetEnv,str], model_cfg:Union[DictConfig, DpDatasetEnv,str], device:str='cuda:0'):
        # =========== Load configuration ===========
        logger.info(f"Init FakeDpRunner")
        self.device = torch.device(device)
        if isinstance(env_cfg, str):
            env_cfg = OmegaConf.load(env_cfg)
        if isinstance(model_cfg, str):
            model_cfg = OmegaConf.load(model_cfg)
        if isinstance(env_cfg, DictConfig):
            self.env:DpDatasetEnv = build_dp_dataset_env(env_cfg)
        if isinstance(model_cfg, DictConfig):
            self.policy:DiffusionUnetImagePolicy = build_dp_model(model_cfg)

        # =========== Load checkpoint ===========
        cprint(f"Loading checkpoint from: {ckpt_path}", "yellow")
        payload = torch.load(ckpt_path, map_location=self.device)

        
        # Load the model weights
        self.policy.load_state_dict(payload['state_dicts']['model'])
        
        
        # Move policy to the correct device and set to evaluation mode
        self.policy.to(self.device)
        self.policy.eval()

        # =========== Initialize observation buffer ===========
        # self.n_obs_steps = policy_cfg.n_obs_steps
        self.n_obs_steps = self.policy.n_obs_steps
        # Get the observation keys from the training config's shape_meta
        self.n_action_steps = self.policy.n_action_steps
        # self.key_to_shape = train_cfg.shape_meta['obs']
        self.key_to_shape = self.policy.obs_encoder.shape_meta['obs']
        print("after init")


    def run(self):
        """Main inference loop."""
        print("Start inference loop...")
        input_dict = dict()
        plot_steps = 2

        try:
            # rossub_thread = threading.Thread(target=self.env.ros_thread, daemon=True)
            # rossub_thread.start()
            step_count = 0
            # while True:
            for _ in range(plot_steps):
                obs = self.env.get_obs()
                if obs is None:
                    continue
                else:
                    # Stack multi-frame observations into a batch
                    if isinstance(obs, list):
                        obs_processed = {
                            key: torch.from_numpy(np.stack([o[key] for o in obs])).unsqueeze(0).to(self.device)
                            for key in obs[0].keys()
                        }
                    else:
                        obs_processed = {
                            key: torch.from_numpy(obs[key]).unsqueeze(0).to(self.device)
                            for key in obs.keys()
                        }
                    # Data Processing
                    for key in obs_processed.keys():
                        if 'img' in key:
                            # print(obs_processed[key].shape)
                            # exit()
                            obs_processed[key] = obs_processed[key].permute(0, 1, 4, 2, 3)  # BNHWC -> BNCHW
                            obs_processed[key] = obs_processed[key].float() / 255.0 # real env needs / 255, make sure in dataset we have * 255, because the data in dataset is usually normalized
                    for key in input_key_list:
                        input_dict[key] = obs_processed[key]
                    # print("---------------")
                    # print(input_dict['left_robot_tcp_pose'])
                    # print(f"-----------------")
                    
                    # Data Processing
                    # Use the model to predict actions.
                    with torch.no_grad():
                        action_dict = self.policy.predict_action(input_dict)

                    # Extract the action sequence.
                    action_sequence = action_dict['action'].detach().cpu().numpy()[0]
                    
                    # Execute each action in the sequence.
                    # self.env.update_obs(action_dict['action_pred'].detach().cpu().numpy()[0])
                    for i in range(min(self.n_action_steps, len(action_sequence))):
                        action_step = action_sequence[i]
                        self.env.execute_action(action_step)
                    self.env.end_of_chunk()
                    
                    
                    step_count += 1
                    # if step_count >= self.env.max_steps:
                    #     print(f"Executed {50} steps; inference loop finished.")
                    #     break  # Or use break to exit while True.
            self.env.save_plot_actions()

        except KeyboardInterrupt:
            print("Inference interrupted by user.")
        finally:
            print("Program finished.")



@hydra.main(
    version_base=None,
    config_path="../configs/runner",
    # config_name="dp_infer"
    config_name="fake_dp_runner"
)
def main(cfg: Union[DictConfig, str]):
    # Create the inference runner and start the loop
    # OmegaConf.set_struct(cfg, False)
    # cfg.inference = {
    #     'ckpt_path': cfg.load_ckpt_path,
    #     'pca_path': cfg.load_pca_path,
    #     'robot_ip': '192.168.1.239'
    # }
    # OmegaConf.set_struct(cfg, True)
    # runner = FakeDpRunner(cfg)
    if isinstance(cfg, str):
        cfg = OmegaConf.load(cfg)
    print(cfg)
    runner = hydra.utils.instantiate(cfg)
    # runner = FakeDpRunner(ckpt_path=cfg['ckpt_path'], model_cfg=cfg['model_cfg'], env_cfg=cfg[])
    print(type(runner))
    runner.run()

if __name__ == "__main__":
    main()
 