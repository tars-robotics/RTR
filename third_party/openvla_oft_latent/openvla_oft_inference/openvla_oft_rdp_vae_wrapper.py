from typing import Dict, Any, Union, Optional, List
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from rtr_async_sys.core.model_wrapper_base import AbsModelWrapper
from rtr_async_sys.utils.action_utils import ortho6d_to_rotation_matrix
import transforms3d as t3d

# from hello_model import HelloModel
import pickle
from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector, get_vla, get_vla_action, get_noisy_action_projector
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

from rtr_async_sys.utils.image_utils import decompress_image, decompress_video_frames
from rtr_async_sys.utils.action_utils import ortho6d_to_rotation_matrix
import os
from rtr_async_sys.models.reactive_diffusion_policy.model.vae.model import VAE

import PIL
from PIL import Image

def convert_9d_state_to_8d_state(state:np.ndarray):
    if state.shape == (8,):
        return state
    assert state.shape == (9, )

    state = state[None,:]
    zeros_array = np.zeros((state.shape[0], 2), dtype=state.dtype)

    left_rot_mat_batch = ortho6d_to_rotation_matrix(state[:, 3:9])  #(action_steps, 3, 3)
    left_euler_batch = np.array([t3d.euler.mat2euler(rot_mat) for rot_mat in left_rot_mat_batch])  # (action_steps, 3)
    left_trans_batch = state[:, :3]  # (action_steps, 3)
    left_action_8d = np.concatenate([left_trans_batch, left_euler_batch, zeros_array], axis=1) # (action_steps, 8)
    
    return left_action_8d[0]



class OpenvlaOftModelWrapper(AbsModelWrapper):
    def __init__(
            self, 
            model, 
            device, 
            ckpt_path="data/ckpts/openvla-oft/vase_sponge_test1_oft/openvla-7b-oft-finetuned-libero-spatial+vase_sponge_test1_oft_dataset+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--45000_chkpt",
            vae_ckpt_path=None,
            return_raw_action = False, need_refine = False, build_policy=False, prompt = "wipe the vase.",
            map_dict = None,
            compress_obs:bool = False,
            unnorm_key="vase_sponge_test1_oft_dataset",
            dataset_statistics_path="data/ckpts/vase_sponge_test1_60hz/ckpts_abs/openvla_oft/vae/horizon48_compress4_n_embed_10/dataset_statistics.json",
            normalize_latent=True,
            use_diffusion=False,
            block_reuse:bool=False,
            reuse_action_num:int=24,
            openloop_eval:bool=False
        ):
        super().__init__(None, device, ckpt_path=None, return_raw_action=return_raw_action, need_refine=need_refine, build_policy=build_policy, compress_obs=compress_obs)
        vae = model
        if isinstance(vae, str):
            vae = OmegaConf.load(vae)
        # vae.normalize_latent = normalize_latent
        self.normalize_latent = normalize_latent
        if isinstance(vae, DictConfig):
            vae = hydra.utils.instantiate(vae)
        self.vae:VAE = vae
        payload = torch.load(vae_ckpt_path, weights_only=False, map_location="cpu")
        self.vae.load_state_dict(payload['state_dicts']['model'])
        self.dataset_statistics_path = dataset_statistics_path
        self.vae._load_latent_dataset_statistics(self.dataset_statistics_path)
        # Move vae to the correct device and set to evaluation mode, just on cpu
        self.vae.to(device)
        self.vae.eval()
        
        if map_dict == None:
            map_dict = {
                'full_image': 'left_wrist_img',
                'state': 'left_robot_tcp_pose',
            }
        self.map_dict = map_dict

        self.language_instruction = prompt
        use_l1_regression = not use_diffusion

        self.vla_cfg = GenerateConfig(
            # pretrained_checkpoint = "moojink/openvla-7b-oft-finetuned-libero-spatial",
            pretrained_checkpoint=ckpt_path,
            # use_l1_regression = True,
            # use_diffusion = False,
            use_film = False,
            num_images_in_input = 1,
            use_proprio = True,
            load_in_8bit = False,
            load_in_4bit = False,
            # center_crop = True,
            # num_open_loop_steps = NUM_ACTIONS_CHUNK,
            unnorm_key=unnorm_key,
            use_l1_regression=use_l1_regression,
            use_diffusion=use_diffusion
            # unnorm_key = "libero_spatial_no_noops",
        )
        self.use_diffusion = use_diffusion


        self.vla = get_vla(self.vla_cfg)
        self.processor = get_processor(self.vla_cfg)
        # Load MLP action head to generate continuous actions (via L1 regression)
        self.action_head = get_action_head(self.vla_cfg, llm_dim=self.vla.llm_dim)
        # Load proprio projector to map proprio to language embedding space
        self.proprio_projector = get_proprio_projector(self.vla_cfg, llm_dim=self.vla.llm_dim, proprio_dim=PROPRIO_DIM)
        if use_diffusion:
            self.noisy_action_projector = get_noisy_action_projector(self.vla_cfg, llm_dim=self.vla.llm_dim)
        else:
            self.noisy_action_projector = None
        
        ### for block_reuse
        self.block_reuse = block_reuse
        self.reuse_action_num = reuse_action_num
        self.last_action_chunk = None
        self.openloop_eval = openloop_eval

    def reset(self):
        self.last_action_chunk = None
    
    def predict_action_chunk(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Call preprocess_obs and postprocess_action inside this function.
        """
        observation = self.preprocess_obs(obs_dict)

        if not self.block_reuse:
            action_chunk = get_vla_action(
                self.vla_cfg, 
                self.vla, 
                self.processor, 
                observation, 
                observation["language_instruction"], 
                self.action_head, 
                self.proprio_projector,
                noisy_action_projector = self.noisy_action_projector
            )
            action_chunk = torch.tensor(action_chunk).to(self.device)
            if self.normalize_latent:
                print("normalize latent in oft-latent")
                action_chunk = self.vae.denormalize_from_dataset(action_chunk.to(torch.float32)[None,:], is_latent=True)[0]
            action_chunk = self.vae.decode_from_latent(action_chunk.to(torch.float32)[None,:])
            action_chunk = action_chunk.detach().cpu().numpy()[0]
        else:
            action_chunk = get_vla_action(
                self.vla_cfg, 
                self.vla, 
                self.processor, 
                observation, 
                observation["language_instruction"], 
                self.action_head, 
                self.proprio_projector,
                noisy_action_projector = self.noisy_action_projector
            )
            action_chunk = torch.tensor(action_chunk).to(self.device)
            if self.normalize_latent:
                print("normalize latent in oft-latent")
                action_chunk = self.vae.denormalize_from_dataset(action_chunk.to(torch.float32)[None,:], is_latent=True)[0]
            action_chunk = self.vae.decode_from_latent(action_chunk.to(torch.float32)[None,:]).to(self.device)

            if self.last_action_chunk is None:
                # store action_chunk to be reused in next inference
                self.last_action_chunk = action_chunk.detach()
                action_chunk = action_chunk.detach().cpu().numpy()[0]
            else:
                if not self.openloop_eval:
                    now_action_chunk = action_chunk.detach()
                    reuse_action_chunk = self.last_action_chunk[:, -self.reuse_action_num:, :] # (b, reuse_action_num, action_dim)
                    concate_action_chunk = torch.concatenate([reuse_action_chunk, now_action_chunk[:, self.reuse_action_num:, :]], dim=1) # (b, horizon, action_dim)
                    
                    concate_action_latent_chunk = self.vae.encode_to_latent(concate_action_chunk)
                    concate_action_chunk = self.vae.decode_from_latent(concate_action_latent_chunk).to(self.device)

                    self.last_action_chunk = concate_action_chunk.detach()
                    action_chunk = concate_action_chunk.detach().cpu().numpy()[0]
                else:  # need to handle `switch episode`
                    now_action_chunk = action_chunk.detach()

                    this_x = now_action_chunk[0,0,0]
                    if self.last_x < 0 and this_x > 0:
                        self.last_action_chunk = action_chunk.detach()
                        action_chunk = action_chunk.detach().cpu().numpy()[0]
                    else:
                        reuse_action_chunk = self.last_action_chunk[:, -self.reuse_action_num:, :] # (b, reuse_action_num, action_dim)
                        concate_action_chunk = torch.concatenate([reuse_action_chunk, now_action_chunk[:, self.reuse_action_num:, :]], dim=1) # (b, horizon, action_dim)
                        concate_action_latent_chunk = self.vae.encode_to_latent(concate_action_chunk)
                        concate_action_chunk = self.vae.decode_from_latent(concate_action_latent_chunk).to(self.device)

                        self.last_action_chunk = concate_action_chunk.detach()
                        action_chunk = concate_action_chunk.detach().cpu().numpy()[0]

            self.last_x = action_chunk[0][0]


        action_chunk = self.postprocess_action(action_chunk)

        return action_chunk

    def preprocess_obs(self, obs: Dict[str, Any]) -> PIL.Image:
        # Stack multi-frame observations into a batch
        if isinstance(obs, list):
            if self.compress_obs:
                for obs_item in obs:  # Real-robot env returns a list, so images can be compressed conveniently
                    for key in obs_item.keys():
                        if 'img' in key:
                            obs_item[key] = decompress_image(obs_item[key])

            obs = obs[-1]
        else:
            if self.compress_obs:
                for key in obs.keys():
                    if 'img' in key:
                        obs[key] = decompress_video_frames(obs[key])

            for key in obs.keys():
                if len(obs[key].shape) == 4:# image [t,h,w,c]
                    obs[key] = obs[key][-1]
                if len(obs[key].shape) == 2:# state [t,dim]
                    obs[key] = obs[key][-1]

        new_obs = {}
        for key in self.map_dict:
            map_key = self.map_dict[key]
            item = obs[map_key]
            if key == 'state':
                item = convert_9d_state_to_8d_state(item)
            new_obs[key] = item
        
        new_obs['language_instruction'] = self.language_instruction
        
        return new_obs

    def postprocess_action(self, raw_action_chunk: Union[np.ndarray, List]) -> np.ndarray:
        """
        Currently only single-arm is supported. \\
        Input is a (10,)-shaped action: xyz + 6d_rotation + gripper. \\
        Outputs: 1. raw model output for dataset_env, used to compute L1 loss and other open-loop metrics; 2. executable action for real_env [x,y,z,yaw,pitch,roll] or [x,y,z,yaw,pitch,roll,gripper_width,gripper_force]. \\
        Angles are in radians; convert to degrees inside the env as needed.
        """
        if isinstance(raw_action_chunk, List):
            raw_action_chunk = np.stack(raw_action_chunk)
        assert len(raw_action_chunk.shape) == 2, "input in postprocess_action should be action chunk"  # (action_steps, d_a)
        if raw_action_chunk.shape[1] == 10:
            left_rot_mat_batch = ortho6d_to_rotation_matrix(raw_action_chunk[:, 3:9])  # (action_steps, 3, 3)
            left_euler_batch = np.array([t3d.euler.mat2euler(rot_mat) for rot_mat in left_rot_mat_batch])  # (action_steps, 3)
            left_trans_batch = raw_action_chunk[:, :3]  # (action_steps, 3)
            left_action_7d = np.concatenate([left_trans_batch, left_euler_batch, raw_action_chunk[:,9:]], axis=1) # (action_steps, 7)
            return left_action_7d
        else: 
            assert raw_action_chunk.shape[1] == 7, ""
            return raw_action_chunk

@hydra.main(
    config_path="configs/model_wrapper",
    config_name="openvla_oft_model_wrapper",
    version_base=None
)
def main(cfg: DictConfig):
    if isinstance(cfg, str):
        cfg = OmegaConf.load(cfg)
    # cfg.device = "cuda:1" # openvla-oft does not seem to support setting the device via cfg; use CUDA_VISIBLE_DEVICES=xxx instead
    model_wrapper:OpenvlaOftModelWrapper = hydra.utils.instantiate(cfg)
    print(model_wrapper)

    image_path = os.environ.get("OPENVLA_OFT_SAMPLE_IMAGE", "data/sample_images/wipe.png")
    image = Image.open(image_path)
    image = image.convert("RGB")
    image_np = np.array(image)
    state = np.zeros((8,), dtype=np.float32)

    obs_dict = {"left_wrist_img": image_np, "left_robot_tcp_pose": state}
    # import pdb; pdb.set_trace()
    action_chunk = model_wrapper.predict_action_chunk(obs_dict)
    print(action_chunk)

if __name__ == '__main__':
    main()
