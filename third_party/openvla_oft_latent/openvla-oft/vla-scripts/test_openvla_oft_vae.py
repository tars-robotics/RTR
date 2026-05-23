from omegaconf import DictConfig, OmegaConf
import hydra
import torch

vae_config_path: str = "src/rtr_async_sys/configs/user/model_wrapper/model/rdp/openvla_oft_vae.yaml"
ckpt_path = "data/ckpts/vase_sponge_test1_60hz/ckpts_abs/rdp_vae/n_embed_10/horizon48/latest.ckpt"

vae_config = OmegaConf.load(vae_config_path)
vae = hydra.utils.instantiate(vae_config)
print(vae)
if ckpt_path != None:
    payload = torch.load(ckpt_path)
    # Load the model weights
    vae.load_state_dict(payload['state_dicts']['model'])
print(vae)