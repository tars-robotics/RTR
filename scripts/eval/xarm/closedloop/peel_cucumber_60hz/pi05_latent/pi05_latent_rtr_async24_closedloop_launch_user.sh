cd third_party/lerobot/lerobot

ckpt_path="outputs/pi05_training_logl1_latent_rdp_vae_cucumber60hz_steps40000_world1/checkpoints/040000/pretrained_model"
dataset_id="sadpiggy/xarm_peel_cucumber_state9_60hz"
chunk_size=12

vae_config_path="configs/rdp_vae/rdp_vae.yaml"
latent_dataset_statistics="../../../data/ckpts/peel_cucumber_60hz/ckpts_abs/rdp_vae/n_embed_10/horizon48/dataset_stats.json"
vae_load_path="../../../data/ckpts/peel_cucumber_60hz/ckpts_abs/rdp_vae/n_embed_10/horizon48/latest.ckpt"

block_reuse=True
reuse_action_num=24
openloop_eval=False
compress_obs=False


python -m rtr_async_sys.user.simple_user \
    --config-dir configs --config-name pi0_5_passive_user \
    user_hz=10 \
    model_wrapper=pi0_5_latent_rdp_vae_model_wrapper \
    model_wrapper.cfg.policy.pretrained_path=${ckpt_path} \
    model_wrapper.cfg.policy.chunk_size=${chunk_size} \
    model_wrapper.cfg.policy.n_action_steps=${chunk_size} \
    model_wrapper.cfg.dataset.repo_id=${dataset_id} \
    model_wrapper.vae_config_path=${vae_config_path} \
    model_wrapper.vae_load_path=${vae_load_path} \
    model_wrapper.latent_dataset_statistics=${latent_dataset_statistics} \
    model_wrapper.compress_obs=${compress_obs} \
    model_wrapper.block_reuse=${block_reuse} \
    model_wrapper.reuse_action_num=${reuse_action_num} \
    model_wrapper.openloop_eval=${openloop_eval} \
