cd third_party/lerobot/lerobot

ckpt_path="outputs/pi05_training_logl1_latent_rdp_vae_cucumber60hz_steps40000_world1/checkpoints/040000/pretrained_model"
dataset_id="sadpiggy/xarm_peel_cucumber_state9_60hz"
chunk_size=12

vae_config_path="configs/rdp_vae/rdp_vae.yaml"
latent_dataset_statistics="../../../data/ckpts/peel_cucumber_60hz/ckpts_abs/rdp_vae/n_embed_10/horizon48/dataset_stats.json"
vae_load_path="../../../data/ckpts/peel_cucumber_60hz/ckpts_abs/rdp_vae/n_embed_10/horizon48/latest.ckpt"

# port
ctrl_user=30022
user_sched=30052
user_ctrl_port="tcp://127.0.0.1:${ctrl_user}"
user_sched_port="tcp://127.0.0.1:${user_sched}"

python -m rtr_async_sys.user.simple_user \
    --config-dir configs --config-name pi0_5_passive_user \
    user_hz=20 \
    model_wrapper=pi0_5_latent_rdp_vae_model_wrapper \
    controller_endpoint=${user_ctrl_port} \
    scheduler_endpoint=${user_sched_port} \
    model_wrapper.compress_obs=False \
    model_wrapper.cfg.policy.pretrained_path=${ckpt_path} \
    model_wrapper.cfg.policy.chunk_size=${chunk_size} \
    model_wrapper.cfg.policy.n_action_steps=${chunk_size} \
    model_wrapper.cfg.dataset.repo_id=${dataset_id} \
    model_wrapper.vae_config_path=${vae_config_path} \
    model_wrapper.vae_load_path=${vae_load_path} \
    model_wrapper.latent_dataset_statistics=${latent_dataset_statistics} \