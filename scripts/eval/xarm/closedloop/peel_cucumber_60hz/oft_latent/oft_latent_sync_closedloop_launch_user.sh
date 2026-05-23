cd third_party/openvla_oft_latent/openvla-oft

export PYTHONPATH="../../LIBERO:${PYTHONPATH}"
vae_config_path="configs/rdp_vae/rdp_vae.yaml"
vae_ckpt_path="../../../data/ckpts/peel_cucumber_60hz/ckpts_abs/rdp_vae/n_embed_10/horizon48/latest.ckpt"
vae_latent_dataset_statistics_path="../../../data/ckpts/peel_cucumber_60hz/ckpts_abs/rdp_vae/n_embed_10/horizon48/dataset_stats.json"
dataset_name="peel_cucumber60hz_oft6drotate_dataset"
ckpt_path="../../openvla_oft/openvla-oft/data/peel_cucumber_60hz_oft_6drotate_latent_with_rdp_vae/peel_cucumber_60hz_6drorate_latent_with_rdp_vae--50000_chkpt"
normalize_latent=True
compress_obs=False


python -m rtr_async_sys.user.simple_user \
    --config-dir configs --config-name openvla_oft_passive_user \
    model_wrapper=openvla_oft_rdp_vae_model_wrapper \
    model_wrapper.ckpt_path=${ckpt_path} \
    model_wrapper.compress_obs=${compress_obs} \
    model_wrapper.unnorm_key=${dataset_name} \
    model_wrapper.model=${vae_config_path} \
    model_wrapper.vae_ckpt_path=${vae_ckpt_path} \
    model_wrapper.dataset_statistics_path=${vae_latent_dataset_statistics_path} \
    model_wrapper.normalize_latent=${normalize_latent} \
    model_wrapper.use_diffusion=False
