# horizon 48 (horizon in openvla is 8), latent, compress 4
export PYTHONPATH="$(pwd)/third_party/openvla_oft/openvla-oft/LIBERO:${PYTHONPATH}"

# not-two-stage
vae_config_path="configs/rdp_vae/rdp_vae.yaml"
vae_ckpt_path="data/ckpts/vase_sponge_test1_60hz/ckpts_abs/rdp_vae/n_embed_10/horizon48/latest.ckpt"
vae_latent_dataset_statistics_path="data/ckpts/vase_sponge_test1_60hz/ckpts_abs/rdp_vae/n_embed_10/horizon48/dataset_stats.json"


# normalize latent, image_aug
ckpt_path="data/ckpts/openvla_oft_latent/vase_sponge_test1_60hz_oft_6drotate_img_aug_normalize_latent_rdp_vae/60hz_6drotate_horizon48_latent_compress4_img_aug_true_normalize_latent--50000_chkpt"
# dataset_statistics_path="data/ckpts/openvla_oft_latent/vase_sponge_test1_60hz_oft_6drotate_aug_normalize_latent/60hz_6drotate_horizon48_latent_compress4_img_aug_true_normalize_latent--50000_chkpt/dataset_statistics.json"

normalize_latent=True

CUDA_VISIBLE_DEVICES=0 python -m rtr_async_sys.user.simple_user \
    --config-dir "$(pwd)/third_party/openvla_oft_latent/openvla-oft/configs" --config-name openvla_oft_user \
    model_wrapper=openvla_oft_rdp_vae_model_wrapper \
    model_wrapper.ckpt_path=${ckpt_path} \
    model_wrapper.compress_obs=False \
    model_wrapper.unnorm_key="vase_sponge_test1_oft_6drotate_dataset" \
    model_wrapper.model=${vae_config_path} \
    model_wrapper.vae_ckpt_path=${vae_ckpt_path} \
    model_wrapper.dataset_statistics_path=${vae_latent_dataset_statistics_path} \
    model_wrapper.normalize_latent=${normalize_latent} \
    model_wrapper.use_diffusion=False
