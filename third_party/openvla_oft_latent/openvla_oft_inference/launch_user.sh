# horizon 48 (horizon in openvla is 8), latent, compress 4
export PYTHONPATH="$(pwd)/third_party/openvla_oft/openvla-oft/LIBERO:${PYTHONPATH}"

# not-two-stage
vae_ckpt_path="data/ckpts/vase_sponge_test1_60hz/ckpts_abs/openvla_oft/vae/horizon48_compress4_n_embed_10/latest.ckpt"
# two-stage
# vae_ckpt_path="data/ckpts/openvla_oft_latent/vase_sponge_test1_60hz_oft_openvla_oft_vae_two_stage/20251225-040257_60hz_6drotate_horizon48_openvla_oft_vae_two_stage/checkpoints/latest.ckpt"


# not normalize latent, image_aug
# ckpt_path="data/ckpts/openvla_oft_latent/vase_sponge_test1_60hz_oft_6drotate_latent/60hz_6drotate_horizon48_latent_compress4--50000_chkpt"
# dataset_statistics_path="data/ckpts/vase_sponge_test1_60hz/ckpts_abs/openvla_oft/vae/horizon48_compress4_n_embed_10/dataset_statistics.json"
# normalize_latent=False

# not normalize latent, not image_aug
# ckpt_path="data/ckpts/openvla_oft_latent/vase_sponge_test1_60hz_oft_6drotate_latent_image_aug_false/60hz_6drotate_horizon48_latent_compress4_image_aug_false--50000_chkpt"
# dataset_statistics_path="data/ckpts/openvla_oft_latent/vase_sponge_test1_60hz_oft_6drotate_latent_image_aug_false/60hz_6drotate_horizon48_latent_compress4_image_aug_false--50000_chkpt/dataset_statistics.json"
# normalize_latent=False

# normalize latent, image_aug
ckpt_path="data/ckpts/openvla_oft_latent/vase_sponge_test1_60hz_oft_6drotate_aug_normalize_latent/60hz_6drotate_horizon48_latent_compress4_img_aug_true_normalize_latent--50000_chkpt"
# ckpt_path="data/ckpts/openvla_oft_latent/vase_sponge_test1_60hz_oft_6drotate_aug_normalize_latent/60hz_6drotate_horizon48_latent_compress4_img_aug_true_normalize_latent--50000_chkpt"
dataset_statistics_path="data/ckpts/openvla_oft_latent/vase_sponge_test1_60hz_oft_6drotate_aug_normalize_latent/60hz_6drotate_horizon48_latent_compress4_img_aug_true_normalize_latent--50000_chkpt/dataset_statistics.json"
# normalize_latent=True

# normalize latent, not image_aug
# ckpt_path="data/ckpts/openvla_oft_latent/vase_sponge_test1_60hz_oft_6drotate_not_aug_normalize_latent/60hz_6drotate_horizon48_latent_compress4_img_aug_normalize_latent--50000_chkpt"
# ckpt_path="data/ckpts/openvla_oft_latent/vase_sponge_test1_60hz_oft_6drotate_aug_normalize_latent_diffusion/60hz_6drotate_horizon48_latent_compress4_img_aug_true_normalize_latent_diffusion--50000_chkpt"

# dataset_statistics_path="data/ckpts/openvla_oft_latent/vase_sponge_test1_60hz_oft_6drotate_not_aug_normalize_latent/60hz_6drotate_horizon48_latent_compress4_img_aug_normalize_latent--50000_chkpt/dataset_statistics.json"
normalize_latent=True

CUDA_VISIBLE_DEVICES=0 python -m rtr_async_sys.user.simple_user \
    --config-dir "$(pwd)/third_party/openvla_oft_latent/openvla-oft/configs" --config-name openvla_oft_user \
    model_wrapper.ckpt_path=${ckpt_path} \
    model_wrapper.compress_obs=False \
    model_wrapper.unnorm_key="vase_sponge_test1_oft_6drotate_dataset" \
    model_wrapper.model="$(pwd)/src/rtr_async_sys/configs/user/model_wrapper/model/rdp/openvla_oft_vae.yaml" \
    model_wrapper.vae_ckpt_path=${vae_ckpt_path} \
    model_wrapper.dataset_statistics_path=${dataset_statistics_path} \
    model_wrapper.normalize_latent=${normalize_latent} \
    model_wrapper.use_diffusion=False
