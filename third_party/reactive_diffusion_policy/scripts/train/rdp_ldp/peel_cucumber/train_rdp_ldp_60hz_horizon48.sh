# #!/bin/bash

GPU_ID=4

horizon=48

TASK_NAME="wipe"
DATASET_PATH="../../data/ckpts/peel_cucumber_60hz/rdp_zarr"
LOGGING_MODE="online"
TIMESTAMP=peel_cucumber_rdp_ldp_60hz_horizon${horizon}
SEARCH_PATH="./data/outputs"



AT_LOAD_DIR="../../data/ckpts/peel_cucumber_60hz/ckpts_abs/rdp_vae/n_embed_10/horizon48/latest.ckpt"

echo ""
echo "Stage 2: training Latent Diffusion Policy..."
CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch train.py \
    --config-name=train_latent_diffusion_unet_real_image_workspace \
    +policy.change_kernel_size=False \
    policy.noise_scheduler.num_train_timesteps=30 \
    policy.num_inference_steps=30 \
    task=real_${TASK_NAME}_image_gelsight_emb_ldp_24fps \
    task.dataset_path=${DATASET_PATH} \
    task.dataset.relative_action=False \
    task.name=real_${TASK_NAME}_ldp_${TIMESTAMP} \
    at=at_wipe_lift \
    at_load_dir=${AT_LOAD_DIR} \
    logging.mode=${LOGGING_MODE} \
    at.dataset_obs_temporal_downsample_ratio=1 \
    at.horizon=${horizon} \
    at.n_obs_steps=1 \
    at.policy.use_rnn_decoder=False \
    at.policy.n_embed=10
