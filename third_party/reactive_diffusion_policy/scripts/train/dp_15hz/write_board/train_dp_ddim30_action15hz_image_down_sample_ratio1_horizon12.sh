# a#!/bin/bash
# a#!/bin/bash
n_obs_steps=1
horizon=12 # corresponds to horizon 96 at 60hz


CUDA_VISIBLE_DEVICES=0 accelerate launch train.py \
    --config-name=train_diffusion_unet_real_image_workspace \
    task=real_wipe_image_gelsight_emb_dp_ablation_ensemble_absolute_12fps \
    task.dataset_path=../../data/ckpts/write_board_15hz/rdp_zarr \
    task.name=dp_ddim30_15hz_obs1_write_board_horizon${horizon} \
    logging.mode=online \
    policy.noise_scheduler.num_train_timesteps=30 \
    policy.num_inference_steps=30 \
    horizon=${horizon} \
    n_action_steps=${horizon} \
    +task.dataset.image_downsample_ratio=1 \
    +policy.image_downsample_ratio=1 \
    n_obs_steps=${n_obs_steps}