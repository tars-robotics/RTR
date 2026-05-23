world_size=2 #4                  # = GPU count when training on multiple GPUs
export CUDA_VISIBLE_DEVICES=1,2
per_device_batch_size=4 # 8 for 6-GPU setup
global_batch_size=$((per_device_batch_size * world_size))

dataset_root_dir=${TFDS_DATA_DIR}
dataset_name="write_board60hz_oft6drotate_dataset"
run_root_dir="../../openvla_oft/openvla-oft/data/write_board_60hz_oft_6drotate_latent_with_rdp_vae"
vla_path="moojink/openvla-7b-oft-finetuned-libero-spatial"
run_id_override="write_board_60hz_6drorate_latent_with_rdp_vae"

# vae
vae_config_path="configs/rdp_vae/rdp_vae.yaml"
vae_ckpt_path="../../../data/ckpts/write_board_60hz/ckpts_abs/rdp_vae/n_embed_10/horizon48/latest.ckpt"
vae_latent_dataset_statistics_path="../../../data/ckpts/write_board_60hz/ckpts_abs/rdp_vae/n_embed_10/horizon48/dataset_stats.json"
horizon=48
normalize_latent=True


export PYTHONPATH="."

# save_freq 5000

torchrun --standalone --nnodes 1 --nproc-per-node ${world_size} vla-scripts/finetune_latent_oft_with_rdp_vae.py \
                                    --data_root_dir ${dataset_root_dir} \
                                    --dataset_name ${dataset_name} \
                                    --run_root_dir ${run_root_dir} \
                                    --vla_path ${vla_path} \
                                    --batch_size ${per_device_batch_size} \
                                    --save_freq 25000 \
                                    --use_lora True \
                                    --num_images_in_input 1 \
                                    --use_proprio True \
                                    --wandb_entity wkykaixin-shanghai-jiao-tong-university \
                                    --wandb_project openvla-oft-rdp-vae \
                                    --max_steps 50000 \
                                    --num_steps_before_decay 25000 \
                                    --run_id_override ${run_id_override} \
                                    --horizon ${horizon} \
                                    --vae_config_path ${vae_config_path} \
                                    --vae_ckpt_path ${vae_ckpt_path} \
                                    --image_aug True \
                                    --normalize_latent ${normalize_latent} \
                                    --vae_latent_dataset_statistics_path ${vae_latent_dataset_statistics_path}
