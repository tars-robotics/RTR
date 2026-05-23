dataset_root_dir=${TFDS_DATA_DIR}

world_size=2                  # = GPU count when training on multiple GPUs
export CUDA_VISIBLE_DEVICES=4,5
per_device_batch_size=4 # 8 for 6-GPU setup
global_batch_size=$((per_device_batch_size * world_size))

dataset_name="wipe_vase15hz_oft6drotate_dataset"
run_root_dir="./data/wipe_vase_15hz_oft_6drotate_horizon48_not_latent"
run_id_override="wipe_vase_15hz_not_latent_worldsize${world_size}"
vla_path="moojink/openvla-7b-oft-finetuned-libero-spatial"



export PYTHONPATH="."

# save_freq 5000

torchrun --standalone --nnodes 1 --nproc-per-node ${world_size} vla-scripts/finetune.py \
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
                                    --wandb_project openvla-oft \
                                    --max_steps 50000 \
                                    --num_steps_before_decay 25000 \
                                    --run_id_override ${run_id_override}
