steps=40000
save_freq=20000

dataset="<your_username>/xarm_peel_cucumber_state9_15hz"

world_size=1 
run_id=train_logl1_cucumber_15hz_world${world_size}_steps${steps} # pi05_training
output_dir="outputs/pi05_training_logl1_cucumber_15hz_steps${steps}_world${world_size}"


# rm -rf ${output_dir}

# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/codes/train_pi0.5/lerobot_train.py \
TORCHDYNAMO_DISABLE=1 CUDA_VISIBLE_DEVICES=2 accelerate launch --num_processes=${world_size} scripts/codes/train_pi0.5/lerobot_train.py \
    --dataset.repo_id=${dataset} \
    --policy.type=pi05 \
    --output_dir=${output_dir} \
    --job_name=${run_id} \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --policy.dtype=bfloat16 \
    --steps=${steps} \
    --save_freq=${save_freq} \
    --policy.device=cuda \
    --batch_size=20 \
    --policy.chunk_size=12 \
    --policy.n_action_steps=12 \
    --policy.max_action_dim=10 \
    --policy.max_state_dim=9 \
    --policy.repo_id="<your_username>/xarm_peel_cucumber_15hz_pi0.5" \
    --policy.push_to_hub=False \
    --wandb.project="lerobot_pi0.5" \
    --wandb.run_id=${run_id} \
    --log_freq=5 \
    --eval_freq=10
    # --rename_map '{"observation.images.image":"image"}'