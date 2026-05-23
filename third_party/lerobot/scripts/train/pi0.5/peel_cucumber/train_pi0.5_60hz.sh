
steps=40000
save_freq=20000

dataset="<your_username>/xarm_peel_cucumber_state9_60hz"

world_size=1 
run_id=train_logl1_cucumber_60hz_world${world_size}_steps${steps} # pi05_training
output_dir="outputs/pi05_training_logl1_cucumber_60hz_steps${steps}_world${world_size}"


TORCHDYNAMO_DISABLE=1 CUDA_VISIBLE_DEVICES=4 accelerate launch --num_processes=${world_size} scripts/codes/train_pi0.5/lerobot_train.py \
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
    --policy.chunk_size=48 \
    --policy.n_action_steps=48 \
    --policy.max_action_dim=10 \
    --policy.max_state_dim=9 \
    --policy.repo_id="<your_username>/xarm_peel_cucumber_60hz_pi0.5" \
    --policy.push_to_hub=False \
    --wandb.project="lerobot_pi0.5" \
    --wandb.run_id=${run_id} \
    --log_freq=5 \
    --eval_freq=10
    # --rename_map '{"observation.images.image":"image"}'