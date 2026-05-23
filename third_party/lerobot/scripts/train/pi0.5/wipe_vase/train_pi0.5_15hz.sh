# When debugging the code, disable torch.compile first: TORCHDYNAMO_DISABLE=1 bash scripts/train/pi0.5/xarm_sponge_test1_15hz/test.sh
# state: xyz+6d rorate; action: xyz+6drotate+1d_gripper

# steps=50000
# save_freq=5000
steps=40000
save_freq=20000

dataset="<your_username>/xarm_wipe_vase_state9_15hz"
# With torch.compile disabled, batchsize=20, 3000 steps. world_size 1: 1h44min; 2: 2h14min; 4: 2h18min
# for 20000steps. 1: 13h8min; 4: 15h20min
# With torch.compile enabled, batchsize=20, 3000 steps. world_size 1: xxh; 2: xx; 4: xx
## VRAM usage grows with world_size. batchsize=20 on 4 GPUs is the memory limit, even though a single GPU only uses ~34 GB.
world_size=1 
run_id=train_logl1_wipe_vase_15hz_world${world_size}_steps${steps} # pi05_training
output_dir="outputs/pi05_training_logl1_wipe_vase_15hz_steps${steps}_world${world_size}"


# rm -rf ${output_dir}

# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/codes/train_pi0.5/lerobot_train.py \
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
    --policy.chunk_size=12 \
    --policy.n_action_steps=12 \
    --policy.max_action_dim=10 \
    --policy.max_state_dim=9 \
    --policy.repo_id="<your_username>/xarm_wipe_vase_15hz_pi0.5" \
    --policy.push_to_hub=False \
    --wandb.project="lerobot_pi0.5" \
    --wandb.run_id=${run_id} \
    --log_freq=5 \
    --eval_freq=10
    # --rename_map '{"observation.images.image":"image"}'