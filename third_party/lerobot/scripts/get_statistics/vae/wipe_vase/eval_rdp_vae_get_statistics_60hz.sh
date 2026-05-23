steps=500000
save_freq=2000
batch_size=128

dataset="<your_username>/xarm_wipe_vase_state9_60hz_wo_img"
output_dir="outputs/vae/eval/pi05_eval_rdp_vae_wipe_vase_steps${steps}"

world_size=1 
run_id=test_vae_world${world_size} # pi05_training


vae_load_path="../../../data/ckpts/wipe_vase_60hz/ckpts_abs/rdp_vae/n_embed_10/horizon48/latest.ckpt"
vae_config_path="configs/rdp_vae/rdp_vae.yaml"


TORCHDYNAMO_DISABLE=1 CUDA_VISIBLE_DEVICES=1 python scripts/codes/eval_pi0.5/lerobot_eval_rdp_vae.py \
    --dataset.repo_id=${dataset} \
    --policy.type=pi05 \
    --output_dir=${output_dir} \
    --job_name=${run_id} \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=false \
    --policy.dtype=bfloat16 \
    --steps=${steps} \
    --save_freq=${save_freq} \
    --policy.device=cuda \
    --batch_size=${batch_size} \
    --policy.chunk_size=48 \
    --policy.n_action_steps=48 \
    --policy.max_action_dim=10 \
    --policy.max_state_dim=9 \
    --policy.repo_id="<your_username>/xarm_vase_sponge_test1_60hz_pi0.5_vae" \
    --policy.push_to_hub=False \
    --wandb.project="lerobot_pi0.5_vae" \
    --wandb.run_id=${run_id} \
    --log_freq=5 \
    --vae_load_path ${vae_load_path} \
    --get_latent_statistics True \
    --vae_config_path ${vae_config_path}