cd third_party/lerobot/lerobot

ckpt_path="outputs/pi05_training_logl1_wipe_vase_60hz_steps40000_world1/checkpoints/040000/pretrained_model"
dataset_id="sadpiggy/xarm_wipe_vase_state9_60hz"
chunk_size=48

python -m rtr_async_sys.user.simple_user \
    --config-dir configs --config-name pi0_5_passive_user \
    user_hz=10 \
    model_wrapper.compress_obs=False \
    model_wrapper.cfg.policy.pretrained_path=${ckpt_path} \
    model_wrapper.cfg.policy.chunk_size=${chunk_size} \
    model_wrapper.cfg.policy.n_action_steps=${chunk_size} \
    model_wrapper.cfg.dataset.repo_id=${dataset_id}