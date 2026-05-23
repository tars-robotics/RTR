cd third_party/lerobot/lerobot

ckpt_path="outputs/pi05_training_logl1_write_board_15hz_steps40000_world1/checkpoints/040000/pretrained_model"
dataset_id="sadpiggy/xarm_write_board_state9_15hz"
chunk_size=12

# port
ctrl_user=30020
user_sched=30050
user_ctrl_port="tcp://127.0.0.1:${ctrl_user}"
user_sched_port="tcp://127.0.0.1:${user_sched}"

CUDA_VISIBLE_DEVICES=4 python -m rtr_async_sys.user.simple_user \
    --config-dir configs --config-name pi0_5_passive_user \
    user_hz=20 \
    controller_endpoint=${user_ctrl_port} \
    scheduler_endpoint=${user_sched_port} \
    model_wrapper.compress_obs=False \
    model_wrapper.cfg.policy.pretrained_path=${ckpt_path} \
    model_wrapper.cfg.policy.chunk_size=${chunk_size} \
    model_wrapper.cfg.policy.n_action_steps=${chunk_size} \
    model_wrapper.cfg.dataset.repo_id=${dataset_id}