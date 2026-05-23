cd third_party/openvla_oft/openvla-oft

export PYTHONPATH="../../LIBERO:${PYTHONPATH}"
ckpt_path="data/wipe_vase_15hz_oft_6drotate_horizon48_not_latent/wipe_vase_15hz_not_latent_worldsize2--50000_chkpt"
compress_obs=False

# port
ctrl_user=40020
user_sched=40050
user_ctrl_port="tcp://127.0.0.1:${ctrl_user}"
user_sched_port="tcp://127.0.0.1:${user_sched}"


python -m rtr_async_sys.user.simple_user \
    --config-dir configs --config-name openvla_oft_passive_user \
    user_hz=20 \
    controller_endpoint=${user_ctrl_port} \
    scheduler_endpoint=${user_sched_port} \
    model_wrapper.compress_obs=${compress_obs} \
    model_wrapper.ckpt_path=${ckpt_path} \
    model_wrapper.unnorm_key="wipe_vase15hz_oft6drotate_dataset"