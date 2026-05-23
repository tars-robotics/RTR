cd third_party/openvla_oft/openvla-oft

export PYTHONPATH="../../LIBERO:${PYTHONPATH}"
ckpt_path="data/wipe_vase_60hz_oft_6drotate_horizon48_not_latent/wipe_vase_60hz_not_latent_worldsize2--50000_chkpt"
compress_obs=False




python -m rtr_async_sys.user.simple_user \
    --config-dir configs --config-name openvla_oft_passive_user \
    user_hz=20 \
    model_wrapper.compress_obs=${compress_obs} \
    model_wrapper.ckpt_path=${ckpt_path} \
    model_wrapper.unnorm_key="wipe_vase60hz_oft6drotate_dataset"