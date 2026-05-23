# Example checkpoint paths (replace with your own):
# ckpt_path="data/ckpts/openvla_oft/vase_sponge_test1_oft/openvla-7b-oft-finetuned-libero-spatial+vase_sponge_test1_oft_dataset+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--50000_chkpt"
export PYTHONPATH="$(pwd)/third_party/openvla_oft/openvla-oft/LIBERO:${PYTHONPATH}"
# 15hz, horizon=12 (the on-disk ckpt name says 48 because it was forgotten when training was renamed)
ckpt_path="data/ckpts/openvla_oft/vase_sponge_test1_15hz_oft_6drotate/60hz_6drotate_horizon48--50000_chkpt"

CUDA_VISIBLE_DEVICES=7 python -m rtr_async_sys.user.simple_user \
    --config-dir "$(pwd)/third_party/openvla_oft/openvla-oft/configs" --config-name openvla_oft_user \
    user_hz=10 \
    model_wrapper.ckpt_path=${ckpt_path} \
    model_wrapper.compress_obs=False \
    model_wrapper.unnorm_key="vase_sponge_test1_15hz_oft_6drotate_dataset" \
    model_wrapper.interpolate=True \
    model_wrapper.interpolate_ratio=4
