ckpt_path="data/ckpts/write_board_60hz/ckpts_abs/dp/horizon48/latest.ckpt"
pca_load_dir="data/ckpts/write_board_60hz/rdp_pca"

image_downsample_ratio=1
n_obs_steps=1

horizon=48
execute_hz=50 # Use 50 Hz execution for 60 Hz trajectories; much lower rates can make motion jittery.
control_hz=60

## Async
inference_step_threshold=24
async_num=24


servo_speed=220
interpolate_ratio=1
adaptive_interpolate_ratio=2

log_traj_dir="xarm_outputs/dp/traj/write_board/dp60hz_async${async_num}"

python src/rtr_async_sys/runner/simple_runner.py \
    --config-name simple_runner \
    controller=fix_threshold_async_controller \
    controller.control_horizon=${horizon} \
    controller.control_hz=${control_hz} \
    controller.inference_step_threshold=${inference_step_threshold} \
    scheduler=simple_scheduler \
    executor=async_servo_executor \
    executor.max_merge_len=1 \
    executor.execute_hz=${execute_hz} \
    executor.servo_speed=${servo_speed} \
    executor.interpolate_ratio=${interpolate_ratio} \
    executor.adaptive_interpolate_ratio=${adaptive_interpolate_ratio} \
    executor/env=xarm_env \
    executor.env.servo_mode=True \
    executor.env.pca_load_dir=${pca_load_dir} \
    executor.env.n_obs_steps=${n_obs_steps} \
    executor.env.log_traj_dir=${log_traj_dir} \
    user@user0=dp_passive_user \
    user0.user_hz=10 \
    user/model_wrapper@user0.model_wrapper=dp_wrapper \
    user0.model_wrapper.return_raw_action=False \
    user0.model_wrapper.ckpt_path=${ckpt_path} \
    user0.model_wrapper.model.noise_scheduler.num_train_timesteps=30 \
    user0.model_wrapper.model.num_inference_steps=30 \
    user0.model_wrapper.model.horizon=${horizon} \
    user0.model_wrapper.model.n_action_steps=${horizon} \
    user0.model_wrapper.model.image_downsample_ratio=${image_downsample_ratio} \
    user0.model_wrapper.model.n_obs_steps=${n_obs_steps}
