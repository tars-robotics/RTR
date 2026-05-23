pca_load_dir="data/ckpts/peel_cucumber_60hz/rdp_pca"

horizon=48
execute_hz=50 # Use 50 Hz execution for 60 Hz trajectories; much lower rates can make motion jittery.
control_hz=60

servo_speed=220
interpolate_ratio=1
adaptive_interpolate_ratio=2

compress_obs=False

# servo
servo_mode=True

## Async
inference_step_threshold=24

async_num=24

log_traj_dir="xarm_outputs/oft/traj/peel_cucumber/oft_latent_rtr_async${async_num}"

# Start the user process first; the system process starts quickly.
sleep 3

python src/rtr_async_sys/runner/simple_runner.py \
    --config-name simple_runner_without_user \
    controller=fix_threshold_async_controller \
    controller.control_horizon=${horizon} \
    controller.control_hz=${control_hz} \
    controller.inference_step_threshold=${inference_step_threshold} \
    scheduler=simple_scheduler \
    executor=async_servo_executor \
    executor.execute_hz=${execute_hz} \
    executor.servo_speed=${servo_speed} \
    executor.interpolate_ratio=${interpolate_ratio} \
    executor.adaptive_interpolate_ratio=${adaptive_interpolate_ratio} \
    executor/env=xarm_env \
    executor.env.compress_obs=${compress_obs} \
    executor.env.servo_mode=${servo_mode} \
    executor.env.pca_load_dir=${pca_load_dir} \
    executor.env.n_obs_steps=1 \
    executor.env.log_traj_dir=${log_traj_dir} \