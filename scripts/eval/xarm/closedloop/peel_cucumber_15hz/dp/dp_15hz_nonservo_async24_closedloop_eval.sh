pca_load_dir="data/ckpts/peel_cucumber_15hz/rdp_pca"
ckpt_path="data/ckpts/peel_cucumber_15hz/ckpts_abs/dp/horizon12/latest.ckpt"

image_downsample_ratio=1
n_obs_steps=1

horizon=12
execute_hz=50 # Use 50 Hz execution for 60 Hz trajectories; much lower rates can make motion jittery.
control_hz=60

## Async
inference_step_threshold=6
async_num=6


log_traj_dir="xarm_outputs/dp/traj/peel_cucumber/dp15hz_nonservo_async${async_num}"


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
    executor/env=xarm_env \
    executor.servo_mode=${servo_mode} \
    executor.env.servo_mode=${servo_mode} \
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
    user0.model_wrapper.model.horizon=12 \
    user0.model_wrapper.model.n_action_steps=12 \
    user0.model_wrapper.model.image_downsample_ratio=${image_downsample_ratio} \
    user0.model_wrapper.model.n_obs_steps=${n_obs_steps}

    
