ckpt_path="data/ckpts/write_board_60hz/ckpts_abs/dp/horizon48/latest.ckpt"
pca_load_dir="data/ckpts/write_board_60hz/rdp_pca"
dataset_path="data/ckpts/write_board_60hz/rdp_zarr"

image_downsample_ratio=1
n_obs_steps=1
horizon=48

sample_stride=1

## for dataset eval
save_plot_action_path="outputs/openloop_eval/write_board/dp/dp_60hz_stride1.pkl"
eval_length=100 # 100 for test, using dataset_length for eval

inference_step_threshold=48
control_hz=1000

## port
ctrl_exec=20010
ctrl_user=20020
ctrl_sched=20030
exec_sched=20040
user_sched=20050
ctrl_exec_port="tcp://127.0.0.1:${ctrl_exec}"
ctrl_user_port="tcp://127.0.0.1:${ctrl_user}"
ctrl_sched_port="tcp://*:${ctrl_sched}"
exec_ctrl_port="tcp://*:${ctrl_exec}"
exec_sched_port="tcp://*:${exec_sched}"
user_ctrl_port="tcp://127.0.0.1:${ctrl_user}"
user_sched_port="tcp://127.0.0.1:${user_sched}"
sched_ctrl_port="tcp://127.0.0.1:${ctrl_sched}"
sched_exec_port="tcp://127.0.0.1:${exec_sched}"
sched_user_port="[\"tcp://127.0.0.1:${user_sched}\"]"



python src/rtr_async_sys/runner/simple_runner.py \
    --config-name simple_runner \
    controller=fix_threshold_async_controller \
    controller.control_horizon=${horizon} \
    controller.inference_step_threshold=${inference_step_threshold} \
    controller.control_hz=${control_hz} \
    controller.open_loop_eval=True \
    scheduler=simple_scheduler \
    executor=sync_nonservo_executor \
    executor/env=dp_dataset_env \
    executor.env.pca_load_dir=${pca_load_dir} \
    executor.env.relative_action=False \
    executor.env.dataset.dataset_path=${dataset_path} \
    executor.env.dataset.horizon=${horizon} \
    executor.env.n_obs_steps=${n_obs_steps} \
    executor.env.dataset.n_obs_steps=${n_obs_steps} \
    +executor.env.dataset.image_downsample_ratio=${image_downsample_ratio} \
    executor.env.save_plot_action_path=${save_plot_action_path} \
    executor.env.eval_length=${eval_length} \
    +executor.env.dataset.sample_stride=${sample_stride} \
    user@user0=dp_passive_user \
    user0.user_hz=20 \
    user/model_wrapper@user0.model_wrapper=dp_wrapper \
    user0.model_wrapper.return_raw_action=False \
    user0.model_wrapper.ckpt_path=${ckpt_path} \
    +executor.env.visualize_image=False \
    user0.model_wrapper.model.noise_scheduler.num_train_timesteps=30 \
    user0.model_wrapper.model.num_inference_steps=30 \
    user0.model_wrapper.model.horizon=${horizon} \
    user0.model_wrapper.model.n_action_steps=${horizon} \
    user0.model_wrapper.model.image_downsample_ratio=${image_downsample_ratio} \
    user0.model_wrapper.model.n_obs_steps=${n_obs_steps} \
    controller.exec_endpoint=${ctrl_exec_port} \
    controller.passive_user_endpoint=${ctrl_user_port} \
    controller.sched_bind=${ctrl_sched_port} \
    executor.ctrl_bind=${exec_ctrl_port} \
    executor.sched_bind=${exec_sched_port} \
    user0.controller_endpoint=${user_ctrl_port} \
    user0.scheduler_endpoint=${user_sched_port} \
    scheduler.controller_endpoint=${sched_ctrl_port} \
    scheduler.executor_endpoint=${sched_exec_port} \
    scheduler.user_endpoints=${sched_user_port}

    