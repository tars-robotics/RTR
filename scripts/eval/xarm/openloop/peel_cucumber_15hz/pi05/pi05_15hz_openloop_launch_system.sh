pca_load_dir="data/ckpts/peel_cucumber_15hz/rdp_pca"
dataset_path="data/ckpts/peel_cucumber_15hz/rdp_zarr"
horizon=12

compress_obs=False
sample_stride=1
eval_length=100 # 100 for test, using dataset_length for eval

## for dataset eval
save_plot_action_path="outputs/openloop_eval/peel_cucumber/pi05/pi05_15hz_stride${sample_stride}.pkl"

inference_step_threshold=${horizon}
control_hz=1000

## port
ctrl_exec=30010
ctrl_user=30020
ctrl_sched=30030
exec_sched=30040
user_sched=30050
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
    --config-name simple_runner_without_user \
    controller=fix_threshold_async_controller \
    controller.control_horizon=${horizon} \
    controller.inference_step_threshold=${inference_step_threshold} \
    controller.control_hz=${control_hz} \
    controller.open_loop_eval=True \
    executor=sync_nonservo_executor \
    executor/env=dp_dataset_env \
    executor.env.pca_load_dir=${pca_load_dir} \
    executor.env.dataset.dataset_path=${dataset_path} \
    executor.env.n_obs_steps=1 \
    executor.env.dataset.n_obs_steps=1  \
    executor.env.dataset.horizon=${horizon} \
    executor.env.compress_obs=${compress_obs} \
    +executor.env.dataset.sample_stride=${sample_stride} \
    executor.env.save_plot_action_path=${save_plot_action_path} \
    executor.env.eval_length=${eval_length} \
    controller.exec_endpoint=${ctrl_exec_port} \
    controller.passive_user_endpoint=${ctrl_user_port} \
    controller.sched_bind=${ctrl_sched_port} \
    executor.ctrl_bind=${exec_ctrl_port} \
    executor.sched_bind=${exec_sched_port} \
    scheduler.controller_endpoint=${sched_ctrl_port} \
    scheduler.executor_endpoint=${sched_exec_port} \
    scheduler.user_endpoints=${sched_user_port}