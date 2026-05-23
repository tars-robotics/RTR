# src/rtr_async_sys/executor/non_servo_executor.py
from collections import deque
from typing import Dict, Any, Tuple

import numpy as np
from loguru import logger
import time
import threading
import zmq

from rtr_async_sys.core.executor_base import AbsExecutor
from rtr_async_sys.env.dp_dataset_env import DpDatasetEnv


class AsyncNonServoExecutor(AbsExecutor):
    """
    sync: controller.exec_action -> executor.exec_action -> env.exec_action -> executor.reply -> controller \\
    async: controller.exec_action -> executor.execute_queue.add(action) | executor.merge -> executor.exec_action -> env.exec_action \\
    Here, async means asynchronous control/execution, not asynchronous inference/control. \\
    When inference latency exceeds execution latency, asynchronous inference/control helps more;
    when execution latency dominates, asynchronous control/execution helps more. Both can be
    async at the same time.
    """
    def __init__(
        self, 
        env, 
        ctrl_bind="tcp://*:10010", 
        sched_bind="tcp://*:10011", 
        max_merge_len=1, 
        execute_hz=10, 
        servo_mode:bool=False,
        servo_speed:int=150,
        interpolate_ratio:int=1,
        adaptive_interpolate_ratio:int=2
    ):
        """
        Input actions are in meters. Only Cartesian control is currently supported;
        joint-angle control is not supported.
        
        :param servo_speed: mm/s. Default is 100 mm/s. execute_hz * max_stride == servo_speed,
            which gives max_stride for execution speed limiting.
        :param interpolate_ratio: uniform interpolation ratio.
        :param adaptive_interpolate_ratio: higher adaptive interpolation ratio for special cases
            such as chunk switches.
        """
        super().__init__(env=env, ctrl_bind=ctrl_bind, sched_bind=sched_bind)
        self._queue = deque()
        self._queue_lock = threading.Lock() # Queue updates also update timestep, so the lock must cover both.
        self._env_lock = threading.Lock()
        self._history = []
        self._last_obs = None
        self.max_merge_len = max_merge_len
        self.execute_hz = execute_hz
        self.execute_interval = 1.0 / self.execute_hz 

        self.servo_mode = servo_mode
        if self.servo_mode:
            assert max_merge_len ==1, "not use merge for servo_mode"
        self.servo_speed = servo_speed
        self.max_stride = (servo_speed/self.execute_hz)
        self.last_xyz = None
        self.interpolate_ratio = interpolate_ratio
        self.adaptive_interpolate_ratio = adaptive_interpolate_ratio
        self.last_position = None # [x,y,z] in mm
        self.begin_exec = False
        # assert self.interpolate_ratio ==1 or self.interpolate_ratio==2, f"interpolate ratio must be 1 or 2, but got {self.interpolate_ratio}"
        # assert self.adaptive_interpolate_ratio ==1 or self.adaptive_interpolate_ratio==2, f"adaptive_interpolate_ratio ratio must be 1 or 2, but got {self.adaptive_interpolate_ratio}"

        logger.info(f"init AsyncNonservoExecutor servo_mode is {self.servo_mode}  max_merge_len is {self.max_merge_len} execute_hz is {self.execute_hz} execute_interval is {self.execute_interval} max_stride is {self.max_stride} servo_speed is {servo_speed}" )
        logger.info(f"self.interpolate_ratio is {self.interpolate_ratio}. interpolate logic is implemented in add_action()")
        logger.info(f"self.adaptive_interpolate_ratio is {self.adaptive_interpolate_ratio}. interpolate logic is implemented in add_action()")

        
        self._exec_thread = threading.Thread(
            target=self.execute_pending_actions_thread,
            daemon=True       # set as daemon so it exits with the main process
        )
        # self._exec_thread.start()
    


    # ==== Controller-facing API ====

    def exec_get_obs_and_timestep(self, n_obs_steps=None) -> Dict[str, Any]:
        """
        asynchronous
        """
        if n_obs_steps == None:
            return {"obs": self._last_obs, "timestep": self._timestep}
        else:# for refine action. This code is not beautiful, TODO: add `sync_obs` tag in async_executor
            raise NotImplementedError("only get obs after each execute_action.")
            with self._env_lock:
                obs = self.env.get_obs(n_obs_steps)
            return {"obs": obs, "timestep": self._timestep}


    def add_action_servo(self, action: np.ndarray, adaptive_interpolate:bool = False) -> None:
        """
        For the actions of the end-point control, the unit is meters.
        """
        # print(f"adaptive_interpolate is {adaptive_interpolate}")
        # FOR DEBUG. always use adaptive_interpolate
        # adaptive_interpolate = True
        if self.last_xyz is None:
            self.last_xyz = self._last_obs[-1]['left_robot_tcp_pose'][0:3]

        with self._queue_lock:
            if adaptive_interpolate == False:
                if self.last_position is None:
                    if isinstance(self._last_obs,list):
                        self.last_position = self._last_obs[-1]['left_robot_tcp_pose'][0:3]*1000
                    else:
                        self.last_position = self._last_obs['left_robot_tcp_pose'][0:3]*1000 # in xarm, _last_obs is 9d state, so we only use 0:3 xyz(in m), then convert it to mm
                if self.interpolate_ratio == 1:
                    action_dict = {
                        'action':action,
                        'step':1
                    }
                    self._queue.append(action_dict)
                else:
                    # Support integer interpolation: ratio = r inserts r - 1 points between adjacent points.
                    r = int(self.interpolate_ratio)
                    if r < 1:
                        raise ValueError(f"interpolate_ratio must be >= 1, got {self.interpolate_ratio}")
                    if self.interpolate_ratio != r:
                        raise ValueError(f"interpolate_ratio must be an integer, got {self.interpolate_ratio}")

                    prev_mm = self.last_position.astype(np.float32)          # (3,) mm
                    target_mm = (action[0:3] * 1000).astype(np.float32)      # (3,) mm

                    # Generate r - 1 intermediate points, excluding the start and end points.
                    for k in range(1, r):
                        alpha = k / r
                        interp_mm = (1 - alpha) * prev_mm + alpha * target_mm

                        mid_action = action.copy()
                        mid_action[0:3] = interp_mm / 1000.0  # back to meters
                        action_dict = {
                            'action':mid_action,
                            'step':0
                        }
                        self._queue.append(action_dict)

                    # Enqueue the endpoint action.
                    action_dict = {
                        'action':action,
                        'step':1
                    }
                    self._queue.append(action_dict)


                self.last_position = action[0:3]*1000# in mm
            else: # Adaptive interpolation used at chunk switches.
                if self.adaptive_interpolate_ratio == 1:
                    action_dict = {
                        'action':action,
                        'step':1
                    }
                    self._queue.append(action_dict)
                else: # Currently only supports interpolate_ratio values of 1 or 2.
                    # TODO: support interpolation ratios >= 2.
                    r = int(self.adaptive_interpolate_ratio)
                    if r < 1:
                        raise ValueError(f"adaptive_interpolate_ratio must be >= 1, got {self.adaptive_interpolate_ratio}")
                    if self.adaptive_interpolate_ratio != r:
                        raise ValueError(f"adaptive_interpolate_ratio must be an integer, got {self.adaptive_interpolate_ratio}")

                    if self.last_position is None:
                        if isinstance(self._last_obs, list):
                            self.last_position = self._last_obs[-1]['left_robot_tcp_pose'][0:3] * 1000
                        else:
                            self.last_position = self._last_obs['left_robot_tcp_pose'][0:3] * 1000  # mm

                    # Compute stride in mm using the maximum xyz step.
                    target_mm = (action[:3] * 1000).astype(np.float32)
                    prev_mm = self.last_position.astype(np.float32)
                    stride = float(np.max(np.abs(target_mm - prev_mm)))
                    self.env.log_switch_list.append((prev_mm, target_mm))

                    # Too large: even r-way interpolation would still exceed max_stride.
                    if stride > self.max_stride * r:
                        logger.info(
                            f"[adaptive interpolate exceed]: stride {stride} exceed self.max_stride*adaptive_interpolate_ratio "
                            f"{self.max_stride*r}. NOT interpolate for this action"
                        )
                        self.env.log_print_list.append(
                            f"[adaptive interpolate exceed]: stride {stride} exceed self.max_stride*adaptive_interpolate_ratio {self.max_stride*r}. NOT interpolate for this action"
                        )
                        action_dict = {
                            'action':action,
                            'step':1
                        }
                        self._queue.append(action_dict)
                    else:
                        # No interpolation needed.
                        if stride <= self.max_stride:
                            action_dict = {
                                'action':action,
                                'step':1
                            }
                            self._queue.append(action_dict)
                        else:
                            # Split this segment so each part is <= max_stride.
                            need = int(np.ceil(stride / self.max_stride))  # Number of required segments.
                            # Allow at most r segments, meaning r - 1 inserted points.
                            segs = min(need, r)
                            logger.info(f"adaptive interpolate, interpolate for {segs} segs")

                            # Insert segs - 1 intermediate points.
                            for k in range(1, segs):
                                alpha = k / segs
                                interp_mm = (1 - alpha) * prev_mm + alpha * target_mm

                                mid_action = action.copy()
                                mid_action[0:3] = interp_mm / 1000.0  # meters
                                action_dict = {
                                    'action':mid_action,
                                    'step':0
                                }
                                self._queue.append(action_dict)

                            # Endpoint
                            action_dict = {
                                'action':action,
                                'step':1
                            }
                            self._queue.append(action_dict)
                self.last_position = action[0:3]*1000    
    

    def add_action(self, action: np.ndarray, adaptive_interpolate:bool = False) -> None:
        """
        For the actions of the end-point control, the unit is meters.
        """
        logger.debug("add_action_non_servo")
        with self._queue_lock:
            action_dict = {
                'action':action,
                'step':1
            }
            self._queue.append(action_dict)                
    

    # ==== Scheduler-facing API ====

    def reply_history(self) -> Dict[str, Any]:
        return {"actions": list(self._history), "timestep": self._timestep}

    def recv_schedule(self, schedule_dict: Dict[str, Any]) -> None:
        logger.info(f"[NonServoExecutor] Received schedule: {schedule_dict}")
        # TODO: adapt internal logic to the scheduling policy.

    # ==== Internal execution ====
    def execute_pending_actions_thread(self) -> None:
        while True:
            start = time.time()

            self.execute_pending_actions()
            
            real_interval = time.time() - start
            if real_interval < self.execute_interval:
                time.sleep(self.execute_interval-real_interval)


    def execute_pending_actions(self) -> None:
        """
        asynchronous: start a thread and execute at a fixed frequency.
        """
        if not self._queue:
            if self.begin_exec:
                logger.info("[queue is empty] waiting for control")
            return
        self.begin_exec = True
        # action = self._queue.popleft()
        merged_len = self.merge()
        if merged_len == 0:
            return
        with self._queue_lock:
            action_dict = self._queue[merged_len-1]#0-base
            action = action_dict['action']
            action_bind_step = action_dict['step']
        #     for _ in range(merged_len):
        #         action = self._queue.popleft()

        # TODO: remove the sleep in end_of_chunk and verify behavior.
        action_clone = action.copy()
        if self.servo_mode:
            not_append_traj = (action_bind_step == 0) # Do not record interpolated trajectory points.
            if self.last_xyz is not None:
                logger.debug(f"action is {action}")
                logger.debug(f"self.last_xyz is {self.last_xyz}")
                stride = max(abs(action[:3]*1000-self.last_xyz[:3]*1000))
                if stride > self.max_stride:
                    logger.info("="*100)
                    logger.info(f"[execute exceed] stride exceed max stride. switch to normal mode. stride is {stride}, max_stride is {self.max_stride}")
                    self.env.log_print_list.append(f"[execute exceed] stride exceed max stride. switch to normal mode. stride is {stride}, max_stride is {self.max_stride}")
                    time.sleep(0.05)
                    self.env.set_mode(servo_mode=False)
                    self.env.execute_action(action, not_append_traj=not_append_traj)
                    self.env.set_mode(servo_mode=True)
                else:
                    logger.info(f"stride is {stride} which in max_stride {self.max_stride}")
                    self.env.execute_action(action, not_append_traj=not_append_traj)
            else:
                raise ValueError("last_xyz should not be None")
        else:
            self.env.execute_action(action)
        self.last_xyz = action_clone
        
        # Update obs only after the action finishes, not midway through execution.
        with self._env_lock:
            obs = self.env.get_obs()
        # Pop after execution.
        with self._queue_lock:
            for _ in range(merged_len):
                self._queue.popleft()
            # These operations happen nearly simultaneously; they can be forced to run together later.
            self._last_obs = obs
            self._history.append(action)
            if not self.servo_mode:
                self._timestep += merged_len
            else:
                self._timestep += action_bind_step
        logger.debug(f"new_timestep is {self._timestep}; merged_len is {merged_len}")

    def merge(self) -> int:
        """
        Execute first, then pop; otherwise ordering bugs are likely.
        return: \\
        merged_action: np.ndarray \\
        merged_len: int
        """
        if not self._queue:
            return 0
        merged_len = min(self.max_merge_len, len(self._queue))

        return merged_len

    def abort(self) -> None:
        with self._queue_lock:
            self._queue.clear()

    def clear(self) -> None:
        """
        [debug] For a sync controller, call end_of_chunk() after each action_chunk for plotting
        instead of calling clear.
        """
        with self._queue_lock:
            self._queue.clear()
        self._history.clear()
        self._timestep = 0
        self.env.clear()
    
    def end_of_chunk(self):
        """
        sync_controller sends a whole chunk to async_executor. Since the controller is synchronous
        and cannot align timesteps, it waits until the whole chunk is updated before continuing. \\
        An async controller does not need end_of_chunk to ensure the whole chunk has finished executing.
        """
        # if not isinstance(self.env, DpDatasetEnv):
        #     return

        while True:
            with self._queue_lock: # This lock may not be necessary here.
                if len(self._queue) == 0:
                    break
            time.sleep(0.05)
 
        self.env.end_of_chunk()
        # with self._env_lock:
            # self._last_obs = self.env.get_obs()# 
    

    def _handle_controller_request(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Controller(REQ) to Executor(REP) message dispatch.
        Core logic: decouple controller action enqueueing from executor action execution, then run
        execution asynchronously for higher throughput.

        Message types:
        1) Get obs+timestep:
           {"type": "exec_get_obs_and_timestep"}
           -> exec_get_obs_and_timestep()
           <- {"status": "ok", "obs": ..., "timestep": int}

        2) Add action:
           {"type": "add_action", "action": np.ndarray}
           -> add_action() + execute_pending_actions()
           <- {"status": "ok"}

        3) clear:
           {"type": "clear"}
           -> clear()
           <- {"status": "ok"}

        4) abort:
           {"type": "abort"}
           -> abort()
           <- {"status": "ok"}

        5) shutdown:
           {"type": "shutdown"}
           -> set _running=False
           <- {"status": "ok"}
        """
        msg_type = msg.get("type", None)

        if msg_type == "exec_get_obs_and_timestep":
            n_obs_steps = msg.get("n_obs_steps", None)
            data = self.exec_get_obs_and_timestep(n_obs_steps)
            return {"status": "ok", **data}

        elif msg_type == "add_action":
            action = msg["action"]
            adaptive_interpolate = msg.get("adaptive_interpolate", False)
            if not self.servo_mode:
                self.add_action(action)
            else:
                self.add_action_servo(action,adaptive_interpolate)
            # Default: try to execute once immediately
            # self.execute_pending_actions()
            return {"status": "ok"}

        elif msg_type == "clear":
            self.clear()
            return {"status": "ok"}

        elif msg_type == "abort":
            self.abort()
            return {"status": "ok"}

        elif msg_type == "shutdown":
            logger.info("[Executor] Received shutdown from Controller.")
            self._running = False
            return {"status": "ok"}
        
        elif msg_type == "end_of_chunk":
            self.end_of_chunk()
            return {"status": "ok"}
        
        else:
            logger.error(f"[Executor] Unknown controller msg type: {msg_type}")
            return {
                "status": "error",
                "msg": f"Unknown controller msg type: {msg_type}",
            }


    def serve_forever(self, poll_timeout_ms: int = 100) -> None:
        """
        Executor ZMQ loop:
        - poll Controller REP socket + Scheduler REP socket
        - recv request
        - dispatch to _handle_controller_request / _handle_scheduler_request
        - send response
        """
        self.env.start()
        
        while self._last_obs == None:
            with self._env_lock:
                self._last_obs = self.env.get_obs()
            if self._last_obs != None:
                break
            time.sleep(0.1)
            logger.debug("self.env.get_obs() get None. wait for 0.1s then retry")

        self._init_sock()

        self._exec_thread.start()

        logger.info("[Executor] Start serve_forever()")
        self._running = True

        while self._running:
            try:
                events = dict(self._poller.poll(timeout=poll_timeout_ms))
            except zmq.ZMQError as e:
                logger.exception(f"[Executor] poll error: {e}")
                continue

            # ---- Requests from Controller ----
            if self._sock_ctrl in events and events[self._sock_ctrl] & zmq.POLLIN:
                try:
                    msg = self._sock_ctrl.recv_pyobj()
                    reply = self._handle_controller_request(msg)
                except Exception as e:
                    logger.exception(
                        f"[Executor] handle controller msg error: {e}"
                    )
                    reply = {"status": "error", "msg": str(e)}
                self._sock_ctrl.send_pyobj(reply)

            # ---- Requests from Scheduler ----
            if self._sock_sched in events and events[self._sock_sched] & zmq.POLLIN:
                try:
                    msg = self._sock_sched.recv_pyobj()
                    reply = self._handle_scheduler_request(msg)
                except Exception as e:
                    logger.exception(
                        f"[Executor] handle scheduler msg error: {e}"
                    )
                    reply = {"status": "error", "msg": str(e)}
                self._sock_sched.send_pyobj(reply)

        logger.info("[Executor] Stopped serve_forever().")


if __name__ == "__main__":
    executor = AsyncNonServoExecutor(env="src/rtr_async_sys/configs/env/dp_dataset_env.yaml")
    executor.serve_forever()
