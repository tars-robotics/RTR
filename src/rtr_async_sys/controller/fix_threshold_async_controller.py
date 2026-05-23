"""
fix_threshold_async_controller.py

Controller actively requests action_chunk from PassiveUser:
- request_action_chunk_thread @ 100Hz: pull obs+timestep and request inference when conditions are met.
- After receiving action_chunk: drop stale actions and enqueue the rest.
- control_add_action_thread @ control_hz: pop actions from the queue and send them to Executor.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List
from collections import deque
import threading
import time
import pickle

import numpy as np
import zmq
from loguru import logger

from rtr_async_sys.core.controller_base import AbsController


class FixThresholdAsyncController(AbsController):
    def __init__(
        self,
        exec_endpoint: str = "tcp://127.0.0.1:10010",
        # This is the passive_user endpoint; PassiveUser binds it as REP.
        passive_user_endpoint: str = "tcp://127.0.0.1:10020",
        sched_bind: str = "tcp://*:10021",
        context=None,
        ensemble_mode: str = "new",
        control_hz: float = 10,
        control_horizon: int = 48,
        warm_up_step: int = 0,
        open_loop_eval: bool = False,
        request_obs_hz: int = 100,
        inference_step_threshold:int = 36
    ):
        super().__init__(
            exec_endpoint=exec_endpoint,
            user_bind="",   # ROUTER/DEALER user_bind is no longer used.
            sched_bind=sched_bind,
            context=context,
            control_horizon=control_horizon,
        )

        # Action queue: deque[List[action_dict]].
        self._queue: deque[List] = deque()
        self._queue_lock = threading.Lock()

        self.ensemble_mode = ensemble_mode
        self.control_hz = control_hz
        self.control_interval = 1.0 / self.control_hz
        self.control_horizon = control_horizon

        self.warm_up_step = warm_up_step
        self.open_loop_eval = open_loop_eval

        # Request thread.
        self.request_obs_hz = request_obs_hz
        self.request_interval = 1.0 / self.request_obs_hz
        self.passive_user_endpoint = passive_user_endpoint

        # Chunk-boundary management.
        self.inference_step_threshold = inference_step_threshold
        self.now_chunk_end_step: int = inference_step_threshold

        # for open_loop_eval
        self._last_infer_obs_timestep: int = -1

        # Avoid REQ/REP reentry.
        self._infer_lock = threading.Lock()
        self._inflight = False

        self._control_thread = threading.Thread(
            target=self._control_add_action_thread,
            daemon=True,
        )
        self._request_thread = threading.Thread(
            target=self.request_action_chunk_thread,
            daemon=True,
        )

        logger.info("[FixThresholdAsyncController] Initialized.")

    # =====================================================================
    # Internal queue management, following AsyncController logic.
    # =====================================================================

    def _add_action_chunk_in_queue(
        self,
        action_chunk: np.ndarray,
        action_chunk_timestep: int,
        need_refine: bool = False,
        user_id=None,
    ) -> None:
        with self._queue_lock:
            horizon = action_chunk.shape[0]
            horizon = min(horizon, self.control_horizon)

            out_of_data_step_num = self._timestep - action_chunk_timestep
            # The queue may become empty if too many actions are stale.
            action_chunk = action_chunk[out_of_data_step_num:horizon, :]

            logger.debug(
                f"[FixThresholdAsyncController] append {action_chunk.shape[0]} actions into queue, "
                f"timestep={self._timestep}, action_chunk_timestep={action_chunk_timestep}"
            )

            for idx, action in enumerate(action_chunk):
                if idx >= len(self._queue):
                    self._queue.append([])

                start_of_new_chunk = (idx == 0)
                action_dict = {
                    "action": action,
                    "need_refine": need_refine,
                    "user_id": user_id,
                    "start_of_new_chunk": start_of_new_chunk,
                }
                self._queue[idx].append(action_dict)

    def _get_action_from_queue(self) -> Optional[Dict[str, Any]]:
        with self._queue_lock:
            if len(self._queue) == 0:
                return None
            actions_in_this_step = self._queue[0]
            if self.ensemble_mode == "new":
                return actions_in_this_step[-1]
            raise NotImplementedError(
                f"[FixThresholdAsyncController] ensemble mode {self.ensemble_mode} not implemented"
            )

    def _control_add_action_thread(self) -> None:
        """
        Dedicated thread: pop actions from the queue at control_hz and send them to Executor.
        """
        while True:
            start = time.time()
            action_dict = self._get_action_from_queue()

            if action_dict is not None:
                action = action_dict["action"]
                need_refine = action_dict["need_refine"]
                user_id = action_dict["user_id"]
                start_of_new_chunk = action_dict["start_of_new_chunk"]

                # Refine logic: the current PassiveUser version has no refine channel.
                # Keep the interface for now, but do not use it by default.
                if need_refine:
                    raise NotImplementedError(
                        "[FixThresholdAsyncController] need_refine=True not supported in passive_user mode yet."
                    )

                # Use adaptive_interpolate at the start of a new chunk to smooth the gap.
                self._rpc_exec_add_action(action, adaptive_interpolate=start_of_new_chunk)

                with self._queue_lock:
                    self._queue.popleft()
                    self._timestep += 1
                if self.open_loop_eval:
                    if self._timestep % self.control_horizon == 0:
                        self._rpc_exec_end_of_chunk() 

            real_interval = time.time() - start
            if real_interval < self.control_interval:
                time.sleep(self.control_interval - real_interval)

    # =====================================================================
    # Actively request action_chunk from PassiveUser.
    # =====================================================================

    def _rpc_user_get_action_chunk(self, obs: Dict[str, Any], timestep: int) -> Dict[str, Any]:
        """
        ZMQ REQ → PassiveUser( REP ):
        send: {"type":"get_action_chunk","obs":obs,"timestep":timestep}
        recv: {"status":"ok","action_chunk":...,"timestep":...}
        """
        req = {"type": "get_action_chunk", "obs": obs, "timestep": timestep}
        self._sock_user.send(pickle.dumps(req))
        raw = self._sock_user.recv()
        reply = pickle.loads(raw)
        return reply

    def request_action_chunk_thread(self) -> None:
        """
        Pull obs+timestep at 100 Hz.
        When either condition is met:
          - timestep == 0
          - or obs_timestep >= now_chunk_end_step
        Request one inference and update now_chunk_end_step = obs_timestep + horizon.
        """
        logger.info(
            f"[FixThresholdAsyncController] request_action_chunk_thread @ {self.request_obs_hz} Hz"
        )

        while True:
            tick = time.time()

            # 1) Pull observations.
            try:
                data = self._rpc_exec_get_obs_and_timestep()
                obs = data["obs"]
                obs_timestep = int(data["timestep"])
            except Exception as e:
                logger.exception(f"[FixThresholdAsyncController] get_obs failed: {e}")
                time.sleep(0.01)
                continue

            # 2) Decide whether to trigger inference.
            if not self.open_loop_eval:
                should_infer = ((obs_timestep == 0) or (obs_timestep >= self.now_chunk_end_step)) and (obs_timestep != self._last_infer_obs_timestep) # when obs_timestep is 0, may inference based on obs 0 for more than once
                self._last_infer_obs_timestep = obs_timestep
                if should_infer:
                    logger.debug(f"NOT open_loop_eval, obs_timestep is {obs_timestep}, timestep is {self._timestep}")
            else:
                should_infer = (obs_timestep % self.control_horizon == 0) and (obs_timestep != self._last_infer_obs_timestep)
                self._last_infer_obs_timestep = obs_timestep
                if should_infer:
                    logger.debug(f"open_loop_eval, obs_timestep is {obs_timestep}, timestep is {self._timestep}")

            if should_infer:
                # Avoid duplicate triggers: REQ sockets require strict serial send/recv.
                if self._infer_lock.acquire(blocking=False):
                    try:
                        if self._inflight:
                            # This should not happen because the lock protects it, but keep the guard.
                            pass
                        self._inflight = True

                        # horizon = int(self.control_horizon)
                        # Update the boundary first to avoid retriggering on the same obs_timestep.
                        # self.now_chunk_end_step = obs_timestep + horizon
                        self.now_chunk_end_step = obs_timestep + self.inference_step_threshold

                        t0 = time.time()
                        reply = self._rpc_user_get_action_chunk(obs, obs_timestep)
                        infer_dt = time.time() - t0

                        if reply.get("status") != "ok":
                            logger.warning(f"[FixThresholdAsyncController] user error reply: {reply}")
                        else:
                            action_chunk = reply["action_chunk"]
                            # By default, action_chunk is aligned to obs_timestep.
                            action_chunk_timestep = int(reply.get("timestep", obs_timestep))
                            need_refine = bool(reply.get("need_refine", False))

                            if not isinstance(action_chunk, np.ndarray):
                                raise TypeError("action_chunk should be np.ndarray")

                            # Warm-up logic follows the previous version.
                            if (self._timestep >= self.warm_up_step) or (len(self._queue) == 0):
                                self._add_action_chunk_in_queue(
                                    action_chunk,
                                    action_chunk_timestep,
                                    need_refine=need_refine,
                                    user_id=None,
                                )
                                logger.debug(
                                    f"[FixThresholdAsyncController] infer ok, obs_timestep={obs_timestep}, "
                                    f"now_chunk_end_step={self.now_chunk_end_step}, _rpc_user_get_action_chunk latency={infer_dt:.4f}s"
                                )
                            else:
                                logger.info(
                                    f"[FixThresholdAsyncController] warm up: warm_step={self.warm_up_step}, "
                                    f"timestep={self._timestep}"
                                )

                    except Exception as e:
                        logger.exception(f"[FixThresholdAsyncController] request_action_chunk failed: {e}")
                    finally:
                        self._inflight = False
                        self._infer_lock.release()

            # 3) Rate control.
            elapsed = time.time() - tick
            if elapsed < self.request_interval:
                time.sleep(self.request_interval - elapsed)

    # =====================================================================
    # Scheduler: keep REP bind and poll.
    # =====================================================================

    def reply_history(self) -> Dict[str, Any]:
        return {}

    def recv_schedule(self, schedule_dict: Dict[str, Any]) -> None:
        logger.info(f"[FixThresholdAsyncController] Received schedule: {schedule_dict}")

    # =====================================================================
    # sockets / serve_forever
    # =====================================================================

    def _init_sock(self):
        self._context = self.context or zmq.Context.instance()

        # === REQ socket connected to Executor ===
        self._sock_exec = self._context.socket(zmq.REQ)
        logger.info(f"[FixThresholdAsyncController] Connecting to Executor at {self.exec_endpoint}")
        self._sock_exec.connect(self.exec_endpoint)

        # === REQ socket connected to PassiveUser ===
        self._sock_user = self._context.socket(zmq.REQ)
        logger.info(
            f"[FixThresholdAsyncController] Connecting to PassiveUser at {self.passive_user_endpoint}"
        )
        self._sock_user.connect(self.passive_user_endpoint)

        # === Scheduler-facing REP socket socket ===
        self._sock_sched = self._context.socket(zmq.REP)
        logger.info(f"[FixThresholdAsyncController] Binding Scheduler REP at {self.sched_bind}")
        self._sock_sched.bind(self.sched_bind)

        # === Poller: listen only to Scheduler ===
        self._poller = zmq.Poller()
        self._poller.register(self._sock_sched, zmq.POLLIN)

    def serve_forever(self, poll_timeout_ms: int = 100) -> None:
        self._init_sock()

        # Start threads.
        self._control_thread.start()
        self._request_thread.start()

        logger.info("[FixThresholdAsyncController] Start serve_forever()")
        self._running = True

        while self._running:
            try:
                events = dict(self._poller.poll(timeout=poll_timeout_ms))
            except zmq.ZMQError as e:
                logger.exception(f"[FixThresholdAsyncController] poll error: {e}")
                continue

            # ---- Messages from Scheduler ----
            if self._sock_sched in events and events[self._sock_sched] & zmq.POLLIN:
                try:
                    raw = self._sock_sched.recv()
                    msg = pickle.loads(raw)
                    reply = self._handle_scheduler_request(msg)
                except Exception as e:
                    logger.exception(f"[FixThresholdAsyncController] handle scheduler msg error: {e}")
                    reply = {"status": "error", "msg": str(e)}

                try:
                    self._sock_sched.send(pickle.dumps(reply))
                except Exception as e:
                    logger.exception(f"[FixThresholdAsyncController] send scheduler reply error: {e}")

        logger.info("[FixThresholdAsyncController] Stopped serve_forever().")


if __name__ == "__main__":
    controller = FixThresholdAsyncController(
        exec_endpoint="tcp://127.0.0.1:10010",
        passive_user_endpoint="tcp://127.0.0.1:10020",
        sched_bind="tcp://*:10021",
        control_hz=10,
        control_horizon=10,
        request_obs_hz=100,
    )
    controller.serve_forever()
