# src/rtr_async_sys/controller/sync_controller.py
from __future__ import annotations

from typing import Dict, Any, Optional,List

import numpy as np
from loguru import logger
import time
from collections import deque
import threading
import zmq

from rtr_async_sys.core.controller_base import AbsController


class AsyncController(AbsController):
    """
    SyncController
    --------------
    Minimal Controller implementation:
    - When User requests observations, pull directly from Executor
    - When User sends an action chunk, send it directly to Executor
    - Scheduler history requests return an empty dictionary
    - Scheduler schedules are logged only and do not affect behavior

    Properties:
    - Fully synchronous with almost no caching or queue handling
    - Simple behavior that does not rewrite User behavior
    - Suitable as a baseline or starter controller

    --------
    These methods are not called unless _handle_user_request is changed
    """

    def __init__(
        self,
        exec_endpoint: str = "tcp://127.0.0.1:10010",
        user_bind: str = "tcp://*:10020",
        sched_bind: str = "tcp://*:10021",
        context=None,
        ensemble_mode:str = "new",
        control_hz:float = 10,
        control_horizon:int = 10,
        warm_up_step:int = 0
    ):
        """
        execute_action_hz: frequency for _control_add_action_thread, which calls _rpc_exec_add_action. \\
        When the data starts with a pause, the trained model may produce unstable actions in the first few steps, so warm_up uses synchronous behavior.
        """
        super().__init__(
            exec_endpoint=exec_endpoint,
            user_bind=user_bind,
            sched_bind=sched_bind,
            context=context,
            control_horizon=control_horizon
        )
        self._queue:deque[List] = deque() # Each item is a list sorted by obs timestep for ensemble support. List[0] is {'action', 'need_refine', 'user_id' ...}
        self._queue_lock = threading.Lock()

        self.ensemble_mode = ensemble_mode
        self.control_hz = control_hz
        self.control_interval = 1.0 / self.control_hz

        self.warm_up_step = warm_up_step

        self._control_thread = threading.Thread(
            target=self._control_add_action_thread,
            daemon=True
        )
        # self._control_thread.start()

        logger.info("[AsyncController] Initialized.")

    # ===============================================import zmq======================
    #                 User-facing interface (called by _handle_user_request in the base class)
    # =====================================================================

    # The minimal version does not cache chunks
    def add_action_chunk(self, action_chunk: np.ndarray) -> None:
        """
        High-level interface called by User.
        Forward the chunk directly to Executor without caching.
        """
        logger.debug(
            f"[SyncController] add_action_chunk called, shape={getattr(action_chunk, 'shape', None)}"
        )
        self._rpc_exec_add_action(action_chunk)

    def send_action(self, action: Optional[np.ndarray] = None) -> None:
        """
        As with add_action_chunk, the minimal implementation does not handle action=None,
        because SyncController has no local cache.
        """
        if action is None:
            logger.warning("[SyncController] send_action(None) ignored.")
            return
        logger.debug("[SyncController] send_action called.")
        self._rpc_exec_add_action(action)

    def get_obs_and_timestep(self) -> Dict[str, Any]:
        """
        Call the Executor RPC directly.
        """
        return self._rpc_exec_get_obs_and_timestep()

    # =====================================================================
    #                 Scheduler-facing API
    # =====================================================================

    def reply_history(self) -> Dict[str, Any]:
        """
        Minimal SyncController implementation:
        - Does not maintain local history
        - Does not pull history from Executor; Scheduler queries Executor directly
        Future extensions can add:
        - controller-generated events
        - user request information
        - user/timestep to action mapping
        """
        logger.debug("[SyncController] reply_history called.")
        return {}  # minimal version returns empty

    def recv_schedule(self, schedule_dict: Dict[str, Any]) -> None:
        """
        Minimal implementation: only log Scheduler schedules without applying them.
        Possible future extensions:
        - change the controller merge policy
        - change send frequency
        - change action filtering
        """
        logger.info(f"[SyncController] Received schedule: {schedule_dict}")
        # TODO: in a real system, this would change the controller behavior policy
        pass

    # =====================================================================
    #        user ---> async ---> controller ---> aysnc ---> executor
    # =====================================================================

    def _handle_user_request(
        self, user_id: bytes, msg: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Default User request handling; subclasses can override it. This implementation uses asynchronous logic.

        Expected User message format (via DEALER -> ROUTER):
        1) Request obs + timestep:
           {"type": "get_obs_and_timestep"}

           Returns:
           {"status": "ok", "obs": ..., "timestep": int}

        2) Send action chunk:
           {"type": "send_action_chunk", "action_chunk": np.ndarray, "timestep": int}

           Default behavior: forward directly to Executor through _rpc_exec_add_action
           Returns:
           {"status": "ok"}
        """
        msg_type = msg.get("type", None)

        # ===== User registers refine endpoint =====
        if msg_type == "register_refiner":
            endpoint = msg["endpoint"]  # e.g. "tcp://127.0.0.1:12345"
            logger.info(f"[Controller] register_refiner from user={user_id}, endpoint={endpoint}")
            # Create a REQ socket for this user
            sock = self._context.socket(zmq.REQ)
            sock.connect(endpoint)
            self._user_refiners[user_id] = sock
            return {"status": "ok"}

        elif msg_type == "get_obs_and_timestep":
            data = self._rpc_exec_get_obs_and_timestep()
            # while True:
            #     # get_obs is synchronous, but can be made asynchronous.
            #     data = self._rpc_exec_get_obs_and_timestep()
            #     obs_timestep = data['timestep']
            #     if obs_timestep < self._timestep:# executor have not executed all of the actions. Sync controller will wait # only used in sync controller. and this situation can happen in aysnc-executor
            #         time.sleep(0.05)
            #     else:
            #         break

            return {"status": "ok", **data}

        elif msg_type == "send_action_chunk":
            action_chunk = msg["action_chunk"]
            timestep = msg.get("timestep", None)
            need_refine = msg.get("need_refine", False)
            # The default implementation does not use timestep and simply forwards actions to Executor.
            # Override this method in subclasses if timestep-based queue alignment is needed.
            logger.debug(
                f"[Controller] Received action_chunk from user={user_id}, "
                f"timestep={timestep}, shape={getattr(action_chunk, 'shape', None)}"
            )
            if not isinstance(action_chunk, np.ndarray):
                raise TypeError("action_chunk should be np.adarray")
            
            if (self._timestep >= self.warm_up_step) or (len(self._queue)==0):
                self._add_action_chunk_in_queue(action_chunk, timestep, need_refine, user_id)
            else:
                logger.info(f"warm up. warm_step is {self.warm_up_step}. timestep is {self._timestep}")
            # predict_horizon = action_chunk.shape[0]
            # Synchronous flow: after receiving a chunk, execute all actions with _rpc_exec_add_action and call end_of_chunk to ensure the whole chunk finishes.
            # This now uses async logic: a dedicated thread executes actions at a fixed frequency.
            # for i in range(predict_horizon):
            #     self._rpc_exec_add_action(action_chunk[i])
            #     self._timestep += 1
            #     logger.debug(f"self._timestep is {self._timestep}")
            # # self._rpc_exec_clear()
            # self._rpc_exec_end_of_chunk()
            return {"status": "ok"}

        else:
            logger.error(f"[Controller] Unknown user msg type: {msg_type}")
            return {
                "status": "error",
                "msg": f"Unknown user msg type: {msg_type}",
            }

    # =====================================================================
    #       async_executor internal method.
    # =====================================================================
    def _add_action_chunk_in_queue(self, action_chunk:np.ndarray, action_chunk_timestep, need_refine=False, user_id=None)->None:
        with self._queue_lock:
            horizon = action_chunk.shape[0]
            horizon = min(horizon, self.control_horizon)
            out_of_data_step_num = self._timestep - action_chunk_timestep # This can produce an empty array if too many steps are stale.
            action_chunk = action_chunk[out_of_data_step_num:horizon, :]# align
            logger.debug(f"append {action_chunk.shape[0]} actions into queue, timestep is {self._timestep}, action_chunk_timestep is {action_chunk_timestep}")
            for idx, action in enumerate(action_chunk):
                if idx >= len(self._queue):
                    self._queue.append([])
                start_of_new_chunk = False
                if idx == 0:
                    start_of_new_chunk = True
                action_dict = {
                    'action': action,
                    'need_refine': need_refine,
                    'user_id': user_id,
                    'start_of_new_chunk':start_of_new_chunk
                }
                self._queue[idx].append(action_dict)
    
    def _get_action_from_queue(self)->np.ndarray:
        """
        shape: [action_dim]
        """
        with self._queue_lock:
            if len(self._queue) == 0:
                return None
            actions_in_this_step = self._queue[0]
            if self.ensemble_mode == 'new':
                return actions_in_this_step[-1]
            else:
                raise NotImplementedError(f"[AsyncController] have not implemented ensemble mode {self.ensemble_mode}")

    def _control_add_action_thread(self) -> None:
        """
        Dedicated thread that pops actions from the queue at a fixed frequency, calls _rpc_exec_add_action, then pops left.
        """
        while True:
            start = time.time()
            action_dict = self._get_action_from_queue()

            
            if action_dict is not None:
                action = action_dict['action']
                need_refine = action_dict['need_refine']
                user_id = action_dict['user_id']
                start_of_new_chunk = action_dict['start_of_new_chunk']
                if need_refine:
                    action_latent = action
                    tcp_extended_obs_step = int(action_latent[-1])
                    # Get the observation needed for refine
                    data = self._rpc_exec_get_obs_and_timestep(tcp_extended_obs_step)
                    obs = data['obs']
                    action = self._rpc_user_refine_action(user_id, action_latent, obs)

                self._rpc_exec_add_action(action, adaptive_interpolate=start_of_new_chunk)# start_of_new_chunk uses adaptive_interpolate to smooth the gap
                with self._queue_lock:
                    self._queue.popleft()
                    self._timestep += 1

            real_interval = time.time() - start
            if real_interval < self.control_interval:
                time.sleep(self.control_interval - real_interval)

    def serve_forever(self, poll_timeout_ms: int = 100) -> None:
        """
        Main event loop:
        - poll ROUTER(User) + REP(Scheduler)
        - call _handle_user_request for User messages
        - call _handle_scheduler_request for Scheduler messages
        """
        self._init_sock()
        self._control_thread.start()

        logger.info("[Controller] Start serve_forever()")
        self._running = True

        while self._running:
            try:
                events = dict(self._poller.poll(timeout=poll_timeout_ms))
            except zmq.ZMQError as e:
                logger.exception(f"[Controller] poll error: {e}")
                continue

            # ---- Messages from User ----
            if self._sock_user in events and events[self._sock_user] & zmq.POLLIN:
                try:
                    identity, msg = self._recv_user_msg()
                    reply = self._handle_user_request(identity, msg)
                except Exception as e:
                    logger.exception(f"[Controller] handle user msg error: {e}")
                    reply = {"status": "error", "msg": str(e)}
                self._send_user_reply(identity, reply)

            # ---- Messages from Scheduler ----
            if self._sock_sched in events and events[self._sock_sched] & zmq.POLLIN:
                try:
                    msg = self._recv_scheduler_msg()
                    reply = self._handle_scheduler_request(msg)
                except Exception as e:
                    logger.exception(
                        f"[Controller] handle scheduler msg error: {e}"
                    )
                    reply = {"status": "error", "msg": str(e)}
                self._send_scheduler_reply(reply)

        logger.info("[Controller] Stopped serve_forever().")


if __name__ == "__main__":
    controller = AsyncController(
        exec_endpoint="tcp://127.0.0.1:10010",
        user_bind="tcp://*:10020",
        sched_bind="tcp://*:10021",
    )
    controller.serve_forever()