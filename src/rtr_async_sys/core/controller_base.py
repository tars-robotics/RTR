# src/rtr_async_sys/core/controller_base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

import pickle
import numpy as np
import zmq
from loguru import logger

import time
import threading


class AbsController(ABC):
    """
    Controller:
    - One-to-one connection to Executor (ZMQ REQ -> REP)
    - Many-to-one service for Users (ZMQ ROUTER <- DEALER)
    - One-to-one service for Scheduler (ZMQ REP <- REQ)

        Communication topology (default ports are examples and can be overridden through __init__ arguments):

      User(DEALER)  -->  Controller(ROUTER, bind tcp://*:10020)
      Scheduler(REQ)->  Controller(REP,    bind tcp://*:10021)
      Controller(REQ)-> Executor(REP,      connect tcp://executor:10010)
    """

    def __init__(
        self,
        # ----- Executor channel -----
        exec_endpoint: str = "tcp://127.0.0.1:10010",
        # ----- User channel (ROUTER)-----
        user_bind: str = "tcp://*:10020",
        # ----- Scheduler channel (REP)-----
        sched_bind: str = "tcp://*:10021",
        context: Optional[zmq.Context] = None,
        control_horizon:int = 10# maximum number of actions executed per chunk
    ) -> None:
        self._timestep = 0
        self.control_horizon = control_horizon

        self._exec_socket_lock = threading.Lock() # both the main process and _control_add_action_thread use exec_socket, so access must be locked to preserve ZMQ send/recv ordering
        self.exec_endpoint = exec_endpoint
        self.user_bind = user_bind
        self.sched_bind = sched_bind
        self.context = context

        self._running: bool = False

    # =====================================================================
    #                 Low-level RPC wrappers for Executor communication
    # =====================================================================

    def _rpc_exec_get_obs_and_timestep(self, n_obs_steps=None) -> Dict[str, Any]:
        """
        Request obs + timestep from Executor.
        Expected Executor-side messages:
          send: {"type": "exec_get_obs_and_timestep"}
          recv: {"status": "ok", "obs": ..., "timestep": int}
        """
        with self._exec_socket_lock:
            self._sock_exec.send_pyobj({"type": "exec_get_obs_and_timestep", "n_obs_steps":n_obs_steps})
            reply = self._sock_exec.recv_pyobj()
        if reply.get("status") != "ok":
            logger.error(f"[Controller] exec_get_obs_and_timestep failed: {reply}")
            raise RuntimeError(f"Executor error: {reply}")
        return {"obs": reply["obs"], "timestep": reply["timestep"]}

    def _rpc_exec_add_action(self, action: np.ndarray, adaptive_interpolate:bool = False) -> None:
        """
        Send action to Executor.
        Expected Executor-side messages:
          send: {"type": "add_action", "action": np.ndarray}
          recv: {"status": "ok"} or {"status":"error", ...}
        """
        with self._exec_socket_lock:
            self._sock_exec.send_pyobj({"type": "add_action", "action": action, "adaptive_interpolate":adaptive_interpolate})
            reply = self._sock_exec.recv_pyobj()
        if reply.get("status") != "ok":
            logger.error(f"[Controller] add_action failed: {reply}")
            raise RuntimeError(f"Executor error: {reply}")

    def _rpc_exec_clear(self) -> None:
        with self._exec_socket_lock:
            self._sock_exec.send_pyobj({"type": "clear"})
            reply = self._sock_exec.recv_pyobj()
        if reply.get("status") != "ok":
            logger.error(f"[Controller] clear failed: {reply}")
            raise RuntimeError(f"Executor error: {reply}")
    
    def _rpc_exec_end_of_chunk(self) -> None:
        with self._exec_socket_lock:
            self._sock_exec.send_pyobj({"type": "end_of_chunk"})
            reply = self._sock_exec.recv_pyobj()
        if reply.get("status") != "ok":
            logger.error(f"[Controller] end_of_chunk failed: {reply}")
            raise RuntimeError(f"Executor error: {reply}")

    def _rpc_exec_abort(self) -> None:
        with self._exec_socket_lock:
            self._sock_exec.send_pyobj({"type": "abort"})
            reply = self._sock_exec.recv_pyobj()
        if reply.get("status") != "ok":
            logger.error(f"[Controller] abort failed: {reply}")
            raise RuntimeError(f"Executor error: {reply}")

    def rpc_exec_shutdown(self) -> None:
        """Helper: send shutdown signal to Executor."""
        logger.info("[Controller] Sending shutdown to Executor...")
        with self._exec_socket_lock:
            self._sock_exec.send_pyobj({"type": "shutdown"})
            reply = self._sock_exec.recv_pyobj()
        if reply.get("status") != "ok":
            logger.error(f"[Controller] shutdown failed: {reply}")
            raise RuntimeError(f"Executor error: {reply}")

    # =====================================================================
    #                       Scheduler-facing abstract interface
    # =====================================================================

    @abstractmethod
    def reply_history(self) -> Dict[str, Any]:
        """
        Called by Scheduler over the REP channel:
        - Controller returns its own or Executor-related history
        - Subclasses decide the content and format
        """
        pass

    @abstractmethod
    def recv_schedule(self, schedule_dict: Dict[str, Any]) -> None:
        """
        Called by Scheduler:
        - Receive schedule information and adjust Controller policy/behavior
        """
        pass

    # =====================================================================
    #               Controller calls User refine (REQ→REP)
    # =====================================================================
    def _rpc_user_refine_action(
        self, user_id: bytes, action: np.ndarray, obs: Dict[str, Any]
    ) -> np.ndarray:
        """
        Call a specific User refine_action through the refiner REQ/REP channel.
        """
        sock = self._user_refiners.get(user_id, None)
        if sock is None:
            raise RuntimeError(f"[Controller] No refiner registered for user_id={user_id}")

        sock.send_pyobj(
            {
                "type": "refine_action",
                "action": action,
                "obs": obs,
            }
        )
        reply = sock.recv_pyobj()
        if reply.get("status") != "ok":
            raise RuntimeError(f"[Controller] refine_action failed: {reply}")
        return reply["action"]


    # =====================================================================
    #                      Default User-facing behavior (overrideable)
    # =====================================================================

    def _handle_user_request(
        self, user_id: bytes, msg: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Default User request handling; subclasses can override it. The default is synchronous.
        [Synchronous flow]: User.get_obs_timestep -> controller.get_obs_timestep -> executor.rep_obs_timestep(interacts with env) \\
        -> controller.rep_obs_timestep -> User.predict_action_chunk -> User.send_action_chunk -> controller.add_action \\
        -> executor.execute_action(interacts with env)
        [Synchronous flow]: Call exec_clear after the whole action_chunk is executed

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
            while True:
                data = self._rpc_exec_get_obs_and_timestep()
                obs_timestep = data['timestep']
                if obs_timestep < self._timestep:# executor have not executed all of the actions. Sync controller will wait # only used in sync controller. and this situation can happen in aysnc-executor
                    time.sleep(0.05)
                else:
                    break

            return {"status": "ok", **data}

        elif msg_type == "send_action_chunk":
            action_chunk = msg["action_chunk"]
            timestep = msg.get("timestep", None)
            need_refine = msg.get("need_refine", False) # If need_refine is set, action must be refined before execution
            # The default implementation does not use timestep and simply forwards actions to Executor.
            # Override this method in subclasses if timestep-based queue alignment is needed.
            logger.debug(
                f"[Controller] Received action_chunk from user={user_id}, "
                f"timestep={timestep}, shape={getattr(action_chunk, 'shape', None)}"
            )
            if not isinstance(action_chunk, np.ndarray):
                raise TypeError("action_chunk should be np.adarray")
            predict_horizon = action_chunk.shape[0]
            predict_horizon = min(predict_horizon, self.control_horizon)
            for i in range(predict_horizon):
                if need_refine:
                    action_latent = action_chunk[i]
                    tcp_extended_obs_step = int(action_latent[-1])
                    # Get the observation needed for refine
                    data = self._rpc_exec_get_obs_and_timestep(tcp_extended_obs_step)
                    obs = data['obs']
                    # start_refine=time.time()
                    action = self._rpc_user_refine_action(user_id, action_latent, obs)
                    # logger.info(f"_rpc_user_refine_action time is {time.time()-_rpc_user_refine_action}")
                else:
                    action = action_chunk[i]
                if i == 0:# first-of-the-chunk, use adaptive_interpolate to prevent the gap of the first-chunk
                    adaptive_interpolate = True
                else:
                    adaptive_interpolate = False
                self._rpc_exec_add_action(action, adaptive_interpolate=adaptive_interpolate)
                self._timestep += 1
            # self._rpc_exec_clear()
            logger.debug("[controller] after _rpc_exec_end_of_chunk")
            self._rpc_exec_end_of_chunk()
            return {"status": "ok"}

        else:
            logger.error(f"[Controller] Unknown user msg type: {msg_type}")
            return {
                "status": "error",
                "msg": f"Unknown user msg type: {msg_type}",
            }

    # =====================================================================
    #                      Scheduler-facing message dispatch
    # =====================================================================

    def _handle_scheduler_request(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scheduler to Controller (REQ->REP) message dispatch.

        Expected Scheduler message format:
        1) Get history:
           {"type": "scheduler_request_info"}
           -> call reply_history()
           <- {"status": "ok", "history": <dict>}

        2) Send schedule:
           {"type": "scheduler_send_schedule", "schedule": {...}}
           -> call recv_schedule(schedule)
           <- {"status": "ok"}

        Override this method in subclasses to extend it.
        """
        msg_type = msg.get("type", None)

        if msg_type == "scheduler_request_info":
            hist = self.reply_history()
            return {"status": "ok", "history": hist}

        elif msg_type == "scheduler_send_schedule":
            schedule = msg["schedule"]
            self.recv_schedule(schedule)
            return {"status": "ok"}

        else:
            logger.error(f"[Controller] Unknown scheduler msg type: {msg_type}")
            return {
                "status": "error",
                "msg": f"Unknown scheduler msg type: {msg_type}",
            }

    # =====================================================================
    #                           ROUTER / REP I/O
    # =====================================================================

    def _recv_user_msg(self) -> Tuple[bytes, Dict[str, Any]]:
        """
        Receive a User message from ROUTER:
        - return (identity, msg_dict)
        """
        frames = self._sock_user.recv_multipart()
        if len(frames) != 2:
            raise RuntimeError(f"[Controller] Invalid user frames: {frames}")
        identity, data = frames
        msg = pickle.loads(data)
        return identity, msg

    def _send_user_reply(self, identity: bytes, reply: Dict[str, Any]) -> None:
        data = pickle.dumps(reply)
        self._sock_user.send_multipart([identity, data])

    def _recv_scheduler_msg(self) -> Dict[str, Any]:
        """Scheduler uses REQ/REP and receives pyobj directly."""
        return self._sock_sched.recv_pyobj()

    def _send_scheduler_reply(self, reply: Dict[str, Any]) -> None:
        self._sock_sched.send_pyobj(reply)

    # =====================================================================
    #                              Main loop
    # =====================================================================

    def _init_sock(self):
        """
        If subclasses need extra sockets, add that logic here rather than in __init__; __init__ may be called by the runner before ports are ready.
        """
        self._context = self.context or zmq.Context.instance()

        # === REQ socket connected to Executor ===
        self._sock_exec = self._context.socket(zmq.REQ)
        logger.info(f"[Controller] Connecting to Executor at {self.exec_endpoint}")
        self._sock_exec.connect(self.exec_endpoint)

        # === User-facing ROUTER socket ===
        self._sock_user = self._context.socket(zmq.ROUTER)
        logger.info(f"[Controller] Binding User ROUTER at {self.user_bind}")
        self._sock_user.bind(self.user_bind)

        # === User-facing REQ socket ===
        self._user_refiners: Dict[bytes, zmq.Socket] = {}  # added

        # === Scheduler-facing REP socket socket ===
        self._sock_sched = self._context.socket(zmq.REP)
        logger.info(f"[Controller] Binding Scheduler REP at {self.sched_bind}")
        self._sock_sched.bind(self.sched_bind)

        # === Poller listens to User and Scheduler inputs ===
        self._poller = zmq.Poller()
        self._poller.register(self._sock_user, zmq.POLLIN)
        self._poller.register(self._sock_sched, zmq.POLLIN)


    def serve_forever(self, poll_timeout_ms: int = 100) -> None:
        """
        Main event loop:
        - poll ROUTER(User) + REP(Scheduler)
        - call _handle_user_request for User messages
        - call _handle_scheduler_request for Scheduler messages
        """
        self._init_sock()

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

    def stop(self) -> None:
        """stop() can be called externally to terminate the loop."""
        self._running = False

    def close(self) -> None:
        """Clean up resources."""
        logger.info("[Controller] Closing sockets.")
        self._poller.unregister(self._sock_user)
        self._poller.unregister(self._sock_sched)
        self._sock_user.close(0)
        self._sock_sched.close(0)
        self._sock_exec.close(0)
