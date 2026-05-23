# src/rtr_async_sys/core/executor_base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Optional

from omegaconf import DictConfig, OmegaConf
from loguru import logger
import hydra

import numpy as np
import zmq

from rtr_async_sys.core.env_base import AbsEnv

OmegaConf.register_new_resolver("eval", eval, replace=True)


class AbsExecutor(ABC):
    """Executor is a separate process.
    It manages Env and executes actions.

    Responsibilities:
    - Own and drive the Env instance (get_obs / execute_action)
    - Communicate with the Controller (actions and observations) and Scheduler (history and schedules) over ZMQ
    - Subclasses implement the action queue, merge policy, and history statistics

    Communication topology (default ports can be overridden in __init__):

      Controller(REQ)  →  Executor(REP, bind tcp://*:10010)
      Scheduler(REQ)   →  Executor(REP, bind tcp://*:10011)
    """

    def __init__(
        self,
        env: Union[AbsEnv, str, DictConfig],
        ctrl_bind: str = "tcp://*:10010",   # controller channel
        sched_bind: str = "tcp://*:10011",  # scheduler channel
        context: Optional[zmq.Context] = None,
    ) -> None:
        """
        The base __init__ instantiates the env and sets up communication sockets. Subclasses implement the abstract methods and override _handle_controller_request when needed. \\
        The default _handle_controller_request is synchronous; asynchronous subclasses should override it.
        """
        # -------- Env instantiation --------
        if isinstance(env, str):
            env = OmegaConf.load(env)
        if isinstance(env, DictConfig):
            env = hydra.utils.instantiate(env)

        if not isinstance(env, AbsEnv):
            raise TypeError(f"env must be AbsEnv or its config, got {type(env)}")

        self._timestep = 0

        self.env: AbsEnv = env
        self.input_context = context
        self.ctrl_bind = ctrl_bind
        self.sched_bind = sched_bind
        # self.env.start()

        # # -------- ZMQ Context & Sockets --------
        # self._context = context or zmq.Context.instance()

        # # Controller-facing REP socket
        # self._sock_ctrl = self._context.socket(zmq.REP)
        # logger.info(f"[Executor] Binding Controller REP at {ctrl_bind}")
        # self._sock_ctrl.bind(ctrl_bind)

        # # Scheduler-facing REP socket
        # self._sock_sched = self._context.socket(zmq.REP)
        # logger.info(f"[Executor] Binding Scheduler REP at {sched_bind}")
        # self._sock_sched.bind(sched_bind)

        # # Poller listens to both input channels
        # self._poller = zmq.Poller()
        # self._poller.register(self._sock_ctrl, zmq.POLLIN)
        # self._poller.register(self._sock_sched, zmq.POLLIN)

        self._running: bool = False

    # =====================================================================
    #                     Controller-facing abstract interface
    # =====================================================================

    @abstractmethod
    def exec_get_obs_and_timestep(self, n_obs_steps=None) -> Dict[str, Any]:
        """Return obs + timestep to the Controller.

        Expected return format:
        {
            "obs": <any serializable structure>,
            "timestep": int
        }
        """
        pass

    @abstractmethod
    def add_action(self, action: np.ndarray) -> None:
        """Receive action from Controller and enqueue."""
        pass

    # =====================================================================
    #                     Scheduler-facing abstract interface
    # =====================================================================

    @abstractmethod
    def reply_history(self) -> Dict[str, Any]:
        """Return history after a Scheduler request."""
        pass

    @abstractmethod
    def recv_schedule(self, schedule_dict: Dict[str, Any]) -> None:
        """Adjust internal state after the Scheduler sends a schedule."""
        pass

    # =====================================================================
    #                        Internal execution abstract interface
    # =====================================================================

    @abstractmethod
    def execute_pending_actions(self) -> None:
        """Pop an action from the internal queue and call env.execute_action."""
        pass

    @abstractmethod
    def merge(self) -> None:
        """Merge buffered actions if needed (e.g. chunk assembly)."""
        pass

    @abstractmethod
    def abort(self) -> None:
        """Abort current execution and reset."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear action buffer and reset executor internal state."""
        pass
    
    @abstractmethod
    def end_of_chunk(self) -> None:
        """call self.env.end_of_chunk. For DEBUG"""
        pass

    # =====================================================================
    #                     Message handling: from Controller
    # =====================================================================

    def _handle_controller_request(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Controller(REQ) to Executor(REP) message dispatch.

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
        
        6) end_of_chunk:
           {"type": "end_of_chunk"}
           -> end_of_chunk()
           <- {"status": "ok"}
        """
        msg_type = msg.get("type", None)

        if msg_type == "exec_get_obs_and_timestep":
            n_obs_steps = msg.get("n_obs_steps", None)
            data = self.exec_get_obs_and_timestep(n_obs_steps)
            return {"status": "ok", **data}

        elif msg_type == "add_action":
            action = msg["action"]
            self.add_action(action)
            # Default: try to execute once immediately
            self.execute_pending_actions()
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

    # =====================================================================
    #                     Message handling: from Scheduler
    # =====================================================================

    def _handle_scheduler_request(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scheduler(REQ) to Executor(REP) message dispatch.

        Message types:
        1) Request history:
           {"type": "scheduler_request_info"}
           -> reply_history()
           <- {"status": "ok", "history": {...}}

        2) Send schedule:
           {"type": "scheduler_send_schedule", "schedule": {...}}
           -> recv_schedule(schedule)
           <- {"status": "ok"}

        3) shutdown(optional):
           {"type": "shutdown"}
           -> _running=False
           <- {"status": "ok"}
        """
        msg_type = msg.get("type", None)

        if msg_type == "scheduler_request_info":
            hist = self.reply_history()
            return {"status": "ok", "history": hist}

        elif msg_type == "scheduler_send_schedule":
            schedule = msg["schedule"]
            self.recv_schedule(schedule)
            return {"status": "ok"}

        elif msg_type == "shutdown":
            logger.info("[Executor] Received shutdown from Scheduler.")
            self._running = False
            return {"status": "ok"}

        else:
            logger.error(f"[Executor] Unknown scheduler msg type: {msg_type}")
            return {
                "status": "error",
                "msg": f"Unknown scheduler msg type: {msg_type}",
            }

    def _init_sock(self):
        # -------- ZMQ Context & Sockets --------
        self._context = self.input_context or zmq.Context.instance()

        # Controller-facing REP socket
        self._sock_ctrl = self._context.socket(zmq.REP)
        logger.info(f"[Executor] Binding Controller REP at {self.ctrl_bind}")
        self._sock_ctrl.bind(self.ctrl_bind)

        # Scheduler-facing REP socket
        self._sock_sched = self._context.socket(zmq.REP)
        logger.info(f"[Executor] Binding Scheduler REP at {self.sched_bind}")
        self._sock_sched.bind(self.sched_bind)

        # Poller listens to both input channels
        self._poller = zmq.Poller()
        self._poller.register(self._sock_ctrl, zmq.POLLIN)
        self._poller.register(self._sock_sched, zmq.POLLIN)


    # =====================================================================
    #                          Main loop: serve_forever
    # =====================================================================

    def serve_forever(self, poll_timeout_ms: int = 100) -> None:
        """
        Executor ZMQ loop:
        - poll Controller REP socket + Scheduler REP socket
        - recv request
        - dispatch to _handle_controller_request / _handle_scheduler_request
        - send response
        """
        self.env.start()
        self._init_sock()

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

    def stop(self) -> None:
        """stop() can be called externally to terminate the loop."""
        self._running = False

    def close(self) -> None:
        """Clean up resources."""
        logger.info("[Executor] Closing sockets.")
        self._poller.unregister(self._sock_ctrl)
        self._poller.unregister(self._sock_sched)
        self._sock_ctrl.close(0)
        self._sock_sched.close(0)
