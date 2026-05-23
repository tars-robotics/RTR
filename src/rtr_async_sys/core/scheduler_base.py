# src/rtr_async_sys/core/scheduler_base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import zmq
import pickle
from loguru import logger
import time


class AbsScheduler(ABC):
    """
    Centralized Scheduler (REQ -> REP)
    - REQ/REP with Controller
    - REQ/REP with Executor
    - REQ/REP with multiple Users (one socket per User)

    Runtime behavior:
    - Scheduler actively communicates with modules and does not receive external requests
    - Poll all module states at a fixed frequency
    - Generate a unified schedule and broadcast it to all Users
    """

    def __init__(
        self,
        controller_endpoint: str = "tcp://127.0.0.1:10021",  # e.g. tcp://127.0.0.1:10040
        executor_endpoint: str = "tcp://127.0.0.1:10011",    # e.g. tcp://127.0.0.1:10050
        user_endpoints: List[str] = ["tcp://127.0.0.1:10030"], # e.g. ["tcp://127.0.0.1:10060", "tcp://127.0.0.1:10061"]
        context: Optional[zmq.Context] = None,
        schedule_hz: float = 0.05
    ) -> None:
        self.schedule_hz = schedule_hz

        self.input_context = context
        self.controller_endpoint = controller_endpoint
        self.executor_endpoint = executor_endpoint
        self.user_endpoints = user_endpoints
        # self._context = context or zmq.Context.instance()

        # # ========== Controller socket ==========
        # self._sock_ctrl = self._context.socket(zmq.REQ)
        # self._sock_ctrl.connect(controller_endpoint)
        # logger.info(f"[Scheduler] Connected to Controller at {controller_endpoint}")

        # # ========== Executor socket ==========
        # self._sock_exec = self._context.socket(zmq.REQ)
        # self._sock_exec.connect(executor_endpoint)
        # logger.info(f"[Scheduler] Connected to Executor at {executor_endpoint}")

        # # ========== Users sockets ==========
        # self._sock_users: Dict[str, zmq.Socket] = {}
        # for idx, ep in enumerate(user_endpoints):
        #     sock = self._context.socket(zmq.REQ)
        #     sock.connect(ep)
        #     uid = f"user_{idx}"
        #     self._sock_users[uid] = sock
        #     logger.info(f"[Scheduler] Connected to User[{uid}] at {ep}")

        # running flag
        self._running = False

    # =====================================================================
    #               Subclasses implement the scheduling logic below
    # =====================================================================

    @abstractmethod
    def collect_controller_info(self, info: Dict[str, Any]) -> None:
        """Handle information received from Controller"""
        pass

    @abstractmethod
    def collect_executor_info(self, info: Dict[str, Any]) -> None:
        """Handle information received from Executor"""
        pass

    @abstractmethod
    def collect_user_history(self, uid: str, history: Dict[str, Any]) -> None:
        """Handle history received from User"""
        pass

    @abstractmethod
    def make_schedule(self) -> Dict[str, Any]:
        """Generate a scheduling policy from all collected state"""
        pass

    # =====================================================================
    #                     Internal communication helpers
    # =====================================================================

    def _req_rep(self, sock: zmq.Socket, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Minimal Scheduler-to-module REQ-REP model"""
        sock.send(pickle.dumps(msg))
        raw = sock.recv()
        return pickle.loads(raw)

    def _init_sock(self):
        self._context = self.input_context or zmq.Context.instance()

        # ========== Controller socket ==========
        self._sock_ctrl = self._context.socket(zmq.REQ)
        self._sock_ctrl.connect(self.controller_endpoint)
        logger.info(f"[Scheduler] Connected to Controller at {self.controller_endpoint}")

        # ========== Executor socket ==========
        self._sock_exec = self._context.socket(zmq.REQ)
        self._sock_exec.connect(self.executor_endpoint)
        logger.info(f"[Scheduler] Connected to Executor at {self.executor_endpoint}")

        # ========== Users sockets ==========
        self._sock_users: Dict[str, zmq.Socket] = {}
        for idx, ep in enumerate(self.user_endpoints):
            sock = self._context.socket(zmq.REQ)
            sock.connect(ep)
            uid = f"user_{idx}"
            self._sock_users[uid] = sock
            logger.info(f"[Scheduler] Connected to User[{uid}] at {ep}")

    # =====================================================================
    #                     Main loop
    # =====================================================================

    def serve_forever(self):
        """
        Main scheduling loop:
        1) Request Controller state
        2) Request Executor state
        3) Request User history
        4) Generate scheduling policy
        5) Broadcast to User

        freq: scheduling frequency
        """ 
        self._init_sock()

        freq = self.schedule_hz
        dt = 1.0 / freq
        self._running = True

        logger.info(f"[Scheduler] serve_forever at {freq} Hz, dt is {dt}s")

        while self._running:
            start = time.time()
            logger.info("begin schedule")
            # ========== 1. Controller status ==========
            ctrl_info = self._req_rep(
                self._sock_ctrl,
                {"type": "scheduler_request_info"}
            )
            self.collect_controller_info(ctrl_info)

            # ========== 2. Executor status ==========
            exec_info = self._req_rep(
                self._sock_exec,
                {"type": "scheduler_request_info"}
            )
            self.collect_executor_info(exec_info)

            # ========== 3. Request user histories ==========
            for uid, sock in self._sock_users.items():
                msg = {"type": "scheduler_request_info", "uid": uid}
                history = self._req_rep(sock, msg)
                self.collect_user_history(uid, history)

            # ========== 4. Make schedule ==========
            schedule = self.make_schedule()

            # ========== 5. Broadcast schedule ==========
            for uid, sock in self._sock_users.items():
                msg = {"type": "scheduler_send_schedule", "schedule": schedule}
                _ = self._req_rep(sock, msg)

            # ========== Rate control ==========
            elapsed = time.time() - start
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)

        logger.info("[Scheduler] stopped.")

    def stop(self):
        self._running = False
