# src/rtr_async_sys/scheduler/simple_scheduler.py
from __future__ import annotations

from typing import Dict, Any, List, Optional
import zmq
from loguru import logger
from rtr_async_sys.core.scheduler_base import AbsScheduler


class SimpleScheduler(AbsScheduler):
    """
    A minimal centralized scheduler:
    - Track the latest Controller / Executor state
    - Track each User state (history/info)
    - Broadcast a fixed schedule to users each round
    """

    def __init__(
        self,
        controller_endpoint: str = "tcp://127.0.0.1:10021",  # e.g. tcp://127.0.0.1:10040
        executor_endpoint: str = "tcp://127.0.0.1:10011",    # e.g. tcp://127.0.0.1:10050
        user_endpoints: List[str] = ["tcp://127.0.0.1:10030"], # e.g. ["tcp://127.0.0.1:10060", "tcp://127.0.0.1:10061"]
        context: Optional[zmq.Context] = None,
        schedule_hz: float = 0.05
    ) -> None:
        super().__init__(controller_endpoint, executor_endpoint, user_endpoints, context, schedule_hz)

        # State storage
        self.controller_info: Dict[str, Any] = {}
        self.executor_info: Dict[str, Any] = {}
        self.user_info: Dict[str, Dict[str, Any]] = {}

    # =========================================================
    #                 Receive Controller information
    # =========================================================

    def collect_controller_info(self, info: Dict[str, Any]) -> None:
        """Record the latest Controller information"""
        logger.debug(f"[SimpleScheduler] Controller info")
        self.controller_info = info

    # =========================================================
    #                 Receive Executor information
    # =========================================================

    def collect_executor_info(self, info: Dict[str, Any]) -> None:
        """Record the latest Executor information"""
        logger.debug(f"[SimpleScheduler] Executor info")
        self.executor_info = info

    # =========================================================
    #                 Receive User information (history/info)
    # =========================================================

    def collect_user_history(self, uid: str, history: Dict[str, Any]) -> None:
        """Record each User history/info"""
        logger.debug(f"[SimpleScheduler] User[{uid}] info")
        self.user_info[uid] = history

    # =========================================================
    #                   Generate scheduling policy
    # =========================================================

    def make_schedule(self) -> Dict[str, Any]:
        """
        Minimal implementation:
        - Do not use info for decisions
        - Return a fixed schedule only

        In a real system, this can use:
        - controller_info
        - executor_info
        - user_info (multi-user)
        to make decisions
        """
        schedule = {
            "command": "keep_running",
            "tick": 1,
        }
        logger.debug(f"[SimpleScheduler] Broadcast schedule: {schedule}")
        return schedule

if __name__ == '__main__':
    scheduler = SimpleScheduler()
    scheduler.serve_forever()