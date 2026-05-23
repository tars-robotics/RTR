# src/rtr_async_sys/controller/sync_controller.py
from __future__ import annotations

from typing import Dict, Any, Optional

import numpy as np
from loguru import logger

from rtr_async_sys.core.controller_base import AbsController


class SyncController(AbsController):
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
        debug_message:str=None,
        control_horizon:int = 10
    ):
        super().__init__(
            exec_endpoint=exec_endpoint,
            user_bind=user_bind,
            sched_bind=sched_bind,
            context=context,
            control_horizon=control_horizon
        )

        logger.info("[SyncController] Initialized.")
        logger.info(f"debug msg is {debug_message}")

    # =====================================================================
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


if __name__ == "__main__":
    controller = SyncController(
        exec_endpoint="tcp://127.0.0.1:10010",
        user_bind="tcp://*:10020",
        sched_bind="tcp://*:10021",
    )
    controller.serve_forever()