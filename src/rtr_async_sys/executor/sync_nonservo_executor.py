# src/rtr_async_sys/executor/non_servo_executor.py
from collections import deque
from typing import Dict, Any

import numpy as np
from loguru import logger

from rtr_async_sys.core.executor_base import AbsExecutor


class SyncNonServoExecutor(AbsExecutor):
    def __init__(self, env, ctrl_bind="tcp://*:10010", sched_bind="tcp://*:10011"):
        super().__init__(env=env, ctrl_bind=ctrl_bind, sched_bind=sched_bind)
        self._queue = deque()
        self._history = []

    # ==== Controller-facing API ====

    def exec_get_obs_and_timestep(self, n_obs_steps=None) -> Dict[str, Any]:
        """
        synchronous
        """
        obs = self.env.get_obs(n_obs_steps)
        return {"obs": obs, "timestep": self._timestep}

    def add_action(self, action: np.ndarray) -> None:
        self._queue.append(action)

    # ==== Scheduler-facing API ====

    def reply_history(self) -> Dict[str, Any]:
        return {"actions": list(self._history), "timestep": self._timestep}

    def recv_schedule(self, schedule_dict: Dict[str, Any]) -> None:
        logger.info(f"[NonServoExecutor] Received schedule: {schedule_dict}")
        # TODO: adapt internal logic to the scheduling policy.

    # ==== Internal execution ====

    def execute_pending_actions(self) -> None:
        if not self._queue:
            return
        action = self._queue.popleft()
        self.env.execute_action(action)
        self._history.append(action)
        self._timestep += 1

    def merge(self) -> None:
        # Non-servo implementations can leave this empty; servo variants should override it.
        pass

    def abort(self) -> None:
        self._queue.clear()

    def clear(self) -> None:
        self._queue.clear()
        self._history.clear()
        self._timestep = 0
        self.env.clear()
    
    def end_of_chunk(self) -> None:
        self.env.end_of_chunk()



if __name__ == "__main__":
    executor = SyncNonServoExecutor(env="src/rtr_async_sys/configs/env/dp_dataset_env.yaml")
    executor.serve_forever()
