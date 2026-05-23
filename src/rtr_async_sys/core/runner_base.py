# src/rtr_async_sys/core/runner_base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

from rtr_async_sys.core.user_base import AbsUser
from rtr_async_sys.core.controller_base import AbsController

class AbsRunner(ABC):
    """Runner coordinates User, Controller, and overall loop.
    Runs in main process, separate thread from Controller.
    """

    def __init__(self, users: List[AbsUser], controllers: List[AbsController]):
        self.users = users
        self.controllers = controllers

    @abstractmethod
    def run(self) -> None:
        """
        Main loop:
        - periodically call user.step()
        - control inference frequency, e.g. 50 Hz / 100 Hz
        """
        pass

    