# src/rtr_async_sys/core/env_base.py

from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np

class AbsEnv(ABC):
    """Abstract environment interface.
    Executor interacts with the environment only through this interface.
    """

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the robot."""
        pass

    @abstractmethod
    def restart(self) -> None:
        """Stop the robot."""
        pass

    @abstractmethod
    def set_mode(self) -> None:
        """set the robot's mode."""
        pass

    @abstractmethod
    def get_obs(self, n_obs_steps=None) -> Dict[str, np.ndarray]:
        """Fetch the latest observation from the environment."""
        pass

    @abstractmethod
    def execute_action(self, action: np.ndarray) -> None:
        """Send action command to the environment."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        clear the buffer of real env. and call `end_of_chunk` of dataset_env
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        reset env
        """
        pass
    
    @abstractmethod
    def end_of_chunk(self):
        pass
