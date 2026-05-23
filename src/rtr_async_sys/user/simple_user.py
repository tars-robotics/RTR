# src/rtr_async_sys/user/simple_user.py
from __future__ import annotations
from typing import Any, Dict, Union, Optional
from loguru import logger
from omegaconf import DictConfig, OmegaConf
import hydra
import zmq

from rtr_async_sys.core.user_base import AbsUser
from rtr_async_sys.core.model_wrapper_base import AbsModelWrapper
import numpy as np


class SimpleUser(AbsUser):
    """
    Minimal User subclass:
    - Reuse all communication logic from AbsUser
    - Use ModelWrapper for preprocess/predict/postprocess
    - No additional scheduling policy
    - No local state mutation
    """

    def __init__(
        self,
        model_wrapper: Union[AbsModelWrapper, str, DictConfig],
        controller_endpoint: str = "tcp://127.0.0.1:10020",
        scheduler_endpoint: str = "tcp://127.0.0.1:10030",
        identity: Optional[bytes] = None,
        context: Optional[zmq.Context] = None,
        user_hz: float = 2
    ):
        super().__init__(
            model_wrapper=model_wrapper,
            controller_endpoint=controller_endpoint,
            scheduler_endpoint=scheduler_endpoint,
            identity=identity,
            context=context,
            user_hz=user_hz
        )
        logger.info("[SimpleUser] Initialized.")
        
    # =====================================================================
    # Optional override for custom User-side scheduler handling.
    # =====================================================================

    def recv_schedule(self, schedule_dict: Dict[str, Any]) -> None:
        logger.info(f"[SimpleUser] Received schedule: {schedule_dict}")

    def reply_history(self) -> Dict[str, Any]:
        return {"simple_user_history": self._history}
    
    def predict_action_chunk(self, obs_dict: Dict[str, Any]) -> Any:
        return self.policy.predict_action_chunk(obs_dict)
    
    def refine_action(self, action:np.ndarray, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Input: \\
        1. action to refine \\
        2. obs_dict 
        """
        return self.policy.refine_action(action, obs_dict)


@hydra.main(
    config_path="../configs/user",
    config_name="dp_user",
    version_base=None
)
def main(cfg: DictConfig):
    if isinstance(cfg, str):
        cfg = OmegaConf.load(cfg)
    user = hydra.utils.instantiate(cfg)
    print(user)
    user.serve_forever()

if __name__ == '__main__':
    main()

# if __name__ == '__main__':
#     cfg = "src/rtr_async_sys/configs/user/simple_user.yaml"
#     cfg = OmegaConf.load(cfg)
#     simple_user = hydra.utils.instantiate(cfg)
#     # print(simple_user)
#     simple_user.serve_forever()