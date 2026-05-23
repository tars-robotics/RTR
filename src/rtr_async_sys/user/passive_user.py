"""
passive_user.py

PassiveUser:
- Controller actively sends REQ to User: get_action_chunk(obs, timestep)
- User passively replies to Controller with action_chunk over REP.
- Scheduler keeps the AbsUser REQ/REP scheduling interface, with User as REP.
"""

from __future__ import annotations

from typing import Any, Dict, Union, Optional

import time
import pickle
import zmq
import numpy as np
from loguru import logger
from omegaconf import DictConfig, OmegaConf
import hydra

from rtr_async_sys.core.user_base import AbsUser
from rtr_async_sys.core.model_wrapper_base import AbsModelWrapper


class PassiveUser(AbsUser):
    """
    Passive User:
    - Does not actively pull obs from Controller.
    - Controller sends get_action_chunk requests over REQ, including obs and timestep.
    - User runs inference and replies with action_chunk directly over REP.

    REQ/REP sockets time out after 3 seconds without a reply; reset and reinitialize
    sockets after 10 seconds of silence.
    """

    def __init__(
        self,
        model_wrapper: Union[AbsModelWrapper, str, DictConfig],
        # For PassiveUser, controller_endpoint is the bind address because User is REP.
        controller_endpoint: str = "tcp://127.0.0.1:10020",
        scheduler_endpoint: str = "tcp://127.0.0.1:10030",
        identity: Optional[bytes] = None,
        context: Optional[zmq.Context] = None,
        user_hz: float = 2,  # PassiveUser does not depend on this frequency, but keeps interface consistency.
    ):
        super().__init__(
            model_wrapper=model_wrapper,
            controller_endpoint=controller_endpoint,
            scheduler_endpoint=scheduler_endpoint,
            identity=identity,
            context=context,
            user_hz=user_hz,
        )
        logger.info("[PassiveUser] Initialized.")

    # ============================================================
    # Keep or override these hooks as needed; they match the SimpleUser style.
    # ============================================================

    def recv_schedule(self, schedule_dict: Dict[str, Any]) -> None:
        logger.info(f"[PassiveUser] Received schedule: {schedule_dict}")

    def reply_history(self) -> Dict[str, Any]:
        return {"passive_user_history": self._history}

    def predict_action_chunk(self, obs_dict: Dict[str, Any]) -> Any:
        return self.policy.predict_action_chunk(obs_dict)

    def refine_action(self, action: np.ndarray, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return self.policy.refine_action(action, obs_dict)


    def _close_sockets(self):
        """
        Unregister the current controller/scheduler sockets from the poller and close them,
        preventing duplicate registrations and handle leaks.
        """
        if getattr(self, "_poller", None) is not None:
            for sock in [getattr(self, "_sock_ctrl", None), getattr(self, "_sock_sched", None)]:
                if sock is not None:
                    try:
                        self._poller.unregister(sock)
                    except Exception:
                        pass

        for attr in ["_sock_ctrl", "_sock_sched"]:
            sock = getattr(self, attr, None)
            if sock is not None:
                try:
                    sock.close(0)
                except Exception:
                    pass
                setattr(self, attr, None)
    
    def _reset_due_to_timeout(self, idle_sec: float, idle_timeout: float):
        """
        Called when controller/scheduler send no requests within idle_timeout seconds:
        - Reset model state.
        - Clear history.
        - Close old sockets.
        - Reinitialize sockets and bind endpoints.
        """
        logger.warning(
            f"[PassiveUser] No messages from controller/scheduler for "
            f"{idle_sec:.1f}s (>= {idle_timeout}s). Resetting user state and sockets..."
        )

        # 1) Reset model.
        try:
            if hasattr(self, "policy") and hasattr(self.policy, "reset"):
            # if True:
                self.policy.reset()
                logger.info("[PassiveUser] policy.reset() called.")
            else:
                logger.warning("[PassiveUser] policy has no reset() method.")
        except Exception as e:
            logger.exception(f"[PassiveUser] policy.reset() failed: {e}")

        # 2) Clear history (optional).
        try:
            if hasattr(self, "_history"):
                self._history.clear()
        except Exception:
            pass

        # 3) Close old sockets.
        self._close_sockets()

        # 4) Reinitialize sockets and bind again.
        self._init_sock()

    # ============================================================
    # ZMQ initialization: the key difference is the controller channel.
    # ============================================================

    def _init_sock(self):
        # --------------------------
        # ZMQ Context
        # --------------------------
        self._context = self.input_context or zmq.Context.instance()

        # -----------------------------------------
        # REP socket ← Controller (Controller REQ)
        # -----------------------------------------
        # PassiveUser binds and waits for Controller requests instead of connecting to Controller.
        self._sock_ctrl = self._context.socket(zmq.REP)
        self._sock_ctrl.bind(self.controller_endpoint)
        logger.info(f"[PassiveUser] REP bind ← Controller at {self.controller_endpoint}")

        # -----------------------------------------
        # REP socket ← Scheduler (Scheduler REQ)
        # -----------------------------------------
        self._sock_sched = self._context.socket(zmq.REP)
        self._sock_sched.bind(self.scheduler_endpoint)
        logger.info(f"[PassiveUser] REP bind ← Scheduler at {self.scheduler_endpoint}")

        for s in (self._sock_ctrl, self._sock_sched):
            # Do not block during close.
            # s.setsockopt(zmq.LINGER, 0)
            # Send timeout avoids hangs in extreme cases.
            s.setsockopt(zmq.SNDTIMEO, 8000)
            # Optional: cap queue depth to avoid unbounded buildup.
            # s.setsockopt(zmq.SNDHWM, 1000)

        # -----------------------------------------
        # No extra refine channel for PassiveUser for now.
        # Add another REQ/REP endpoint later if Controller needs passive refine requests.
        # -----------------------------------------
        self._sock_refine = None
        self._refine_endpoint = None

        # Poller listens to controller and scheduler
        self._poller = zmq.Poller()
        self._poller.register(self._sock_ctrl, zmq.POLLIN)
        self._poller.register(self._sock_sched, zmq.POLLIN)

    # ============================================================
    # Controller request handling
    # ============================================================

    def _handle_controller_request(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle Controller -> User requests.
        Expected format:
        - msg = {"type":"get_action_chunk", "obs":..., "timestep":...}
        - reply = {"status":"ok","action_chunk":..., "timestep":...}
        """
        req_type = msg.get("type", None)

        if req_type == "get_action_chunk":
            obs = msg.get("obs", None)
            timestep = msg.get("timestep", None)

            if obs is None or timestep is None:
                return {
                    "status": "error",
                    "error": "missing field: obs or timestep",
                }

            try:
                t0 = time.time()
                action_chunk = self.predict_action_chunk(obs)
                dt = time.time() - t0
                logger.info(f"predict_action_latency is {dt}")

                # Optional: record history.
                self._history.append(
                    {
                        "timestep": timestep,
                        "action_chunk": action_chunk,
                        "latency_sec": dt,
                    }
                )

                return {
                    "status": "ok",
                    "action_chunk": action_chunk,
                    "timestep": timestep,
                    "latency_sec": dt,
                }
            except Exception as e:
                logger.exception(f"[PassiveUser] get_action_chunk failed: {e}")
                return {
                    "status": "error",
                    "error": repr(e),
                }

        elif req_type == "ping":
            return {"status": "ok", "msg": "pong"}

        else:
            return {"status": "error", "error": f"unknown controller request type: {req_type}"}

    # ============================================================
    # Main loop: only poll and respond passively.
    # ============================================================

    def serve_forever(self):
        self._init_sock()
        self._running = True
        logger.info("[PassiveUser] serve_forever started.")

        idle_timeout = 15.0  # Seconds; shared silence threshold for controller/scheduler disconnect detection.
        last_msg_ts = time.time()  # Time of the most recent message from either side.

        while self._running:
            try:
                # Poll for 1 second: responsive enough for exit and timeout detection.
                socks = dict(self._poller.poll(timeout=1000))  # ms
            except zmq.error.ZMQError as e:
                logger.exception(f"[PassiveUser] Poller error: {e}")
                # Sleep briefly after errors to avoid a busy loop.
                time.sleep(1.0)
                continue

            now = time.time()
            got_any_msg = False

            # 1) Controller request
            if socks.get(self._sock_ctrl) == zmq.POLLIN:
                got_any_msg = True
                last_msg_ts = now

                try:
                    raw = self._sock_ctrl.recv()
                    msg = pickle.loads(raw)
                except Exception as e:
                    logger.exception(f"[PassiveUser] Failed to recv/deserialize controller msg: {e}")
                    # REP must reply once, otherwise the Controller REQ will block.
                    try:
                        self._sock_ctrl.send(
                            pickle.dumps({"status": "error", "error": "bad request"})
                        )
                    except Exception:
                        pass
                    # Malformed packets do not imply disconnect because the peer is still sending.
                    continue

                reply = self._handle_controller_request(msg)
                try:
                    self._sock_ctrl.send(pickle.dumps(reply))
                except Exception as e:
                    logger.exception(f"[PassiveUser] Failed to serialize/send reply to controller: {e}")

            # 2) Scheduler request
            if socks.get(self._sock_sched) == zmq.POLLIN:
                got_any_msg = True
                last_msg_ts = now

                try:
                    raw = self._sock_sched.recv()
                    sched_msg = pickle.loads(raw)
                    self._handle_scheduler(sched_msg)
                except Exception as e:
                    logger.exception(f"[PassiveUser] Failed to handle scheduler msg: {e}")
                    # REP must reply once.
                    try:
                        self._sock_sched.send(pickle.dumps({"status": "error", "error": repr(e)}))
                    except Exception:
                        pass

            # 3) If no messages arrive for a long time, treat controller/scheduler as disconnected and reset.
            if not got_any_msg:
                idle_sec = now - last_msg_ts
                if idle_sec >= idle_timeout:
                    self._reset_due_to_timeout(idle_sec=idle_sec, idle_timeout=idle_timeout)
                    # After reset, start timing from now.
                    last_msg_ts = time.time()

        logger.info("[PassiveUser] Stopped.")


@hydra.main(
    config_path="../configs/user",
    config_name="dp_passive_user",  # Switch to a dedicated passive_user.yaml later if needed.
    version_base=None,
)
def main(cfg: DictConfig):
    if isinstance(cfg, str):
        cfg = OmegaConf.load(cfg)
    user = hydra.utils.instantiate(cfg)
    print(user)
    user.serve_forever()


if __name__ == "__main__":
    main()
