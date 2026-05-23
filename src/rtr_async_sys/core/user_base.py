# src/rtr_async_sys/core/user_base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
from omegaconf import DictConfig, OmegaConf
import hydra

import numpy as np
import pickle
import zmq
from loguru import logger
import time
import threading

from rtr_async_sys.core.model_wrapper_base import AbsModelWrapper


class AbsUser(ABC):
    """
    User runs in a separate process with two ZMQ channels:
    - DEALER → Controller (pull obs, push action)
    - REP     → Scheduler (reply history, receive schedule, etc.)
    - REP     → Controller(refine)  (provides refine support)

    User runs a synchronous loop: each frame polls Scheduler commands before running one inference step.
    """

    def __init__(
        self,
        model_wrapper: Union[AbsModelWrapper, str, DictConfig],
        controller_endpoint: str = "tcp://127.0.0.1:10020",
        scheduler_endpoint: str = "tcp://*:10030",   # Scheduler connects to this endpoint
        refine_bind: Optional[str] = None,           # e.g. "tcp://*:10031",optional; set to None to choose a random port
        identity: Optional[bytes] = None,
        context: Optional[zmq.Context] = None,
        user_hz:float = 2 #DP inference is roughly 0.6s; 2 Hz is near the useful upper bound. Lower values can reduce controller queue discontinuity.
    ):
        self.user_hz = user_hz
        # --------------------------
        # Model wrapper instantiation
        # --------------------------
        logger.info(f"before instantiate, model wrapper type is {type(model_wrapper)}")
        if isinstance(model_wrapper, str):
            model_wrapper = OmegaConf.load(model_wrapper)
        if isinstance(model_wrapper, DictConfig):
            logger.info("instantiate model_wrapper")
            model_wrapper = hydra.utils.instantiate(model_wrapper)
        logger.debug(f"model_wrapper is {model_wrapper}")
        assert isinstance(model_wrapper, AbsModelWrapper), type(model_wrapper)
        self.policy = model_wrapper
        self.need_refine = self.policy.need_refine

        self.input_context = context
        self.controller_endpoint = controller_endpoint
        self.scheduler_endpoint = scheduler_endpoint
        self.refine_bind = refine_bind
        self.identity = identity

        # # --------------------------
        # # ZMQ Context
        # # --------------------------
        # self._context = context or zmq.Context.instance()

        # # -----------------------------------------
        # # DEALER socket → Controller
        # # -----------------------------------------
        # self._sock_ctrl = self._context.socket(zmq.DEALER)

        # if identity is None:
        #     import uuid
        #     identity = uuid.uuid4().bytes
        # self.identity = identity

        # self._sock_ctrl.setsockopt(zmq.IDENTITY, self.identity)
        # self._sock_ctrl.connect(controller_endpoint)
        # logger.info(f"[User] DEALER → Controller at {controller_endpoint}")

        # # -----------------------------------------
        # # REP socket ← Scheduler (Scheduler REQ)
        # # -----------------------------------------
        # self._sock_sched = self._context.socket(zmq.REP)
        # self._sock_sched.bind(scheduler_endpoint)
        # logger.info(f"[User] REP bind ← Scheduler at {scheduler_endpoint}")

        # # -----------------------------------------
        # # (refine)REP socket ← Controller (refine REQ)
        # # Enabled only when policy.need_refine is True
        # # -----------------------------------------
        # self._sock_refine: Optional[zmq.Socket] = None
        # self._refine_endpoint: Optional[str] = None
        # if getattr(self.policy, "need_refine", False):
        #     self._sock_refine = self._context.socket(zmq.REP)
        #     bind_addr = refine_bind or "tcp://*:0"   # random or configured port
        #     self._sock_refine.bind(bind_addr)

        #     # Get the actual endpoint, e.g. "tcp://0.0.0.0:12345"
        #     endpoint = self._sock_refine.getsockopt(zmq.LAST_ENDPOINT).decode()
        #     # Controller cannot connect to 0.0.0.0, so replace it with localhost
        #     endpoint = endpoint.replace("0.0.0.0", "127.0.0.1")
        #     self._refine_endpoint = endpoint

        #     logger.info(f"[User] REFINE REP bind ← Controller at {endpoint}")

        #     # Register the refiner endpoint with Controller
        #     # Note: use the existing DEALER channel and perform one send -> recv pair
        #     self._send_ctrl(
        #         {
        #             "type": "register_refiner",
        #             "endpoint": self._refine_endpoint,
        #         }
        #     )
        #     reply = self._recv_ctrl()
        #     if reply.get("status") != "ok":
        #         raise RuntimeError(f"[User] register_refiner failed: {reply}")
        #     logger.info(f"[User] register_refiner OK, endpoint={self._refine_endpoint}")      


        # # Poller listens to controller and scheduler
        # self._poller = zmq.Poller()
        # self._poller.register(self._sock_sched, zmq.POLLIN)
        # if self._sock_refine is not None:
        #     self._poller.register(self._sock_refine, zmq.POLLIN)
        # # Controller DEALER does not need polling because user actively pulls observations

        # --------------------------
        # Internal state
        # --------------------------
        self._running = False
        self._history = []

        # Start step thread
        self._step_thread = threading.Thread(
            target=self.step_thread,
            daemon=True       # set as daemon so it exits with the main process
        )
        freq  = self.user_hz
        dt = 1.0 / freq
        logger.info(f"[User] step @ {freq} Hz")
        # self._step_thread.start()


    # refine thread
    def step_thread(self) -> None:
        freq  = self.user_hz
        dt = 1.0 / freq

        logger.info(f"[User] step @ {freq} Hz")

        while True:
            if self._running:
                start = time.time()
                # ---------------------
                # Run inference step
                # ---------------------
                try:
                    self.step()
                except Exception as e:
                    logger.exception(f"[User] step failed: {e}")

                # Rate control
                elapsed = time.time() - start
                if dt - elapsed > 0:
                    time.sleep(dt - elapsed)


    # =====================================================================
    #                     Controller communication (DEALER)
    # =====================================================================

    def _send_ctrl(self, msg: Dict[str, Any]) -> None:
        self._sock_ctrl.send(pickle.dumps(msg))

    def _recv_ctrl(self) -> Dict[str, Any]:
        return pickle.loads(self._sock_ctrl.recv())

    def get_obs_and_timestep(self) -> Dict[str, Any]:
        self._send_ctrl({"type": "get_obs_and_timestep"})
        reply = self._recv_ctrl()
        if reply.get("status") != "ok":
            raise RuntimeError(f"[User] get_obs_and_timestep error: {reply}")
        return reply

    def send_action_chunk(self, action_chunk:np.ndarray, timestep: int) -> None:
        self._send_ctrl(
            {"type": "send_action_chunk", "action_chunk": action_chunk, "timestep": timestep, "need_refine": self.policy.need_refine}
        )
        reply = self._recv_ctrl()
        if reply.get("status") != "ok":
            raise RuntimeError(f"[User] send_action_chunk error: {reply}")

    # =====================================================================
    #               Scheduler communication (REQ->REP, User is REP)
    # =====================================================================

    def _handle_scheduler(self, msg: Dict[str, Any]):
        """Handle messages received from Scheduler"""
        mtype = msg.get("type")

        if mtype == "scheduler_request_info":
            reply = self.reply_history()
            reply["status"] = "ok"
            self._sock_sched.send(pickle.dumps(reply))
            return

        elif mtype == "scheduler_send_schedule":
            schedule = msg.get("schedule", {})
            self.recv_schedule(schedule)
            self._sock_sched.send(pickle.dumps({"status": "ok"}))
            return

        else:
            # Extension point
            logger.warning(f"[User] Unknown scheduler message: {msg}")
            self._sock_sched.send(pickle.dumps({"status": "unknown_type"}))

    def _handle_refine_request(self, msg: Dict[str, Any]):
        """
        Controller to User refine RPC.
        Message schema:
          {"type": "refine_action", "action": np.ndarray, "obs": Dict[str, np.ndarray]}
        """
        mtype = msg.get("type")
        if mtype == "refine_action":
            action = msg["action"]
            obs_dict = msg["obs"]
            refined = self.refine_action(action, obs_dict)
            self._sock_refine.send_pyobj({"status": "ok", "action": refined})
        else:
            logger.warning(f"[User] Unknown refine msg type: {mtype}")
            self._sock_refine.send_pyobj(
                {"status": "error", "msg": f"unknown refine msg type: {mtype}"}
            )


    # Subclasses may override this
    def reply_history(self) -> Dict[str, Any]:
        return {"history": self._history}

    def recv_schedule(self, schedule_dict: Dict[str, Any]) -> None:
        logger.info(f"[User] Received schedule: {schedule_dict}")

    # =====================================================================
    #                         Model inference interface
    # =====================================================================

    @abstractmethod
    def predict_action_chunk(self, processed_obs: Dict[str, Any]) -> Any:
        pass

    def refine_action(self, action:np.ndarray, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Input: \\
        1. action to refine \\
        2. obs_dict 
        """
        pass

    # =====================================================================
    #                          Full inference step
    # =====================================================================

    def step(self):
        logger.debug("================================= user time profile ================================")
        # 1) Pull observation
        start = time.time()
        data = self.get_obs_and_timestep()
        obs = data["obs"]
        timestep = data["timestep"]
        logger.debug(f"[get_obs_time] is {time.time() - start}")# sync executor: tens of milliseconds of latency
        
        # 2) Full model inference
        start = time.time()
        action = self.predict_action_chunk(obs)
        logger.debug(f"[predict_action_chunk_time] is {time.time() - start}") # sync executor: 600-900 ms

        # 3) Send action chunk
        start = time.time()
        self.send_action_chunk(action, timestep)
        logger.debug(f"[send_action_chunk_time] is {time.time() - start}")# sync executor: end-of-chunk return means the full chunk finished; total latency can exceed 2s without merging

        # 4) Save history
        self._history.append({"timestep": timestep, "action": action})


    def _init_sock(self):
        # --------------------------
        # ZMQ Context
        # --------------------------
        self._context = self.input_context or zmq.Context.instance()

        # -----------------------------------------
        # DEALER socket → Controller
        # -----------------------------------------
        self._sock_ctrl = self._context.socket(zmq.DEALER)

        if self.identity is None:
            import uuid
            self.identity = uuid.uuid4().bytes
        # self.identity = identity

        self._sock_ctrl.setsockopt(zmq.IDENTITY, self.identity)
        self._sock_ctrl.connect(self.controller_endpoint)
        logger.info(f"[User] DEALER → Controller at {self.controller_endpoint}")

        # -----------------------------------------
        # REP socket ← Scheduler (Scheduler REQ)
        # -----------------------------------------
        self._sock_sched = self._context.socket(zmq.REP)
        self._sock_sched.bind(self.scheduler_endpoint)
        logger.info(f"[User] REP bind ← Scheduler at {self.scheduler_endpoint}")

        # -----------------------------------------
        # (refine)REP socket ← Controller (refine REQ)
        # Enabled only when policy.need_refine is True
        # -----------------------------------------
        self._sock_refine: Optional[zmq.Socket] = None
        self._refine_endpoint: Optional[str] = None
        if getattr(self.policy, "need_refine", False):
            self._sock_refine = self._context.socket(zmq.REP)
            bind_addr = self.refine_bind or "tcp://*:0"   # random or configured port
            self._sock_refine.bind(bind_addr)

            # Get the actual endpoint, e.g. "tcp://0.0.0.0:12345"
            endpoint = self._sock_refine.getsockopt(zmq.LAST_ENDPOINT).decode()
            # Controller cannot connect to 0.0.0.0, so replace it with localhost
            endpoint = endpoint.replace("0.0.0.0", "127.0.0.1")
            self._refine_endpoint = endpoint

            logger.info(f"[User] REFINE REP bind ← Controller at {endpoint}")

            # Register the refiner endpoint with Controller
            # Note: use the existing DEALER channel and perform one send -> recv pair
            self._send_ctrl(
                {
                    "type": "register_refiner",
                    "endpoint": self._refine_endpoint,
                }
            )
            reply = self._recv_ctrl()
            if reply.get("status") != "ok":
                raise RuntimeError(f"[User] register_refiner failed: {reply}")
            logger.info(f"[User] register_refiner OK, endpoint={self._refine_endpoint}")      


        # Poller listens to controller and scheduler
        self._poller = zmq.Poller()
        self._poller.register(self._sock_sched, zmq.POLLIN)
        if self._sock_refine is not None:
            self._poller.register(self._sock_refine, zmq.POLLIN)
        # Controller DEALER does not need polling because user actively pulls observations

    # =====================================================================
    #                         Main loop:poll Scheduler + step
    # =====================================================================

    def serve_forever(self):
        """
        Synchronous inference loop with scheduler polling:
        - poll Scheduler before each frame
        - then run step()
        """
        self._init_sock()

        self._step_thread.start()

        self._running = True

        while self._running:
            socks = dict(self._poller.poll(timeout=1))
            # ---------------------
            # Handle Scheduler requests (non-blocking)
            # ---------------------

            if socks.get(self._sock_sched) == zmq.POLLIN:
                raw = self._sock_sched.recv()
                msg = pickle.loads(raw)
                self._handle_scheduler(msg)
            
            # Refine(optional)
            if self._sock_refine is not None and socks.get(self._sock_refine) == zmq.POLLIN:
                msg = self._sock_refine.recv_pyobj()
                self._handle_refine_request(msg)

            # Controller requests are sent from step(), which runs in a separate thread


        logger.info("[User] Stopped.")

    def stop(self):
        self._running = False


    def serve_forever_20s_disconnect(self):
        """
        Synchronous inference loop with scheduler polling:
        - poll Scheduler / refine
        - monitor Controller(DEALER) connection state through the ZMQ monitor
        - Controller disconnected for more than 20s -> treat as offline -> reinitialize sockets and ensure step_thread is running
        Not tested yet
        """
        from zmq.utils.monitor import recv_monitor_message

        # --------------------------
        # helper: cleanly close old sockets/poller
        # --------------------------
        def _safe_close_sock(sock: Optional[zmq.Socket], name: str):
            if sock is None:
                return
            try:
                sock.close(linger=0)
                logger.info(f"[User] closed socket: {name}")
            except Exception as e:
                logger.warning(f"[User] close socket {name} failed: {e}")

        def _teardown_io(monitor_sock: Optional[zmq.Socket], monitor_addr: Optional[str]):
            # unregister poller
            try:
                if hasattr(self, "_poller") and self._poller is not None:
                    try:
                        self._poller.unregister(self._sock_sched)
                    except Exception:
                        pass
                    try:
                        if self._sock_refine is not None:
                            self._poller.unregister(self._sock_refine)
                    except Exception:
                        pass
                    try:
                        if monitor_sock is not None:
                            self._poller.unregister(monitor_sock)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"[User] poller unregister failed: {e}")

            # stop monitor first
            try:
                if hasattr(self, "_sock_ctrl") and self._sock_ctrl is not None:
                    try:
                        # disable monitoring
                        self._sock_ctrl.monitor(None, 0)
                    except Exception:
                        pass
            except Exception:
                pass

            # close sockets
            _safe_close_sock(monitor_sock, "ctrl_monitor")
            _safe_close_sock(getattr(self, "_sock_refine", None), "refine_rep")
            _safe_close_sock(getattr(self, "_sock_sched", None), "sched_rep")
            _safe_close_sock(getattr(self, "_sock_ctrl", None), "ctrl_dealer")

        def _setup_ctrl_monitor() -> tuple[zmq.Socket, str]:
            # unique inproc address (avoid conflicts across restarts)
            addr = f"inproc://ctrl-monitor-{id(self)}-{int(time.time()*1000)}"
            # monitor all connection-related events
            self._sock_ctrl.monitor(
                addr,
                zmq.EVENT_CONNECTED
                | zmq.EVENT_DISCONNECTED
                | zmq.EVENT_CONNECT_DELAYED
                | zmq.EVENT_CONNECT_RETRIED
                | zmq.EVENT_CLOSED
            )
            msock = self._context.socket(zmq.PAIR)
            msock.connect(addr)
            return msock, addr

        def _apply_ctrl_timeouts():
            # Important: prevent step_thread from blocking forever in recv/send
            # This lets serve_forever set _running=False and stop the step thread when disconnected for more than 20s
            try:
                self._sock_ctrl.setsockopt(zmq.RCVTIMEO, 20000)  # 1s
                self._sock_ctrl.setsockopt(zmq.SNDTIMEO, 20000)  # 1s
                self._sock_ctrl.setsockopt(zmq.LINGER, 0)
            except Exception as e:
                logger.warning(f"[User] set ctrl socket timeouts failed: {e}")

        # --------------------------
        # init sockets + monitor
        # --------------------------
        self._init_sock()
        _apply_ctrl_timeouts()

        monitor_sock, monitor_addr = _setup_ctrl_monitor()
        # Register monitor_sock with the poller
        self._poller.register(monitor_sock, zmq.POLLIN)

        # Step thread: start it if it has not started or has stopped
        if not self._step_thread.is_alive():
            self._step_thread = threading.Thread(target=self.step_thread, daemon=True)
            self._step_thread.start()
        else:
            # Do not start it again if already started; Python threads cannot be restarted
            pass

        self._running = True  # allow step_thread to run step()

        # --------------------------
        # connection state tracking
        # --------------------------
        disconnected_since: Optional[float] = None
        last_connected_ts: float = time.time()

        # Distinguish stop() from temporary _running=False during restart
        restarting = False

        while True:
            # stop() sets self._running to False
            if (not self._running) and (not restarting):
                break

            socks = dict(self._poller.poll(timeout=1_000))  # ms

            # ---------------------
            # 1) controller connection monitor events
            # ---------------------
            if socks.get(monitor_sock) == zmq.POLLIN:
                try:
                    evt = recv_monitor_message(monitor_sock, flags=zmq.NOBLOCK)
                    ev = evt.get("event")
                    now = time.time()

                    if ev == zmq.EVENT_CONNECTED:
                        last_connected_ts = now
                        disconnected_since = None
                        logger.info("[User] Controller connected.")

                    elif ev in (zmq.EVENT_DISCONNECTED, zmq.EVENT_CLOSED):
                        if disconnected_since is None:
                            disconnected_since = now
                            logger.warning("[User] Controller disconnected.")

                    elif ev in (zmq.EVENT_CONNECT_DELAYED, zmq.EVENT_CONNECT_RETRIED):
                        # The connection is retrying; it may not be disconnected yet but is unstable
                        if disconnected_since is None:
                            disconnected_since = now
                        logger.warning("[User] Controller connection retrying...")

                except zmq.Again:
                    pass
                except Exception as e:
                    logger.warning(f"[User] monitor recv failed: {e}")

            # ---------------------
            # 2) detect offline > 20s -> restart sockets
            # ---------------------
            if disconnected_since is not None:
                if (time.time() - disconnected_since) > 20.0:
                    logger.error("[User] Controller offline > 20s, restarting sockets...")

                    # Pause step; timeouts make step_thread raise zmq.Again and observe _running=False on the next loop
                    restarting = True
                    self._running = False

                    # teardown old IO
                    _teardown_io(monitor_sock, monitor_addr)

                    # Reinitialize sockets
                    self._init_sock()
                    _apply_ctrl_timeouts()

                    # Recreate monitor and register it with the poller
                    monitor_sock, monitor_addr = _setup_ctrl_monitor()
                    self._poller.register(monitor_sock, zmq.POLLIN)

                    # Ensure the step thread is running
                    if not self._step_thread.is_alive():
                        self._step_thread = threading.Thread(target=self.step_thread, daemon=True)
                        self._step_thread.start()

                    # Resume step
                    disconnected_since = None
                    last_connected_ts = time.time()
                    self._running = True
                    restarting = False

                    logger.info("[User] Restart complete.")
                    continue

            # ---------------------
            # 3) handle Scheduler (non-blocking)
            # ---------------------
            if socks.get(self._sock_sched) == zmq.POLLIN:
                raw = self._sock_sched.recv()
                msg = pickle.loads(raw)
                self._handle_scheduler(msg)

            # ---------------------
            # 4) handle refine (optional)
            # ---------------------
            if self._sock_refine is not None and socks.get(self._sock_refine) == zmq.POLLIN:
                msg = self._sock_refine.recv_pyobj()
                self._handle_refine_request(msg)

        logger.info("[User] Stopped.")