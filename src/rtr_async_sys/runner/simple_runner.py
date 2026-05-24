# src/rtr_async_sys/runner/simple_runner.py
from __future__ import annotations

import multiprocessing as mp
import signal
import time
from loguru import logger
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Union, List

# Compatibility with the previous directory layout.
from rtr_async_sys.core.controller_base import AbsController
from rtr_async_sys.core.executor_base import AbsExecutor
from rtr_async_sys.core.scheduler_base import AbsScheduler
from rtr_async_sys.core.user_base import AbsUser
# from rtr_async_sys.controller.sync_controller import SyncController
# from rtr_async_sys.executor.sync_nonservo_executor import SyncNonServoExecutor
# from rtr_async_sys.scheduler.simple_scheduler import SimpleScheduler
# from rtr_async_sys.user.simple_user import SimpleUser


class SimpleRunner:
    """
    SimpleRunner responsibilities:
    - Start Controller
    - Start Executor
    - Start Scheduler
    - Start Users (multiple users supported)
    - Manage all processes (start/stop)
    """

    def __init__(
        self,
        controller: Union[DictConfig, str, AbsController],
        executor: Union[DictConfig, str],
        scheduler: Union[DictConfig, str],
        **kwargs,
        # users: List[Union[DictConfig, str]],
    ):  
        if isinstance(controller, str):
            controller = OmegaConf.load(controller)
        if isinstance(executor, str):
            executor = OmegaConf.load(executor)
        if isinstance(scheduler, str):
            scheduler = OmegaConf.load(scheduler)
        # processed_users = []
        # for user_cfg in users:
        #     if isinstance(user_cfg,str):
        #         user_cfg = OmegaConf.load(user_cfg)
        #     processed_users.append(user_cfg)
        users = []
        for key in kwargs.keys():
            item = kwargs[key]
            if 'user' in key and (isinstance(item, AbsUser) or isinstance(item, DictConfig)):
                users.append(item)

        self.controller = controller
        self.executor = executor
        self.scheduler = scheduler
        self.users = users

        # process handler list
        self.processes = []

    # ----------------------------------------------------------------------
    # Start modules.
    # ----------------------------------------------------------------------

    def _start_controller(self):
        """Start Controller."""
        # cfg = self.controller
        def run():
            if isinstance(self.controller, DictConfig):
                self.controller = hydra.utils.instantiate(self.controller)
            self.controller.serve_forever()

        p = mp.Process(target=run, daemon=False)
        p.start()
        logger.info(f"[Runner] Controller started (pid={p.pid})")
        self.processes.append(p)

    def _start_executor(self):
        """Start Executor."""
        # cfg = self.executor

        def run():
            if isinstance(self.executor, DictConfig):
                self.executor = hydra.utils.instantiate(self.executor)
            self.executor.serve_forever()

        p = mp.Process(target=run, daemon=False)
        p.start()
        logger.info(f"[Runner] Executor started (pid={p.pid})")
        self.processes.append(p)

    def _start_scheduler(self):
        """Start Scheduler."""
        # cfg = self.scheduler

        def run():
            if isinstance(self.scheduler, DictConfig):
                self.scheduler = hydra.utils.instantiate(self.scheduler)
            self.scheduler.serve_forever()

        p = mp.Process(target=run, daemon=False)
        p.start()
        logger.info(f"[Runner] Scheduler started (pid={p.pid})")
        self.processes.append(p)

    def _start_users(self):
        """
        Start all Users.
        """
        for idx, user in enumerate(self.users):
            def run(user=user):
                if isinstance(user, DictConfig):
                    user = hydra.utils.instantiate(user)
                user.serve_forever()

            p = mp.Process(target=run, daemon=False)
            p.start()
            logger.info(f"[Runner] User[{idx}] started on port {user.scheduler_endpoint} (pid={p.pid})")
            self.processes.append(p)

    # ----------------------------------------------------------------------
    # Main startup flow.
    # ----------------------------------------------------------------------

    def launch_all(self):
        logger.info("[Runner] Launching all components...")
        start_interval_time = 1
        self._start_controller()
        time.sleep(start_interval_time)   # Ensure Controller starts first.

        self._start_executor()
        time.sleep(start_interval_time)

        self._start_scheduler()
        time.sleep(start_interval_time)

        self._start_users()
        time.sleep(start_interval_time)

        logger.info("[Runner] All components launched successfully.")

    # ----------------------------------------------------------------------
    # Stop all processes.
    # ----------------------------------------------------------------------

    def stop_all(self):
        logger.warning("[Runner] Stopping all processes…")

        for p in self.processes:
            try:
                p.terminate()
            except Exception:
                pass

        for p in self.processes:
            p.join(timeout=2)

        logger.info("[Runner] All processes terminated.")


# ----------------------------------------------------------------------
# Hydra Entry: main()
# ----------------------------------------------------------------------

@hydra.main(
    config_path="../configs",
    config_name="simple_runner",
    version_base=None
)
def main(cfg: DictConfig):
    if isinstance(cfg, str):
        cfg = OmegaConf.load(cfg)
    runner = hydra.utils.instantiate(cfg, _recursive_=False)
    if isinstance(runner, DictConfig):
        print("*"*100)
        print(runner)
        runner = SimpleRunner(
            controller=cfg.controller,
            executor=cfg.executor,
            scheduler=cfg.scheduler,
            # users=cfg.users,
        )

    # Ctrl+C stops all child processes.
    def handle_sigint(sig, frame):
        logger.warning("[Runner] Caught SIGINT, shutting down…")
        runner.stop_all()
        exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    # Start the full system.
    runner.launch_all()

    # Keep the Runner main thread blocked; if any child process exits, shut everything down.
    while True:
        for p in runner.processes:
            if p.exitcode is not None:
                logger.warning(
                    f"[Runner] Child process exited (pid={p.pid}, exitcode={p.exitcode}), shutting down..."
                )
                exitcode = int(p.exitcode)
                runner.stop_all()
                raise SystemExit(exitcode)
        time.sleep(1)


if __name__ == "__main__":
    # main("src/rtr_async_sys/configs/runner/simple_runner.yaml")
    main()
