"""
Temporal Ensemble.
"""

import torch
import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation as R, Slerp
from ...reactive_diffusion_policy.common.space_utils import ortho6d_to_rotation_matrix
import time
import threading

class AlignLazyController:
    """
    Alignment-enabled EnsembleBuffer. When using this buffer, do not truncate in advance with a custom `latency_step`. \\
    This is an experimental lazy alignment implementation called by inference and execution processes, rather than by inference/execution logic directly. \\
    write_lock prevents the model and executor from modifying the action queue simultaneously. Only write locking is used now, so executor reads can still race with model writes. \\
    TODO: implement ActiveAlignEnsembleBuffer. \\
    The lazy version does not fit `Batchstep` well because the slow system receives latent_action_chunk; compare latent similarity instead of action distance.

    """
    def __init__(self,
                 ensemble_mode = "new",
                 execute_horizon = 5,
                 n_obs_steps = 2,
                 obs_temporal_downsample_ratio = 2,
                ):
        assert ensemble_mode in ["new", "old", "avg", "act", "hato"], f"Ensemble mode {ensemble_mode} not supported now."
        self.mode = ensemble_mode
        self.timestep = 0
        self.last_update_timestep = -execute_horizon
        self.latest_obs = None
        self.actions = []
        self.action_shape = None

        self.execute_hoziron = execute_horizon
        self.n_obs_steps = n_obs_steps
        self.obs_temporal_downsample_ratio = obs_temporal_downsample_ratio

        self.write_lock = threading.Lock()

    def clear(self):
        """
        Clear the ensemble buffer.
        """
        self.timestep = 0
        self.latest_obs = None
        self.actions = []
        self.last_update_timestep = -self.execute_hoziron
    

    def need_update(self):
        return self.timestep - self.last_update_timestep >= self.execute_hoziron
    
    def get_latest_obs_timestep(self):
        """
        if last_obs is None, please get the obs manually!!!
        """
        return (self.latest_obs, self.timestep)

    def add_action(self, action_chunk, inf_timestep):
        """
        Add action to the ensemble buffer:

        Parameters:
        - action_chunk: horizon x action_dim (...);
        - inf_timestep: inference_timestep
        """
        if self.timestep - self.last_update_timestep < self.execute_hoziron:
            print(f"It's not the time to add_action")
            return
        
        with self.write_lock:
            action_chunk = np.array(action_chunk)
            if self.action_shape == None:
                self.action_shape = action_chunk.shape[1:]
                assert len(self.action_shape) == 1, "Only support action with 1D shape."
            else:
                assert self.action_shape == action_chunk.shape[1:], "Incompatible action shape."

            start = self.timestep - inf_timestep  # Number of executed steps since inference started.
            horizon = action_chunk.shape[0]

            self.actions = []
            print("============================== add action ======================================")
            # time.sleep(3)
            # horizon = start + self.execute_horizon # Debugging: pause the fast system when the slow system sleeps.
            for i in range(start, horizon):
                # print(f"action_chunk[{i}] is {action_chunk[i][0:3]}")
                if len(self.actions) > i-start:
                    self.actions[i-start] = action_chunk[i]
                else:
                    self.actions.append(action_chunk[i])
            
            self.last_update_timestep = self.timestep
                
            print(f"Added a new action chunk to the queue, horizon is {action_chunk.shape[0]}, timestep is {self.timestep}, inf_timestep is {inf_timestep}, len(self.actions) is {len(self.actions)} start is {start}")
        
    def get_action(self):
        """
        Get ensembled action from buffer. \\
        Call update after executing an action.
        """
        if len(self.actions) == 0:
            return None
        
        action = self.actions[0]
        # action = self.actions.pop(0)
        # self.timestep += 1
        # Check whether each get returns only one action, the latest one, until a new queue item is added
        print(f"[get action] timestep is {self.timestep} action[0:10] is {action[0:3]}")
        return action
    
    def update(self, env):
        """
        Execution finished; update latest_obs.
        """
        with self.write_lock:  # Use a lock to avoid undefined behavior from concurrent writes.
            print("update obs")
            obs = env.get_obs(
                        obs_steps=self.n_obs_steps,
                        temporal_downsample_ratio=self.obs_temporal_downsample_ratio)
            if len(self.actions) != 0:
                self.actions.pop(0)
            self.timestep += 1
            self.latest_obs = obs



