"""
Temporal Ensemble.
"""

import torch
import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation as R, Slerp
from reactive_diffusion_policy.common.space_utils import ortho6d_to_rotation_matrix
import time
import threading

class AlignLazyController:
    """
    Alignment-version of EnsembleBuffer. Do NOT pre-truncate with a custom `latency_step` when using this buffer.
    This is the experimental "lazy" alignment variant: invoked by the inference and the executor threads,
    rather than driving the inference / execution logic itself.
    A write_lock prevents the model and the executor from mutating the action queue simultaneously.
    Only the write lock is enforced; reader-vs-writer conflicts between the executor and the model are
    ignored for now.
    TODO: implement ActiveAlignEnsembleBuffer.
    The lazy version cannot easily implement `Batchstep` because the slow system produces latent_action chunks
    and cannot directly compare actions -- only latent similarities.

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

            start = self.timestep - inf_timestep  # how many steps elapsed since inference began
            horizon = action_chunk.shape[0]

            self.actions = []
            print("============================== add action ======================================")
            # time.sleep(3)
            # horizon = start + self.execute_horizon  # debug: when slow system sleeps(2), also pause fast system
            for i in range(start, horizon):
                # print(f"action_chunk[{i}] is {action_chunk[i][0:3]}")
                if len(self.actions) > i-start:
                    self.actions[i-start] = action_chunk[i]
                else:
                    self.actions.append(action_chunk[i])
            
            self.last_update_timestep = self.timestep
                
            print(f"Appended new action to queue. horizon={action_chunk.shape[0]}, timestep={self.timestep}, inf_timestep={inf_timestep}, len(actions)={len(self.actions)}, start={start}")
        
    def get_action(self):
        """
        Get ensembled action from buffer.

        Remember to call `update` after executing the action.
        """
        if len(self.actions) == 0:
            return None
        
        action = self.actions[0]
        # action = self.actions.pop(0)
        # self.timestep += 1
        # sanity check: every get returns the same (latest) action until a new chunk is appended
        print(f"[get action] timestep is {self.timestep} action[0:10] is {action[0:3]}")
        return action
    
    def update(self, env):
        """
        After executing an action, update `latest_obs`.
        """
        with self.write_lock:  # lock prevents concurrent writes between threads (avoids UB)
            print("update obs")
            obs = env.get_obs(
                        obs_steps=self.n_obs_steps,
                        temporal_downsample_ratio=self.obs_temporal_downsample_ratio)
            if len(self.actions) != 0:
                self.actions.pop(0)
            self.timestep += 1
            self.latest_obs = obs



