import numpy as np
import torch
import torch.nn as nn
from collections import deque
from lc.Reinforce_learning.Basealgos import BaseAlgo, AlgoConfig
from lc.NN.BaseNN import BaseRLModel
from dataclasses import dataclass

@dataclass
class TSAConfig(AlgoConfig):
    """Configuration for Temporal Sample Augmentation"""
    action_hold_k: int = 4       # Hold action for k steps
    stack_frames: int = 4        # Stack last n frames
    omega_base: float = 10.0     # Base bandwidth
    omega_range: float = 5.0     # Range: [base-range, base+range]

class TSA_LADRC_Agent(BaseAlgo):
    """
    RL Agent that tunes LADRC parameters with Temporal Sample Augmentation.
    Based on TD3 algorithm.
    """
    def __init__(self, model: BaseRLModel, optimizer, config: TSAConfig):
        super().__init__(config)
        self.model = model
        self.optimizer = optimizer
        self.cfg = config
        
        # TSA: State Stacking Buffer
        self.state_queue = deque(maxlen=self.cfg.stack_frames)
        
        # TSA: Action Holding Counter
        self.hold_counter = 0
        self.last_action = None

    def process_state(self, state: np.ndarray) -> np.ndarray:
        """TSA: Stack frames to create augmented state"""
        # If queue is empty (reset), fill with current state
        while len(self.state_queue) < self.cfg.stack_frames:
            self.state_queue.append(state)
        
        self.state_queue.append(state) # Push new state, pop old
        
        # Stack: (dim,) -> (dim * k,)
        return np.concatenate(list(self.state_queue), axis=-1)

    def select_action(self, state: np.ndarray, deterministic=False) -> np.ndarray:
        """
        Select action with Action Holding logic.
        Returns: [omega_c_norm, b0_norm] in [-1, 1]
        """
        # TSA: Action Holding
        if self.hold_counter > 0 and self.last_action is not None:
            self.hold_counter -= 1
            return self.last_action
        
        # Reset counter
        self.hold_counter = self.cfg.action_hold_k - 1
        
        # Prepare state (stacking)
        aug_state = self.process_state(state)
        state_t = torch.FloatTensor(aug_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_dist = self.model.forward_dist(state_t)
            action = action_dist.sample() if not deterministic else action_dist.mean
            
        action = action.cpu().numpy()[0]
        self.last_action = action
        return action

    def get_control_params(self, action: np.ndarray):
        """Map RL action [-1, 1] to LADRC params"""
        # Action[0] -> omega_c
        omega_c = self.cfg.omega_base + action[0] * self.cfg.omega_range
        
        # Action[1] -> b0 (optional, can be fixed)
        b0_scale = 1.0 + action[1] * 0.5 # [0.5, 1.5]
        
        return omega_c, b0_scale

    def update(self, batch):
        """Standard TD3 update (omitted for brevity, inherits structure)"""
        # Note: In TSA, the 'state' in batch must be the stacked state!
        # This requires the ReplayBuffer to store stacked states or stack them on sampling.
        pass

    def reset(self):
        self.state_queue.clear()
        self.hold_counter = 0
        self.last_action = None
