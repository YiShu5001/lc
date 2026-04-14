from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lc.control.controllers import AdaptiveLADRCController
from lc.rl.algorithms import MDDPGConfig, MDDPGPolicy


@dataclass
class ControlLADRLAgent:
    """Chapter-3 RL-LADRC agent.

    The policy does not output the control signal directly. It outputs normalized
    LADRC parameters in the order `r / b0 / wc / k`, where `k = wo / wc`.
    The adaptive LADRC wrapper then reconstructs `wo = k * wc` and computes the
    final control signal.
    """

    obs_dim: int
    stack_size: int
    action_hold_steps: int = 1
    n_step: int = 1
    batch_size: int = 128
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    hidden_dim: int = 512
    dropout_p: float = 0.2
    tau: float = 0.05
    soft_update_interval: int = 20
    exploration_noise_schedule: str = "fixed"
    exploration_noise_start: float = 0.1
    exploration_noise_end: float = 0.1
    controller: AdaptiveLADRCController = field(default_factory=AdaptiveLADRCController)

    def __post_init__(self) -> None:
        self.policy = MDDPGPolicy(
            MDDPGConfig(
                state_dim=self.obs_dim,
                action_dim=4,
                stack_size=self.stack_size,
                action_hold_steps=self.action_hold_steps,
                batch_size=self.batch_size,
                actor_lr=self.actor_lr,
                critic_lr=self.critic_lr,
                hidden_dim=self.hidden_dim,
                dropout_p=self.dropout_p,
                tau=self.tau,
                soft_update_interval=self.soft_update_interval,
                expl_noise=self.exploration_noise_start,
                expl_noise_start=self.exploration_noise_start,
                expl_noise_end=self.exploration_noise_end,
                expl_noise_schedule=self.exploration_noise_schedule,
            )
        )

    def reset(self) -> None:
        self.controller.reset()
        self.policy.reset()

    def act(self, stacked_state: np.ndarray, explore: bool = False) -> np.ndarray:
        """Output normalized LADRC parameters in the order: r, b0, wc, k."""
        return self.policy.select_action(stacked_state, explore=explore)


ControlMDDPGAgent = ControlLADRLAgent
