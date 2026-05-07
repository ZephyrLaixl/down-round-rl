"""
Rollout buffer with global state support for MAPPO CTDE architecture.

Stores (obs, global_state, action, log_prob, reward, done) tuples
and converts them to PyTorch tensors for PPO updates.
"""

import numpy as np
import torch


class RolloutBuffer:
    """Rollout buffer supporting global state for CTDE."""

    def __init__(self):
        self.obs = []
        self.global_states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []

    def add(self, obs, global_state, action, log_prob, reward, done):
        """Add one transition to the buffer."""
        self.obs.append(obs)
        self.global_states.append(global_state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)

    def get(self):
        """Return all transitions as PyTorch tensors."""
        obs = torch.FloatTensor(np.array(self.obs))
        global_states = torch.FloatTensor(np.array(self.global_states))
        actions = torch.FloatTensor(np.array(self.actions))
        log_probs = torch.FloatTensor(np.array(self.log_probs))
        rewards = torch.FloatTensor(np.array(self.rewards))
        dones = torch.FloatTensor(np.array(self.dones))

        return obs, global_states, actions, log_probs, rewards, dones

    def clear(self):
        """Clear all stored transitions."""
        self.obs.clear()
        self.global_states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.obs)