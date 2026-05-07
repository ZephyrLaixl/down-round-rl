"""
MAPPO Agent with Centralized Training Decentralized Execution (CTDE)

Architecture:
- DecentralizedActor: Uses only private observation for action selection
- CentralizedCritic: Uses global state for value estimation
- Information asymmetry: Actor sees local info, Critic sees global info

Reference: Yu et al. (2021) "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, List


class DecentralizedActor(nn.Module):
    """
    Decentralized Actor network — uses only private observation.

    Input: [signal, round, prev_price, is_winner, belief_mean, belief_std]
    Output: Action distribution N(mu, sigma)
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Initialize policy head bias near plausible bid (~30.0)
        nn.init.constant_(self.mean_layer.bias, 30.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature(obs)
        mean = self.mean_layer(features)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std

    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple:
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            mean, std = self.forward(obs_tensor)

            if deterministic:
                action = mean
            else:
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()

            dist = torch.distributions.Normal(mean, std)
            log_prob = dist.log_prob(action).sum(dim=-1)

        return action.numpy().flatten(), log_prob.item()

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple:
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()
        return log_prob, entropy


class CentralizedCritic(nn.Module):
    """
    Centralized Critic network — uses global state for value estimation.

    Input: [signal_A, signal_B, round, prev_price, prev_winner, ownership_A, ownership_B]
    Output: State value V(s)
    """

    def __init__(self, global_state_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.network(global_state)


class MAPPOAgent:
    """
    MAPPO (Multi-Agent PPO) Agent with CTDE architecture.

    Core features:
    1. Centralized Training: Critic uses global state
    2. Decentralized Execution: Actor uses private observation
    3. Information asymmetry: leverage global info during training, local info during execution
    """

    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 global_state_dim: int,
                 lr_actor: float = 3e-4,
                 lr_critic: float = 1e-3,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 hidden_dim_actor: int = 64,
                 hidden_dim_critic: int = 128,
                 device: str = 'cpu'):
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.actor = DecentralizedActor(obs_dim, action_dim, hidden_dim_actor).to(self.device)
        self.critic = CentralizedCritic(global_state_dim, hidden_dim_critic).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=1000, gamma=0.9)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=1000, gamma=0.9)

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple:
        """Select action using private observation only (decentralized execution)."""
        return self.actor.get_action(obs, deterministic)

    def get_value(self, global_state: np.ndarray) -> float:
        """Estimate state value using global state (centralized evaluation)."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)
            value = self.critic(state_tensor)
        return value.item()

    def compute_gae(self,
                    rewards: List[float],
                    values: List[float],
                    dones: List[bool],
                    next_value: float) -> Tuple[List[float], List[float]]:
        """Generalized Advantage Estimation."""
        advantages = []
        gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        returns = [a + v for a, v in zip(advantages, values)]
        return advantages, returns

    def update(self,
               obs: torch.Tensor,
               global_states: torch.Tensor,
               actions: torch.Tensor,
               old_log_probs: torch.Tensor,
               advantages: torch.Tensor,
               returns: torch.Tensor,
               n_epochs: int = 4,
               batch_size: int = 32) -> Dict[str, float]:
        """
        PPO update with CTDE architecture.

        Actor uses private observation (obs).
        Critic uses global state (global_states).
        """
        dataset_size = obs.size(0)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(n_epochs):
            indices = torch.randperm(dataset_size, device=self.device)

            for start in range(0, dataset_size, batch_size):
                end = min(start + batch_size, dataset_size)
                mb_indices = indices[start:end]

                mb_obs = obs[mb_indices]
                mb_global_states = global_states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                new_log_probs, entropy = self.actor.evaluate(mb_obs, mb_actions)
                values = self.critic(mb_global_states).squeeze()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * ((values - mb_returns) ** 2).mean()

                actor_loss = policy_loss - self.entropy_coef * entropy
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        self.actor_scheduler.step()
        self.critic_scheduler.step()

        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'lr_actor': self.actor_optimizer.param_groups[0]['lr'],
            'lr_critic': self.critic_optimizer.param_groups[0]['lr']
        }

    def save(self, path: str):
        """Save agent checkpoint."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load agent from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])