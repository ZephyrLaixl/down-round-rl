"""
Multi-Bidder VC Auction Environment with Data-Calibrated Parameters

N-bidder Vickrey auction with stochastic true value, inter-round V shifts,
and realistic signal noise. Calibrated to match empirical DR statistics:
- DR rate (consecutive): 49.35%
- DR magnitude (mean): -62.24%, (median): -68.11%
- Ownership alpha: 0.17 (data median)

Key features:
1. Stochastic true value V per episode (log-normal, calibrated to data)
2. Inter-round V shift via log-normal distribution
3. N bidders (default 3, matching typical VC round investor count)
4. Adaptive signal noise (scales with V)
5. Optional DR penalty mechanism
6. CTDE-compatible global state output
"""

import numpy as np
import gymnasium as gym
from typing import Tuple, Dict, Optional, List


class MultiBidderVCAuctionEnv(gym.Env):
    """
    N-bidder multi-period VC auction environment (data-calibrated).

    Observation (per bidder): [signal, round, prev_price, is_winner, belief_mean, belief_std]
    Global state: [signals_N, round, prev_price, prev_winner, ownership_N]
    Action: bid in [0, budget]
    """

    def __init__(self,
                 true_value_mu: float = 100.0,
                 true_value_sigma_log: float = 1.19,
                 noise_std: float = 20.0,
                 n_bidders: int = 3,
                 n_rounds: int = 2,
                 budget: float = 250.0,
                 initial_ownership: float = 0.17,
                 discount_factor: float = 0.95,
                 v_shift_mu: float = 0.0,
                 v_shift_sigma: float = 1.0,
                 dr_penalty: float = 0.0,
                 seed: int = 42):
        """
        Args:
            true_value_mu: Median V (log-normal distribution)
            true_value_sigma_log: Log-std of V (calibrated sigma_log=1.19 from data)
            noise_std: Signal noise sigma (sigma/V ~ 20%)
            n_bidders: Number of bidders per round (default 3)
            n_rounds: Number of financing rounds (default 2)
            budget: Maximum bid constraint
            initial_ownership: Equity fraction per round (alpha=0.17, data median)
            discount_factor: Inter-period discount factor gamma
            v_shift_mu: Log-mean of inter-round V shift (0 = no drift)
            v_shift_sigma: Log-std of inter-round V shift (calibrated sigma=1.0)
            dr_penalty: DR penalty coefficient (reputation cost, signaling effect)
            seed: Random seed
        """
        super().__init__()

        self.true_value_mu = true_value_mu
        self.true_value_sigma_log = true_value_sigma_log
        self.noise_std = noise_std
        self.n_bidders = n_bidders
        self.n_rounds = n_rounds
        self.budget = budget
        self.initial_ownership = initial_ownership
        self.discount_factor = discount_factor
        self.v_shift_mu = v_shift_mu
        self.v_shift_sigma = v_shift_sigma
        self.dr_penalty = dr_penalty
        self.rng = np.random.RandomState(seed)

        self.action_space = gym.spaces.Box(
            low=0.0, high=budget, shape=(1,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([500.0, float(n_rounds), 500.0, 1.0, 500.0, 100.0]),
            dtype=np.float32
        )
        self.global_state_dim = n_bidders + 1 + 1 + 1 + n_bidders  # 2N + 3

        self.current_round = 0
        self.episode_true_value = 100.0
        self.signals = [0.0] * n_bidders
        self.prices = [0.0] * n_rounds
        self.winners = [-1] * n_rounds
        self.ownership = [0.0] * n_bidders
        self.beliefs = [{'mean': true_value_mu, 'std': noise_std} for _ in range(n_bidders)]

    def _sample_true_value(self) -> float:
        """Sample V from log-normal distribution (median=mu, sigma_log=1.19)."""
        mu_log = np.log(self.true_value_mu)
        z = self.rng.normal(0, 1)
        V = np.exp(mu_log + self.true_value_sigma_log * z)
        return np.clip(V, 10.0, 500.0)

    def _shift_true_value(self) -> float:
        """Inter-round V shift: V_new = V_old * exp(N(v_shift_mu, v_shift_sigma))."""
        log_shift = self.rng.normal(self.v_shift_mu, self.v_shift_sigma)
        new_V = self.episode_true_value * np.exp(log_shift)
        return np.clip(new_V, 10.0, 500.0)

    def _generate_signals(self):
        """Generate N private signals with adaptive noise."""
        adaptive_noise = max(1.0, self.noise_std * (self.episode_true_value / self.true_value_mu))
        self.signals = [
            np.clip(self.episode_true_value + self.rng.normal(0, adaptive_noise), 0.0, self.budget)
            for _ in range(self.n_bidders)
        ]

    def reset(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """Reset environment. Returns (observations_list, global_state)."""
        self.current_round = 0
        self.episode_true_value = self._sample_true_value()
        self._generate_signals()

        self.prices = [0.0] * self.n_rounds
        self.winners = [-1] * self.n_rounds
        self.ownership = [0.0] * self.n_bidders
        self.beliefs = [
            {'mean': self.true_value_mu, 'std': self.noise_std}
            for _ in range(self.n_bidders)
        ]

        observations = [self._get_observation(i) for i in range(self.n_bidders)]
        global_state = self._get_global_state()
        return observations, global_state

    def _get_observation(self, bidder_id: int) -> np.ndarray:
        """Get private observation for bidder_id."""
        prev_price = self.prices[self.current_round - 1] if self.current_round > 0 else 0.0
        is_winner = 1.0 if (self.current_round > 0 and self.winners[self.current_round - 1] == bidder_id) else 0.0
        belief = self.beliefs[bidder_id]
        return np.array([
            self.signals[bidder_id],
            float(self.current_round),
            prev_price,
            is_winner,
            belief['mean'],
            belief['std']
        ], dtype=np.float32)

    def _get_global_state(self) -> np.ndarray:
        """Get global state for Centralized Critic."""
        prev_price = self.prices[self.current_round - 1] if self.current_round > 0 else 0.0
        prev_winner = float(self.winners[self.current_round - 1]) if self.current_round > 0 else -1.0
        state = []
        state.extend(self.signals)
        state.append(float(self.current_round))
        state.append(prev_price)
        state.append(prev_winner)
        state.extend(self.ownership)
        return np.array(state, dtype=np.float32)

    def step(self, actions: List[float]) -> Tuple:
        """Execute one auction round. Returns (obs_list, global_state, rewards, done, info)."""
        bids = [np.clip(float(a), 0.0, self.budget) for a in actions]
        winner, price = self._second_price_auction(bids)

        self.prices[self.current_round] = price
        self.winners[self.current_round] = winner

        if winner >= 0:
            if self.current_round == 0:
                self.ownership[winner] = self.initial_ownership
            else:
                remaining = 1.0 - sum(self.ownership)
                self.ownership[winner] += remaining * self.initial_ownership

        rewards = self._calculate_rewards(bids, winner, price)

        if self.current_round == 0:
            self._update_beliefs(price)

        # Inter-round V shift
        self.current_round += 1
        if self.current_round < self.n_rounds:
            self.episode_true_value = self._shift_true_value()
            self._generate_signals()
            for i in range(self.n_bidders):
                self.beliefs[i]['mean'] = self.episode_true_value

        done = self.current_round >= self.n_rounds

        down_round = False
        down_round_pct = 0.0
        if self.n_rounds >= 2 and self.prices[0] > 0 and self.prices[1] > 0:
            down_round = self.prices[1] < self.prices[0]
            down_round_pct = (self.prices[1] - self.prices[0]) / self.prices[0] * 100

        observations = [self._get_observation(i) for i in range(self.n_bidders)]
        global_state = self._get_global_state()

        info = {
            'round': self.current_round - 1,
            'winner': winner,
            'price': price,
            'bids': bids.copy(),
            'down_round': down_round,
            'down_round_pct': down_round_pct,
            'prices': self.prices.copy(),
            'winners': self.winners.copy(),
            'ownership': self.ownership.copy(),
            'beliefs': [{'mean': b['mean'], 'std': b['std']} for b in self.beliefs],
            'true_value': self.episode_true_value
        }

        return observations, global_state, rewards, done, info

    def _second_price_auction(self, bids: list) -> Tuple[int, float]:
        """Vickrey (second-price) auction for N bidders."""
        sorted_bids = sorted([(b, i) for i, b in enumerate(bids)], reverse=True)
        if len(sorted_bids) >= 2:
            winner = sorted_bids[0][1]
            price = sorted_bids[1][0]
        elif len(sorted_bids) == 1:
            winner = sorted_bids[0][1]
            price = sorted_bids[0][0]
        else:
            winner = -1
            price = 0.0

        max_bid = sorted_bids[0][0]
        tied = [i for i, b in enumerate(bids) if b == max_bid]
        if len(tied) > 1:
            winner = int(self.rng.choice(tied))
            price = max_bid

        return winner, price

    def _calculate_rewards(self, bids: list, winner: int, price: float) -> List[float]:
        """Compute rewards with dilution and optional DR penalty."""
        rewards = [0.0] * self.n_bidders
        if winner < 0:
            return rewards

        if self.current_round == 0:
            current_payoff = self.initial_ownership * self.episode_true_value - price
            future_option = (1 - self.initial_ownership) * self.initial_ownership * self.episode_true_value * 0.5 / self.n_bidders
            rewards[winner] = current_payoff + self.discount_factor * future_option
        else:
            total_ownership_except_winner = sum(self.ownership[i] for i in range(self.n_bidders) if i != winner)
            remaining_for_winner = 1.0 - total_ownership_except_winner
            new_share = remaining_for_winner * self.initial_ownership
            rewards[winner] = new_share * self.episode_true_value - price

            if self.dr_penalty > 0 and self.prices[0] > 0 and price < self.prices[0]:
                dr_pct = (price - self.prices[0]) / self.prices[0]
                penalty = self.dr_penalty * abs(dr_pct)
                rewards[winner] -= penalty * 1.5
                for i in range(self.n_bidders):
                    if i != winner:
                        rewards[i] -= penalty * 0.5

        return rewards

    def _update_beliefs(self, price: float):
        """Bayesian belief update based on observed price."""
        observation_noise = self.noise_std * 1.5
        for bidder_id in range(self.n_bidders):
            prior_mean = self.beliefs[bidder_id]['mean']
            prior_std = self.beliefs[bidder_id]['std']
            prior_var = prior_std ** 2
            obs_var = observation_noise ** 2
            post_var = 1.0 / (1.0 / prior_var + 1.0 / obs_var)
            post_mean = post_var * (prior_mean / prior_var + price / obs_var)
            self.beliefs[bidder_id]['mean'] = post_mean
            self.beliefs[bidder_id]['std'] = np.sqrt(post_var)

    @property
    def global_state_space(self):
        return gym.spaces.Box(
            low=-np.ones(self.global_state_dim) * 500,
            high=np.ones(self.global_state_dim) * 500,
            shape=(self.global_state_dim,),
            dtype=np.float32
        )

    def render(self, mode='human'):
        """Print current environment state."""
        print(f"\n=== Round {self.current_round} ===")
        print(f"True Value: {self.episode_true_value:.2f}")
        for i in range(self.n_bidders):
            print(f"  Bidder {i}: signal={self.signals[i]:.2f}, ownership={self.ownership[i]:.2%}")
        if self.current_round >= 2 and self.prices[0] > 0:
            dr = "YES" if self.prices[1] < self.prices[0] else "NO"
            dr_pct = (self.prices[1] - self.prices[0]) / self.prices[0] * 100
            print(f"  Down Round: {dr} ({dr_pct:.2f}%)")