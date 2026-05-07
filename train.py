"""
MAPPO Training Script for Multi-Bidder VC Auction (V3, Data-Calibrated)

Trains N=3 MAPPO agents with CTDE architecture in a data-calibrated
auction environment with stochastic V, inter-round V shifts, and
adaptive signal noise. Tracks DR rate and magnitude.

Usage: python train.py
Output: models/ (agent checkpoints), logs/training_history.json
"""

import yaml
import numpy as np
import torch
from pathlib import Path
from collections import deque
import json

from src.environment.vc_auction_env import MultiBidderVCAuctionEnv
from src.agents.mappo_agent import MAPPOAgent
from src.utils.buffer import RolloutBuffer


def train(config_path: str = "config/train_config.yaml"):
    """Train N MAPPO agents in the multi-bidder VC auction environment."""

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("Multi-Bidder VC Auction MAPPO Training (Data-Calibrated)")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Environment: {config['environment']}")
    print(f"  PPO: {config['ppo']}")
    print(f"  Training: {config['training']}")
    print()

    # Create environment
    env_config = config['environment']
    env = MultiBidderVCAuctionEnv(**env_config)
    n_bidders = env_config.get('n_bidders', 3)

    # Create N MAPPO agents
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    global_state_dim = env.global_state_dim

    mappo_config = config['ppo'].copy()
    if 'lr' in mappo_config:
        lr = mappo_config.pop('lr')
        mappo_config['lr_actor'] = lr
        mappo_config['lr_critic'] = lr * 3
    if 'hidden_dim' in mappo_config:
        hidden_dim = mappo_config.pop('hidden_dim')
        mappo_config['hidden_dim_actor'] = hidden_dim
        mappo_config['hidden_dim_critic'] = hidden_dim * 2

    agents = []
    buffers = []
    for i in range(n_bidders):
        agents.append(MAPPOAgent(
            obs_dim=obs_dim, action_dim=action_dim,
            global_state_dim=global_state_dim, **mappo_config
        ))
        buffers.append(RolloutBuffer())

    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}, Global state dim: {global_state_dim}")
    print(f"N bidders: {n_bidders}")
    for i, agent in enumerate(agents):
        n_actor = sum(p.numel() for p in agent.actor.parameters())
        n_critic = sum(p.numel() for p in agent.critic.parameters())
        print(f"  Agent {i}: Actor={n_actor} params, Critic={n_critic} params")
    print()

    # Training parameters
    n_episodes = config['training']['n_episodes']
    update_interval = config['training']['update_interval']
    print_interval = config['logging']['print_interval']

    episode_rewards = [deque(maxlen=100) for _ in range(n_bidders)]
    down_round_history = deque(maxlen=100)
    down_round_pct_history = deque(maxlen=100)
    win_histories = [deque(maxlen=100) for _ in range(n_bidders)]

    last_losses = [{'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0} for _ in range(n_bidders)]

    history = {
        'episode': [],
        'avg_reward': [],
        'down_round_rate': [],
        'down_round_pct_mean': [],
        'down_round_pct_median': [],
        'policy_loss_avg': [],
        'value_loss_avg': [],
    }

    print("Starting training...")
    print("=" * 60)

    for episode in range(n_episodes):
        obs_list, global_state = env.reset()
        episode_reward = [0.0] * n_bidders

        for step in range(config['training']['steps_per_episode']):
            actions = []
            log_probs = []
            for i in range(n_bidders):
                action, log_prob = agents[i].select_action(obs_list[i])
                actions.append(action)
                log_probs.append(log_prob)

            values = [agents[i].get_value(global_state) for i in range(n_bidders)]

            bid_values = [a[0] for a in actions]
            next_obs_list, next_global_state, rewards, done, info = env.step(bid_values)

            for i in range(n_bidders):
                buffers[i].add(obs_list[i], global_state, actions[i], log_probs[i], rewards[i], done)
                episode_reward[i] += rewards[i]

            obs_list = next_obs_list
            global_state = next_global_state

            if done:
                down_round_history.append(1.0 if info['down_round'] else 0.0)
                if info['down_round']:
                    down_round_pct_history.append(info['down_round_pct'])
                for i in range(n_bidders):
                    win_histories[i].append(1.0 if info['winners'][0] == i else 0.0)
                break

        for i in range(n_bidders):
            episode_rewards[i].append(episode_reward[i])

        # Update policy when all buffers reach threshold
        all_buffers_ready = all(len(b) >= update_interval for b in buffers)
        if all_buffers_ready:
            losses_list = []
            for i in range(n_bidders):
                next_value = agents[i].get_value(global_state)

                obs_batch, gs_batch, actions_batch, old_log_probs, rewards_batch, dones_batch = buffers[i].get()

                obs_batch = obs_batch.to(agents[i].device)
                gs_batch = gs_batch.to(agents[i].device)
                actions_batch = actions_batch.to(agents[i].device)
                old_log_probs = old_log_probs.to(agents[i].device)

                with torch.no_grad():
                    values = agents[i].critic(gs_batch).squeeze().cpu().numpy()

                advantages, returns = agents[i].compute_gae(
                    rewards_batch.tolist(), values.tolist(), dones_batch.tolist(), next_value
                )

                advantages = torch.FloatTensor(advantages).to(agents[i].device)
                returns = torch.FloatTensor(returns).to(agents[i].device)

                losses = agents[i].update(
                    obs_batch, gs_batch, actions_batch, old_log_probs,
                    advantages, returns,
                    n_epochs=config['training']['n_epochs'],
                    batch_size=config['training']['batch_size']
                )
                losses_list.append(losses)
                buffers[i].clear()

            last_losses = losses_list
        else:
            losses_list = last_losses

        # Logging
        if (episode + 1) % print_interval == 0:
            avg_rewards = [np.mean(episode_rewards[i]) if len(episode_rewards[i]) > 0 else 0.0
                         for i in range(n_bidders)]
            avg_reward = np.mean(avg_rewards)
            dr_rate = np.mean(down_round_history) if len(down_round_history) > 0 else 0.0
            dr_pct_mean = np.mean(down_round_pct_history) if len(down_round_pct_history) > 0 else 0.0
            dr_pct_median = np.median(down_round_pct_history) if len(down_round_pct_history) > 0 else 0.0
            win_rates = [np.mean(win_histories[i]) if len(win_histories[i]) > 0 else 0.0
                        for i in range(n_bidders)]

            avg_policy_loss = np.mean([l['policy_loss'] for l in losses_list])
            avg_value_loss = np.mean([l['value_loss'] for l in losses_list])

            print(f"Episode {episode + 1}/{n_episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Win Rates: {[f'{w:.2%}' for w in win_rates]}")
            print(f"  Down Round Rate: {dr_rate:.2%} (target: ~49%)")
            print(f"  DR Magnitude: mean={dr_pct_mean:.2f}%, median={dr_pct_median:.2f}%")
            print(f"  Policy Loss avg: {avg_policy_loss:.4f}, Value Loss avg: {avg_value_loss:.4f}")
            print()

            history['episode'].append(episode + 1)
            history['avg_reward'].append(avg_reward)
            history['down_round_rate'].append(dr_rate)
            history['down_round_pct_mean'].append(dr_pct_mean)
            history['down_round_pct_median'].append(dr_pct_median)
            history['policy_loss_avg'].append(avg_policy_loss)
            history['value_loss_avg'].append(avg_value_loss)

        # Save model checkpoints
        if (episode + 1) % config['logging']['save_interval'] == 0:
            Path("./models").mkdir(exist_ok=True)
            for i in range(n_bidders):
                agents[i].save(f"./models/agent_{i}_ep{episode+1}.pt")

    # Save final models
    Path("./models").mkdir(exist_ok=True)
    for i in range(n_bidders):
        agents[i].save(f"./models/agent_{i}_final.pt")

    # Save training history
    Path("./logs").mkdir(exist_ok=True)
    with open("./logs/training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print("=" * 60)
    print("Training completed!")
    final_dr_rate = np.mean(down_round_history) if len(down_round_history) > 0 else 0.0
    final_dr_pct_mean = np.mean(down_round_pct_history) if len(down_round_pct_history) > 0 else 0.0
    final_dr_pct_median = np.median(down_round_pct_history) if len(down_round_pct_history) > 0 else 0.0
    print(f"Final DR Rate (last 100): {final_dr_rate:.2%}")
    print(f"Final DR Magnitude: mean={final_dr_pct_mean:.2f}%, median={final_dr_pct_median:.2f}%")
    print(f"Models saved to ./models/")
    print(f"Logs saved to ./logs/")


if __name__ == "__main__":
    train()