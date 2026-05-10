"""
Visualization script for multi-bidder VC auction (V3, data-calibrated).

Figure 1: Equilibrium Policy Functions (N=3 agents, Round 0 and Round 1 strategies)
Figure 2: DR Emergence Heatmap (DR rate vs sigma/V and P0 deviation)

Usage: python visualize.py  (requires trained models from train.py)
Output: figures/fig1_equilibrium_policy.png, figures/fig2_emergence_heatmap.png
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
from pathlib import Path

from src.environment.vc_auction_env import MultiBidderVCAuctionEnv
from src.agents.mappo_agent import MAPPOAgent

# Load calibrated parameters from config
with open("config/train_config.yaml", 'r', encoding='utf-8') as f:
    _config = yaml.safe_load(f)

CALIBRATED_PARAMS = {
    'true_value_mu': _config['environment'].get('true_value_mu', 100.0),
    'true_value_sigma_log': _config['environment'].get('true_value_sigma_log', 1.19),
    'noise_std': _config['environment'].get('noise_std', 20.0),
    'n_bidders': _config['environment'].get('n_bidders', 3),
    'budget': _config['environment'].get('budget', 250.0),
    'initial_ownership': _config['training'].get('initial_ownership', 0.17),
    'discount_factor': _config['training'].get('discount_factor', 0.95),
    'v_shift_sigma': _config['environment'].get('v_shift_sigma', 1.0),
}

N_BIDDERS = CALIBRATED_PARAMS['n_bidders']

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150


def plot_equilibrium_policy_function(agents, env,
                                     save_path='figures/fig1_equilibrium_policy.png'):
    """Figure 1: Equilibrium policy functions for N bidders."""
    fig = plt.figure(figsize=(14, 5), constrained_layout=True)
    gs = GridSpec(1, 2, figure=fig, wspace=0.3)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Left: Round 0 bid vs private signal
    ax1 = fig.add_subplot(gs[0, 0])
    signals = np.linspace(30, 200, 100)

    for i in range(N_BIDDERS):
        bids_r0 = []
        for signal in signals:
            obs = np.array([signal, 0.0, 0.0, 0.0, CALIBRATED_PARAMS['true_value_mu'], CALIBRATED_PARAMS['noise_std']])
            action, _ = agents[i].select_action(obs, deterministic=True)
            bids_r0.append(action[0])
        ax1.plot(signals, bids_r0, color=colors[i], linewidth=2.5,
                label=f'Agent {i} (Round 0)', alpha=0.8)

    ax1.plot(signals, signals * CALIBRATED_PARAMS['initial_ownership'], 'k:',
            linewidth=1.5, label=f'b = {CALIBRATED_PARAMS["initial_ownership"]}*theta (Stake)', alpha=0.5)

    ax1.set_xlabel('Private Signal theta_i', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Round 0 Bid b_i0', fontsize=12, fontweight='bold')
    ax1.set_title('Round 0 Bidding Strategy vs Private Signal', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right: Round 1 bid vs Round 0 price
    ax2 = fig.add_subplot(gs[0, 1])
    p0_values = np.linspace(5, 40, 100)
    fixed_signal = 100.0

    for i in range(N_BIDDERS):
        bids_r1 = []
        for p0 in p0_values:
            obs = np.array([fixed_signal, 1.0, p0, float(i == 0), CALIBRATED_PARAMS['true_value_mu'], CALIBRATED_PARAMS['noise_std'] * 0.75])
            action, _ = agents[i].select_action(obs, deterministic=True)
            bids_r1.append(action[0])
        ax2.plot(p0_values, bids_r1, color=colors[i], linewidth=2.5,
                label=f'Agent {i} (Round 1)', alpha=0.8)

    ax2.plot(p0_values, p0_values, 'k:', linewidth=1.5,
            label='b1 = P0 (Parity)', alpha=0.5)
    ax2.fill_between(p0_values, 0, p0_values, color='red', alpha=0.05,
                    label='Down Round Region (b1 < P0)')

    ax2.set_xlabel('Round 0 Price P0', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Round 1 Bid b_i1', fontsize=12, fontweight='bold')
    ax2.set_title('Round 1 Bid vs Round 0 Price', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'Figure 1: Equilibrium Policy Functions (N={N_BIDDERS} Bidders, Data-Calibrated)',
                fontsize=16, fontweight='bold', y=1.02)

    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure 1 saved: {save_path}")
    plt.close()


def plot_down_round_emergence_heatmap(save_path='figures/fig2_emergence_heatmap.png'):
    """Figure 2: DR emergence heatmap (V3, data-calibrated)."""
    fig, ax = plt.subplots(figsize=(11, 5.5))

    p0_deviations = np.linspace(-0.3, 0.3, 50)
    noise_levels = np.linspace(0.05, 0.50, 50)

    true_value = CALIBRATED_PARAMS['true_value_mu']
    down_round_rates = np.zeros((len(noise_levels), len(p0_deviations)))

    print("\nCalculating DR Emergence Heatmap (V3)...")
    for i, noise_ratio in enumerate(noise_levels):
        for j, p0_dev in enumerate(p0_deviations):
            noise_std = noise_ratio * true_value
            base_p0 = true_value * CALIBRATED_PARAMS['initial_ownership']
            p0_target = base_p0 * (1 + p0_dev)

            env = MultiBidderVCAuctionEnv(
                true_value_mu=true_value, true_value_sigma_log=CALIBRATED_PARAMS['true_value_sigma_log'],
                noise_std=noise_std, n_bidders=N_BIDDERS,
                initial_ownership=CALIBRATED_PARAMS['initial_ownership'],
                v_shift_sigma=CALIBRATED_PARAMS['v_shift_sigma'],
                seed=42 + i * 20 + j
            )

            n_samples = 200
            down_count = 0

            for _ in range(n_samples):
                obs_list, _ = env.reset()
                bids_r0 = [p0_target] * N_BIDDERS
                obs_list, _, _, done, info = env.step(bids_r0)

                if not done:
                    bids_r1 = [obs[0] * CALIBRATED_PARAMS['initial_ownership'] for obs in obs_list]
                    _, _, _, done, info = env.step(bids_r1)

                if done and info['down_round']:
                    down_count += 1

            down_round_rates[i, j] = down_count / n_samples

        if (i + 1) % 10 == 0:
            print(f"  Progress: {(i+1)/len(noise_levels)*100:.0f}%")

    # Smooth heatmap to reduce Monte Carlo noise artifacts
    down_round_rates = gaussian_filter(down_round_rates, sigma=1.0)

    # Set color scale to actual data range for better contrast
    vmin_val = max(0.0, np.percentile(down_round_rates, 5))
    vmax_val = min(1.0, np.percentile(down_round_rates, 95))

    im = ax.imshow(down_round_rates, aspect='auto', origin='lower',
                   extent=[p0_deviations[0]*100, p0_deviations[-1]*100,
                          noise_levels[0]*100, noise_levels[-1]*100],
                   cmap='RdYlGn_r', vmin=vmin_val, vmax=vmax_val, interpolation='nearest')

    contours = ax.contour(p0_deviations*100, noise_levels*100, down_round_rates,
                          levels=[0.3, 0.5, 0.7], colors='black', linewidths=1.5, alpha=0.6)
    ax.clabel(contours, inline=True, fontsize=10, fmt='%.1f')

    ax.text(-25, 40, 'High DR Region\n(Systematic)', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    ax.text(15, 10, 'Low DR Region\n(Random)', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))

    # Mark calibrated point
    ax.scatter(0, CALIBRATED_PARAMS['noise_std']/true_value*100, s=100, color='blue',
               marker='*', zorder=5, label=f'Calibrated Point (sigma/V={CALIBRATED_PARAMS["noise_std"]/true_value*100:.0f}%)')

    ax.set_xlabel('P0 Deviation (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Noise Intensity sigma/V (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Figure 2: DR Emergence Heatmap (N={N_BIDDERS}, v_shift sigma={CALIBRATED_PARAMS["v_shift_sigma"]})',
                 fontsize=14, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Down Round Rate', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure 2 saved: {save_path}")
    plt.close()

    print(f"\nPhase Transition Analysis:")
    print(f"  Max DR Rate: {np.max(down_round_rates):.2%}")
    print(f"  Min DR Rate: {np.min(down_round_rates):.2%}")


def generate_narrative():
    """Generate full narrative: evaluate agents, produce figures."""

    print("=" * 60)
    print("Generating Down Round Emergence Analysis (V3, Data-Calibrated)")
    print(f"  Params: sigma={CALIBRATED_PARAMS['noise_std']}, "
          f"v_shift={CALIBRATED_PARAMS['v_shift_sigma']}, "
          f"N={N_BIDDERS}, alpha={CALIBRATED_PARAMS['initial_ownership']}")
    print("=" * 60)

    model_path_dir = Path("models")
    agents = None

    final_models = list(model_path_dir.glob("agent_*_final.pt"))
    if len(final_models) >= N_BIDDERS:
        print("\nFound trained models, loading...")

        env = MultiBidderVCAuctionEnv(
            true_value_mu=CALIBRATED_PARAMS['true_value_mu'],
            true_value_sigma_log=CALIBRATED_PARAMS['true_value_sigma_log'],
            noise_std=CALIBRATED_PARAMS['noise_std'],
            n_bidders=N_BIDDERS,
            initial_ownership=CALIBRATED_PARAMS['initial_ownership'],
            v_shift_sigma=CALIBRATED_PARAMS['v_shift_sigma'],
        )
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        global_state_dim = env.global_state_dim

        agents = []
        for i in range(N_BIDDERS):
            agent = MAPPOAgent(obs_dim, action_dim, global_state_dim)
            agent.load(str(model_path_dir / f"agent_{i}_final.pt"))
            agents.append(agent)

        # RL-trained evaluation
        print("\nRunning 1000-episode RL evaluation...")
        dr_count = 0
        dr_pcts = []
        for _ in range(1000):
            obs_list, gs = env.reset()
            bids_r0 = [agents[i].select_action(obs_list[i], deterministic=True)[0][0] for i in range(N_BIDDERS)]
            obs_list, gs, rewards, done, info = env.step(bids_r0)
            if not done:
                bids_r1 = [agents[i].select_action(obs_list[i], deterministic=True)[0][0] for i in range(N_BIDDERS)]
                _, _, rewards, done, info = env.step(bids_r1)
            if info['down_round']:
                dr_count += 1
            if info['down_round'] and info['down_round_pct'] != 0.0:
                dr_pcts.append(info['down_round_pct'])

        rl_dr_rate = dr_count / 1000
        print(f"  RL-trained DR Rate: {rl_dr_rate:.2%}")
        if dr_pcts:
            print(f"  RL-trained DR Magnitude (mean): {np.mean(dr_pcts):.1f}%")
            print(f"  RL-trained DR Magnitude (median): {np.median(dr_pcts):.1f}%")

        # Random baseline
        print("\nRunning 1000-episode random baseline...")
        dr_count_random = 0
        dr_pcts_random = []
        for _ in range(1000):
            obs_list, gs = env.reset()
            bids_r0 = [obs[0] * CALIBRATED_PARAMS['initial_ownership'] + np.random.normal(0, 3) for obs in obs_list]
            obs_list, gs, rewards, done, info = env.step(bids_r0)
            if not done:
                bids_r1 = [obs[0] * CALIBRATED_PARAMS['initial_ownership'] + np.random.normal(0, 3) for obs in obs_list]
                _, _, rewards, done, info = env.step(bids_r1)
            if info['down_round']:
                dr_count_random += 1
            if info['down_round'] and info['down_round_pct'] != 0.0:
                dr_pcts_random.append(info['down_round_pct'])

        random_dr_rate = dr_count_random / 1000
        print(f"  Random baseline DR Rate: {random_dr_rate:.2%}")
        if dr_pcts_random:
            print(f"  Random baseline DR Magnitude (mean): {np.mean(dr_pcts_random):.1f}%")
            print(f"  Random baseline DR Magnitude (median): {np.median(dr_pcts_random):.1f}%")
        print(f"  Strategic DR component: {rl_dr_rate - random_dr_rate:.2%}")

        print("\nGenerating Figure 1: Equilibrium Policy Functions...")
        plot_equilibrium_policy_function(agents, env)
    else:
        print("\nTrained models not found, skipping Figure 1")
        print("  Please run train.py first")

    print("\nGenerating Figure 2: Down Round Emergence Heatmap...")
    plot_down_round_emergence_heatmap()

    print("\n" + "=" * 60)
    print("Analysis Generation Completed!")
    print("=" * 60)


if __name__ == "__main__":
    generate_narrative()