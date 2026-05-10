# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-agent reinforcement learning research project investigating whether down rounds in venture capital financing are statistical noise or rational strategic behavior. Uses MAPPO-trained agents in a data-calibrated multi-bidder Vickrey auction environment. Key finding: strategic devaluation emerges as optimal behavior, adding a 33% component beyond the statistical baseline.

## Quick Start Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train MAPPO agents (~10 min on CPU, produces models/)
python train.py

# Generate analysis figures (requires trained models)
python visualize.py

# One-command reproduction (train + visualize)
bash reproduce.sh

# Compile paper (requires LaTeX)
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Architecture

### MAPPO CTDE (Centralized Training Decentralized Execution)

The system trains N=3 agents in a 2-round VC auction using the CTDE paradigm:

- **Decentralized Actor** (`src/agents/mappo_agent.py:DecentralizedActor`): 2-layer MLP (64 hidden, tanh). Takes 6-dim private observation `[signal, round, prev_price, is_winner, belief_mean, belief_std]` and outputs action distribution N(mu, sigma). Outputs bid in [0, budget].
- **Centralized Critic** (`src/agents/mappo_agent.py:CentralizedCritic`): 3-layer MLP (128->128->64, tanh). Takes 9-dim global state `[signals_N, round, prev_price, prev_winner, ownership_N]` and outputs V(s).
- **Rollout Buffer** (`src/utils/buffer.py`): Stores (obs, global_state, action, log_prob, reward, done) transitions for PPO updates with GAE.

### Environment (`src/environment/vc_auction_env.py`)

`MultiBidderVCAuctionEnv` — a Gymnasium-compatible environment implementing:
- **N-bidder Vickrey (second-price) auction**: highest bidder wins, pays second-highest price
- **Stochastic true value V**: log-normal distribution per episode (median=100, sigma_log=1.19)
- **Inter-round V shift**: log-normal multiplicative shift between rounds (sigma=1.0)
- **Bayesian belief updates**: agents update beliefs based on observed auction prices
- **Dilution-aware reward**: winner's payoff accounts for ownership dilution across rounds
- **Optional DR penalty**: configurable reputation cost for down rounds (disabled by default)

### Training Pipeline (`train.py`)

1. Load config from `config/train_config.yaml`
2. Create environment and N MAPPO agents (one per bidder)
3. Run 10,000 episodes with PPO updates every 256 steps
4. Track down round rate and magnitude metrics
5. Save model checkpoints every 1,000 episodes to `models/`
6. Export training history to `logs/training_history.json`

### Visualization (`visualize.py`)

- **Figure 1**: Equilibrium policy functions (bid vs signal for Round 0, bid vs P0 for Round 1)
- **Figure 2**: DR emergence heatmap (DR rate vs noise intensity and P0 deviation)
- Evaluates RL-trained agents vs random baseline (1,000 episodes each)

## Key Files

| File | Purpose |
|------|---------|
| `train.py` | Main training script |
| `visualize.py` | Figure generation and evaluation |
| `config/train_config.yaml` | All calibrated environment + training hyperparameters |
| `src/environment/vc_auction_env.py` | VC auction environment |
| `src/agents/mappo_agent.py` | MAPPO agent with actor/critic networks |
| `src/utils/buffer.py` | Rollout buffer for PPO |

## Calibration Parameters (from config/train_config.yaml)

| Parameter | Value | Source |
|-----------|-------|--------|
| N (bidders) | 3 | Typical VC round investor count |
| noise_std | 20.0 | sigma/V ~ 20% realistic uncertainty |
| true_value_sigma_log | 1.19 | Openbook data calibration |
| v_shift_sigma | 1.0 | Inter-round V shift |
| initial_ownership | 0.17 | Data median (funding/valuation) |
| budget | 250.0 | Bid constraint |

## Important Notes

- Raw VC data (Crunchbase/Openbook) is NOT included — environment is self-contained with calibrated parameters
- Training runs on CPU (no GPU required) — takes ~10 minutes for 10,000 episodes
- All randomness is seeded for reproducibility (seed=42)
- Model checkpoints are PyTorch state dicts saved to `models/`
- No external API keys needed
