# Down Round as Emergent Strategy: Multi-Agent RL Evidence from VC Auctions

> [!NOTE]
> **DOTE 6635 (Spring 2026) — AI for Business Research | CUHK Business School**
> Final Project · Team 11 — Xinglin Lai · [Working Paper](paper/main.pdf) · [Slides](slides/presentation.pdf)

Are down rounds statistical noise or rational strategy? We use MAPPO-trained agents in a data-calibrated multi-bidder Vickrey auction to show that strategic devaluation **emerges** as optimal behavior, adding a 33% component beyond the statistical baseline. Future work will integrate structured and unstructured VC data for text-numerical alignment and DPO fine-tuning.

## Key Findings

| Statistic | Random Baseline | RL-Trained | Strategic Component |
|-----------|----------------|------------|-------------------|
| DR rate | 49% | **82%** | +33% |
| DR magnitude (mean) | -40% | **-29%** | milder |
| DR magnitude (median) | -38% | **-24%** | milder |

Down rounds emerge as rational optimal strategies in data-calibrated MAPPO agents with N=3 bidders, adding a 33% strategic component beyond the 49% statistical baseline. RL-trained agents produce frequent but mild DRs (-29% mean), while real data shows severe DRs (-62% mean) — a "magnitude paradox" likely explained by behavioral biases.

## Quick Start

```bash
pip install -r requirements.txt
python train.py      # ~10 min on CPU, produces models/
python visualize.py  # produces figures/
```

Or one-command reproduction:

```bash
bash reproduce.sh
```

## Repository Structure

```
down-round-rl/
├── paper/           # LaTeX paper source + figures + references
├── slides/          # Beamer presentation source + PDF
├── src/
│   ├── environment/ # Multi-bidder VC auction environment (V3, data-calibrated)
│   ├── agents/      # MAPPO CTDE agent
│   └── utils/       # Rollout buffer
├── config/          # Training configuration (calibrated parameters)
├── train.py         # Training script (N=3 bidders)
├── visualize.py     # Figure generation
├── reproduce.sh     # One-command reproduction
├── figures/         # Output figures (generated)
├── models/          # Output models (generated)
└── logs/            # Output logs (generated)
```

## Calibration Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| N (bidders) | 3 | Typical VC round investor count |
| sigma/V (noise_std) | 20% | Realistic signal uncertainty |
| sigma_log (V distribution) | 1.19 | Openbook data calibration |
| v_shift_sigma | 1.0 | Inter-round V shift (log-std) |
| V (true value) | Log-normal, median=100 | Data-calibrated |
| alpha (ownership) | 0.17 | Data median (funding/valuation) |
| budget | 250.0 | Bid constraint |

## Architecture

**MAPPO CTDE (Centralized Training Decentralized Execution)**:
- **Decentralized Actor**: 6-dim private observation -> bid (2-layer MLP, 64 hidden)
- **Centralized Critic**: 9-dim global state -> V(s) (3-layer MLP, 128 hidden)

## Future Work (Phase B)

The current Phase A demonstrates strategic emergence in a simulated auction. Phase B plans to:
- Integrate structured and unstructured VC data (Crunchbase, news, filings)
- Perform text-numerical matching and alignment
- Use DPO-trained LLMs as "decision interpreters" to quantify behavioral bias contributions (anchoring, loss aversion, overconfidence, herding)
- Extend from 2-round to realistic 3-7 round sequences

## Note on Data

This repository contains only the simulation environment and RL training code. Raw venture capital data (Crunchbase/Openbook) is not included due to data sensitivity. The environment is self-contained — all statistics emerge from the calibrated simulation parameters.

## Citation

This is an unpublished working paper. Any citation or use requires the author's explicit consent. For inquiries, contact xinglinlai@ln.hk or xinglinzephyrlai@gmail.com.

```bibtex
@article{lai2026downround,
  title={Down Round as Emergent Strategy: Multi-Agent RL Evidence from Venture Capital Auctions},
  author={Lai, Xinglin},
  journal={Working Paper},
  year={2026}
}
```

## License

All rights reserved. See [LICENSE](LICENSE) for details.
