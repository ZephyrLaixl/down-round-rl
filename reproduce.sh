#!/bin/bash
# One-command reproduction script for Down Round RL project
# Runs training (10,000 episodes) and then generates figures

echo "=== Down Round RL Reproduction Script ==="
echo ""

# Check Python
if ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please install Python 3.8+."
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Train MAPPO agents
echo ""
echo "=== Step 1: Training MAPPO agents (10,000 episodes) ==="
echo "This takes ~10 minutes on CPU."
python train.py

if [ ! -f "models/agent_a_final.pt" ]; then
    echo "Error: Training failed. No model checkpoints found."
    exit 1
fi

echo ""
echo "=== Step 2: Generating figures ==="
python visualize.py

echo ""
echo "=== Reproduction Complete ==="
echo "Models: models/agent_a_final.pt, agent_b_final.pt"
echo "Figures: figures/fig1_equilibrium_policy.png, fig2_emergence_heatmap.png"
echo "Logs: logs/training_history.json"
echo ""
echo "To compile the paper: cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex"