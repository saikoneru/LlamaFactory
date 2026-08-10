#!/bin/bash
#SBATCH --job-name=benx_lora_sft
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plggdvps01-gpu-gh200
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=119G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Prevent login node variables (like x86 paths) from leaking into ARM compute node
#SBATCH --export=NONE

# Allow aliases/functions (like 'module') to work in non-interactive shell
shopt -s expand_aliases

# Safely source system environment profile without letting hidden micro-errors trip set -e
set +e
source /etc/profile
set -e

# Fail immediately if training or loading modules breaks
set -euo pipefail

echo "===== NODE INFO ====="
echo "Running on node: $(hostname)"
echo "Architecture:   $(uname -m)"

echo "===== LOADING ENVIRONMENT ====="
module load ML-bundle/24.06a
source /net/storage/pr3/plgrid/plggdvps/skoneru/envs/llama/bin/activate

echo "===== LAUNCHING TRAINING ====="
bash /net/home/plgrid/plgskoneru/LLaMA-Factory/examples/dvps/run_benx.sh