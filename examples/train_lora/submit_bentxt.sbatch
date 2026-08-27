#!/bin/bash
#SBATCH --job-name=bentxt
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plggdvps01-gpu-gh200
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=119G
#SBATCH --time=18:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Allow environment variables passed via sbatch --export
# (Note: Remove or adjust --export=NONE in header if passing variables)

shopt -s expand_aliases
set +e
source /etc/profile
set -e
set -euo pipefail

module load ML-bundle/24.06a
source /net/storage/pr3/plgrid/plggdvps/skoneru/envs/terramind_llama/bin/activate
mkdir -p logs

export CUDA_VISIBLE_DEVICES=0

# CONFIG_PATH comes from the command line sbatch --export call
llamafactory-cli train "$CONFIG_PATH"