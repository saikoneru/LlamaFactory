#!/bin/bash
#SBATCH --job-name=four_enc_train
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plggdvps01-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=119G
#SBATCH --time=24:00:00
#SBATCH --output=/net/storage/pr3/plgrid/plggdvps/rcorrente/dvps-fm/outputs/four_encoder_mixed/logs/%x-%j.out
#SBATCH --error=/net/storage/pr3/plgrid/plggdvps/rcorrente/dvps-fm/outputs/four_encoder_mixed/logs/%x-%j.err
#SBATCH --export=NONE
#
# Usage:
#   sbatch LLaMA-Factory/examples/dvps/run_four_encoder_mixed.sbatch

shopt -s expand_aliases
set +e
source /etc/profile
set -euo pipefail

ROOT=/net/storage/pr3/plgrid/plggdvps/rcorrente/dvps-fm
YAML="${1:-${ROOT}/LLaMA-Factory/examples/dvps/four_encoder_mixed.yaml}"
OUT="${ROOT}/outputs/four_encoder_mixed"

echo "===== NODE INFO ====="
echo "node: $(hostname)   arch: $(uname -m)"
echo "job:  ${SLURM_JOB_ID:-none}"
echo "YAML: ${YAML}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

source "${ROOT}/omni_composite/scripts/env_omniavsr.sh"
export PYTHONPATH="${ROOT}/LLaMA-Factory/src:${PYTHONPATH:-}"
export DISABLE_VERSION_CHECK=1
export WANDB_PROJECT=dvps-fm
export WANDB_NAME=four-encoder-mixed-projectors
if [[ -f "${HOME}/.wandb_api_key" ]]; then
  export WANDB_API_KEY="$(cat "${HOME}/.wandb_api_key")"
fi

rm -rf "${HF_HOME}/modules/transformers_modules/qwen_omni_four" || true
mkdir -p "${OUT}/logs" "${HF_HOME}"

cd "${ROOT}/LLaMA-Factory"

echo "===== TRAIN ====="
python -m llamafactory.cli train "${YAML}"
