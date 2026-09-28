#!/bin/bash
#SBATCH --job-name=phase4_train
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plggdvps01-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --mem-per-cpu=4GB
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH --output=/net/storage/pr3/plgrid/plggdvps/rcorrente/dvps-fm/outputs/phase4/logs/%x-%j.out
#SBATCH --error=/net/storage/pr3/plgrid/plggdvps/rcorrente/dvps-fm/outputs/phase4/logs/%x-%j.err
#SBATCH --export=NONE
#
# Usage:
#   sbatch --job-name=benx_tm_lora \
#     examples/dvps/phase4/run_train.sbatch \
#     examples/dvps/phase4/benx_terramind_lora.yaml

shopt -s expand_aliases
set +e
source /etc/profile
set -euo pipefail

YAML="${1:?usage: sbatch run_train.sbatch <yaml>}"

echo "===== NODE INFO ====="
echo "Running on node: $(hostname)"
echo "Architecture:   $(uname -m)"
echo "YAML:           ${YAML}"

module load ML-bundle/25.10
source /net/storage/pr3/plgrid/plggdvps/skoneru/envs/terramind_llama/bin/activate

export PYTHONPATH=/net/storage/pr3/plgrid/plggdvps/rcorrente/dvps-fm/LLaMA-Factory/src:${PYTHONPATH:-}
export DISABLE_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/net/scratch/hscra/plgrid/plgriccardocorr/.cache/huggingface
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_MODULES_CACHE="${HF_HOME}/job_modules/${SLURM_JOB_ID}"
export HF_DATASETS_CACHE="${HF_HOME}/job_datasets/${SLURM_JOB_ID}"
export WANDB_ENTITY=DVPS-ai
export WANDB_PROJECT=GI-finetuning
if [[ -f "${HOME}/.wandb_api_key" ]]; then
  export WANDB_API_KEY="$(cat "${HOME}/.wandb_api_key")"
fi
# Continue an existing W&B run when WANDB_RUN_ID is injected via sbatch --export.
# Projector YAML defaults to the crashed 24015089 run so a plain resubmit keeps the same chart.
if [[ -z "${WANDB_RUN_ID:-}" && "${YAML}" == *benx_terramind_projector.yaml ]]; then
  WANDB_RUN_ID=musyfqwu
  WANDB_RESUME="${WANDB_RESUME:-must}"
fi
if [[ -n "${WANDB_RUN_ID:-}" ]]; then
  export WANDB_RESUME="${WANDB_RESUME:-must}"
  export WANDB_RUN_ID
  echo "W&B continue: WANDB_RESUME=${WANDB_RESUME} WANDB_RUN_ID=${WANDB_RUN_ID}"
fi

mkdir -p /net/storage/pr3/plgrid/plggdvps/rcorrente/dvps-fm/outputs/phase4/logs
mkdir -p "${HF_HOME}" "${HF_MODULES_CACHE}" "${HF_DATASETS_CACHE}"

cd /net/storage/pr3/plgrid/plggdvps/rcorrente/dvps-fm/LLaMA-Factory

if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  NPROC_PER_NODE="${SLURM_GPUS_ON_NODE##*:}"
elif [[ -n "${SLURM_NTASKS_PER_NODE:-}" ]]; then
  NPROC_PER_NODE="${SLURM_NTASKS_PER_NODE}"
else
  NPROC_PER_NODE=4
fi
export NPROC_PER_NODE
CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NPROC_PER_NODE - 1)))"
export CUDA_VISIBLE_DEVICES
export FORCE_TORCHRUN=1
echo "GPUs/procs: NPROC_PER_NODE=${NPROC_PER_NODE} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 \
  python -m llamafactory.cli train "${YAML}"
