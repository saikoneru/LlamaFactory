#!/bin/bash -l
#SBATCH --job-name=bentxt_multigpu
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plggdvps01-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --time=18:00:00
#SBATCH --output=logs/%x-%j.out

# Allow environment variables passed via sbatch --export
# (Note: Remove or adjust --export=NONE in header if passing variables)

shopt -s expand_aliases
set -euo pipefail

ENV_PATH=$1
CONFIG_PATH=$2

module load ML-bundle/25.10 CUDA/13.0.2
source $ENV_PATH/bin/activate
mkdir -p logs

CACHE_ROOT="$SCRATCH/llamafactory_cache"
mkdir -p "$CACHE_ROOT/huggingface" "$CACHE_ROOT/tmp" "$CACHE_ROOT/torchinductor"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_DATASETS_CACHE="$CACHE_ROOT/huggingface/datasets"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/huggingface/hub"
export TMPDIR="$CACHE_ROOT/tmp"
export TMP="$CACHE_ROOT/tmp"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor"
export NCCL_ENABLE_C2C_GDR=0

# wandb: API key is picked up from ~/.netrc (machine api.wandb.ai).
# Keep run files on $SCRATCH so ~/.cache does not hit the home quota.
mkdir -p "$CACHE_ROOT/wandb"
export WANDB_DIR="$CACHE_ROOT/wandb"
export WANDB_ENTITY="${WANDB_ENTITY:-DVPS-ai}"
export WANDB_PROJECT="${WANDB_PROJECT:-GI-finetuning}"
# Attach to the crashed GI run instead of opening a new W&B page.
# Override with sbatch --export=ALL,WANDB_RUN_ID=... or unset WANDB_RUN_ID for a fresh run.
if [[ "${CONFIG_PATH}" == *qwen_encoders_GI_all.yaml ]]; then
  export WANDB_RUN_ID="${WANDB_RUN_ID:-yeubb6i0}"
  export WANDB_RESUME="${WANDB_RESUME:-must}"
fi

export FORCE_TORCHRUN=1
export NNODES=$SLURM_NNODES
export NPROC_PER_NODE=$SLURM_GPUS_ON_NODE
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500


export NCCL_NET_GDR_C2C=0 
export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
export TORCHDYNAMO_CAPTURE_DYNAMIC_OUTPUT_SHAPE_OPS=1
export TORCH_LOGS="graph_breaks,recompiles"
srun bash -c "NODE_RANK=\$SLURM_NODEID llamafactory-cli train $CONFIG_PATH"
