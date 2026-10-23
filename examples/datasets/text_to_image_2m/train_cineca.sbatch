#!/bin/bash
#SBATCH --job-name=omni3b-unilip
#SBATCH --account=EUHPC_B39_014
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/leonardo_work/EUHPC_B39_014/skoneru/LlamaFactory
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

source /leonardo_work/EUHPC_B39_014/skoneru/envs/llama_gpu/bin/activate

export T2I_2M_ROOT="${T2I_2M_ROOT:-/leonardo_work/EUHPC_B39_014/skoneru/huggingface_cache/hub/datasets--jackyhate--text-to-image-2M/snapshots/d6e16ffc2d9051fa91c08b4ebca82554287c28f3}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/leonardo_work/EUHPC_B39_014/skoneru/huggingface_cache/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/leonardo_work/EUHPC_B39_014/skoneru/huggingface_cache/datasets}"

# One Slurm task launches four torchrun workers; avoid launching four copies of torchrun.
export FORCE_TORCHRUN=1
export NNODES=1
export NODE_RANK=0
export NPROC_PER_NODE=4
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Keep this batch run separate from an interactive run using the same YAML.
run_output="${UNILIP_OUTPUT_DIR:-$PWD/saves/omni3b_unilip_text_to_image_2m_${SLURM_JOB_ID}}"
printf 'Job %s on %s; output: %s\n' "$SLURM_JOB_ID" "$(hostname)" "$run_output"

# Extra CLI overrides can follow the sbatch filename (e.g. resume_from_checkpoint=...).
exec srun --ntasks=1 --cpu-bind=none llamafactory-cli train \
  examples/datasets/text_to_image_2m/train_omni3b.yaml \
  "output_dir=$run_output" "$@"
