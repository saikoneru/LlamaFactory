#!/bin/bash
# LEONARDO COPY of examples/dvps/run_train.sbatch (rcorrente). Changed: Slurm resources, env,
# LLaMA-Factory source, caches, offline HF/W&B, log/output paths, single-node NCCL settings.
#SBATCH --job-name=phase4_train
#SBATCH --partition=boost_usr_prod
#SBATCH --account=trans_f3
#SBATCH --nodes=2   # 8 A100 (2 x 4); see per_device_train_batch_size in the configs
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=/leonardo_scratch/fast/trans_f3/dvps-multimodal/dvps-fm/logs/%x-%j.log
#SBATCH --export=NONE
#
# Usage:
#   sbatch --job-name=benx_tm_lora \
#     examples/dvps/run_train.sbatch \
#     /absolute/path/to/config.yaml \
#     GI-finetuning

shopt -s expand_aliases
set +e
source /etc/profile
set -euo pipefail

YAML="${1:?usage: sbatch run_train.sbatch <yaml> <wandb-project>}"
WANDB_PROJECT="${2:?usage: sbatch run_train.sbatch <yaml> <wandb-project>}"
case "${WANDB_PROJECT}" in
  GI-finetuning|CD-finetuning|LC-finetuning|dvps-fm) ;;
  *)
    echo "Unsupported W&B project: ${WANDB_PROJECT}" >&2
    exit 2
    ;;
esac

echo "===== NODE INFO ====="
echo "Running on node: $(hostname)"
echo "Architecture:   $(uname -m)"
echo "YAML:           ${YAML}"
echo "W&B project:    ${WANDB_PROJECT}"

# x86_64 / A100 env pinned to the Helios package versions (torch 2.13.0+cu126); CUDA comes from the wheels.
module load python/3.11.7
unset PYTHONPATH  # the module's PYTHONPATH would shadow the venv
source /leonardo_scratch/fast/trans_f3/dvps-multimodal/envs/dvps_train/bin/activate

# rcorrente's LLaMA-Factory checkout on Leonardo (same commit as Helios), used read-only.
export PYTHONPATH=/leonardo_scratch/fast/trans_f3/dvps-multimodal/dvps-fm/LLaMA-Factory/src
export DISABLE_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/leonardo_scratch/fast/trans_f3/dvps-multimodal/dvps-fm/.cache/huggingface
# Booster compute nodes have no internet; the checkpoint is self-contained.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_MODULES_CACHE="${HF_HOME}/job_modules/${SLURM_JOB_ID}"
export HF_DATASETS_CACHE="${HF_HOME}/job_datasets/${SLURM_JOB_ID}"
# A fixed Inductor cache lets restarted jobs reuse kernels compiled by earlier runs.
export TORCHINDUCTOR_CACHE_DIR=/leonardo_scratch/fast/trans_f3/dvps-multimodal/dvps-fm/.cache/torchinductor
export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
export TORCHDYNAMO_CAPTURE_DYNAMIC_OUTPUT_SHAPE_OPS=1
export TORCH_LOGS="graph_breaks,recompiles"
export WANDB_ENTITY=DVPS-ai
export WANDB_PROJECT
# No internet on compute nodes: log offline, then `wandb sync` from a login node.
export WANDB_MODE=offline
export WANDB_DIR=/leonardo_scratch/fast/trans_f3/dvps-multimodal/dvps-fm/wandb
mkdir -p "${WANDB_DIR}"
if [[ -f "${HOME}/.wandb_api_key" ]]; then
  export WANDB_API_KEY="$(cat "${HOME}/.wandb_api_key")"
fi
# Optional YAML comment "# wandb_run_id: <id>" appends to that existing W&B run.
WANDB_RUN_ID="$(awk -F': ' '$0 ~ "^#[[:space:]]+wandb_run_id:" { sub(/^#[[:space:]]*[^:]+:[[:space:]]*/, ""); print; exit }' "${YAML}")"
if [[ -n "${WANDB_RUN_ID}" ]]; then
  export WANDB_RUN_ID
  export WANDB_RESUME=allow
  echo "W&B resume: ${WANDB_ENTITY}/${WANDB_PROJECT}/${WANDB_RUN_ID}"
fi

mkdir -p /leonardo_scratch/fast/trans_f3/dvps-multimodal/dvps-fm/logs /leonardo_scratch/fast/trans_f3/dvps-multimodal/dvps-fm/outputs
mkdir -p "${HF_HOME}" "${HF_MODULES_CACHE}" "${HF_DATASETS_CACHE}" "${TORCHINDUCTOR_CACHE_DIR}"

# Refuse configs that still point to Helios (HELIOS_GUARD).
if grep -vE '^[[:space:]]*#' "${YAML}" | grep -q '/net/storage'; then
  echo "${YAML} still contains Helios paths" >&2  # HELIOS_GUARD
  exit 2
fi

cd /leonardo_scratch/fast/trans_f3/dvps-multimodal/dvps-fm

# Populate custom Transformers modules once before torchrun starts. Loading an
# empty shared module cache concurrently from every rank can expose partially
# copied Python modules.
python - "${YAML}" <<'PY'
import importlib
import importlib
import json
import sys
from pathlib import Path

import yaml
from transformers.dynamic_module_utils import get_class_from_dynamic_module


def safe_invalidate_caches():
    for finder in sys.meta_path:
        invalidate = getattr(finder, "invalidate_caches", None)
        if invalidate is None:
            continue

        try:
            invalidate()
        except TypeError as error:
            if "missing 1 required positional argument: 'cls'" not in str(error):
                raise
            invalidate(finder)


# The environment's metadata backport registers an unbound class method in
# sys.meta_path. Transformers refreshes these caches while creating its dynamic
# module cache, so handle that malformed finder without skipping cache refresh.
importlib.invalidate_caches = safe_invalidate_caches

training_config = yaml.safe_load(Path(sys.argv[1]).read_text())
model_path = Path(training_config["model_name_or_path"])
class_references = set()

# The normal invalidation walk is broken by mixed system/importlib_metadata
# finders in this environment. Clearing the importer cache provides the
# required refresh without invoking those incompatible finder methods.
importlib.invalidate_caches = sys.path_importer_cache.clear

if model_path.is_dir():
    for filename in ("config.json", "processor_config.json", "tokenizer_config.json"):
        metadata_path = model_path / filename
        if not metadata_path.is_file():
            continue

        auto_map = json.loads(metadata_path.read_text()).get("auto_map", {})
        for reference in auto_map.values():
            references = reference if isinstance(reference, list) else [reference]
            class_references.update(item for item in references if item)

for class_reference in sorted(class_references):
    get_class_from_dynamic_module(class_reference, str(model_path))

if class_references:
    print(f"Preloaded {len(class_references)} custom Transformers classes from {model_path}")
PY

if [[ "${DVPS_PRELOAD_ONLY:-false}" == "true" ]]; then
  exit 0
fi

if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  NPROC_PER_NODE="${SLURM_GPUS_ON_NODE##*:}"
elif [[ -n "${SLURM_NTASKS_PER_NODE:-}" && "${SLURM_NTASKS_PER_NODE}" -gt 1 ]]; then
  NPROC_PER_NODE="${SLURM_NTASKS_PER_NODE}"
else
  NPROC_PER_NODE=4
fi
export NPROC_PER_NODE
export NNODES="${SLURM_NNODES:-1}"
export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
export MASTER_PORT="${MASTER_PORT:-29500}"
export FORCE_TORCHRUN=1

if [[ "${NNODES}" -gt 1 ]]; then
  # One torchrun per node. Rendezvous assigns ranks, so this does not depend on
  # SLURM_NODEID surviving srun --export. IB stays enabled; C2C GDR is off.
  export RDZV_ID="${SLURM_JOB_ID}"
  export NCCL_NET_GDR_C2C=0
  export NCCL_ENABLE_C2C_GDR=0
  # A rank segfaulted in its first cross-node collective once; these print the native stack and NCCL's reason.
  export PYTHONFAULTHANDLER=1
  export NCCL_DEBUG=WARN
  echo "Multi-node: NNODES=${NNODES} NPROC_PER_NODE=${NPROC_PER_NODE} MASTER=${MASTER_ADDR}:${MASTER_PORT}"
  unset CUDA_VISIBLE_DEVICES
  srun --export=ALL \
    --ntasks="${NNODES}" \
    --ntasks-per-node=1 \
    --gpus-per-task="${NPROC_PER_NODE}" \
    python -m llamafactory.cli train "${YAML}"
else
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NPROC_PER_NODE - 1)))"
  export CUDA_VISIBLE_DEVICES
  echo "GPUs/procs: NPROC_PER_NODE=${NPROC_PER_NODE} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  # Helios disabled IB/P2P here; on Leonardo NVLink P2P stays on (4-GPU NCCL check passed).
    python -m llamafactory.cli train "${YAML}"
fi
