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
#SBATCH --output=/net/storage/pr3/plgrid/plggdvps/rcorrente/dvps-fm/dvps-fm-v1-ablations/logs/%x-%j.log
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
export WANDB_PROJECT
if [[ -f "${HOME}/.wandb_api_key" ]]; then
  export WANDB_API_KEY="$(cat "${HOME}/.wandb_api_key")"
fi

mkdir -p /net/storage/pr3/plgrid/plggdvps/rcorrente/dvps-fm/outputs/phase4/logs
mkdir -p "${HF_HOME}" "${HF_MODULES_CACHE}" "${HF_DATASETS_CACHE}"

cd /net/storage/pr3/plgrid/plggdvps/rcorrente/dvps-fm/LLaMA-Factory

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
