#!/bin/bash
#SBATCH --job-name=dvps_train
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plggdvps01-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=119G
#SBATCH --time=48:00:00
#SBATCH --output=/net/storage/pr3/plgrid/plggdvps/rcorrente/dvps-fm/outputs/dvps_train_logs/%x-%j.out
#SBATCH --error=/net/storage/pr3/plgrid/plggdvps/rcorrente/dvps-fm/outputs/dvps_train_logs/%x-%j.err
#SBATCH --export=NONE
#
# One launcher for the single-GPU recipes in this directory.
#
#   sbatch --job-name=autoavsr_video \
#     LLaMA-Factory/examples/dvps/run_train.sbatch \
#     LLaMA-Factory/examples/dvps/autoavsr_video.yaml
#
# output_dir and run_name are read from the YAML. Optional YAML comments:
#   # wandb_entity: DVPS-ai
#   # wandb_project: LC-finetuning
# Raise memory at submit time when a recipe needs it: sbatch --mem=180G ...

shopt -s expand_aliases
set +e
source /etc/profile
set -euo pipefail

if [[ -z "${HOME:-}" ]]; then
  export HOME=/net/home/plgrid/plgriccardocorr
fi

ROOT=/net/storage/pr3/plgrid/plggdvps/rcorrente/dvps-fm
YAML="${1:?usage: sbatch run_train.sbatch <yaml>}"
if [[ "${YAML}" != /* ]]; then
  if [[ -f "${YAML}" ]]; then
    YAML="$(readlink -f "${YAML}")"
  else
    YAML="${ROOT}/${YAML}"
  fi
fi

yaml_field() {
  awk -F': ' -v key="$1" '$0 ~ "^" key ":" { sub(/^[^:]+:[[:space:]]*/, ""); print; exit }' "${YAML}"
}
yaml_comment() {
  awk -F': ' -v key="$1" '$0 ~ "^#[[:space:]]+" key ":" { sub(/^#[[:space:]]*[^:]+:[[:space:]]*/, ""); print; exit }' "${YAML}"
}

OUT="$(yaml_field output_dir)"
RUN_NAME="$(yaml_field run_name)"
MODEL="$(yaml_field model_name_or_path)"
WANDB_ENTITY_VALUE="$(yaml_comment wandb_entity || true)"
WANDB_PROJECT_VALUE="$(yaml_comment wandb_project || true)"

echo "===== NODE INFO ====="
echo "node: $(hostname)   arch: $(uname -m)"
echo "job:  ${SLURM_JOB_ID:-none}"
echo "YAML: ${YAML}"
echo "OUT:  ${OUT:-<unset>}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

source "${ROOT}/omni_composite/scripts/env_autoavsr.sh"
export PYTHONPATH="${ROOT}/LLaMA-Factory/src:${PYTHONPATH:-}"
export DISABLE_VERSION_CHECK=1
export HF_MODULES_CACHE="${HF_HOME}/job_modules/${SLURM_JOB_ID}"
if [[ -n "${WANDB_ENTITY_VALUE}" ]]; then
  export WANDB_ENTITY="${WANDB_ENTITY_VALUE}"
fi
if [[ -n "${WANDB_PROJECT_VALUE}" ]]; then
  export WANDB_PROJECT="${WANDB_PROJECT_VALUE}"
fi
if [[ -n "${RUN_NAME}" ]]; then
  export WANDB_NAME="${RUN_NAME}"
fi
if [[ -f "${HOME}/.wandb_api_key" ]]; then
  export WANDB_API_KEY="$(cat "${HOME}/.wandb_api_key")"
fi

if [[ -n "${MODEL}" && -d "${MODEL}" ]]; then
  rm -rf "${HF_HOME}/modules/transformers_modules/$(basename "${MODEL}")" || true
fi
mkdir -p "${ROOT}/outputs/dvps_train_logs" "${HF_HOME}" "${HF_MODULES_CACHE}"
if [[ -n "${OUT}" ]]; then
  mkdir -p "${OUT}/logs"
fi

cd "${ROOT}/LLaMA-Factory"
python -m llamafactory.cli train "${YAML}"
