#!/bin/bash
#SBATCH --job-name=phase4_predict
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plggdvps01-gpu-gh200
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=119G
#SBATCH --time=04:00:00
#SBATCH --output=/net/storage/pr3/plgrid/plggdvps/rcorrente/dvps-fm/outputs/phase4/logs/%x-%j.out
#SBATCH --error=/net/storage/pr3/plgrid/plggdvps/rcorrente/dvps-fm/outputs/phase4/logs/%x-%j.err
#SBATCH --export=NONE
#
# Usage:
#   sbatch examples/dvps/phase4/run_predict.sbatch examples/dvps/phase4/mimic_medgemma_lora_predict.yaml
# Then:
#   bash examples/dvps/phase4/eval_generations.sh <predict_output_dir>/generated_predictions.jsonl

shopt -s expand_aliases
set +e
source /etc/profile
set -euo pipefail

YAML="${1:?usage: sbatch run_predict.sbatch <yaml>}"

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

rm -rf "${HF_HOME}/modules/transformers_modules/qwen_omni_terramind_s1s2" \
       "${HF_HOME}/modules/transformers_modules/qwen_omni_terramind_medgemma" || true

mkdir -p /net/storage/pr3/plgrid/plggdvps/rcorrente/dvps-fm/outputs/phase4/logs
mkdir -p "${HF_HOME}"

cd /net/storage/pr3/plgrid/plggdvps/rcorrente/dvps-fm/LLaMA-Factory

python -m llamafactory.cli train "${YAML}"
