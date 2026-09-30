#!/bin/bash
#SBATCH --job-name=omni_composite_smoke
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plggdvps01-gpu-gh200
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=119G
#SBATCH --time=00:45:00
#SBATCH --output=/net/storage/pr3/plgrid/plggdvps/rcorrente/dvps-fm/outputs/omni_composite_mixed_smoke/%x-%j.out
#SBATCH --error=/net/storage/pr3/plgrid/plggdvps/rcorrente/dvps-fm/outputs/omni_composite_mixed_smoke/%x-%j.err
#SBATCH --export=NONE

shopt -s expand_aliases
set +e
source /etc/profile
set -euo pipefail

echo "===== NODE INFO ====="
echo "Running on node: $(hostname)"
echo "Architecture:   $(uname -m)"

module load ML-bundle/25.10
source /net/storage/pr3/plgrid/plggdvps/skoneru/envs/terramind_llama/bin/activate

export PYTHONPATH=/net/storage/pr3/plgrid/plggdvps/rcorrente/dvps-fm/LLaMA-Factory/src:${PYTHONPATH:-}
export DISABLE_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/net/scratch/hscra/plgrid/plgriccardocorr/.cache/huggingface

mkdir -p /net/storage/pr3/plgrid/plggdvps/rcorrente/dvps-fm/outputs/omni_composite_mixed_smoke

cd /net/storage/pr3/plgrid/plggdvps/rcorrente/dvps-fm/LLaMA-Factory

echo "===== UNIT TESTS ====="
python -m pytest tests/data/test_extra_encoders.py tests/data/test_converter.py -q

echo "===== SMOKE TRAIN ====="
python -m llamafactory.cli train examples/dvps/omni_composite_mixed_smoke.yaml
