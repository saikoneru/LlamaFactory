#!/bin/bash -l
#SBATCH --job-name=build_autoavsr_video_data
#SBATCH --partition=plgrid
#SBATCH --account=plggdvps01-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/net/storage/pr3/plgrid/plggdvps/rcorrente/dvps-fm/outputs/%x-%j.out
#SBATCH --error=/net/storage/pr3/plgrid/plggdvps/rcorrente/dvps-fm/outputs/%x-%j.err
#SBATCH --export=NONE

set -euo pipefail

ROOT=/net/storage/pr3/plgrid/plggdvps/rcorrente/dvps-fm
DIR="${ROOT}/LLaMA-Factory/examples/dvps/multimodal_training"

module load GCC/14.3.0
module load Python-bundle-PyPI/2025.07
module load Arrow/22.0.0

/bin/bash "${DIR}/run_build.sh" \
  --config "${DIR}/encoder_views_autoavsr_video.json" \
  --overwrite
