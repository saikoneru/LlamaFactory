#!/bin/bash -l
#SBATCH --job-name=mm_encoder_views
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

# On another cluster override the #SBATCH lines on the command line and pass
# DVPS_ROOT / BUILD_MODULES with --export=ALL,... (see README).
ROOT="${DVPS_ROOT:-/net/storage/pr3/plgrid/plggdvps/rcorrente/dvps-fm}"
RUNNER="${ROOT}/LLaMA-Factory/examples/dvps/multimodal_training/run_build.sh"

for module_name in ${BUILD_MODULES-GCC/14.3.0 Python-bundle-PyPI/2025.07 Arrow/22.0.0}; do
  module load "${module_name}"
done

echo "Job ${SLURM_JOB_ID} started on $(hostname) at $(date)"
/bin/bash "${RUNNER}" "$@"
echo "Job ${SLURM_JOB_ID} finished at $(date)"
