#!/bin/bash -l

#SBATCH --job-name=make_s1s2_view
#SBATCH --partition=plgrid
#SBATCH --account=plggdvps01-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/make_s1s2_view-%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p "$REPO_DIR/logs"

echo "Job ${SLURM_JOB_ID} on $(hostname) at $(date)"
srun --cpu-bind=cores \
    /net/storage/pr3/plgrid/plggdvps/rcorrente/envs/pqtools/bin/python \
    "$REPO_DIR/examples/dvps/phase4/make_s1s2_view.py" --workers 1
echo "Job finished at $(date)"
