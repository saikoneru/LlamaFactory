#!/bin/bash
set -euo pipefail

ROOT=/net/storage/pr3/plgrid/plggdvps/rcorrente/dvps-fm
SCRIPT_DIR="${ROOT}/LLaMA-Factory/examples/dvps/multimodal_training"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Refusing to run parquet work on a login node." >&2
  echo "Request an interactive CPU allocation or use submit_build.sbatch." >&2
  exit 2
fi

if [[ -n "${VENV_PATH:-}" ]]; then
  if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
    echo "Python environment not found: ${VENV_PATH}" >&2
    exit 2
  fi
  source "${VENV_PATH}/bin/activate"
fi

export PYTHONPATH="${ROOT}/LLaMA-Factory/src:${PYTHONPATH:-}"

if ! python -c "import pyarrow" >/dev/null 2>&1; then
  echo "The active Python does not provide pyarrow." >&2
  echo "Load GCC/14.3.0 and Arrow/22.0.0, or set VENV_PATH." >&2
  exit 2
fi

python "${SCRIPT_DIR}/build_views.py" \
  --config "${SCRIPT_DIR}/encoder_views.json" \
  "$@"
