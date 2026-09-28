#!/bin/bash
# 2-step projector smoke on qwen_omni_four with the five real datasets.
#
#   srun ... bash LLaMA-Factory/examples/dvps/run_four_encoder_mixed_smoke.sh

set +e
source /etc/profile
set -euo pipefail

ROOT=/net/storage/pr3/plgrid/plggdvps/rcorrente/dvps-fm
YAML="${ROOT}/LLaMA-Factory/examples/dvps/four_encoder_mixed_smoke.yaml"
OUT="${ROOT}/outputs/four_encoder_mixed_smoke"

echo "===== NODE INFO ====="
echo "node: $(hostname)   arch: $(uname -m)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

source "${ROOT}/omni_composite/scripts/env_omniavsr.sh"
export PYTHONPATH="${ROOT}/LLaMA-Factory/src:${PYTHONPATH:-}"
export DISABLE_VERSION_CHECK=1

rm -rf "${HF_HOME}/modules/transformers_modules/qwen_omni_four" || true
mkdir -p "${OUT}"

cd "${ROOT}/LLaMA-Factory"

echo "===== DATASET PEEK ====="
python - <<'PY'
from datasets import load_dataset
from pathlib import Path
import json

rows = {
    "lnqa_train": "/net/storage/pr3/plgrid/plggdvps/datasets/LNQA/sharegpt/train",
    "lipcrops_180hours_av_train": "/net/storage/pr3/plgrid/plggdvps/datasets/lipcrops_180hours/lipread/av_train.parquet",
    "mimic_cxr_medgemma_train": "/net/storage/pr3/plgrid/plggdvps/datasets/MIMIC-CXR/medgemma/train.parquet",
    "benx_multimodal_test": "/net/storage/pr3/plgrid/plggdvps/datasets/BigEarthNet.txt/benx_llamafactory_multimodal_test/train.parquet",
    "train_v1": "/net/storage/pr3/plgrid/plggdvps/skoneru/data/v1/train.v1.json",
}

def first(path):
    p = Path(path)
    if p.is_dir():
        ds = load_dataset("parquet", data_files=str(p / "*.parquet"), split="train", streaming=True)
    elif p.suffix == ".json":
        ds = load_dataset("json", data_files=str(p), split="train", streaming=True)
    else:
        ds = load_dataset("parquet", data_files=str(p), split="train", streaming=True)
    return next(iter(ds))

for name, path in rows.items():
    row = first(path)
    msgs = row.get("messages")
    keys = [k for k in row.keys() if k != "messages"]
    print(f"\n-- {name}")
    print("  keys:", keys)
    print("  messages:", json.dumps(msgs, ensure_ascii=False)[:500])
    for k in keys:
        v = row[k]
        if v is None:
            print(f"  {k}: None")
            continue
        s = repr(v)
        print(f"  {k}: {type(v).__name__} {s[:180]}")
PY

echo "===== SMOKE TRAIN ====="
python -m llamafactory.cli train "${YAML}"
