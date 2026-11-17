#!/bin/bash
# Build the DVPS training env for Leonardo (x86_64, A100, driver 535 / CUDA 12.2).
# Mirrors the Helios env skoneru/envs/terramind_llama (aarch64, cu130):
#   torch 2.13.0 / torchvision 0.28.0 / torchaudio 2.11.0 from the cu126 index,
#   everything else pinned to the Helios versions (requirements-helios-pinned.txt).
# Not installed: FlashAttention-3 (Hopper only; A100 uses SDPA), CUDA-13 runtimes,
# editable installs. LLaMA-Factory is used from source via PYTHONPATH.
#
# A venv is not relocatable, so build it at the path it will be used from.
# Login/serial nodes have internet; Booster nodes do not.
#
#   ROOT=/leonardo_scratch/fast/trans_f3/dvps-multimodal
#   sbatch -A trans_f3 -p lrd_all_serial -t 02:00:00 -c 4 --mem=16G \
#     -o $ROOT/dvps-fm/logs/%x-%j.out -J build_dvps_train \
#     $ROOT/dvps-fm/LLaMA-Factory/examples/dvps/leonardo/setup/build_dvps_train_env.sh
set -uo pipefail
umask 002   # shared project folder: keep group write

ROOT=${ROOT:-/leonardo_scratch/fast/trans_f3/dvps-multimodal}
ENV=${ENV:-$ROOT/envs/dvps_train}
LF_SRC=${LF_SRC:-$ROOT/dvps-fm/LLaMA-Factory/src}
REQS=${REQS:-$ROOT/dvps-fm/LLaMA-Factory/examples/dvps/leonardo/setup/requirements-helios-pinned.txt}
LOG=${TMPDIR:-/tmp}/pip-last-$$.log

case "$ENV" in "$ROOT"/envs/*) ;; *) echo "refusing to build outside $ROOT/envs: $ENV" >&2; exit 2 ;; esac
[ -f "$REQS" ] || { echo "missing $REQS" >&2; exit 2; }

module load python/3.11.7
unset PYTHONPATH   # the module's PYTHONPATH shadows the venv's pip
export PIP_CACHE_DIR=$ROOT/dvps-fm/.cache/pip

rm -rf "$ENV"
python3 -m venv "$ENV"
PY=$ENV/bin/python
$PY -m pip install -q --upgrade pip setuptools wheel

echo "=== $(date +%T) torch stack (cu126)"
$PY -m pip install torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu126 || { echo "TORCH INSTALL FAILED"; exit 1; }
$PY -m pip install torchcodec==0.12.0 --index-url https://download.pytorch.org/whl/cu126 \
  || echo "OPTIONAL torchcodec==0.12.0 not available for cu126 (LLaMA-Factory decodes audio with soundfile/librosa)"

echo "=== $(date +%T) pinned packages (one by one, --no-deps)"
failed=()
while read -r req; do
  [[ -z "$req" || "$req" == \#* ]] && continue
  if ! $PY -m pip install -q --no-deps "$req" >"$LOG" 2>&1; then
    failed+=("$req"); echo "FAILED $req: $(tail -2 "$LOG" | tr '\n' ' ')"
  fi
done < "$REQS"
echo "pinned install failures: ${#failed[@]} ${failed[*]:-}"

echo "=== $(date +%T) pip check"
$PY -m pip check || true

echo "=== $(date +%T) import test (CPU)"
PYTHONPATH=$LF_SRC $PY - <<'EOF'
import importlib, sys
mods = ["torch", "torchvision", "torchaudio", "transformers", "peft", "accelerate", "datasets",
        "trl", "liger_kernel", "terratorch", "torchgeo", "timm", "av", "librosa", "soundfile",
        "pyarrow", "qwen_omni_utils", "wandb", "llamafactory"]
bad = 0
for name in mods:
    try:
        module = importlib.import_module(name)
        print(f"ok   {name} {getattr(module, '__version__', '')}")
    except Exception as error:
        bad += 1
        print(f"FAIL {name}: {type(error).__name__}: {error}")
import torch
print("torch cuda build:", torch.version.cuda, "| sdpa available:", hasattr(torch.nn.functional, "scaled_dot_product_attention"))
sys.exit(1 if bad else 0)
EOF
echo "import test exit: $?"
echo "=== $(date +%T) done: $ENV"
