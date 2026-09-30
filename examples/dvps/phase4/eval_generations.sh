#!/bin/bash
# BLEU/ROUGE on LLaMA-Factory generated_predictions.jsonl (caption / report text).
# Domain metrics (BEN mAP, CheXbert, RadGraph) are not in this repo; run those
# separately on the same jsonl if those toolkits are available.
set -euo pipefail

PRED="${1:?usage: eval_generations.sh <generated_predictions.jsonl>}"
LF_ROOT="/net/storage/pr3/plgrid/plggdvps/rcorrente/dvps-fm/LLaMA-Factory"

export PYTHONPATH="${LF_ROOT}/src:${PYTHONPATH:-}"
cd "$(dirname "${PRED}")"
python "${LF_ROOT}/scripts/eval_bleu_rouge.py" "$(basename "${PRED}")"
echo "Wrote predictions_score.json next to ${PRED}"
