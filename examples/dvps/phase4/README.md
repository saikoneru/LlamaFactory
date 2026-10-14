# Phase 4 training configs

Software-first recipes for the first real encoder vs Omni-RGB comparison.
For a short data-path check, copy a training YAML, set `max_steps: 2`, and submit it with `examples/dvps/run_train.sbatch`.

Launch from this directory, using **this** LLaMA-Factory tree (`python -m llamafactory.cli`), not Sai’s install.

```bash
LF=/net/storage/pr3/plgrid/plggdvps/rcorrente/dvps-fm/LLaMA-Factory
cd "$LF"
```

## Matrix

| Recipe | Dataset | Model | YAML |
| --- | --- | --- | --- |
| TerraMind projector-only | `mmtrain_bentxt_terramind_*` (S2L2A; masked S1GRD) | `checkpoints/qwen_omni_terramind_s1s2` | `benx_terramind_projector.yaml` |
| TerraMind projector + Thinker LoRA | same | same | `benx_terramind_lora.yaml` |
| BEN Omni RGB control (same parquet family) | `benx_omnirgb_*` | `Qwen/Qwen2.5-Omni-7B` | `benx_omnirgb_lora.yaml` |
| BEN Omni RGB control (sharegpt-rgb) | `bentxt_*` | same | `bentxt_omnirgb_lora.yaml` (also `dvps-fm/finetune_dvpsfm_bentxt.yaml`) |
| MedGemma projector-only | `mimic_cxr_medgemma_*` | dual `qwen_omni_terramind_medgemma` | `mimic_medgemma_projector.yaml` |
| MedGemma projector-only, single encoder | `mimic_cxr_medgemma_*` | `checkpoints/qwen_omni_medgemma` | `mimic_medgemma_only_projector.yaml` |
| MedGemma projector + Thinker LoRA | same | same | `mimic_medgemma_lora.yaml` |
| MIMIC Omni RGB control | `mimic_cxr_*` | `Qwen/Qwen2.5-Omni-7B` | `mimic_omnirgb_lora.yaml` (also `dvps-fm/finetune_dvpsfm_mimic.yaml`) |
| Mixed MIMIC MedGemma + BEN S1S2/RGB | `mimic_cxr_medgemma_train, benx_terramindl2_omnirgb_train` | dual `qwen_omni_terramind_medgemma` | `mimic_benx_projectors.yaml` |

Each dataset states the marker of the encoder that will run. The `mimic_cxr_medgemma_*` entries read `MIMIC-CXR/medgemma/*.parquet`, a view whose prompts say `<medgemma>` and whose X-rays live in a `medgemma` column. `<image>` always means native Omni vision, so a row can hold both an RGB photo and an X-ray. Regenerate the view with:

```bash
python examples/dvps/phase4/make_medgemma_view.py \
  --src /net/storage/pr3/plgrid/plggdvps/datasets/MIMIC-CXR/train.parquet \
  --out /net/storage/pr3/plgrid/plggdvps/datasets/MIMIC-CXR/medgemma/train.parquet
```

The Omni RGB control (`mimic_cxr_*`) keeps reading the original `<image>` parquet.

## Eval

Training YAMLs log **eval_loss** (teacher-forced caption / report NLL) every 500 steps with early stopping.

After a run, generate with `run_predict.sbatch` from a YAML that sets `do_predict: true`, then score BLEU/ROUGE:

```bash
bash examples/dvps/phase4/eval_generations.sh \
  <predict_output_dir>/generated_predictions.jsonl
```

BEN mAP / CheXbert / RadGraph are not wired in LLaMA-Factory. Point those toolkits at the same `generated_predictions.jsonl` when you have them.

## Submit

Full 4-GPU training:

```bash
sbatch --job-name=benx_tm_proj examples/dvps/phase4/run_train.sbatch \
  examples/dvps/phase4/benx_terramind_projector.yaml
sbatch --job-name=p4_mimic_benx examples/dvps/phase4/run_train.sbatch \
  examples/dvps/phase4/mimic_benx_projectors.yaml
```

Env: `ML-bundle/25.10` + `terramind_llama`. `HF_HOME` is on scratch. After editing checkpoint `*.py`, the sbatch scripts delete the matching `transformers_modules` cache.
