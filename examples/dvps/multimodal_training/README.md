# Encoder-routed multimodal training views

This directory defines derived parquet views under:

```text
/net/storage/pr3/plgrid/plggdvps/datasets/multimodal_training
```

The source datasets are never modified. A route is the pair of a prompt
marker and its parquet payload column:

- BigEarthNet.txt: `<terramind>` + `terramind`
- LipCrops AV audio: `<omniavsr_audio>` + `omniavsr_audio`
- LipCrops AV video: `<omniavsr_video>` + `omniavsr_video`
- RSTeller and GeoChat: `<image>` + `images` (native Qwen vision)

`encoder_views.json` is the source of truth. Custom routes are checked against
their checkpoint encoder declarations before any rows are read. TerraMind is
validated against `qwen_omni_terramind_s1s2`; the remaining custom routes use
`qwen_omni_four`.

BigEarthNet currently provides `S2L2A` only. The mask-capable TerraMind
encoder declares `min_modalities: 1`, so its fixed S1/S2 token layout uses
the pretrained mask token for the absent `S1GRD` slot. The dataset still uses
one `<terramind>` marker and one `terramind` payload column.

## Output behavior

BigEarthNet and LipCrops are streamed by row group and atomically installed.
RSTeller and GeoChat are already correctly routed to native Qwen vision;
after validation, the builder creates symlinks to their existing parquet
shards instead of duplicating embedded image bytes.

Existing outputs are skipped. Pass `--overwrite` to replace a generated
parquet or a non-matching symlink. The last successful materialization writes
`last_build_report.json` in the output root.

## Run only on a CPU allocation

Do not run the Python converter on a login node. For configuration checks and
small validation work, first request an interactive CPU allocation:

```bash
salloc --partition=plgrid --account=plggdvps01-cpu \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=01:00:00
srun --pty bash -l
module load GCC/14.3.0
module load Python-bundle-PyPI/2025.07
module load Arrow/22.0.0
bash /net/storage/pr3/plgrid/plggdvps/rcorrente/dvps-fm/LLaMA-Factory/examples/dvps/multimodal_training/run_build.sh \
  --dry-run
```

Use the batch job for a full scan and conversion:

```bash
cd /net/storage/pr3/plgrid/plggdvps/rcorrente/dvps-fm/LLaMA-Factory
sbatch examples/dvps/multimodal_training/submit_build.sbatch
```

Useful targeted commands, also from a Slurm allocation:

```bash
# Validate without writing.
bash examples/dvps/multimodal_training/run_build.sh \
  --dataset lipcrops_180hours --validate-only

# Rebuild one split.
bash examples/dvps/multimodal_training/run_build.sh \
  --split BigEarthNet.txt:val --overwrite
```
