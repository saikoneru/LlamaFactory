# Encoder-routed multimodal training views

This directory defines derived parquet views under:

```text
/net/storage/pr3/plgrid/plggdvps/datasets/multimodal_training
```

The source datasets are never modified. A route is the pair of a prompt
marker and its parquet payload column:

- BigEarthNet.txt: `<terramind>` + `terramind`
- LipCrops AV audio: `<autoavsr_audio>` + `autoavsr_audio`
- LipCrops AV video: `<autoavsr_video>` + `autoavsr_video`
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

After the Auto-AVSR rename, rebuild the LipCrops views with `--overwrite` so
their columns and markers change from the previous encoder names.

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

## Building on another cluster (Leonardo)

`encoder_views_leonardo.json` defines the views used by `examples/dvps/dvps_v1`..`v4`
with their sources written as Helios paths. Its `path_map` translates them (longest
prefix wins) to Leonardo, and in `rewrite` mode the same rules are applied to the
media paths stored in the routed columns, so the views open files that exist there.

- `--check-paths` checks every media path a view exposes (rewritten paths, or the
  source's own paths for symlinked views) and exits with status 3 if any is missing;
  counts and examples go to `last_build_report.json`.
- Splits that name `dataset_info` entries get them copied from `registry_template`
  into `<output_root>/dataset_info.json` with `file_name` relative to the output root,
  so a training config only needs `dataset_dir: <output_root>`.
- `relative_symlinks: true` makes symlinked views survive a copy of the tree.
- Rows with several payloads of one route (two-ECG comparisons) need
  `"multi_payload": true` on that route binding; text-only datasets need
  `"text_only": true` and no routes.

```bash
OWN=/leonardo_scratch/fast/trans_f3/dvps-multimodal-smestici
sbatch -A trans_f3 -p lrd_all_serial -t 04:00:00 -c 4 --mem=30G \
  -o $OWN/logs/%x-%j.out -e $OWN/logs/%x-%j.err \
  --export=ALL,DVPS_ROOT=$OWN,BUILD_MODULES=python/3.11.7,VENV_PATH=$OWN/envs/views \
  $OWN/LLaMA-Factory/examples/dvps/multimodal_training/submit_build.sbatch \
  --config $OWN/LLaMA-Factory/examples/dvps/multimodal_training/encoder_views_leonardo.json \
  --check-paths
```
