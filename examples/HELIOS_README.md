# Helios with LlamaFactory

The goal of this document is to help partners from AD fine-tune different models as easily as possible. Since they will be providing interesting data, this document also walks through how to fine-tune and run inference/demos with publicly available multimodal foundation models.

## Account Creation

First, log in to Helios with a user account that belongs to the `plggdvps` group. For instructions on setting up your account, see this guide: [Cyfronet Partner Account Guide](https://docs.google.com/document/d/14yOorsjyXjOqEh51g3ut_iqFSJmPDpSKRRFEGFNja7s/edit?tab=t.0) — thanks to our Cyfronet partners for putting this together.

Once your account is set up, clone the repository to your desired location. For example, to clone it into your home directory:

```bash
cd ~
git clone https://github.com/saikoneru/LlamaFactory
cd LlamaFactory
```

## Data Preparation

The next step is preparing data for fine-tuning. This document won't cover data preparation in full detail, but you can see an example of formatting ShareGPT-style data here: [BigEarthNet.txt dataset](https://github.com/DVPS-ai/DVPS-FM/tree/bentxt-wds/datasets/BigEarthNet.txt).

In this example, we'll use image-text paired data. Note that the ShareGPT format also supports multiple images per sample — but always check whether your target model supports this. Qwen2.5-Omni-7B, for instance, can be fine-tuned on this kind of data, and even on combinations of multiple modalities together, such as image-audio-text → text. The main limitation is on the *output* side: Omni currently cannot be fine-tuned to generate images or speech.

For this walkthrough, we'll use the dataset located at:

`/net/storage/pr3/plgrid/plggdvps/datasets/BigEarthNet.txt/sharegpt-rgb`


Let's take a look at what's inside:

```bash
ls /net/storage/pr3/plgrid/plggdvps/datasets/BigEarthNet.txt/sharegpt-rgb
bench.parquet test.parquet train val.parquet
```


The `val`, `test`, and `bench` splits each consist of a single parquet file, while `train` is split across 32 shards. This is because the training split contains roughly 5M samples, and sharding allows the data to be loaded in parallel.

With the dataset in place, the next step is to tell LlamaFactory where to find it and how it's formatted.

First, let's print one sample from `test.parquet` to see the expected structure:

### Requesting an interactive CPU job

Since this is just a quick inspection task (not training), request a small interactive CPU-only job rather than running on the login node:

```bash
srun -p plgrid --nodes=1 --cpus-per-task=2 -A plggdvps01-cpu --gres=gpu:0 --time 0:05:00 --mem=1G --pty /bin/bash -l
```

### Loading the required modules

Once inside the job, load the modules providing `pandas` and `pyarrow`:

```bash
module load GCC/14.3.0
module load SciPy-bundle/2025.07
module load Arrow/22.0.0
```

### Printing a sample from the dataset

```bash
python3 -c 'import pandas as pd; df = pd.read_parquet("/net/storage/pr3/plgrid/plggdvps/datasets/BigEarthNet.txt/sharegpt-rgb/test.parquet"); print(f"Number of samples: {len(df)}\nColumns: {list(df.columns)}\n\n--- Example sample ---"); [print(f"{col}: {df.iloc[0][col]}") for col in df.columns]'
```

You should see:
```bash
Number of samples: 8192
Columns: ['messages', 'images']

--- Example sample ---
messages: [{'role': 'system', 'content': 'You are a helpful assistant that analyzes satellite imagery.'}
 {'role': 'user', 'content': '<image>Are there fewer than two continuous regions of marine waters in this image?'}
 {'role': 'assistant', 'content': 'yes'}]
images: [{'bytes': None, 'path': '/net/storage/pr3/plgrid/plggdvps/datasets/BigEarthNet.txt/webdataset-rgb/test-images/Ireland/Spring/S2A_MSIL2A_20180529T115401_N9999_R023_T29UNB_31_56.png'}]
```


The `<image>` token tells LlamaFactory that an image precedes the text at that position, and to load it accordingly. Note that the number of `<image>` tokens in `messages` and the length of `images` should match.

Once you're done inspecting the data, exit the CPU job to return to the login node:

```bash
exit
```

## Fine-Tuning on BigEarthNet.txt

To fine-tune on this data, we first need to register it in `dataset_info.json`. For simplicity, we'll refer to the one at [saikoneru/LlamaFactory `data/dataset_info.json`](https://github.com/saikoneru/LlamaFactory/blob/main/data/dataset_info.json) — though this file can live anywhere, as LlamaFactory can be told where to load it from.

Add your dataset entry as shown [here](https://github.com/saikoneru/LlamaFactory/blob/05e5131f4f7d31ac50f62746fb1c4a75c0904cd3/data/dataset_info.json#L2). For audio-only data, replace the `images` field accordingly; for combined image-and-audio data, include both fields together.

Now we're ready to fine-tune. Let's first run an interactive GPU job to confirm everything is working end to end.

### Requesting an Interactive GPU Job

```bash
srun -p plgrid-gpu-gh200 --nodes=1 --cpus-per-task=8 -A plggdvps01-gpu-gh200 --gres=gpu:1 --time 1:00:00 --mem=119G --pty /bin/bash -l
```

### Loading the Environment

For convenience, there is already an existing virtual environment with FlashAttention, LlamaFactory, and other dependencies (including audio loading support) pre-installed. Load it as follows:

```bash
module load ML-bundle/24.06a
module load GCCcore/14.3.0
module load FFmpeg/7.1.2
module load CUDA/13.0.2

source /net/storage/pr3/plgrid/plggdvps/skoneru/envs/terramind_llama/bin/activate

ENV="/net/storage/pr3/plgrid/plggdvps/skoneru/envs/terramind_llama"
export LD_LIBRARY_PATH="$ENV/lib/python3.11/site-packages/torch/lib:$ENV/lib/python3.11/site-packages/nvidia/npp/lib:$ENV/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$ENV/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH=/net/storage/pr3/plgrid/plggdvps/skoneru/envs/terramind_llama/lib/python3.11/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
```

That's it for installation — no further setup needed.

### Configuring the Training Run

The only remaining step is the config file. An example is provided here: [`qwen_2.5omni_lora_sft.yaml`](https://github.com/saikoneru/LlamaFactory/blob/main/examples/train_lora/qwen_2.5omni_lora_sft.yaml).

### Running Training Interactively

```bash
cd ~/LlamaFactory
llamafactory-cli train examples/train_lora/qwen_2.5omni_lora_sft.yaml 2>&1 | tee bentxt_llamafactory.log
```

If everything works, you should see training start shortly after. Note that FlashAttention-3 is enabled by default here, specifically optimized for Helios' GPUs.

### Submitting as a SLURM Job

Once you've confirmed training works interactively, exit the interactive job and submit it as a proper SLURM job instead.

An example submission script is provided here: [`submit_job.sh`](https://github.com/saikoneru/LlamaFactory/blob/main/examples/train_lora/submit_job.sh).

Run it from the login node, inside the LlamaFactory root directory:

```bash
./submit_job.sh examples/train_lora/qwen_2.5omni_lora_sft.yaml
```

For multi-GPU training, an example script is provided here: [`multigpu_submit_bentxt.sbatch`](https://github.com/saikoneru/LlamaFactory/blob/main/examples/train_lora/multigpu_submit_bentxt.sbatch) — simply swap in the multi-GPU script in place of the single-GPU one above.

If everything runs successfully, you should see checkpoints appear in the `output_dir` specified in your config file, saving the LoRA adapters as training progresses.

---

**TODO:** Demo and inference code


