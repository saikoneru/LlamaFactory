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

The <image> token allows Llamafactory to infer that the image is given before the text and load the image. Note that <image> and images should be equal length

then exit the cpu job

```bash
exit
```

## Fine-tuning on BigEarthNet.txt

Now, to fine-tune on this data, we need to add the data files to the `dataset_info.json`. For simplicity, we will refer to the one at https://github.com/saikoneru/LlamaFactory/blob/main/data/dataset_info.json . But this could be anywhere and Llamafactory can be told where to load this file from.

