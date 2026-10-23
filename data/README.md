The [dataset_info.json](dataset_info.json) contains all available datasets. If you are using a custom dataset, please **make sure** to add a *dataset description* in `dataset_info.json` and specify `dataset: dataset_name` before training to use it.

The `dataset_info.json` file should be put in the `dataset_dir` directory. You can change `dataset_dir` to use another directory. The default value is `./data`.

Currently we support datasets in **alpaca** and **sharegpt** format. Allowed file types include json, jsonl, csv, parquet, arrow.

```json
"dataset_name": {
  "hf_hub_url": "the name of the dataset repository on the Hugging Face hub. (if specified, ignore script_url, file_name and cloud_file_name)",
  "ms_hub_url": "the name of the dataset repository on the Model Scope hub. (if specified, ignore script_url, file_name and cloud_file_name)",
  "script_url": "the name of the directory containing a dataset loading script. (if specified, ignore file_name and cloud_file_name)",
  "cloud_file_name": "the name of the dataset file in s3/gcs cloud storage. (if specified, ignore file_name)",
  "file_name": "the name of the dataset folder or dataset file in this directory. (required if above are not specified)",
  "formatting": "the format of the dataset. (optional, default: alpaca, can be chosen from {alpaca, sharegpt})",
  "ranking": "whether the dataset is a preference dataset or not. (default: False)",
  "subset": "the name of the subset. (optional, default: None)",
  "split": "the name of dataset split to be used. (optional, default: train)",
  "folder": "the name of the folder of the dataset repository on the Hugging Face hub. (optional, default: None)",
  "num_samples": "the number of samples in the dataset to be used. (optional, default: None)",
  "columns (optional)": {
    "prompt": "the column name in the dataset containing the prompts. (default: instruction)",
    "query": "the column name in the dataset containing the queries. (default: input)",
    "response": "the column name in the dataset containing the responses. (default: output)",
    "history": "the column name in the dataset containing the histories. (default: None)",
    "messages": "the column name in the dataset containing the messages. (default: conversations)",
    "system": "the column name in the dataset containing the system prompts. (default: None)",
    "tools": "the column name in the dataset containing the tool description. (default: None)",
    "images": "the column name in the dataset containing the image inputs. (default: None)",
    "videos": "the column name in the dataset containing the videos inputs. (default: None)",
    "audios": "the column name in the dataset containing the audios inputs. (default: None)",
    "chosen": "the column name in the dataset containing the chosen answers. (default: None)",
    "rejected": "the column name in the dataset containing the rejected answers. (default: None)",
    "kto_tag": "the column name in the dataset containing the kto tags. (default: None)"
  },
  "tags (optional, used for the sharegpt format)": {
    "role_tag": "the key in the message represents the identity. (default: from)",
    "content_tag": "the key in the message represents the content. (default: value)",
    "user_tag": "the value of the role_tag represents the user. (default: human)",
    "assistant_tag": "the value of the role_tag represents the assistant. (default: gpt)",
    "observation_tag": "the value of the role_tag represents the tool results. (default: observation)",
    "function_tag": "the value of the role_tag represents the function call. (default: function_call)",
    "system_tag": "the value of the role_tag represents the system prompt. (default: system, can override system column)"
  }
}
```

## text-to-image-2M WebDataset

`examples/datasets/text_to_image_2m/` contains a native dataset registration,
an Omni-3B training config, a CPU preview, and an optional local repacking tool
for [jackyhate/text-to-image-2M](https://huggingface.co/datasets/jackyhate/text-to-image-2M).
The CINECA recipe reads the already downloaded `data_512_2M` and
`data_1024_10K` tar folders through `T2I_2M_ROOT`. It uses Hugging Face's built-in WebDataset reader;
there is no separate `webdataset` package dependency.

Each record contains `jpg` image bytes and `json.prompt`. The dataset adapter
uses the caption itself as the user request, matching the demo's prompt field,
with an assistant response `<res_512x512><unilip>` or `<res_1024x1024><unilip>`, and an independent
image target. The resolution token follows the actual image dimensions.
The target is never added to input `images` or to an input `<image>` marker.
Matching RGB JPEGs keep their original bytes and pixel dimensions. Other sizes
use the checkpoint's supported bucket/resize policy.

The updated five-layer Omni-3B model explicitly separates output and vision
encoder sizes, keeping all pretrained codec weights:

| Target and generated output | InternViT input | Latent grid | Resolution token |
| --- | --- | --- | --- |
| 512×512 | 448×448 | 16×16 | `<res_512x512>` |
| 1024×1024 | 896×896 | 32×32 | `<res_1024x1024>` |

This uses the DC-AE decoder's native 32-pixel output stride. Existing models
without the new `tokenizer_config.output_pixels_per_latent` setting retain
their previous pixel geometry. The 1024 bucket has four times as many latent
positions as the 512 bucket and needs more compute and memory.

Run from the LlamaFactory directory:

```bash
# Install the UniLIP decoder dependency and the validated Transformers version.
python -m pip install -e '.[unilip]'

# Point to the dataset already downloaded on CINECA. No dataset download is needed.
export T2I_2M_ROOT=/path/to/staged/text-to-image-2M

# Check tokenizer labels, output targets and mixed-resolution collation on CPU.
CUDA_VISIBLE_DEVICES="" python examples/datasets/text_to_image_2m/preview.py --samples 2

# The model is loaded from the private skoneru/omni3b-unilip-connector5 repository.
# Authenticate with Hugging Face on the login node before the first model download.
llamafactory-cli train examples/datasets/text_to_image_2m/train_omni3b.yaml
```

To use a previously downloaded local copy of the model:

```bash
llamafactory-cli train examples/datasets/text_to_image_2m/train_omni3b.yaml \
  model_name_or_path=/path/to/omni3b_unilip_connector5
```

For the provided CINECA account/environment and dataset snapshot, submit a
four-GPU, 48-hour job with
`sbatch examples/datasets/text_to_image_2m/train_cineca.sbatch`.
It uses `boost_qos_lprod` (normal production is limited to 24 hours), one
Slurm task launching four torchrun workers, and a distinct output directory
ending in the job ID. `UNILIP_OUTPUT_DIR` can override that output path.

Alternatively set `data_files` in the dataset entry to explicit local tar
paths/globs or URLs. Relative paths resolve against `dataset_dir`. To check a
few local shards, pass `--data-files /absolute/path/to/data_*.tar` to the preview.
Omitting `data_files` switches the adapter to upstream streaming URLs; the
included CINECA manifest explicitly selects local tar files instead.
Streaming keeps sample shuffling enabled with a bounded per-worker buffer;
`max_steps` controls run length. `preprocessing_num_workers` does not parallelize
streaming; `dataloader_num_workers` controls that work.

For efficient distributed loading, the number of shards should divide evenly
across ranks, with enough shards for the data-loader workers. With the original
47 shards and typical GPU counts, Hugging Face falls back to sample-level
partitioning, which is correct but repeats tar reads on different ranks. The
optional CPU tool creates balanced shards for shard-level partitioning:

```bash
python examples/datasets/text_to_image_2m/prepare_shards.py \
  --input "$T2I_2M_ROOT"/data_512_2M/*.tar "$T2I_2M_ROOT"/data_1024_10K/*.tar \
  --output-dir /path/to/repacked --num-shards 256
llamafactory-cli train examples/datasets/text_to_image_2m/train_omni3b.yaml \
  dataset_dir=/path/to/repacked model_name_or_path=/path/to/omni3b_unilip_connector5
```

Choose a shard count divisible by the total GPU count. Repacking needs storage
for another copy of the tar data; it copies each JPEG and JSON unchanged and
writes a ready-to-use `dataset_info.json`. It does not resize or extract images.
The training profile remains image-loss-only: the resolution token is supplied
from target metadata. Learning to autonomously predict resolution/trigger tokens
also requires language-model supervision in a later training profile.

## Alpaca Format

### Supervised Fine-Tuning Dataset

* [Example dataset](alpaca_en_demo.json)

In supervised fine-tuning, the `instruction` column will be concatenated with the `input` column and used as the user prompt, then the user prompt would be `instruction\ninput`. The `output` column represents the model response.

For reasoning models, if the dataset contains chain-of-thought (CoT), the CoT needs to be placed in the model responses, such as `<think>cot</think>output`.

The `system` column will be used as the system prompt if specified.

The `history` column is a list consisting of string tuples representing prompt-response pairs in the history messages. Note that the responses in the history **will also be learned by the model** in supervised fine-tuning.

```json
[
  {
    "instruction": "user instruction (required)",
    "input": "user input (optional)",
    "output": "model response (required)",
    "system": "system prompt (optional)",
    "history": [
      ["user instruction in the first round (optional)", "model response in the first round (optional)"],
      ["user instruction in the second round (optional)", "model response in the second round (optional)"]
    ]
  }
]
```

Regarding the above dataset, the *dataset description* in `dataset_info.json` should be:

```json
"dataset_name": {
  "file_name": "data.json",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "system": "system",
    "history": "history"
  }
}
```

> [!TIP]  
> If the model has reasoning capabilities (e.g. Qwen3) but the dataset does not contain chain-of-thought (CoT), LLaMA-Factory will automatically add empty CoT to the data. When `enable_thinking` is `True` (slow thinking, by default), the empty CoT will be added to the model responses and loss computation will be considered; otherwise (fast thinking), it will be added to the user prompts and loss computation will be ignored. Please keep the `enable_thinking` parameter consistent during training and inference.
>
> If you want to train data containing CoT with slow thinking and data without CoT with fast thinking, you can set `enable_thinking` to `None`. However, this feature is relatively complicated and should be used with caution.

### Pre-training Dataset

- [Example dataset](c4_demo.jsonl)

In pre-training, only the `text` column will be used for model learning.

```json
[
  {"text": "document"},
  {"text": "document"}
]
```

Regarding the above dataset, the *dataset description* in `dataset_info.json` should be:

```json
"dataset_name": {
  "file_name": "data.json",
  "columns": {
    "prompt": "text"
  }
}
```

### Preference Dataset

Preference datasets are used for reward modeling, DPO training, ORPO and SimPO training.

It requires a better response in `chosen` column and a worse response in `rejected` column.

```json
[
  {
    "instruction": "user instruction (required)",
    "input": "user input (optional)",
    "chosen": "chosen answer (required)",
    "rejected": "rejected answer (required)"
  }
]
```

Regarding the above dataset, the *dataset description* in `dataset_info.json` should be:

```json
"dataset_name": {
  "file_name": "data.json",
  "ranking": true,
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "chosen": "chosen",
    "rejected": "rejected"
  }
}
```

### KTO Dataset

An additional column `kto_tag` is required. Please refer to the [sharegpt](#sharegpt-format) format for details.

### Multimodal Image Dataset

An additional column `images` is required. Please refer to the [sharegpt](#sharegpt-format) format for details.

### Multimodal Video Dataset

An additional column `videos` is required. Please refer to the [sharegpt](#sharegpt-format) format for details.

### Multimodal Audio Dataset

An additional column `audios` is required. Please refer to the [sharegpt](#sharegpt-format) format for details.

## Sharegpt Format

### Supervised Fine-Tuning Dataset

- [Example dataset](glaive_toolcall_en_demo.json)

Compared to the alpaca format, the sharegpt format allows the datasets have **more roles**, such as human, gpt, observation and function. They are presented in a list of objects in the `conversations` column.

Note that the human and observation should appear in odd positions, while gpt and function should appear in even positions. The gpt and function will be learned by the model.

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "user instruction"
      },
      {
        "from": "function_call",
        "value": "tool arguments"
      },
      {
        "from": "observation",
        "value": "tool result"
      },
      {
        "from": "gpt",
        "value": "model response"
      }
    ],
    "system": "system prompt (optional)",
    "tools": "tool description (optional)"
  }
]
```

Regarding the above dataset, the *dataset description* in `dataset_info.json` should be:

```json
"dataset_name": {
  "file_name": "data.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "conversations",
    "system": "system",
    "tools": "tools"
  }
}
```

### Pre-training Dataset

Not yet supported, please use the [alpaca](#alpaca-format) format.

### Preference Dataset

- [Example dataset](dpo_en_demo.json)

Preference datasets in sharegpt format also require a better message in `chosen` column and a worse message in `rejected` column.

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "user instruction"
      },
      {
        "from": "gpt",
        "value": "model response"
      },
      {
        "from": "human",
        "value": "user instruction"
      }
    ],
    "chosen": {
      "from": "gpt",
      "value": "chosen answer (required)"
    },
    "rejected": {
      "from": "gpt",
      "value": "rejected answer (required)"
    }
  }
]
```

Regarding the above dataset, the *dataset description* in `dataset_info.json` should be:

```json
"dataset_name": {
  "file_name": "data.json",
  "formatting": "sharegpt",
  "ranking": true,
  "columns": {
    "messages": "conversations",
    "chosen": "chosen",
    "rejected": "rejected"
  }
}
```

### KTO Dataset

- [Example dataset](kto_en_demo.json)

KTO datasets require a extra `kto_tag` column containing the boolean human feedback.

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "user instruction"
      },
      {
        "from": "gpt",
        "value": "model response"
      }
    ],
    "kto_tag": "human feedback [true/false] (required)"
  }
]
```

Regarding the above dataset, the *dataset description* in `dataset_info.json` should be:

```json
"dataset_name": {
  "file_name": "data.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "conversations",
    "kto_tag": "kto_tag"
  }
}
```

### Multimodal Image Dataset

- [Example dataset](mllm_demo.json)

Multimodal image datasets require an `images` column containing the paths to the input images.

The number of images should be identical to the `<image>` tokens in the conversations.

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "<image>user instruction"
      },
      {
        "from": "gpt",
        "value": "model response"
      }
    ],
    "images": [
      "image path (required)"
    ]
  }
]
```

Regarding the above dataset, the *dataset description* in `dataset_info.json` should be:

```json
"dataset_name": {
  "file_name": "data.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "conversations",
    "images": "images"
  }
}
```

### Multimodal Video Dataset

- [Example dataset](mllm_video_demo.json)

Multimodal video datasets require a `videos` column containing the paths to the input videos.

The number of videos should be identical to the `<video>` tokens in the conversations.

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "<video>user instruction"
      },
      {
        "from": "gpt",
        "value": "model response"
      }
    ],
    "videos": [
      "video path (required)"
    ]
  }
]
```

Regarding the above dataset, the *dataset description* in `dataset_info.json` should be:

```json
"dataset_name": {
  "file_name": "data.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "conversations",
    "videos": "videos"
  }
}
```

### Multimodal Audio Dataset

- [Example dataset](mllm_audio_demo.json)

Multimodal audio datasets require an `audios` column containing the paths to the input audios.

The number of audios should be identical to the `<audio>` tokens in the conversations.

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "<audio>user instruction"
      },
      {
        "from": "gpt",
        "value": "model response"
      }
    ],
    "audios": [
      "audio path (required)"
    ]
  }
]
```

Regarding the above dataset, the *dataset description* in `dataset_info.json` should be:

```json
"dataset_name": {
  "file_name": "data.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "conversations",
    "audios": "audios"
  }
}
```

### OpenAI Format

The openai format is simply a special case of the sharegpt format, where the first message may be a system prompt.

```json
[
  {
    "messages": [
      {
        "role": "system",
        "content": "system prompt (optional)"
      },
      {
        "role": "user",
        "content": "user instruction"
      },
      {
        "role": "assistant",
        "content": "model response"
      }
    ]
  }
]
```

Regarding the above dataset, the *dataset description* in `dataset_info.json` should be:

```json
"dataset_name": {
  "file_name": "data.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "messages"
  },
  "tags": {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant",
    "system_tag": "system"
  }
}
```
