"""Read jackyhate/text-to-image-2M JPEG + JSON(prompt) WebDataset records.

The JPEG is an assistant output target, never an Omni input image. Keep images
encoded until UniLIP target preprocessing; no extraction to individual files or
extra WebDataset dependency is needed.
"""

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from datasets import Features, List, Value

from ..unilip_generation import convert_sharegpt_target


if TYPE_CHECKING:
    from ...hparams import DataArguments
    from ..parser import DatasetAttr


REPO_ID = "jackyhate/text-to-image-2M"
NUM_512_SHARDS = 46
NUM_SHARDS = NUM_512_SHARDS + 1


def resolve_data_files(dataset_attr: "DatasetAttr", dataset_dir: str):
    """Accept default upstream URLs or explicit local/remote tar paths/globs."""
    data_files = dataset_attr.data_files
    if data_files is None:
        if dataset_attr.dataset_name != REPO_ID:
            raise ValueError("WebDataset entries other than text-to-image-2M require explicit data_files.")
        return [
            f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/data_512_2M/data_{i:06d}.tar"
            for i in range(NUM_512_SHARDS)
        ] + [f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/data_1024_10K/data_000000.tar"]

    def resolve(value):
        if isinstance(value, list):
            if not value:
                raise ValueError("WebDataset data_files must not be empty.")
            return [resolve(item) for item in value]
        if not isinstance(value, str) or not value:
            raise ValueError("WebDataset data_files must contain tar paths, URLs or globs.")
        value = os.path.expanduser(os.path.expandvars(value))
        if "://" in value or os.path.isabs(value):
            return value
        return os.path.join(dataset_dir, value)

    if isinstance(data_files, dict):
        if dataset_attr.split not in data_files:
            raise ValueError(f"WebDataset data_files has no split {dataset_attr.split!r}.")
        return {split: resolve(paths) for split, paths in data_files.items()}
    return resolve(data_files)


@dataclass
class TextToImage2MDatasetConverter:
    dataset_attr: "DatasetAttr"
    data_args: "DataArguments"

    # Streaming map cannot infer its output schema without consuming samples.
    # Declare it so the normal SFT tokenization path also works lazily.
    features: ClassVar[Features] = Features(
        {
            "_prompt": List({"role": Value("string"), "content": Value("string")}),
            "_response": List({"role": Value("string"), "content": Value("string")}),
            "_system": Value("string"),
            "_tools": Value("string"),
            "_images": Value("null"),
            "_videos": Value("null"),
            "_audios": Value("null"),
            "_encoders": {},
            "_terramind": Value("null"),
            "_unilip_target": {"bytes": Value("binary"), "height": Value("int64"), "width": Value("int64")},
        }
    )

    def __call__(self, example):
        key = example.get("__key__", "<unknown>")
        metadata = example.get("json")
        if isinstance(metadata, (str, bytes)):
            metadata = json.loads(metadata)
        prompt = metadata.get("prompt") if isinstance(metadata, dict) else None
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"text-to-image-2M sample {key!r} requires a nonempty json.prompt string.")
        image = example.get("jpg")
        if image is None:
            raise ValueError(f"text-to-image-2M sample {key!r} is missing its jpg target.")
        response = [{"role": "assistant", "content": "<unilip>"}]
        target = convert_sharegpt_target(
            response,
            image,
            media_dir=None,
            buckets=self.data_args.unilip_resolution_buckets,
            resize_mode=self.data_args.unilip_resize_mode,
            warn_on_resize=self.data_args.unilip_warn_on_resize,
            preserve_matching_bytes=True,
        )
        return {
            # Match the demo: the entered caption is the user message itself.
            "_prompt": [{"role": "user", "content": prompt.strip()}],
            "_response": response,
            "_system": "",
            "_tools": "",
            "_images": None,
            "_videos": None,
            "_audios": None,
            "_encoders": {},
            "_terramind": None,
            "_unilip_target": target,
        }
