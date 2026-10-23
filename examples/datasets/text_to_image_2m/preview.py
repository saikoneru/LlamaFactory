#!/usr/bin/env python
"""Check a few samples through the real tokenizer and collator, without model weights.

From LlamaFactory: python examples/datasets/text_to_image_2m/preview.py --samples 2
For local shards, add --data-files /path/to/data_*.tar.
"""

import argparse
import io
import json
import tempfile
from itertools import islice
from pathlib import Path

import yaml
from PIL import Image

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.data.unilip_generation import UniLIPDataCollatorWith4DAttentionMask
from llamafactory.hparams import get_train_args
from llamafactory.model import load_tokenizer
from llamafactory.model.loader import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("train_omni3b.yaml"))
    parser.add_argument("--model")
    parser.add_argument("--data-files", nargs="+")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    values = yaml.safe_load(args.config.read_text())
    values.update(
        use_cpu=True,
        report_to="none",
        dataloader_num_workers=0,
        preprocessing_batch_size=min(args.samples, 8),
        buffer_size=min(args.samples, 8),
    )
    if args.model:
        values["model_name_or_path"] = args.model
    with tempfile.TemporaryDirectory(prefix="t2i2m_preview_") as temporary:
        if args.data_files:
            info = {
                "text_to_image_2m": {
                    "webdataset_url": "jackyhate/text-to-image-2M",
                    "formatting": "text_to_image_2m",
                    "data_files": args.data_files,
                }
            }
            manifest = Path(temporary) / "dataset_info.json"
            manifest.write_text(json.dumps(info))
            values["dataset_info"] = str(manifest)
        model_args, data_args, training_args, _, _ = get_train_args(values)
        modules = load_tokenizer(model_args)
        tokenizer = modules["tokenizer"]
        spec = load_config(model_args).vision_decoder
        policy = spec["resolution_policy"]
        data_args.unilip_resolution_buckets = ",".join(f"{h}x{w}" for h, w in policy["supported_sizes"])
        data_args.unilip_resize_mode = policy.get("resize_mode", "center_crop")
        template = get_template_and_fix_tokenizer(tokenizer, data_args)
        datasets = get_dataset(template, model_args, data_args, training_args, "sft", **modules)
        rows = list(islice(datasets["train_dataset"], args.samples))
        if len(rows) != args.samples:
            raise ValueError(f"Only {len(rows)} valid samples were available")
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
        summaries = []
        trigger = tokenizer.convert_tokens_to_ids("<unilip>")
        for index, row in enumerate(rows):
            target = row["unilip_target"]
            size = f"{target['height']}x{target['width']}"
            assert row["images"] == [] and not row["encoders"]
            assert row["input_ids"][-1] == row["labels"][-1] == trigger
            assert row["input_ids"][-2] == spec["resolution_token_ids"][size]
            summary = {
                "size": size,
                "tokens": len(row["input_ids"]),
                "input": tokenizer.decode(row["input_ids"]),
                "labels": tokenizer.decode([x for x in row["labels"] if x != -100]),
            }
            if args.output_dir:
                with Image.open(io.BytesIO(target["bytes"])) as image:
                    image.save(args.output_dir / f"target_{index}_{size}.png")
            summaries.append(summary)
        collator = UniLIPDataCollatorWith4DAttentionMask(
            tokenizer=tokenizer, template=template, processor=modules.get("processor")
        )
        batch = collator(rows)
        groups = [
            {"shape": list(group["target_images"].shape), "rows": group["image_batch_indices"].tolist()}
            for group in batch["unilip_image_groups"]
        ]
        report = {"success": True, "samples": summaries, "image_groups": groups}
        if args.output_dir:
            (args.output_dir / "preview.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
