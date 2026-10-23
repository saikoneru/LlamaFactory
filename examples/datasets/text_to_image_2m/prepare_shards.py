#!/usr/bin/env python
"""Repack staged text-to-image-2M tar files for efficient multi-GPU streaming.

The upstream 47 shards do not divide evenly across typical GPU counts. Native
Hugging Face streaming then falls back to reading and filtering records on every
rank. Repacking once to e.g. 256 balanced shards enables shard-level partitioning.
Images and JSON are copied byte-for-byte, with no image extraction or resizing.

python examples/datasets/text_to_image_2m/prepare_shards.py \
    --input /staged/data_512_2M/*.tar /staged/data_1024_10K/*.tar \
    --output-dir /scratch/t2i2m_shards --num-shards 256
"""

import argparse
import copy
import glob
import json
import tarfile
from contextlib import ExitStack
from pathlib import Path


def iter_samples(archive):
    """Group contiguous jpg/json members; reject incomplete or duplicate pairs."""
    key, parts = None, {}
    for member in archive:
        if not member.isfile() or "." not in member.name:
            continue
        current, extension = member.name.rsplit(".", 1)
        if extension not in {"jpg", "json"}:
            continue
        if current != key and parts:
            if set(parts) != {"jpg", "json"}:
                raise ValueError(f"Incomplete sample {key!r} in {archive.name}")
            yield parts
            parts = {}
        key = current
        if extension in parts:
            raise ValueError(f"Duplicate {extension} for {key!r} in {archive.name}")
        parts[extension] = member
        # TarInfo lists otherwise retain millions of metadata objects.
        archive.members.clear()
    if parts:
        if set(parts) != {"jpg", "json"}:
            raise ValueError(f"Incomplete sample {key!r} in {archive.name}")
        yield parts


def prepare(input_paths, output_dir, num_shards):
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    paths = [Path(path).resolve() for path in input_paths]
    if len(paths) != len(set(paths)):
        raise ValueError("An input archive was specified more than once")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")
    total = 0
    # Uncompressed local tar files let this metadata pass seek over JPEG data.
    for path in paths:
        with tarfile.open(path, "r:") as archive:
            count = sum(1 for _ in iter_samples(archive))
            total += count
            print(f"{path}: {count} samples", flush=True)
    if total < num_shards:
        raise ValueError(f"{total} samples cannot fill {num_shards} nonempty shards")
    output_dir.mkdir(parents=True, exist_ok=True)
    quotient, remainder = divmod(total, num_shards)
    counts = [quotient + (index < remainder) for index in range(num_shards)]
    output_index, written, remaining = 0, 0, counts[0]
    names = [f"data_{index:06d}.tar" for index in range(num_shards)]
    with ExitStack() as stack:
        writer = stack.enter_context(tarfile.open(output_dir / names[0], "w"))
        for path in paths:
            with tarfile.open(path, "r:") as archive:
                for parts in iter_samples(archive):
                    if remaining == 0:
                        stack.close()
                        output_index += 1
                        writer = stack.enter_context(tarfile.open(output_dir / names[output_index], "w"))
                        remaining = counts[output_index]
                    for extension, member in parts.items():
                        renamed = copy.copy(member)
                        renamed.name = f"{written:012d}.{extension}"
                        renamed.pax_headers = {}
                        with archive.extractfile(member) as stream:
                            writer.addfile(renamed, stream)
                    remaining -= 1
                    written += 1
    assert written == total and output_index == num_shards - 1
    info = {
        "text_to_image_2m": {
            "webdataset_url": "jackyhate/text-to-image-2M",
            "formatting": "text_to_image_2m",
            "data_files": {"train": names},
        }
    }
    (output_dir / "dataset_info.json").write_text(json.dumps(info, indent=2))
    report = {
        "samples": total,
        "num_shards": num_shards,
        "samples_per_shard": counts,
        "inputs": [str(path) for path in paths],
    }
    (output_dir / "shard_manifest.json").write_text(json.dumps(report, indent=2))
    print(f"Prepared {total} samples in {num_shards} shards under {output_dir}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=256)
    args = parser.parse_args()
    paths = []
    for pattern in args.input:
        matches = sorted(glob.glob(pattern))
        if not matches:
            parser.error(f"No input archives match {pattern!r}")
        paths.extend(matches)
    prepare(paths, args.output_dir, args.num_shards)


if __name__ == "__main__":
    main()
