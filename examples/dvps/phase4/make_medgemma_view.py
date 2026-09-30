#!/usr/bin/env python
"""Write a MedGemma view of a MIMIC-CXR ShareGPT parquet.

The shared MIMIC parquet is written for native Omni vision: the prompt says
``<image>`` and the X-ray lives in the ``images`` column. MedGemma is a
separate encoder with its own marker and its own dataset column, so this
script emits a second parquet that names both honestly:

    messages   <image> -> <medgemma>
    images     -> medgemma   (bytes are copied unchanged)

The source file is never modified; the native Omni RGB control keeps reading
it as-is.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HF_METADATA_KEY = b"huggingface"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--image-placeholder", default="<image>")
    parser.add_argument("--marker", default="<medgemma>")
    parser.add_argument("--messages-column", default="messages")
    parser.add_argument("--image-column", default="images")
    parser.add_argument("--encoder-column", default="medgemma")
    parser.add_argument(
        "--max-report-errors",
        type=int,
        default=20,
        help="How many offending rows to list before giving up.",
    )
    return parser.parse_args()


def build_output_schema(source: pa.Schema, args: argparse.Namespace) -> pa.Schema:
    """Rename the image column and keep its Hugging Face feature type."""
    schema = pa.schema(
        [
            pa.field(args.messages_column, source.field(args.messages_column).type),
            pa.field(args.encoder_column, source.field(args.image_column).type),
        ]
    )

    metadata = dict(source.metadata or {})
    raw = metadata.get(HF_METADATA_KEY)
    if raw is None:
        return schema

    info = json.loads(raw.decode())
    features = info.get("info", {}).get("features", {})
    if args.image_column in features:
        features[args.encoder_column] = features.pop(args.image_column)

    # Keep only the columns this view writes; stale feature entries make
    # datasets reject the file at load time.
    kept = {name.name for name in schema}
    info.get("info", {})["features"] = {
        name: feature for name, feature in features.items() if name in kept
    }

    metadata[HF_METADATA_KEY] = json.dumps(info).encode()
    return schema.with_metadata(metadata)


def rewrite_messages(
    messages: list[list[dict]],
    num_images: list[int],
    args: argparse.Namespace,
    row_offset: int,
    errors: list[str],
) -> list[list[dict]]:
    """Swap the Omni image placeholder for the MedGemma marker, one per row."""
    rewritten = []
    for row, (turns, image_count) in enumerate(zip(messages, num_images)):
        turns = [dict(turn) for turn in turns]
        marker_count = sum(
            turn["content"].count(args.image_placeholder) for turn in turns
        )
        if image_count < 1 or marker_count != image_count:
            if len(errors) < args.max_report_errors:
                errors.append(
                    f"row {row_offset + row}: {marker_count} "
                    f"{args.image_placeholder} marker(s), {image_count} image(s)"
                )
            continue

        for turn in turns:
            turn["content"] = turn["content"].replace(
                args.image_placeholder,
                args.marker,
            )
        rewritten.append(turns)

    return rewritten


def main() -> None:
    args = parse_args()
    source = pq.ParquetFile(args.src)
    out_schema = build_output_schema(source.schema_arrow, args)
    compression = source.metadata.row_group(0).column(0).compression.lower()
    if compression == "uncompressed":
        compression = "none"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    row_offset = 0

    writer = pq.ParquetWriter(args.out, out_schema, compression=compression)
    try:
        for index in range(source.num_row_groups):
            table = source.read_row_group(
                index,
                columns=[args.messages_column, args.image_column],
            )
            images = table.column(args.image_column)
            turns = rewrite_messages(
                table.column(args.messages_column).to_pylist(),
                [0 if value is None else len(value) for value in images.to_pylist()],
                args,
                row_offset,
                errors,
            )
            if errors:
                break

            writer.write_table(
                pa.table(
                    [
                        pa.array(turns, type=out_schema.field(0).type),
                        images.combine_chunks(),
                    ],
                    schema=out_schema,
                )
            )
            row_offset += table.num_rows
            print(f"{args.out.name}: {row_offset}/{source.metadata.num_rows} rows")
    finally:
        writer.close()

    if errors:
        args.out.unlink(missing_ok=True)
        listed = "\n  ".join(errors)
        raise SystemExit(
            f"{args.src} has rows where the {args.image_placeholder} count "
            f"does not match the image count:\n  {listed}"
        )

    print(f"wrote {row_offset} rows to {args.out}")


if __name__ == "__main__":
    main()
