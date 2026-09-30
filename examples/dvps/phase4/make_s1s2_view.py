"""Write an S1/S2-only view of benx_llamafactory_all_terramind_final.

Each output parquet keeps ``messages`` and ``terramind`` and drops ``images``.
``<image>`` is removed from the message text so the loader does not require a
PNG. The TerraMind rasters are copied unchanged. Source files are not modified.

Train shards go in a directory of their own because LLaMA-Factory treats every
file in a dataset directory as one split.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SRC = Path(
    "/net/storage/pr3/plgrid/plggdvps/datasets/BigEarthNet.txt/"
    "benx_llamafactory_all_terramind_final"
)
DST = Path(
    "/net/storage/pr3/plgrid/plggdvps/datasets/BigEarthNet.txt/"
    "benx_llamafactory_s1s2"
)


def destinations(src: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for path in sorted(src.glob("*.parquet")):
        if path.name.startswith("train-"):
            dest = DST / "train" / path.name
        else:
            dest = DST / path.name
        pairs.append((path, dest))
    return pairs


def _strip_table(table: pa.Table) -> pa.Table:
    messages = table.column("messages").to_pylist()
    for conversation in messages:
        for message in conversation:
            message["content"] = message["content"].replace("<image>", "")
            if "<image>" in message["content"]:
                raise ValueError("An <image> marker survived replacement.")
        markers = sum(message["content"].count("<terramind>") for message in conversation)
        if markers != 1:
            raise ValueError(f"Expected one <terramind> marker, found {markers}.")
    terramind = table.schema.field("terramind")
    return pa.table(
        {
            "messages": pa.array(messages, type=table.schema.field("messages").type),
            "terramind": table.column("terramind"),
        },
        schema=pa.schema(
            [
                table.schema.field("messages"),
                terramind,
            ]
        ),
    )


def convert(src: str, dest: str) -> str:
    source = Path(src)
    target = Path(dest)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    partial.unlink(missing_ok=True)
    if target.is_file():
        existing = pq.ParquetFile(target)
        incoming = pq.ParquetFile(source)
        if existing.metadata.num_rows == incoming.metadata.num_rows:
            return f"skip {target.name} ({existing.metadata.num_rows} rows)"

    reader = pq.ParquetFile(source)
    compression = reader.metadata.row_group(0).column(0).compression.lower()
    if compression == "uncompressed":
        compression = "none"
    writer = None
    written = 0
    try:
        for batch in reader.iter_batches(batch_size=128, columns=["messages", "terramind"]):
            table = _strip_table(pa.Table.from_batches([batch]))
            if writer is None:
                writer = pq.ParquetWriter(partial, table.schema, compression=compression)
            writer.write_table(table)
            written += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    if written != reader.metadata.num_rows:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"{source.name}: wrote {written} of {reader.metadata.num_rows} rows.")
    partial.replace(target)
    return f"wrote {target.name} ({written} rows)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only", nargs="*", default=None, help="Basenames to convert.")
    args = parser.parse_args()

    pairs = destinations(SRC)
    if args.only:
        wanted = set(args.only)
        pairs = [(src, dest) for src, dest in pairs if src.name in wanted]
    if not pairs:
        raise SystemExit("No parquet files selected.")

    print(f"Converting {len(pairs)} file(s) with {args.workers} workers.", flush=True)
    if args.workers == 1:
        for src, dest in pairs:
            print(convert(str(src), str(dest)), flush=True)
        return

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(convert, str(src), str(dest)): src.name for src, dest in pairs}
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
