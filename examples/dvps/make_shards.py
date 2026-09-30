# write_one_shard.py

import sys
from pathlib import Path
import pyarrow.parquet as pq

src = Path(sys.argv[1])
out = Path(sys.argv[2])

start_row = int(sys.argv[3])
num_rows = int(sys.argv[4])

pf = pq.ParquetFile(src)

writer = None
global_row = 0
written = 0

try:
    for rg_idx in range(pf.num_row_groups):
        rg_rows = pf.metadata.row_group(rg_idx).num_rows
        rg_start = global_row
        rg_end = global_row + rg_rows

        wanted_start = start_row
        wanted_end = start_row + num_rows

        # No intersection with requested shard
        if rg_end <= wanted_start:
            global_row = rg_end
            continue

        if rg_start >= wanted_end:
            break

        table = pf.read_row_group(rg_idx)

        # Intersection within this row group
        local_start = max(wanted_start - rg_start, 0)
        local_end = min(wanted_end - rg_start, rg_rows)
        take = local_end - local_start

        if take > 0:
            piece = table.slice(local_start, take)

            if writer is None:
                out.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(
                    out,
                    piece.schema,
                    compression="zstd",
                )

            writer.write_table(piece)
            written += piece.num_rows

        del table
        global_row = rg_end

finally:
    if writer is not None:
        writer.close()

print(f"{out} -> {written:,} rows")
