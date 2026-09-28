#!/bin/bash

BASE=/net/storage/pr3/plgrid/plggdvps/datasets/BigEarthNet.txt/benx_llamafactory_all_terramind

SRC="$BASE/train.parquet"
OUT="$BASE/train_shards"

TOTAL=4674281
SHARD=100000

mkdir -p "$OUT"

i=0
start=0

while [ "$start" -lt "$TOTAL" ]; do
    printf -v idx "%05d" "$i"

    remaining=$((TOTAL - start))

    if [ "$remaining" -lt "$SHARD" ]; then
        count="$remaining"
    else
        count="$SHARD"
    fi

    echo "Writing shard $idx: rows $start .. $((start + count - 1))"

    python make_shards.py \
        "$SRC" \
        "$OUT/train-$idx.parquet" \
        "$start" \
        "$count"

    if [ $? -ne 0 ]; then
        echo "Shard $idx failed"
        exit 1
    fi

    start=$((start + count))
    i=$((i + 1))
done
