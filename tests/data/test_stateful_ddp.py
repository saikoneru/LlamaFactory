#!/usr/bin/env python3

import argparse
import os
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader


DATASET_SIZE = 10_000
BATCH_SIZE = 8
CHECKPOINT_BATCH = 100
NUM_WORKERS = 8
SEED = 12345


class TestDataset(Dataset):
    def __init__(self, size: int):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {
            "id": index,
            "value": index * 10,
        }


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def batch_to_ids(batch):
    ids = batch["id"]
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return list(ids)


def setup_distributed():
    dist.init_process_group(backend="gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    return rank, world_size, local_rank


def cleanup_distributed():
    dist.barrier()
    dist.destroy_process_group()


def create_loader(
    rank: int,
    world_size: int,
    epoch: int = 0,
):
    dataset = TestDataset(DATASET_SIZE)

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=SEED,
        drop_last=False,
    )

    sampler.set_epoch(epoch)

    loader = StatefulDataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )

    return loader, sampler


def compare_sequences(expected, actual):
    if expected == actual:
        return True, None

    max_len = max(len(expected), len(actual))

    for i in range(max_len):
        exp = expected[i] if i < len(expected) else None
        act = actual[i] if i < len(actual) else None

        if exp != act:
            return False, (i, exp, act)

    return False, None


def print_rank(rank, text):
    print(f"[rank {rank}] {text}", flush=True)


def run_test():
    rank, world_size, local_rank = setup_distributed()

    if world_size < 2:
        raise RuntimeError(
            "Run this with at least 2 processes, e.g.:\n"
            "torchrun --standalone --nproc_per_node=2 "
            "test_stateful_ddp.py"
        )

    if rank == 0:
        print("=" * 100)
        print("DDP StatefulDataLoader Resume Test")
        print("=" * 100)
        print(f"torch       : {torch.__version__}")
        print(f"world size  : {world_size}")
        print(f"dataset size: {DATASET_SIZE}")
        print(f"batch size  : {BATCH_SIZE}")
        print(f"workers     : {NUM_WORKERS}")
        print(f"checkpoint  : batch {CHECKPOINT_BATCH}")
        print("=" * 100)

    dist.barrier()

    tmp_root = Path(
        os.environ.get(
            "STATEFUL_TEST_DIR",
            "/tmp/stateful_ddp_test",
        )
    )

    if rank == 0:
        shutil.rmtree(tmp_root, ignore_errors=True)
        tmp_root.mkdir(parents=True, exist_ok=True)

    dist.barrier()

    # ============================================================
    # 1. Uninterrupted baseline
    # ============================================================

    print_rank(rank, "Running uninterrupted baseline")

    set_all_seeds(SEED)

    baseline_loader, baseline_sampler = create_loader(
        rank=rank,
        world_size=world_size,
        epoch=0,
    )

    baseline = []

    for batch_idx, batch in enumerate(baseline_loader):
        ids = batch_to_ids(batch)
        baseline.append(ids)

        if batch_idx < 2:
            print_rank(
                rank,
                f"baseline batch {batch_idx:04d}: {ids}",
            )

    print_rank(
        rank,
        f"baseline batches: {len(baseline)}",
    )

    # ============================================================
    # 2. Run until checkpoint
    # ============================================================

    dist.barrier()

    print_rank(rank, "Running checkpoint phase")

    set_all_seeds(SEED)

    save_loader, save_sampler = create_loader(
        rank=rank,
        world_size=world_size,
        epoch=0,
    )

    iterator = iter(save_loader)

    before = []

    for batch_idx in range(CHECKPOINT_BATCH):
        batch = next(iterator)
        ids = batch_to_ids(batch)
        before.append(ids)

        if batch_idx >= CHECKPOINT_BATCH - 3:
            print_rank(
                rank,
                f"before checkpoint {batch_idx:04d}: {ids}",
            )

    # ============================================================
    # Save state PER RANK
    # ============================================================

    state = save_loader.state_dict()

    state_file = tmp_root / f"dataloader_rank{rank}.pt"

    torch.save(
        {
            "rank": rank,
            "world_size": world_size,
            "epoch": 0,
            "checkpoint_batch": CHECKPOINT_BATCH,
            "dataloader": state,
        },
        state_file,
    )

    print_rank(
        rank,
        f"saved state -> {state_file}",
    )

    del iterator
    del save_loader
    del save_sampler

    dist.barrier()

    # ============================================================
    # 3. Simulate restart
    # ============================================================

    set_all_seeds(999999)

    resume_loader, resume_sampler = create_loader(
        rank=rank,
        world_size=world_size,
        epoch=0,
    )

    checkpoint = torch.load(
        state_file,
        map_location="cpu",
        weights_only=False,
    )

    assert checkpoint["rank"] == rank
    assert checkpoint["world_size"] == world_size
    assert checkpoint["epoch"] == 0

    resume_loader.load_state_dict(
        checkpoint["dataloader"]
    )

    # ============================================================
    # 4. Resume
    # ============================================================

    after = []

    for relative_idx, batch in enumerate(resume_loader):
        ids = batch_to_ids(batch)
        after.append(ids)

        if relative_idx < 5:
            absolute_idx = CHECKPOINT_BATCH + relative_idx

            print_rank(
                rank,
                f"resumed batch {absolute_idx:04d}: {ids}",
            )

    combined = before + after

    passed, mismatch = compare_sequences(
        baseline,
        combined,
    )

    print_rank(
        rank,
        (
            f"baseline={len(baseline)} "
            f"before={len(before)} "
            f"after={len(after)} "
            f"combined={len(combined)}"
        ),
    )

    if passed:
        print_rank(rank, "RESULT: PASS")
    else:
        print_rank(rank, "RESULT: FAIL")

        if mismatch is not None:
            idx, expected, actual = mismatch

            print_rank(
                rank,
                f"first mismatch at batch {idx}",
            )
            print_rank(
                rank,
                f"expected: {expected}",
            )
            print_rank(
                rank,
                f"actual  : {actual}",
            )

    # ============================================================
    # 5. Cross-rank sanity checks
    # ============================================================

    result_tensor = torch.tensor(
        [1 if passed else 0],
        dtype=torch.int64,
    )

    dist.all_reduce(
        result_tensor,
        op=dist.ReduceOp.MIN,
    )

    all_passed = bool(result_tensor.item())

    # Check that ranks are not receiving the same samples around
    # the checkpoint boundary.
    boundary_ids = baseline[CHECKPOINT_BATCH]

    gathered = [None for _ in range(world_size)]

    dist.all_gather_object(
        gathered,
        boundary_ids,
    )

    if rank == 0:
        print()
        print("=" * 100)
        print("CROSS-RANK CHECK")
        print("=" * 100)

        for r, ids in enumerate(gathered):
            print(
                f"rank {r} baseline batch "
                f"{CHECKPOINT_BATCH:04d}: {ids}"
            )

        flattened = []

        for ids in gathered:
            flattened.extend(ids)

        duplicates = (
            len(flattened)
            != len(set(flattened))
        )

        print()
        print(
            "duplicate sample IDs across ranks in this batch:",
            duplicates,
        )

        print()
        print("=" * 100)
        print("FINAL RESULT")
        print("=" * 100)

        if all_passed:
            print("PASS")
            print(
                "Every rank resumed exactly at the correct "
                "StatefulDataLoader position."
            )
        else:
            print("FAIL")
            print(
                "At least one rank did not reproduce its "
                "uninterrupted sequence."
            )

    cleanup_distributed()


if __name__ == "__main__":
    run_test()