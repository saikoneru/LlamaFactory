#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4
export FORCE_TORCHRUN=1

NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_DEBUG=INFO llamafactory-cli train /net/home/plgrid/plgskoneru/LLaMA-Factory/examples/dvps/finetune_dvpsfm_freeze.yaml
