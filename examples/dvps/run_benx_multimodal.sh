#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export FORCE_TORCHRUN=1

NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_DEBUG=INFO llamafactory-cli train /net/home/plgrid/plgskoneru/LLaMA-Factory/examples/dvps/benx_lora_multimodal.yaml
