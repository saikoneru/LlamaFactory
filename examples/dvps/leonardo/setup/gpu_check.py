"""GPU sanity check for the Leonardo dvps_train env; run with torchrun --nproc_per_node=4."""
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
local = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local)
dev = torch.device("cuda", local)
props = torch.cuda.get_device_properties(dev)
lines = [f"rank {rank}/{world} {props.name} sm_{props.major}{props.minor} {props.total_memory / 2**30:.0f} GiB"]

# bf16 matmul throughput
a = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
b = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
for _ in range(3):
    a @ b
torch.cuda.synchronize()
t = time.perf_counter()
for _ in range(20):
    c = a @ b
torch.cuda.synchronize()
tflops = 20 * 2 * 8192**3 / (time.perf_counter() - t) / 1e12
lines.append(f"  bf16 matmul: {tflops:.0f} TFLOP/s, finite={bool(torch.isfinite(c).all())}")

# SDPA: flash backend explicitly, compared with the math reference
q, k, v = (torch.randn(4, 28, 2048, 128, device=dev, dtype=torch.bfloat16) for _ in range(3))
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    out_flash = F.scaled_dot_product_attention(q, k, v, is_causal=True)
with sdpa_kernel(SDPBackend.MATH):
    out_math = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=True)
err = (out_flash.float() - out_math).abs().max().item()
lines.append(f"  SDPA flash backend ok, max |flash - math| = {err:.3e}")

# NCCL all-reduce
x = torch.full((64 * 2**20,), float(rank + 1), device=dev)  # 256 MiB fp32
dist.barrier()
torch.cuda.synchronize()
t = time.perf_counter()
dist.all_reduce(x)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t
expected = world * (world + 1) / 2
lines.append(
    f"  NCCL all_reduce 256 MiB: {elapsed * 1e3:.1f} ms, correct={bool(torch.all(x == expected))}"
)

for r in range(world):
    if r == rank:
        print("\n".join(lines), flush=True)
    dist.barrier()
if rank == 0:
    print(f"torch {torch.__version__} | cuda {torch.version.cuda} | nccl {torch.cuda.nccl.version()} | GPU CHECK PASSED")
dist.destroy_process_group()
