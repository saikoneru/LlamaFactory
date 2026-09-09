# Env setup

1. On GH200 node, simply `ml ML-bundle/25.10` and call `uv sync --no-cache --no-managed-python` . 

2. There is a need for manual install of `flash-attention-3`, to do so go to the github repo and follow the instructions for installing, but generally for gh200 (you need to have your venv active):
- `ml ML-bundle/25.10`
- `ml CUDA/13.0.2`
- ` git clone git@github.com:Dao-AILab/flash-attention.git`
- `cd flash-attention`
- `cd hopper`
- `MAX_JOBS=16 python setup.py install`

Afterwards, it should be visible in venv and used by the scripts.

3. Enable the compilation with regional compile added to the codebase (regional one avoids compiling entire model which results in a lot of graph breaks and recompiles, finally leading to no speedup). Remember to export  

4. You probably need to `export NCCL_ENABLE_C2C_GDR=0` for multinode rest looks fine.


# Data loading

Follow the version with bytes, otherwise you will have serious overheads during the initiailization of the model. 
Adding prefetch or workers won't really help as you will be compute bound, so the data loading is not the bottleneck.

# Training scripts

Scripts were a bit to be corrrected, remember to always use  `/bin/bash -l` as shebang as otherwise incorrect env will be set up. 


Do not separate stderr and out (not error but its easier to debug), I also did some evn vars changes so that the number of GPUs per task etc is correctly set.

# Code changes

1. I have added regional compile to the codebase (could not find out if this is supported by llmafactory or not). 
2. I have added NVTX markers for the sake of profiling, NVTX is enabled with `export LLAMAFACTORY_NVTX=1` and then you can call nsys on it. 
3. Cold start of the model can be tedious and slow due to the compile ops happening, and also venv loading can take a bit of time. Consider setting up `export TORCHINDUCTOR_CACHE_DIR=/some/costant/dir/maybe/scratch` so at least part of the compiled operations can be rely

# Profiling
1. You can use `nsys` to profile the training, but remember to set `export LLAMAFACTORY_NVTX=1` so that the NVTX markers are enabled.

2. Then you will have nsys-rep file you need to open with nsys locally, remember to set the `-y` (delay) and `-d` (duration) flags to avoid profiling the warmup, if you profile like 20-30 batches it should be fine.

4. Command:
```bash
 nsys profile -o file-name.nsys-rep -f true -y 120 -d 60 llamafactory-cli train examples/train_lora/qwen_cyfronet_bytes_2.5omni_lora_sft.yaml 
 ```

