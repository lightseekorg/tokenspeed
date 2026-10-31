# Offline Batch Inference Recipe

This recipe demonstrates how to run offline batched inference and evaluation with
TokenSpeed directly in Python, bypassing HTTP networking and serialization overhead.

In-process generation is suited for offline dataset evaluation, synthetic data
generation, and batch benchmarking pipelines where a persistent web server is
unnecessary.

---

## Prerequisites

Run on an accelerated host (NVIDIA GPU with `sm90`/`sm100` support or AMD GPU
with ROCm 7.2) inside the runner container:

```bash
docker run -itd \
  --shm-size 32g \
  --gpus all \
  --ipc=host \
  --network=host \
  --name tokenspeed \
  lightseekorg/tokenspeed-runner:latest \
  /bin/bash
```

Ensure TokenSpeed is installed:

```bash
python -m pip install --upgrade tokenspeed --extra-index-url https://lightseek.org/whl/nightly
```

---

## Companion Script

The companion script lives in [`examples/offline_batch_inference.py`](file:///home/codespace/tokenspeed/examples/offline_batch_inference.py).

### Basic Execution

```bash
python examples/offline_batch_inference.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --output-file results.json
```

### Multi-GPU Tensor Parallelism

To shard a large model across multiple GPUs (e.g. 4 GPUs):

```bash
python examples/offline_batch_inference.py \
  --model Qwen/Qwen2.5-72B-Instruct \
  --tensor-parallel-size 4 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.92 \
  --output-file qwen72b_eval.json
```

---

## Core Implementation Pattern

The core integration pattern uses `tokenspeed.runtime.entrypoints.engine.Engine`
with explicit configuration parameters:

```python
import sys
import time
from typing import Any
from tokenspeed.runtime.entrypoints.engine import Engine
from tokenspeed.runtime.utils.server_args import ServerArgs

# 1. Configure production-safe execution parameters explicitly
server_args = ServerArgs(
    model="meta-llama/Llama-3.1-8B-Instruct",
    attn_tp_size=1,
    max_model_len=8192,
    gpu_memory_utilization=0.90,
    log_level="error",
)

# 2. Instantiate Engine in-process
engine = Engine(server_args=server_args)

prompts = [
    "Explain the split between C++ control plane and Python execution plane.",
    "Implement an LRU cache with O(1) complexity in Python.",
]

sampling_params = {
    "temperature": 0.0,       # Deterministic greedy sampling for reproducible evaluation
    "max_new_tokens": 256,
    "ignore_eos": False,
}

try:
    # 3. Execute batched generation
    start_time = time.perf_counter()
    outputs = engine.generate(prompt=prompts, sampling_params=sampling_params)
    duration = time.perf_counter() - start_time

    # 4. Measure token throughput
    total_tokens = sum(len(out.get("output_ids", [])) for out in outputs)
    throughput = total_tokens / duration if duration > 0.0 else 0.0
    print(f"Generated {total_tokens} tokens in {duration:.2f}s ({throughput:.2f} tokens/s)")

    for prompt, output in zip(prompts, outputs):
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {output['text'].strip()}")

finally:
    # 5. Guarantee subprocess and IPC resource reclamation
    engine.shutdown()
```

---

## Operational Details

### Determinism and Reproducibility
For benchmarking or benchmark evaluation, set `temperature=0.0`. TokenSpeed's
scheduler guarantees deterministic output ordering for batches given fixed seeds
and identical model configurations.

### Memory Sizing
* `--gpu-memory-utilization`: Controls the fraction of GPU VRAM dedicated to
  weights and the TokenSpeed LCM KV cache pool. Defaults to `0.90`.
* `--max-model-len`: Upper bound on sequence length (prompt + output). Align this
  with the maximum expected evaluation context to prevent memory fragmentation.

### Resource Cleanup
TokenSpeed starts background scheduler and detokenizer subprocesses using
multiprocessing and ZMQ IPC. Always invoke `engine.shutdown()` inside a `finally`
block to guarantee all child processes terminate cleanly and release GPU memory.

### Backend Selection
For portable execution across architectures (such as NVIDIA Blackwell Server/Workstation
edition GPUs or AMD ROCm):

```bash
python examples/offline_batch_inference.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --attention-backend triton \
  --sampling-backend triton \
  --output-file results.json
```

---

## Serverless Execution with Modal

For on-demand cloud GPU execution (e.g. on an NVIDIA RTX 6000 or H100):

```python
import modal

app = modal.App("tokenspeed-offline-batch")

image = (
    modal.Image.from_registry(
        "lightseekorg/tokenspeed-runner:cu130-torch-2.14.0-flashinfer-0.6.18",
        add_python="3.12",
    )
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
    })
    .pip_install(
        "torch==2.14.0",
        "tokenspeed",
        extra_index_url="https://lightseek.org/whl/nightly",
    )
)

@app.function(gpu="RTX-PRO-6000", image=image, timeout=900)
def run_batch():
    from examples.offline_batch_inference import run_batch_inference

    prompts = [
        "Explain the split between C++ control plane and Python execution plane.",
        "Implement an LRU cache with O(1) complexity in Python.",
    ]
    return run_batch_inference(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        tensor_parallel_size=1,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        temperature=0.0,
        max_new_tokens=128,
        prompts=prompts,
        attention_backend="triton",
        sampling_backend="triton",
    )
```
