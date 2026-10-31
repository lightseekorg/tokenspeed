# TokenSpeed Production Recipes and Examples

This directory contains standalone, production-oriented recipes and examples demonstrating how to integrate and run [TokenSpeed](https://github.com/lightseekorg/tokenspeed).

Following the project's [Contribution Ethos](../CONTRIBUTING.md), these examples focus on **reproducible performance**, **correctness**, and **production-oriented engineering** rather than ad-hoc demonstrations.

---

## Prerequisites

TokenSpeed requires an accelerated environment (NVIDIA GPU with `sm90`/`sm100` support or AMD GPU with ROCm 7.2):

```bash
# Recommended: Launch inside the official runner container
docker run -itd \
  --shm-size 32g \
  --gpus all \
  --ipc=host \
  --network=host \
  --name tokenspeed \
  lightseekorg/tokenspeed-runner:latest \
  /bin/bash
```

Inside the environment, install TokenSpeed nightly packages as described in the [Getting Started Guide](../docs/guides/getting-started.md):

```bash
python -m pip install --upgrade tokenspeed --extra-index-url https://lightseek.org/whl/nightly
```

---

## Available Examples

### 1. In-Process Offline Batched Inference (`offline_batch_inference.py`)

Demonstrates how to instantiate the TokenSpeed inference engine directly in Python without starting an external HTTP server daemon. Suitable for offline evaluation benchmarks, synthetic data generation pipelines, or embedded workflows.

**Usage:**

```bash
python examples/offline_batch_inference.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --output-file results.json
```

**Key Features:**
* Explicit configuration of memory limits and tensor parallelism.
* Guaranteed resource reclamation and subprocess cleanup via `engine.shutdown()`.
* Throughput reporting (`tokens/s`) and optional JSON evaluation export.
