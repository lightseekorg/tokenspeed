# Getting Started

This guide brings up a TokenSpeed development environment and verifies that the
runtime can start.

## Prerequisites

- NVIDIA GPU host
- Docker with GPU support
- enough shared memory for model serving
- access to the model checkpoints you plan to serve

## Start a Runner Container

```bash
docker pull lightseekorg/tokenspeed-runner:latest

docker run -itd \
  --shm-size 32g \
  --gpus all \
  -v /raid/cache:/home/runner/.cache \
  --ipc=host \
  --network=host \
  --pid=host \
  --privileged \
  --name tokenspeed \
  lightseekorg/tokenspeed-runner:latest \
  /bin/bash
```

Inside the container:

```bash
git clone https://github.com/lightseekorg/tokenspeed.git
cd tokenspeed
```

## Install Packages

Install the Python runtime:

```bash
export PIP_BREAK_SYSTEM_PACKAGES=1
pip install -e "./python" --no-build-isolation
```

Install the kernel package. Its Python package metadata installs the CUDA
kernel dependencies automatically.

```bash
pip install -e tokenspeed-kernel/python/ --no-build-isolation
```

Install the scheduler package:

```bash
pip install -e tokenspeed-scheduler/
```

## Verify

```bash
tokenspeed env
tokenspeed serve --help
```

## Launch

```bash
tokenspeed serve openai/gpt-oss-20b \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1
```

For model-specific examples, continue with [Model Recipes](../recipes/models.md).
For the full list of subcommands, see [CLI Commands](./cli-commands.md).
