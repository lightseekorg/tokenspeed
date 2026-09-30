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

### Nightly wheels

In an activated Python 3.10–3.13 environment on a CUDA 13 GPU host:

```bash
python -m pip install --upgrade tokenspeed --extra-index-url https://lightseek.org/whl/nightly
```

Each TokenSpeed nightly uses a `.postYYYYMMDD` version and depends on the exact
same-date `tokenspeed-kernel` nightly. To select a particular date, for example:

```bash
python -m pip install "tokenspeed==0.1.0.post20260930" --extra-index-url https://lightseek.org/whl/nightly
```

Kernel builds are scheduled at 02:00 UTC and TokenSpeed at 03:00 UTC. TokenSpeed
checks the nightly index for its exact kernel dependency before building and
publishing; it waits up to one hour and fails without publishing if the kernel
is unavailable. Historical nightly wheels remain available. Nightlies publish
to the wheel index, not PyPI.

For a manual nightly, run **Build and Release TokenSpeed** from `main` with
`nightly=true` and `publish_github=true`. `version_date` can select an already
published kernel nightly date. Pull request branches can use
`publish_github=false` to build without publishing.

### From source

For H100/H200 with CUDA Toolkit 12.9, use the
[Hopper / CUDA 12.9 source-install recipe](hopper-cu129.md). In an activated
Python 3.11 environment, run:

```bash
CUDA_VARIANT=cu129 bash test/ci_system/install_deps.sh
```

For the default runner environment, follow the package installation steps below.

Install the Python runtime:

```bash
export PIP_BREAK_SYSTEM_PACKAGES=1
pip install -e "./python" --no-build-isolation
```

Install the kernel package. Its Python package metadata installs the selected
backend dependencies automatically.

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
