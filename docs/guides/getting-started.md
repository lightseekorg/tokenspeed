# Getting Started

This guide brings up a TokenSpeed development environment and verifies that the
runtime can start.

## Prerequisites

- NVIDIA GPU host for the runner container below, or a ROCm 7.2 GPU host for the ROCm nightly wheels
- Docker with GPU support for the runner container
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

On a CUDA 13 GPU host, in an activated Python 3.10–3.13 environment:

```bash
python -m pip install --upgrade tokenspeed --extra-index-url https://lightseek.org/whl/nightly
```

Each TokenSpeed nightly uses a `.postYYYYMMDD` version and depends on the exact
same-date `tokenspeed-kernel` nightly. To select a particular date, for example:

```bash
python -m pip install "tokenspeed==0.1.0.post20260930" --extra-index-url https://lightseek.org/whl/nightly
```

For ROCm 7.2 on Linux x86_64, first install ROCm PyTorch in a fresh Python
3.10–3.13 environment, then use the separate ROCm nightly index:

```bash
python -m pip install "torch==2.14.0" "torchvision==0.29.0" \
  --index-url https://download.pytorch.org/whl/rocm7.2
python -m pip install --upgrade tokenspeed \
  --extra-index-url https://lightseek.org/whl/nightly/rocm7.2
python -c 'import torch; assert torch.version.hip, torch.__version__'
python -m pip check
```

The ROCm index supplies ROCm `tokenspeed-kernel` wheels, while both indexes
link the same pure-Python TokenSpeed wheel. Do not configure both the CUDA and
ROCm nightly indexes at once: kernel wheels have the same version but different
backend dependencies. Preinstalling ROCm PyTorch matters because pip's extra
index does not take priority over PyPI when selecting dependencies. You can
pin `tokenspeed==0.1.0.postYYYYMMDD` once that date is present in both ROCm
nightly package indexes; avoid accidentally choosing a newer CUDA-only PyPI
release when upgrading.

Kernel builds are scheduled at 02:00 UTC and TokenSpeed at 03:00 UTC. TokenSpeed
checks the CUDA nightly index for its exact kernel dependency before building
and publishing; it checks the ROCm index before adding the same wheel there.
Each check waits up to one hour, and the ROCm check cannot block CUDA
publication. Historical nightly wheels remain available. Nightlies publish to
the wheel indexes, not PyPI.

For a manual nightly, run **Build and Release tokenspeed-kernel for ROCm**
from `main` with `nightly=true` and `publish_github=true`, then run **Build and
Release TokenSpeed** with the same settings and `version_date`. Pull request
branches can use `publish_github=false` to build without publishing.

To rebuild and replace an existing TokenSpeed nightly, select `main` in
**Build and Release TokenSpeed** and set these **Run workflow** inputs:

| Input | Value |
| --- | --- |
| `nightly` | `true` |
| `replace_nightly` | `true` |
| `publish_github` | `true` |
| `version_date` | The date to replace as `YYYYMMDD`, or blank for today (UTC) |

This rebuilds the distributions and refreshes their hashes and download URLs in
both the CUDA and ROCm indexes. Replacement defaults to off; normal reruns keep
existing files. TokenSpeed uses one pure-Python wheel for both backends, so it
does not need the kernel workflow's CUDA build-variant inputs. To install a
replacement of an already installed version, add `--no-cache-dir --force-reinstall`
to the corresponding pinned installation command above.

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

### AMD RDNA4 (`gfx1201`)

`gfx1201` devices, including the Radeon AI PRO R9700, use the existing portable
Triton kernels and ROCm library implementations. Install a ROCm PyTorch build
that includes `gfx1201` device support. Dense BF16 inference is the initial
supported path; the `gfx950` and `gfx1250` specialized kernels are not enabled
for this architecture.

For a small single-GPU model, start with Triton attention and eager execution:

```bash
tokenspeed serve Qwen/Qwen3-0.6B \
  --host 127.0.0.1 \
  --dtype bfloat16 \
  --world-size 1 \
  --attention-backend triton \
  --enforce-eager \
  --disable-prefill-graph
```

Architecture detection reports RDNA4 separately from CDNA4/CDNA5. CDNA-specific
matrix and async-copy capabilities remain disabled for `gfx1201`.
