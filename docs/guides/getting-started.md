# Getting started

This guide brings up a TokenSpeed development environment and verifies that the
runtime can start.

## Prerequisites

- NVIDIA GPU host for the runner container below, or a ROCm 7.2 GPU host for the ROCm nightly wheels
- Docker with GPU support for the runner container
- Enough shared memory for model serving
- Access to the model checkpoints you plan to serve

## Start a runner container

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

## Install packages

### Nightly wheels

On a CUDA 13 GPU host, in an activated Python 3.11–3.13 environment:

```bash
python -m pip install --upgrade tokenspeed --extra-index-url https://lightseek.org/whl/nightly
```

Each TokenSpeed nightly uses a `.postYYYYMMDD` version and depends on the exact
same-date `tokenspeed-kernel` nightly. To select a particular date, for example:

```bash
python -m pip install "tokenspeed==0.1.0.post20260930" --extra-index-url https://lightseek.org/whl/nightly
```

For ROCm 7.2 on Linux x86_64, first install ROCm PyTorch in a fresh Python
3.11–3.13 environment, then use the separate ROCm nightly index:

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
pin `tokenspeed==0.1.1.postYYYYMMDD` once that date is present in both ROCm
nightly package indexes. When upgrading, avoid accidentally choosing a newer
CUDA-only PyPI release.

The **Publish nightly** workflow is the single nightly entry point. It runs
daily at 01:00 UTC, or manually from `main`. It publishes, in order,
`tokenspeed-kernel-amd`, the CUDA and ROCm `tokenspeed-kernel` wheels,
`tokenspeed-engine`, and `tokenspeed`. Every stage uses the run's source commit
and date, and TokenSpeed pins the exact kernel and engine versions. Nightlies
publish to the wheel indexes, not PyPI. Stable release workflows remain separate.

For a manual nightly, open **Publish nightly**, select `main`, and choose:

| Input | Meaning |
| --- | --- |
| `version_date` | `YYYYMMDD`, or blank for the run's creation date in UTC |
| `replace_nightly` | Defaults to `false`; enable to replace existing assets for every package |
| `release_candidate` | Optional `N` or `rcN` for a CUDA-only candidate; leave blank for both backends |

By default, an existing version from the same commit is resumed: published files
keep their bytes and hashes, missing assets are uploaded, and indexes are repaired.
An existing version from another commit, or without source provenance, fails the
preflight before publication. A regular nightly's engine uses the same date;
re-running it does not silently allocate tomorrow's engine version.

With `replace_nightly=true`, all four packages are rebuilt and existing assets
are overwritten. Both CUDA and ROCm indexes receive the new hashes and cache-busting
download URLs. Replacement is not atomic across assets or packages: if a stage
fails after uploading, installs may fail hash verification until its index is
repaired. Use **Re-run all jobs** on the same run to retain the original source,
date and replacement setting and complete the entire downstream chain. A new run
on `main` uses its new source commit. To reinstall an already installed version,
add `--no-cache-dir --force-reinstall` to the pinned installation command above.

A release candidate publishes only the CUDA kernel, engine and TokenSpeed
candidates, all using `rcN.postYYYYMMDD` versions.
For example, candidate `1` gives `tokenspeed==0.1.1rc1.post20261108` and
`tokenspeed-kernel==0.1.4rc1.post20261108`. It sorts below the normal nightly of the
same base version. AMD and ROCm are skipped for candidates; the replacement option
replaces all three candidate packages without consuming a future nightly date. Scheduled and manual nightly runs share one concurrency group.

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

Install the engine package, which builds the C++ scheduler in. It needs Rust,
CMake and OpenSSL development headers (see `tokenspeed-engine/README.md`):

```bash
pip install -e tokenspeed-engine/
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
supported path. TokenSpeed does not enable the `gfx950` and `gfx1250`
specialized kernels for this architecture.

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
