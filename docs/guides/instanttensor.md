# Loading Weights with InstantTensor

[InstantTensor](https://github.com/scitix/InstantTensor) accelerates loading
safetensors weights on NVIDIA GPUs through distributed loading, pipelined
prefetching, and direct I/O. It also supports GPUDirect Storage (GDS) when
available, which lets it fully utilize the bandwidth of high-speed networked
storage (e.g. 400 Gbps).

InstantTensor only changes *how* the safetensors shards are read off disk and
moved onto the GPU — the resulting weights are bit-for-bit identical to the
default safetensors loader, so model accuracy is unaffected.

## Installation

InstantTensor is an optional extra, because PyPI carries x86_64 wheels only:

```bash
pip install "tokenspeed[instanttensor]"
```

On aarch64 hosts (GB200, GB300) there is no published wheel, but the sdist
builds from source in about a minute. The extension reaches CUDA, cuFile, and
NCCL through `dlopen` and vendors its own Boost, libaio, and liburing, so it
needs a C++ toolchain and `make` -- not `nvcc`, and not a matching CUDA
toolkit:

```bash
pip install --no-binary instanttensor "tokenspeed[instanttensor]"
```

This is why the dependency is an extra rather than a core one: as a core
dependency every ARM install would compile it, whether or not the format is
ever requested.

It is imported lazily, so it is loaded only when you select
`--load-format instanttensor`.

## Usage

Pass `--load-format instanttensor`. It works with any parallelism
configuration; when the job spans multiple ranks, the world process group is
handed to InstantTensor so reads are sharded across ranks.

```bash
tokenspeed serve Qwen/Qwen3-30B-A3B --load-format instanttensor
```

```bash
tokenspeed serve deepseek-ai/DeepSeek-R1 \
  --load-format instanttensor \
  --tensor-parallel-size 8 \
  --enable-expert-parallel
```

## Measured effect

Weight-load phase for Kimi-K3 (1.42 TiB over 96 shards, 497k tensors) on two
GB300 nodes with `--tensor-parallel-size 8`, checkpoint on shared storage:

| `--load-format` | weight-load phase |
| --- | --- |
| `auto` (safetensors) | 869-1106 s |
| `instanttensor` | 231-234 s |

The ranges are run-to-run variance on the shared filesystem. Both formats
serve identical output, as the loaders are bit-for-bit equivalent.

## Memory considerations

InstantTensor reads each checkpoint tensor **directly onto the GPU**, whereas
the default safetensors loader stages the full tensor in host (CPU) memory and
copies only the current rank's shard to the GPU. InstantTensor's own overhead is
small: it uses a GPU staging buffer (dynamically sized, configurable) that is
released before the KV cache is sized, plus a little fixed runtime overhead, so
its post-load GPU footprint is close to the default loader.

Because tensors land on the GPU, a model's `load_weights` must **consume the
weight iterator lazily** — copying each tensor into its (pre-allocated)
parameter and then releasing it. A `load_weights` that instead collects the
whole iterator into a list keeps every loaded tensor resident on the GPU at once
and will OOM during loading on large models. This stays hidden with the
CPU-staging loaders, where the buffered tensors live in plentiful host RAM.

The gpt-oss MXFP4 path streams its expert tensors for this reason. Paths that
still buffer — notably `_load_normal_weights`, which sorts the iterator, and so
every bf16 gpt-oss checkpoint — hold the whole checkpoint on the device and are
correspondingly limited by GPU memory.

Tuning:

- `INSTANTTENSOR_BUFFER_SIZE` / `INSTANTTENSOR_MAX_FREE_MEM_USAGE` bound
  InstantTensor's GPU I/O staging buffer, trading a little throughput for lower
  peak memory.
- `--gpu-memory-utilization` only sizes the KV cache *after* weights are loaded;
  it does not change peak memory during loading.

## Notes

- InstantTensor requires NVIDIA GPUs. Requesting it on a non-NVIDIA platform
  raises an error.
- Only `*.safetensors` checkpoints are supported (same shard selection as
  `--load-format safetensors`).
- Checkpoints declaring a sub-byte safetensors dtype (`F4`, `F6_E2M3`,
  `F6_E3M2`) are rejected up front, because InstantTensor reads them at twice
  their true length without raising. This does not affect NVFP4 or MXFP4
  checkpoints, which store their packed 4-bit values as byte-aligned `U8`.

For benchmarks and implementation details, see the
[InstantTensor repository](https://github.com/scitix/InstantTensor).
