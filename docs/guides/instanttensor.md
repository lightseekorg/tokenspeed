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

InstantTensor ships as a dependency of TokenSpeed, so a normal install already
includes it — no extra step is needed. It is a CUDA-only package and is
imported lazily, so it is only loaded when you actually select
`--load-format instanttensor` on an NVIDIA GPU.

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
TokenSpeed's model loaders stream the iterator for this reason.

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

For benchmarks and implementation details, see the
[InstantTensor repository](https://github.com/scitix/InstantTensor).
