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

InstantTensor is an optional, NVIDIA-only dependency and is not part of the
core install. Install it explicitly:

```bash
pip install instanttensor
```

or via the extra:

```bash
pip install "tokenspeed[instanttensor]"
```

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

InstantTensor reads each checkpoint tensor **directly onto the GPU**. The
default safetensors loader instead stages the full tensor in host (CPU) memory
and copies only the current rank's shard to the GPU, so for tensor-parallel
models its peak GPU usage during load is roughly one rank's shard. InstantTensor
trades that host-RAM staging for speed, which means it needs meaningfully more
**free VRAM** while loading.

As a result, InstantTensor is best for models that leave headroom on the device.
For a model that already nearly fills GPU memory — especially a large
tensor-parallel model whose unsharded tensors are big — InstantTensor can run
out of memory **during weight loading**, before the KV cache is ever allocated.

If you hit a CUDA OOM inside the loader:

- `--gpu-memory-utilization` will **not** help — it only sizes the KV cache
  *after* weights are loaded; the OOM here happens during the load itself.
- `INSTANTTENSOR_BUFFER_SIZE` / `INSTANTTENSOR_MAX_FREE_MEM_USAGE` bound
  InstantTensor's I/O staging buffer, not the resident weight tensors, so they
  do not resolve a load-time OOM caused by the weights themselves.
- The reliable fix is to fall back to the default loader (drop
  `--load-format instanttensor`), which keeps full tensors in host RAM.

## Notes

- InstantTensor requires NVIDIA GPUs. Requesting it on a non-NVIDIA platform
  raises an error.
- Only `*.safetensors` checkpoints are supported (same shard selection as
  `--load-format safetensors`).

For benchmarks and implementation details, see the
[InstantTensor repository](https://github.com/scitix/InstantTensor).
