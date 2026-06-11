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

## Notes

- InstantTensor requires NVIDIA GPUs. Requesting it on a non-NVIDIA platform
  raises an error.
- Only `*.safetensors` checkpoints are supported (same shard selection as
  `--load-format safetensors`).

For benchmarks and implementation details, see the
[InstantTensor repository](https://github.com/scitix/InstantTensor).
