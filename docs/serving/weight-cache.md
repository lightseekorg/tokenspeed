# Weight Cache Daemon (Fast Recovery)

The weight cache daemon keeps a model's **post-quantized, tensor-parallel-sharded
weights resident in GPU memory** in a small, long-lived process per rank. When an
engine starts (or restarts), each engine rank maps those weights **zero-copy via
CUDA IPC** instead of reading, dtype-converting, and quantizing checkpoints from
disk. This turns a multi-minute weight load into a sub-second attach, which makes
engine restarts (config changes, crashes, rolling upgrades) fast.

::: warning Linux + NVIDIA GPU only
The daemon relies on CUDA IPC handles and a POSIX parent-death signal. It is a
single-node, NVIDIA-GPU feature. On other platforms `--weight-cache-mode` should
be left `off`.
:::

## How it works

- One daemon runs per **global rank** (`mapping.rank`) and owns that rank's GPU.
- The daemon loads the model once, runs `process_weights_after_loading`, then
  exports every parameter and persistent buffer as a CUDA IPC handle over a
  per-rank Unix domain socket (`/tmp/tokenspeed_weight_cache_rank{rank}.sock`).
- An engine rank connects, sends a **`CacheConfig` fingerprint** (model path,
  architecture, parallelism topology, quant method + config hash, dtype,
  revision, device capability, torch version), and only maps the weights if the
  fingerprint matches exactly. Any mismatch falls back to a disk load (client
  mode) or hard-errors (daemon mode).
- The engine initializes the model on the meta device (no allocation) and swaps
  in the imported IPC tensors, so no weight bytes are copied.

### Supported quantization

Zero-copy sharing only exports raw tensor data, so it is correct **only** when
`process_weights_after_loading` is fully captured by that data. TokenSpeed
enforces an allowlist:

| Quantization | Supported |
| --- | --- |
| Unquantized (`bf16`/`fp16`) | Yes |
| Block-wise FP8 (`weight_block_size` set) | Yes |
| Per-tensor FP8, AWQ, GPTQ, Marlin, NVFP4, … | No — hard error |

Methods that transpose/repack weights or stamp Python-side metadata the
meta-initialized engine cannot reproduce are rejected up front rather than
silently serving wrong numerics. For an unsupported model, disable the cache
with `--weight-cache-mode off`.

## Usage

### Daemon mode (engine-managed)

The engine launches the daemons for you, waits until they finish loading, then
attaches every rank via IPC. Use this for the first start on a fresh host:

```bash
tokenspeed serve <model> \
  --attn-tp-size 8 \
  --weight-cache-mode daemon
```

The daemons keep running after the engine attaches, so a later restart of the
engine reattaches in under a second.

### Client mode (external daemons)

Start the daemons out-of-band once, then point one or more short-lived engine
processes at them. Use this when you restart the engine frequently and want the
weights to survive across restarts:

```bash
# 1. Launch the daemons once (blocks until every rank is ready).
python -m tokenspeed.runtime.weight_cache.daemon \
  --model-path <model> \
  --attn-tp-size 8

# 2. Start (and later restart) the engine against the running daemons.
tokenspeed serve <model> \
  --attn-tp-size 8 \
  --weight-cache-mode client
```

In client mode, if no daemon socket is present the engine falls back to a normal
disk load instead of failing.

## Parameters

| Parameter | Use |
| --- | --- |
| `--weight-cache-mode` | `off` (default), `daemon` (engine launches daemons), or `client` (attach to pre-running daemons). |
| `--weight-cache-socket` | Override the per-rank Unix socket path. Defaults to `/tmp/tokenspeed_weight_cache_rank{rank}.sock`. |

The standalone launcher (`python -m tokenspeed.runtime.weight_cache.daemon`)
accepts the parallelism topology directly:

| Flag | Use |
| --- | --- |
| `--model-path` | Model to load and cache. |
| `--attn-tp-size` / `--dense-tp-size` / `--moe-tp-size` | Layer-family tensor parallel sizes (mirror the engine's `Mapping`). |
| `--ep-size` / `--dp-size` | Expert- and data-parallel sizes. |
| `--nnodes` / `--node-rank` / `--base-gpu-id` / `--gpu-id-step` | Multi-node / GPU placement. |
| `--rank` | Run a single rank's daemon (omit to launch all local ranks). |
| `--load-format` / `--dtype` / `--quantization` / `--revision` | Weight load options; must match the engine to pass the fingerprint check. |
| `--force` | Kill and take over a wedged daemon that still holds the socket. |

## Operational notes

- **Topology must match.** The engine and its daemons must use the same
  parallelism sizes, dtype, quantization, and model revision, or the fingerprint
  check will reject the attach.
- **Memory.** Weights live in the daemon's GPU memory; the engine shares them
  read-only. The engine therefore skips the CPU weight backup used by
  `release_memory_occupation`, so sleep/wake that offloads weights is not
  combined with the weight cache.
- **Allocator.** CUDA IPC is incompatible with `expandable_segments`; the daemon
  refuses to start if that allocator mode is set.
- **Lifecycle.** Each daemon installs a parent-death signal in daemon mode and
  writes a `*.ready` file recording its PID. Stale `*.sock`/`*.ready` files from a
  crashed daemon are cleaned up automatically on the next launch; a still-running
  daemon is left untouched unless `--force` is passed.
- **Multi-node.** `--weight-cache-mode daemon` is single-node only. For
  multi-node, pre-launch daemons on each node with the standalone launcher and
  start the engine with `--weight-cache-mode client`.
