# TensorRT-LLM NVFP4 MegaMoE

Source: https://github.com/NVIDIA/TensorRT-LLM/tree/d1166c722ad3b81d7809c9ea98c05e0f1f860248/tensorrt_llm/_torch/cute_dsl_kernels/mega_moe_nvfp4

Revision: `d1166c722ad3b81d7809c9ea98c05e0f1f860248`.
Upstream files retain their Apache-2.0 notices; see LICENSE.
Local adaptation: package imports use the TokenSpeed kernel namespace; formatting follows the repository hooks.
`runner.py` is the TokenSpeed host adapter; it owns symmetric workspaces and launches.

`dynamic_mainloop.py` separates NVVM operand-version detection from the CTA group enum rename for CuTeDSL 4.7.

The kernel tail uses one NVLink barrier to publish both combine outputs and
cleared receive counters. The dispatch barrier completes all peer writes to
these counters. A local grid barrier then waits for every local reader. CTA 0 clears the
counters and publishes both their zero values and the completed combine
writes through the NVLink barrier. Its final grid barrier holds the other
CTAs until CTA 0 has acquired every peer signal. This avoids a second grid
barrier to collect counter zeros from all CTAs.

The grid-sync counter retains its phase across invocations. Its arrival count
resets inside the barrier. Clearing its phase at the kernel tail could hide a
completed barrier from a slower CTA. Bulk FC2 stores wait for destination
writes to complete before the tail publishes output readiness.

Autotuning includes N64, N128 and N256 token tiles. The small-capacity OFF
policy below is specific to K3 SiTU on SM100/SM103 EP16: 896 experts, hidden 3584,
intermediate 3072, top-k 16, beta 4 and linear beta 25. It uses the regimes
observed in the SM103 autotune sweep. These optimizations use the shared
Blackwell kernel implementation. The active CTA count comes from the device;
it is not fixed to an SM103 SM count. Hardware validation covers this K3 shape
on SM100 and SM103. The SM100 paired OFF sweep covers capacities 1 through 2048.
The measured latency gains are stable at capacities 1 through 64.
Other shapes retain the original policy.
Explicit tactics retain priority. All entries use epi_warps and flag_batch 1.

| Tokens per rank (power-of-two capacity) | Tile N | Group hint | Scheduler |
| --- | --- | --- | --- |
| 1 | 64 | 512 | static |
| 2 through 32 | 64 | 1024 | atomic_counter |
| 64 | 128 | 512 | atomic_counter |

For that same K3 shape, capacities 4 through 32 reduce BF16 top-k outputs on
the dispatch workers after the existing tail acquire. This preserves the
original FP32 accumulation order and BF16 rounding. Other capacities retain
the standalone reduction. The kernel receives an explicit reduction flag.
The cache revision separates these kernels from earlier autotune results.

Autotuning can overwrite producer views with synthetic inputs. In a tuning
context, the runner preserves inputs that share workspace storage before
profiling. The selected call stages those saved inputs back. Normal serving
and graph capture do not add these backup copies. The GPU correctness test
starts with producer views and an empty tuning cache. Its torchrun entry
requires an explicit intermediate size; use 3072 for full K3 geometry.

For the same K3 SM100/SM103 EP16 shape, capacities 1, 2 and 64 use two BF16 hidden
elements per reduction worker. This increases reduction parallelism while
preserving the original FP32 top-k accumulation order. Fused capacities 4
through 32 and other shapes keep eight elements per worker. Both kernel
constructors receive the worker width explicitly, and the cache key includes it.

For the same K3 SM100/SM103 EP16 shape, capacities 1, 2 and 4 read the small peer
routing-ID table and build the original receive counts and metadata locally.
The caller supplies one capacity bound for all EP ranks. Unused token rows
contain -1 IDs. The IDs use the existing symmetric input allocation, so
producer views need no extra staging copy. The entry retains two grid barriers
and one peer handshake. The TMA activation pull and kernel tail stay unchanged.
A fixed maximum scratch region supports all capacity buckets. Capacities 1
and 2 use the original smaller view and one ballot per routing job. Capacity 4
uses two ballots and publishes each job's packed count once. Every view writes
the part it reads before the grid barrier. Other capacities keep the original
routing work.
