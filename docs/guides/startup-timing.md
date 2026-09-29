# Startup timing

Set `TOKENSPEED_STARTUP_TIMING=1` when launching a server to emit
`startup_timing` JSON records through the runtime logger. It is off by default.
The instrumentation adds no device synchronization or distributed collectives.

Each span emits `start` and `end` records with a PID, caller-supplied rank and
role, span ID, parent ID, and wall-clock timestamp (`wall_time_ns`). An end
record includes inclusive host wall time (`duration_s`) and success/error
status. Errors propagate unchanged. IDs are local to a process; group records
by host and PID before matching spans. Cross-host timestamps require clock
synchronization. Nested spans overlap: do not sum them to calculate startup.
Host timing includes existing waits but does not wait for newly enqueued GPU
work. It is not a GPU kernel duration measurement.

The scheduler records these phases:

| Phase | Boundary |
| --- | --- |
| `scheduler.init` | CUPTI preparation and event-loop construction |
| `model.config` | Model configuration resolution, target or draft |
| `distributed.init` | Device and distributed initialization |
| `weights.target`, `weights.draft` | Model-runner construction, including loading |
| `weights.allocate` | Default loader model construction |
| `weights.read_copy` | Default loader iterator consumption and weight assignment |
| `weights.postprocess` | Default loader quantization and other post-load transforms |
| `weights.post_quant_warmup` | Default loader post-quantization warmup |
| `multimodal.init` | Applicable multimodal runtime preparation |
| `communication.prepare` | Target/draft persistent communication preparation |
| `kv.build` | Attention backend and KV pool construction, including rebinds |
| `executor.init` | Executor construction |
| `kernels.autotune` | Startup tactic selection |
| `graph.probe_rebind` | Optional partial capture probe, release and KV rebind |
| `graph.capture` | Final serving graph capture |

Detailed weight phases currently cover the default loader. Other loaders still
have the enclosing target/draft span. `weights.read_copy` includes any lazy
checkpoint resolution and model-specific transforms performed by `load_weights`;
it does not isolate disk I/O from H2D. Optional phases may be near-zero no-ops.
These spans do not cover launcher imports, process spawning, frontend tokenizer
initialization, the encode-only loop, the final DP barrier, or HTTP readiness.
Measure process launch to readiness and to the first successful generation
separately; `scheduler.init` is not an end-to-end server startup metric.

With the existing Triton compile monitor enabled, end records also contain
`triton_compiles` and `triton_compile_s`: changes in its startup counters over
the span. A disabled monitor produces `null`, not zero. These are process-wide,
inclusive observations, not a cache-hit ratio or total backend compilation time.
In particular, zero Triton compilations does not establish that a FlashInfer,
CuTeDSL or DeepGEMM cache was hit. Keep the serving JIT checks enabled.

For an initial audit, hold source, model revision, image, hardware, parallelism
and serving parameters fixed. Compare compile-cache-cold and compile-cache-warm
runs, recording the separate checkpoint/page-cache state. Do not clear a shared
host's page cache. Preserve each backend's cache directory across containers,
and use its own diagnostics to establish cache hits. Separate deployment setup
(image pulls, package installation and downloads) from server startup.

Use the slowest rank and the enclosing wall-clock measurements to identify the
critical path. In particular, distinguish the partial graph-memory probe from
final capture before optimizing either. Validate the first real request and
steady-state behavior as well as readiness: moving compilation into serving is
not an equivalent startup improvement.
