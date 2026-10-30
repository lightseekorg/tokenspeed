# Op and layer profiling

The profiling path records eager op/layer metadata and evaluates explicit cost
rules after collection. It does not change kernel selection or execution.
An optional CUDA collector adds measured event windows and native Nsight Systems
NVTX ranges containing MFU, Estimated MBU and collective busbw when the collection window finishes.

## Collection

`ProfileSession` owns a window identified by run, model, rank and iteration.
Bind layer instances to their global layer IDs before recording.
`@profiled(level="layer")` tracks repeated calls and restores parent context on
exit, including exceptions. `inner_id` increases within an iteration/layer;
`occurrence` identifies each layer call.

`register_kernel(..., profile=OpProfile(cost_rule="..."))` opts a synchronous
function into input/output tensor metadata collection. Omitting `profile` leaves
the registered function unchanged; `cost_rule=None` records metadata only.
Wrappers preserve arguments, returns and exceptions. Tensor descriptions retain
shape, dtype, stride, device and logical size, not tensor storage. Nested
registered calls record only the outermost op; its rule must cover the entire op.

Both wrappers use `record_func`. The required `backend` selects `none` (metadata
only), `nvtx` or `torch`; the caller owns the external profiler lifetime.
Labels carry readable layer/op/rank/call identity and an encoded unique key
shared with the metadata export. CPU scope timing includes collection overhead
and cannot be used as GPU latency.

Collection supports synchronous eager windows on one execution thread. Nested
sessions, repeated iteration IDs and profiled stateful kernel classes are rejected.
CUDA graph replay and cross-thread attribution are not implemented. Output
metadata describes returned tensors; in-place-only outputs need a separate adapter.

## Cost analysis

`CostRegistry` applies named rules after collection. `DenseMatmulCost` models
plain dense 2D GEMM with explicit input/output tensor roles. Costs distinguish
useful and executed FLOPs, logical reads/writes and rule assumptions.
Failed calls, missing rules and unsupported shapes retain a missing-cost reason.
Layer summaries aggregate recorded ops per iteration/layer across occurrences,
report coverage, and expose a complete cost only when every recorded op is covered.
This coverage describes opted-in calls, not all work in a model.

Logical bytes do not measure HBM traffic. MFU and Estimated MBU remain unavailable
without device timing and matching hardware peaks. The ordinary metadata path
adds no GPU synchronization or device timing. A rule can provide `compute_type`
and `estimated_hbm_bytes` explicitly for utilization analysis; storage dtype does
not establish the compute engine, precision, sparsity or HBM traffic.

## Communication bandwidth and FLOP conventions

`CollectiveTraffic` records a collective name, full-array `total_bytes`, group
size and group identity. `CollectiveCost` obtains bytes from tensor metadata,
which already includes dtype size. For AllReduceSum and ReduceScatterSum, use
the full input tensor; for AllGather, use the full gathered output, including
all ranks. The latter is `world_size * local_send_bytes`, not local send bytes.
In-place output buffers can be selected from recorded input arguments.

With `S` in bytes, group size `P`, and measured duration `t` in seconds:

| Collective | Normalized bus bytes per rank | busbw (byte/s) |
| --- | --- | --- |
| AllReduceSum | `2*S*(P-1)/P` | `2*S*(P-1)/(P*t)` |
| ReduceScatterSum | `S*(P-1)/P` | `S*(P-1)/(P*t)` |
| AllGather | `S*(P-1)/P` | `S*(P-1)/(P*t)` |

`algbw = S/t`. If the input is an element count, convert it to bytes once with
`numel * element_size`. A single rank has zero bus bytes. JSON stores byte/s;
NVTX displays decimal GB/s (`1e9` byte/s). These are the
[NCCL-tests normalized bandwidth definitions](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md),
not physical link counters, cluster-wide traffic, or HBM bandwidth. Do not
multiply again for send plus receive or compare to an unrelated duplex peak.
No hardware peak is needed to report busbw.

Register explicit rules for each group and buffer role:

```python
from tokenspeed_kernel.profile_costs import CollectiveCost

costs.register("tp_sum", CollectiveCost(
    collective="all_reduce_sum", tensor="x", tensor_source="input",
    world_size=tp_size, group="tp-group-0",
))
costs.register("tp_gather", CollectiveCost(
    collective="all_gather", tensor="return", tensor_source="output",
    world_size=tp_size, group="tp-group-0",
))
```

The recording wrapper must cover completion on the declared CUDA stream. An
asynchronous collective that returns before its work joins that stream cannot
supply a valid event-window denominator. The integration test explicitly waits
on NCCL work on the current stream before returning from each registered call.
This feature does not automatically instrument all distributed calls.

A layer's busbw sums unique recorded collective bus bytes and divides by that
layer's own CUDA event window, including intervening compute and waits. It is
window-average normalized communication throughput, not the average of op
bandwidths or a communication-only link speed. Mixed groups/sizes, missing costs,
invalid timing and failed scopes remain unavailable. Pure collectives have no
model MFU; they do not invalidate the compute type of a mixed compute/communication
layer. HBM estimates remain independently unavailable unless explicitly modeled.

Dense `M x K` by `K x N` GEMM uses `2*M*N*K` FLOPs with an FMA counted as two;
a batched GEMM additionally sums over batches. This is an operation count, not
FLOP/s: divide by seconds for FLOP/s and by `1e12` for TFLOP/s. Dtype does not
halve FLOP counts; it selects the applicable hardware compute peak.

`AttentionMatmulCost` covers fixed-length `[B,H,S,D]` tensors, with explicit
Q/K/V/output roles and mask alignment. It counts the two forward matrix products:
`2*B*Hq*pairs*(Dqk + Dv)`. GQA uses query heads `Hq`, not KV heads. Scaling,
softmax, projections, padding and recompute are excluded; executed FLOPs and HBM
traffic remain unknown. For equal head dimensions and full attention, this is
`4*B*Hq*Q*K*D`. With square causal self-attention (`Q=K=L`), the exact number of
valid pairs is `L*(L+1)/2`; dividing the full count by two is a large-L approximation.
For lower-right causal alignment with `K>=Q`, pairs are
`Q*K-Q*(Q-1)/2`. Single-token decode therefore uses all `K` keys, with no halving.
The rule also handles `Q>K` and explicit upper-left alignment. Variable-length,
paged, sparse and sliding-window workloads require their own metadata adapters.
See [GEMM FLOP counting](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)
and [FlashAttention causal alignment](https://github.com/Dao-AILab/flash-attention#21-change-behavior-of-causal-flag).

## CPU validation workload

Run from the repository root in the project's Python environment:

```bash
PYTHONPATH=tokenspeed-kernel/python python tokenspeed-kernel/test/profile_layers.py \
  --backend torch --output /tmp/op-layer/records.json

PYTHONPATH=tokenspeed-kernel/python python -m pytest \
  tokenspeed-kernel/test/test_op_profile.py -q
```

The example executes a two-GEMM layer twice and checks its result: four ops,
4096 FLOPs, 2816 logical read bytes and 768 logical write bytes. `--backend none`
writes metadata and costs only; `torch` also writes `records.trace.json`.
The illustrative layer ID is bound explicitly. Tests exercise real CPU ops and
torch profiler, covering repeated/nested calls, cost coverage, argument forwarding
and exception recovery.

## Native Nsight timeline

`NvtxTimeline` attaches to an unused `ProfileSession(backend="nvtx")`. Existing
decorators and registered op wrappers use the same `record_func` entry. It
records CPU timestamps in the NVTX tool's clock and CUDA events around each op
and layer. Event capacity is allocated and warmed before the measured iterations.
There is no per-op synchronization or cost evaluation.

On context exit, while the external Nsight capture is still active, it waits
once for the declared stream, computes costs and utilization, writes the metrics
JSON, and submits complete NVTX push/pop ranges with their original CPU start
and end timestamps. NVIDIA's public `nvtxEventSubmit` API supplies the final
labels. Ranges belong to the original execution thread and retain layer nesting
and Nsight's CPU-to-GPU projection. No report rewriting, binary patching, writer
merge or user-run postprocessing script is involved. Explicit `finish()` is
idempotent and must run outside iteration scopes.

```python
from tokenspeed_kernel.profile_nvtx import NvtxTimeline

with NvtxTimeline(
    session,
    costs=costs,
    hardware=hardware,
    stream=torch.cuda.current_stream(),
    max_scopes=128,  # Count layer and op calls together.
    library=bridge_path,
    output=metrics_path,
):
    with session.iteration(iter_id=0):
        result = layer(inputs)
```

Example NVTX label:

```text
gemm.mm | MFU=31.36% Est.MBU=2.58%
```

Layer ranges show only their index, such as `L7`. Op labels show the op name
and utilization; model, rank, iteration and call identities remain in the JSON.
Labels are not unique identifiers: repeated calls and ranks can share a label.
Use the execution thread and time range to distinguish them in a native report.
The labels and JSON consume the same analysis results. Layer costs in the JSON
sum unique descendant ops over each call's own measured window. Their coverage
is limited to opted-in calls, not every operation in a model. Missing costs, timing or
compatible peaks produce `N/A` with reasons in JSON. Mixed compute types report
per-type fractions and no overall MFU. Ratios are not clamped to 100%.

The denominator is a **CUDA event window**, including stream waits, host launch
gaps and event overhead, not the sum of kernel durations or a CPU range duration.
Profiler and instrumentation overhead affects this window, especially for small
ops. Nsight's projected kernel span is a separate measurement. Estimated MBU
uses only the rule's explicit HBM estimate; logical IO is never silently promoted
to measured HBM traffic. The example's optional `logical_io_once` policy excludes
cache reuse, rereads and workspace, so it is a rough estimate.

### One-command CUDA validation workload

Requires Linux, CUDA, a C compiler, and **Nsight Systems 2026.5+ CLI and viewer**.
This path was qualified with 2026.5.1. Older CLIs may accept a schema while
silently omitting deferred events; the launcher rejects them. The optional bridge
builds against the selected Nsight installation's `nvtx/include` before capture;
it does not download dependencies or alter the system profiler.

Supply a hardware JSON with `name`, `compute_flops_per_second`,
`hbm_bytes_per_second` and `source`. Compute peaks are dense FLOP/s keyed by rule
compute type (the example uses `bf16_tensor`); bandwidth is byte/s. Specify actual
per-device/partition capacities and their source. Missing peaks may be omitted
from the compute mapping; unknown HBM bandwidth is `null`. TiB/s uses `2**40`
bytes/s, whereas TB/s uses `10**12`.

```bash
PYTHONPATH=tokenspeed-kernel/python python tokenspeed-kernel/test/nvidia/profile_layers_cuda.py \
  --nsys /path/to/nsys \
  --hardware /path/to/device.json \
  --output-dir /tmp/native-profile \
  --hbm-policy logical_io_once --size 1024 --iterations 3
```

Use a new output directory. This single command builds the bridge, launches
Nsight, executes and checks real BF16 CUDA GEMMs, and produces `timeline.nsys-rep`
and `metrics.json`. Open the report's execution thread NVTX rows and GPU NVTX
projection in Nsight Systems. Warmup and numerical validation lie outside the
annotated iterations. The workload is an illustrative two-GEMM layer, not a
full-model serving benchmark or an unprofiled performance baseline.

The collector supports one execution thread and one declared CUDA stream per
session. Other streams must join that stream before the op returns; unjoined
work is outside the supported contract. Switching the current stream, entering
CUDA graph capture, reusing sessions or exhausting event capacity is rejected.
Graph replay does not rerun Python decorators and has no op attribution here.
Each distributed rank needs its own session, device peaks and output path.
Keep capture active until context exit; do not call `cudaProfilerStop` first.

The optional integration test uses an actual CUDA capture and the official
`nvtx_gpu_proj_trace` export to check thread IDs, nesting, kernel attribution,
and consistency between labels and metrics:

```bash
TOKENSPEED_TEST_NSYS=/path/to/nsys PYTHONPATH=tokenspeed-kernel/python python -m pytest \
  tokenspeed-kernel/test/nvidia/test_profile_nvtx.py -q
```

References: [NVTX deferred event API](https://nvidia.github.io/NVTX/doxygen/nv_tools_ext_payload_8h.html)
and [Nsight NVTX projection](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html).
