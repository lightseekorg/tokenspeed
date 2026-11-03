# Op and layer profiling

The profiling path records eager op/layer metadata and evaluates explicit cost
rules after collection. It does not change kernel selection or execution.
An optional collector emits native NVTX identities. After capture, Nsight Systems
GPU activity timestamps supply the denominator for MFU, Estimated MBU and busbw.
The collector never inserts CUDA Events to measure op or layer latency.

## Collection

`ProfileSession` owns a window identified by run, model, rank and iteration.
Bind layer instances to their global layer IDs before recording.
`@profiled(level="layer")` tracks repeated calls and restores parent context on
exit, including exceptions. `inner_id` increases within an iteration/layer;
`occurrence` identifies each layer call.

`register_kernel(..., profile=OpProfile(cost_rule="..."))` opts a synchronous
function into input/output tensor metadata collection. Omitting `profile` leaves
the registered function unchanged; `cost_rule=None` records metadata only.
Optional `scalar_args=(...)` records named host scalars or tuples of scalars,
including defaults, for rules needing group ranks or other non-tensor geometry.
Singleton JSON-scalar values from existing `KernelSpec.traits` are captured once
at binding and retained as `static_traits`. Fixed-format rules reuse these instead
of collecting redundant arguments. For multi-valued traits, same-named host
arguments are collected automatically, including defaults; the supported set is
never treated as the actual choice. Non-scalar automatic arguments are skipped.
`scalar_args` only needs to declare additional host parameters not already fixed
by the implementation contract. Explicit arguments take precedence when a rule
uses both sources.
Device tensors cannot be used as scalar arguments; that path never reads them back.
Wrappers preserve arguments, returns and exceptions. Tensor descriptions include
omitted tensor defaults as well as explicitly supplied arguments, retaining
shape, dtype, stride, device and logical size, not tensor storage. Defaults are
resolved only for metadata; the original call's arguments are forwarded unchanged.
Nested registered calls record only the outermost op; its rule must cover the entire op.
Tensor fields inside dataclass argument/return holders are described using dotted
paths, such as `cache.kv_cache` and `return.query`; no tensor values are copied.

Dynamic geometry can explicitly opt into `integer_tensor_args=("cache_seqlens",)`.
These snapshots accept contiguous 1-D int32/int64 vectors of at most 4096 elements.
CPU values become immutable tuples immediately. CUDA values require `NvtxTimeline`,
which queues same-stream D2H copies before the op's NVTX range into a preallocated
pinned buffer (`max_metadata_bytes`, default 1 MiB). Values are resolved after the
window's final synchronization for metadata copies only. Reusing a device buffer preserves each
call's original lengths. Ordinary metadata collection never reads CUDA values;
unavailable snapshots remain `null` with a missing-cost reason. Snapshot overhead
affects enclosing layer timing but is outside the op NVTX range. Capacity
exhaustion raises explicitly.

Existing direct entry points can use `wrap_op_for_profiling` with a `KernelSpec`
for identity only; this does not register another kernel or alter dispatch.
Direct RMSNorm and attention imports use this binding without additional registry
entries; existing registered kernels only add their `profile` declaration.
`current_op_metadata()` lets the original dispatcher append its actual group or
fusion decision while collecting; it returns `None` outside a profiled op.

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

Device analysis also exports `useful_flops_per_second` and
`estimated_hbm_bytes_per_second`, dividing each scope's costs by its Nsight GPU
projection. These rates do not require hardware peaks; divide by `1e12` for TFLOP/s
or `1e9` for decimal GB/s. The latter remains an IO estimate, not measured HBM
traffic. Mixed-type layer FLOP/s sums useful FLOPs across recorded ops without
choosing a shared normalization peak. Missing costs, invalid timing or failed
scopes leave rates unavailable. NVTX labels retain only the compact utilization
display; the rates are in the existing metrics output.

The registered causal-convolution prefill rule counts `2*T*C*W` nominal FLOPs
for packed `x[C,T]` and depthwise weights `[C,W]`; `C` includes all packed Q/K/V
channels. Bias and SiLU arithmetic are excluded. Its logical IO includes inputs,
weights, output, optional bias and `C*(W-1)` history elements per sequence for
both reading and writing, using each tensor's dtype size. It never charges the
entire state-pool capacity. The history estimate assumes initialized nonempty
sequences; device masks are not read back, so initial chunks or padding may
overestimate traffic. `logical_io_once` exposes this as Estimated MBU. The Triton
implementation's FP32 accumulation selects `fp32_simt`; provide that GPU's FP32
SIMT FMA reference peak in `compute_flops_per_second` to get MFU alongside MBU.
FP16/BF16 products may round before accumulation, so this is model FLOPs divided
by a reference compute peak, not mixed-instruction pipeline utilization. Storage
dtype does not select a Tensor Core peak. A missing `fp32_simt` peak leaves MFU
unavailable without borrowing a different peak.

`GatedDeltaPrefillCost` covers packed Q/K/V with a value-major recurrent state.
It counts the equivalent recurrence's state decay, state/key product, correction,
rank-one update and readout: `T*H*(7*Dk*Dv + Dv)` FLOPs. This is not the executed
chunk factorization's FLOP count. Normalization, gate activations and featurewise
beta scaling are excluded. Its optional IO estimate includes boundary tensors and
initial/final state, excluding intermediate chunk buffers. Mixed-precision chunk
implementations leave MFU unavailable rather than infer a peak from storage dtype.

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

The recording wrapper must enclose the CUDA API calls that launch the collective.
Nsight correlates those launches with GPU execution even when execution completes
after the host range ends; work launched on another host thread is not attributed. The integration test explicitly waits
on NCCL work on the current stream before returning from each registered call.
The runtime NCCL backend profiles its existing `all_reduce` and `all_reduce_two`
methods, preserving PyNccl/PyTorch dispatch and recording group ranks/name plus
the reduction kind. Each entry has one outer scope; tuple and grouped-pair traffic
sum each tensor's bytes once. The grouped scope includes coalescing completion;
its fallback to ordinary reductions does not create nested op records.
Runtime capture registers its cost rule automatically; non-SUM costs remain
unavailable. Its optional `logical_io_once` HBM estimate counts one
input read plus one output write, excluding protocol buffers and reloads; busbw
uses the independent normalized communication formula above. Other distributed
calls still need their own adapters.

A layer's busbw sums unique recorded collective bus bytes and divides by that
layer's own Nsight GPU projection, including gaps between its GPU activities. It is
window-average normalized communication throughput, not the average of op
bandwidths or a communication-only link speed. Mixed groups/sizes, missing costs,
invalid timing and failed scopes remain unavailable. Pure collectives have no
model MFU; they do not invalidate the compute type of a mixed compute/communication
layer. HBM estimates remain independently unavailable unless explicitly modeled.
A fused communication/compute cost must carry both `communication` and useful
FLOPs; its op label retains MFU, Estimated MBU and normalized busbw. For example,
RMSNorm over `[T,H]` contributes about `4*T*H` FLOPs, or `5*T*H` with a residual
add, excluding per-row scalar and reciprocal-square-root work. Count the fused
boundary tensors once for a logical IO estimate; do not invent intermediate HBM
writes removed by fusion. The common residual variant reads input, residual and
gamma, and writes normalized output plus updated residual. These are logical IO
estimates, not communication protocol traffic. A fused rule must account for its
actual variants and compute precision. Both bandwidth and MFU use the entire fused scope's
GPU window, not an inferred communication-only or norm-only time.

The existing TRT-LLM `allreduce_residual_rmsnorm` dispatcher records one outer
op and its actual fused/fallback decision. Its native call, allocation, return
values and fallback function are unchanged. No additional execution wrapper or
repeated one-shot/two-shot selection is introduced. Group identity and relevant
variant flags accompany the record; the fallback is not reported as fused.
Both bandwidth and MFU use the dispatcher's correlated GPU projection.
CPU-only preparation is not charged as GPU execution time.
`register_collective_costs` installs the communication rules for runtime capture.

`AllReduceRMSNormCost` counts `5*T*H` useful FLOPs and full-input AllReduce bus
bytes. In these implementations both one-shot and two-shot run the full epilogue
on every rank; only the reduction is sharded. The FP32 RMSNorm arithmetic selects
`fp32_simt`, with per-row scalar/rsqrt work and collective additions excluded.
The fused IO estimate counts input/residual/gamma reads and normalized/residual
writes. The partial-output pattern writes only its local normalized rows instead
of the allocated full `norm_out` buffer; it still executes the full epilogue.
Fallback intermediate
traffic is not modeled, so its Estimated MBU stays unavailable. Quantized output
and reduce-scattered residual variants retain metadata with an explicit missing
cost reason. This adapter does not cover sandwich norms, latent norms, MoE
finalization or the separate FlashInfer/CuteDSL fusion interface.

Dense `M x K` by `K x N` GEMM uses `2*M*N*K` FLOPs with an FMA counted as two;
a batched GEMM additionally sums over batches. This is an operation count, not
FLOP/s: divide by seconds for FLOP/s and by `1e12` for TFLOP/s. Dtype does not
halve FLOP counts; it selects the applicable hardware compute peak.

The registered DeepGEMM block-FP8 projection opts into `fp8_block_mm.v1`.
`BlockScaledMatmulCost` counts `2*M*N*K` and boundary IO including its FP32
activation/weight scales (1x128 and 128x128), with BF16 output. Matching E4M3 GPU
operands select `fp8_e4m3_tensor`; missing hardware peaks leave MFU unavailable.
Activation quantization is outside this scope. Scale layout conversion is inside,
but its temporary traffic, padding and reloads are excluded from the logical-IO
HBM estimate. A native trace can therefore contain several kernels for one op.
The MLA value-projection fallback retains its direct `torch.bmm` call; it has no
independent op-cost annotation.

TRT-LLM's 128-wide FP8 token-group quantizers record their values and FP32 scales.
Their traits and actual granularity/encoding identify the format. Both reject
other group sizes before launching, so the rule validates [M,K/128] scale geometry
without collecting `group_size` again. Failed calls have no cost.
Their cost counts one FP32 rescaling operation per input element; amax comparisons,
scale formation and FP8 conversion are excluded. The output dtype does not select
an FP8 Tensor Core peak. Logical IO counts the input read and value/scale writes.
The canonical entry's contiguous-scale copy is inside its scope, but temporary
scale storage, padding and layout-copy traffic are excluded from the boundary-IO
HBM estimate. Other granularities and scale encodings retain a missing-cost reason.

The Triton absorbed MLA prologue also records its typed cache/output holders.
For NoPE with a native cache and unmasked writes, its rule counts the query-tail
copy, new latent rows and slot indices. Untouched query-prefix channels and the
allocated cache capacity are excluded. RoPE, expanded attention, masked writes
and quantized caches require other cost rules. This copy/store scope has zero
model FLOPs; it reports Estimated MBU without selecting a compute peak, and does
not invalidate the MFU of a containing compute scope.

The existing MLA cache scatter/gather functions use the same direct binding.
`MLACacheCopyCost` counts one combined cache row per slot lookup and the split
latent/RoPE tensors, using each side's storage dtype plus index bytes. Output
arguments are writes, although these functions return `None`. Untouched cache
capacity and page gaps are excluded; repeated slots count per access. These
copy/cast operations have zero model FLOPs. Masked writes retain a missing-cost
reason without reading the device mask. The optional HBM estimate excludes
physical reloads and cache effects.

FlashMLA's dense absorbed decode entry snapshots its actual `cache_seqlens` and
records Q, cache, table, output and LSE descriptions. Its internal metadata adapter
reuses `AttentionMatmulCost`'s QK/PV formula, summing requests with their actual
lengths: `2*Hq*sum(valid_query_key_pairs)*(Dqk+Dv)` with lower-right causal alignment;
single-token decode uses every visible key. Its IO estimate reads each request's
visible latent rows once, shared by QK and PV and all query heads/positions. The
512-dimensional value prefix is not charged again alongside the 576-dimensional
key. Only referenced page-table entries and actual length entries are counted,
plus Q reads and output/LSE writes. Softmax, scaling, split/combine arithmetic,
padding, intermediate workspace, reloads and cross-request cache reuse are
excluded. Both native kernels belong to one attention op window. Sparse, FP8
cache and sink variants retain an explicit missing-cost reason.

FA3's packed variable-length entry uses the same QK/PV formula, with explicit
snapshots of both cumulative-length vectors. Each request is counted separately,
including lower-right causal alignment when query and key lengths differ.
This covers normal MLA prefill's causal current-chunk pass and noncausal prefix
passes without charging the full history on every pass. Q/K/V IO uses their
expanded per-head tensors; returned LSE and the length arguments are included.
Prefix KV projections, cache gathers, output copies and attention-state merges
remain separate work. Native BF16/FP16 full or causal attention is supported;
sliding windows, chunk masks, sinks, used-length overrides and FP8 variants retain
a missing-cost reason. This metadata adapter does not add a second attention
formula or change the vendor call's arguments/return values.

The existing CUDA and Triton attention-state merge registrations share
`attention_merge_state.v1`. It counts `3*N` useful FLOPs for combining two
partial value tensors with normalized row weights, using the FP32 SIMT peak.
Weight formation, normalization division, exp/log and LSE arithmetic are
excluded; this is not an executed-instruction count. Logical IO reads both
value/LSE pairs and writes the merged pair once, including in-place writes.

FlashInfer RMSNorm and native CUDA parallel RMSNorm also register at their
existing kernel entry points. `RMSNormCost` counts `4*N` nominal useful FLOPs
over the normalized elements, excluding per-row scalars, rsqrt and padding.
Their FP32 arithmetic selects `fp32_simt`, independently of FP16/BF16 storage.
The parallel call writes two output arguments and returns `None`; its rule
counts those writes, including in-place outputs, without treating output buffers
as additional reads. The optional HBM estimate counts each input and weight
once plus each output write; physical reloads and cache effects remain unknown.

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

## CPU validation

Run from the repository root in the project's Python environment. The shared
test configuration imports vendor modules, so a CPU-only run supplies an explicit
platform description before pytest loads it. This replaces device discovery;
the tensors and arithmetic still run on CPU.

```bash
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=tokenspeed-kernel/python python - <<'PY'
import pytest
import torch
from tokenspeed_kernel.platform import ArchVersion, Platform, PlatformInfo

Platform.override(PlatformInfo(
    vendor="nvidia", arch_version=ArchVersion(9, 0),
    device_name="CPU test fixture", device_count=0,
    total_memory=0, memory_bandwidth=0, sm_count=0,
    max_threads_per_sm=0, max_shared_memory_per_sm=0,
))
status = pytest.main([
    "tokenspeed-kernel/test/test_op_profile.py",
    "tokenspeed-kernel/test/test_profile_utilization.py", "-q",
])
assert not torch.cuda.is_initialized()
raise SystemExit(status)
PY
```

These tests exercise real CPU ops and torch profiler, covering repeated/nested
calls, cost coverage, argument forwarding, exception recovery and analytical
utilization from explicit timing fixtures. They do not qualify GPU timing or
kernel attribution. The illustrative layers remain inside the tests; no separate
example script is required.

## Native Nsight timeline

`NvtxTimeline` attaches to an unused `ProfileSession(backend="nvtx")`. Existing
op wrappers and layer decorators emit native NVTX push/pop ranges. Nsight owns
their timestamps and the CUDA activity timestamps. No CUDA Events are allocated,
recorded or read by this collector. CPU bookkeeping durations remain separate
metadata and are never GPU-time fallbacks.

Visible labels are short (`gemm.mm`, `L7`). A process-local monotonic numeric
NVTX payload joins each range to its full metadata identity, including repeated
calls and iterations. Process and thread IDs prevent cross-rank joins. The C
bridge records ordinary NVTX ranges; it does not submit invented timestamps.
Wrappers supply the display name separately from the serialized identity; the
collector does not parse identity strings to decide what to show. Offline analysis
reads the projection CSV and this process/thread's native ranges separately,
then joins payload IDs to Nsight range IDs before calculating metrics.

At context exit, the collector evaluates costs and writes the sidecar with
`device_analysis.timing_kind="pending_nsight"`. It synchronizes only when explicit
integer-tensor snapshots need to be read back. There is no synchronization for
op timing. `finish()` is idempotent and runs outside iteration scopes.

```python
with NvtxTimeline(
    session, costs=costs, hardware=hardware,
    stream=torch.cuda.current_stream(), max_scopes=128,
    library=bridge_path, output=metadata_path,
):
    with session.iteration(iter_id=0):
        result = layer(inputs)
```

After `nsys profile` finishes the report, run the single analysis command:

```bash
python -m tokenspeed_kernel.profile_nsys \
  --nsys /path/to/nsys --report timeline.nsys-rep \
  --metadata rank-0.metrics.json --output rank-0.nsight.json
```

The command exports SQLite and runs the official `nvtx_gpu_proj_trace` report in
a temporary directory. For each op/layer it joins the native NVTX identity,
process/thread and original range boundaries to that projection. The denominator
is exactly `Projected Duration (ns)`: **last associated GPU activity end minus
first associated GPU activity start**. Associated activities are those whose
launch APIs are enclosed by the CPU NVTX range. This includes gaps between GPU
activities and is not the sum of kernel durations. The evidence fields retain
GPU start/end, activity count, native range ID and native label.

Missing GPU activity leaves utilization unavailable; absent or ambiguous NVTX
ranges and mismatched captures fail explicitly. CPU scope duration, prior Event
measurements and previously computed utilization are never substituted. Nested
layers aggregate descendant op costs once using Nsight's CPU range nesting,
then use their own GPU projection as denominator. Uncovered model work remains
outside cost coverage. CUDA graph replay and cross-thread launches remain
unsupported by this eager collector.

The output has `device_analysis.timing_kind="nsight_gpu_projection"`. It contains
MFU, Estimated MBU, busbw, rates and a computed compact display label. The original
`.nsys-rep` remains unchanged: its native NVTX row contains the identity, not
metrics calculated after capture. `native_nvtx_label` and `nsight_range_id`
identify the corresponding original event. This command does not claim to
rewrite the native report or render the calculated labels into its NVTX row.
Historical Event-based labels in old captures must not be treated as the new
metrics; offline recalculation does not alter those embedded labels.

Estimated MBU uses the cost rule's explicit HBM estimate. `logical_io_once`
excludes cache reuse, rereads and workspace traffic. Missing peaks or costs
produce unavailable metrics with reasons; ratios are not clamped to 100%.

Supply hardware JSON or YAML with `name`, `compute_flops_per_second`,
`hbm_bytes_per_second` and `source`. Compute peaks are FLOP/s keyed by compute
type; bandwidth is byte/s. Unknown HBM bandwidth is `null`. YAML support uses
PyYAML's safe loader through `tokenspeed-kernel[profiling]`. JSON needs no new
dependency. Use the Nsight CLI version matching the capture when exporting.

The GPU integration tests exercise real captures and compare every denominator
to the official projection export. They are skipped without CUDA and
`TOKENSPEED_TEST_NSYS`; a skip is not GPU validation. CPU tests cover native
identity collection without Events, repeated calls, process isolation, projection
mismatches and missing activity. Existing captures can be analyzed without GPU
execution.

Reference: [Nsight NVTX GPU projection](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html#nvtx-gpu-proj-trace).
