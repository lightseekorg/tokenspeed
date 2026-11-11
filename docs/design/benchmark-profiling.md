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
These snapshots accept contiguous 1-D/2-D int32/int64 tensors. The per-argument
default is 4096 values. Grouped experts opt into 8192 route IDs. Matrices use a
flat alias view. Explicit dotted argument paths traverse
stored fields, dictionaries and NamedTuple fields, without evaluating properties.
Without a collector, CPU values become immutable tuples immediately. CUDA values
require an observer. The eager `NvtxTimeline`
queues same-stream D2H copies before the op's NVTX range into a preallocated
pinned buffer (`max_metadata_bytes`, default 1 MiB). Standalone values are
resolved after the window's final metadata synchronization. Serving uses the
existing result-copy event. Reusing a device buffer preserves each call's
original lengths. Unavailable snapshots remain `null` with a missing-cost
reason. Standalone capacity exhaustion raises. Serving reports it per argument.

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
The session does not infer graph replay or cross-thread attribution. The runtime
collector below supplies graph identities and the exact async result gate. Output
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

For a positive scope GPU duration `T` in seconds, Estimated MBU is
`sum(estimated_hbm_bytes) / (hbm_bytes_per_second * T)`. The explicit
`logical_io_once` policy counts the rule's useful boundary reads and writes once;
it excludes protocol/workspace traffic, reloads and allocated-but-untouched
capacity. Host-mapped reads remain separate from estimated HBM bytes. An unknown
HBM estimate or peak leaves MBU unavailable, independently of compute or
communication normalization.

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

The source-bound ordinary Lite recurrence keeps `T*H*(7*D*D+D)` useful FLOPs
when a negative read seeds zero state; a negative write suppresses only its
state store. Logical state IO counts only actual nonnegative source/destination
rows (`4*H*D*D` bytes each), plus the explicit boundary tensors and actual
output. Missing snapshots, invalid positive pages and cross-sequence write
hazards remain unavailable; this does not extend tree/verify policies.

For the ordinary Triton output-bank convolution, only the declared
`pad_slot_id=-1` source returns before data/output access. Its padded row counts
zero useful FLOPs and one logical source-index read, with no output/history
write. Live rows retain `2*C*W` useful FLOPs and their boundary/history IO; negative
live destinations suppress only history stores. A skipped row leaves the
output buffer unchanged, and its actual graph execution remains timed.

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
halve FLOP counts. Proven execution precision and engine select the compatible
compute peak; storage dtype alone supplies neither.

The registered DeepGEMM block-FP8 projection opts into `fp8_block_mm.v1`.
`BlockScaledMatmulCost` counts `2*M*N*K` and boundary IO including its FP32
activation/weight scales (1x128 and 128x128), with BF16 output. E4M3 storage alone
leaves its positive FLOPs unnormalized; a reviewed selected-engine contract and
matching hardware peak are required for `fp8_e4m3_tensor` MFU.
Activation quantization is outside this scope. Scale layout conversion is inside,
but its temporary traffic, padding and reloads are excluded from the logical-IO
HBM estimate. A native trace can therefore contain several kernels for one op.
The MLA value-projection fallback retains its direct `torch.bmm` call and uses
`mla_project_value.v2`; engine normalization still requires execution evidence.

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
snapshots of both cumulative-length vectors and any supplied `seqused_q` or
`seqused_k`. Each used length must fit its packed sequence capacity. Each request
is counted separately, including lower-right causal alignment when query and key
lengths differ.
This covers normal MLA prefill's causal current-chunk pass and noncausal prefix
passes without charging the full history on every pass. Q/K/V IO uses their
expanded per-head tensors; returned LSE and the length arguments are included.
Prefix KV projections, cache gathers, output copies and attention-state merges
remain separate work. The rule supports native BF16/FP16 full or causal
attention. Sliding windows, chunk masks, sinks, missing used lengths and FP8
variants retain a missing-cost reason. The adapter uses the existing attention
formula and forwards the vendor arguments unchanged.

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
calls and iterations. Process and thread IDs prevent cross-rank joins. The
NVIDIA Python NVTX package emits same-thread push/pop ranges in the
`TokenSpeed` domain. Its `get_domain(...).push_range(message=..., payload=...)`
API stores Python integer payloads as signed 64-bit integers. The collector
accepts scope IDs from `0` to `2**63 - 1` and rejects overflow before emitting
a range. Wrappers supply the display name separately from the serialized
identity. Offline analysis joins the projection CSV to this process/thread's
native ranges by payload and Nsight range ID. It reads `int64Value` for Python
sidecars and `uint64Value` for older C sidecars, preserving integer precision
through the join. [NVIDIA Python NVTX payload types](https://nvidia.github.io/NVTX/python/annotation_attributes.html#payload).

Install the optional `tokenspeed-kernel[nvtx]` extra, which pins
`nvtx==0.2.17`, or install that NVIDIA package into an existing environment.
Run eager collection under Nsight Systems 2026.5 or newer with
`--trace=cuda,nvtx`. Remove `library` or `nvtx_library` from existing timeline
configs. The Python NVTX package is loaded when eager native ranges are
requested. [NVIDIA nvtx 0.2.17](https://pypi.org/project/nvtx/0.2.17/).

At context exit, the collector evaluates costs and writes the sidecar with
`device_analysis.timing_kind="pending_nsight"`. It synchronizes only when explicit
integer-tensor snapshots need to be read back. There is no synchronization for
op timing. `finish()` is idempotent and runs outside iteration scopes.

```python
with NvtxTimeline(
    session, costs=costs, hardware=hardware,
    stream=torch.cuda.current_stream(), max_scopes=128,
    output=metadata_path,
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

The raw `ops` and `layers` arrays retain CPU call bookkeeping, including
`cpu_duration_ns`; their top-level `timing_kind` remains `cpu_scope`. GPU timing
and utilization are in `device_analysis.scopes`, joined by `trace_label`.
`gpu_duration_ns` there is the Nsight GPU projection used by MFU/Estimated MBU.

To display those calculated metrics in the native Nsight viewer, append analysis
ranges to a new report with the matching installation's `ImportNvtxt` utility:

```bash
python -m tokenspeed_kernel.profile_native_report \
  --report timeline.nsys-rep --metrics rank-0.nsight.json \
  --output timeline-utilization.nsys-rep \
  --import-nvtxt /path/to/ImportNvtxt --target-id HW:VM --nsys /path/to/nsys
```

Use the actual `HW:VM` target returned by the matching installation's
`ImportNvtxt --cmd info -i timeline.nsys-rep`. In the new report, expand the
per-rank `MFU / Est.MBU(HBM) / Est.CommBWUtil | derived GPU analysis` lanes.
These imported analysis lanes show op and layer labels over their correlated
GPU spans, including gaps, alongside the original CUDA activities. Their
timestamps use the same capture's relative time base. The original capture
and original launch ranges are preserved; no GPU rerun or Event timing is
introduced. Missing metrics retain N/A and their reasons; mixed compute scopes
also show the available per-type MFU contributions. The JSON must come from the
supplied capture. Model, layer, nested module, and op levels occupy separate
lanes within each rank, so the native viewer displays them simultaneously.
Successive layers share a lane; layer IDs remain in each label. Recorded parent
IDs determine levels. Scopes spanning multiple layer IDs form model lanes.
Overlapping spans within a level use additional lanes without changing their
measured boundaries; no lane relies on nested-range rendering.
Rank 0 annotations occupy a separate synthetic process group. Other ranks share
an independently collapsible group, and the all-rank aggregate has its own group.
Collapsing these groups hides no measurements from the report. Initial expansion
state is controlled by the Nsight viewer, not by the NVTXT import format.
The runtime imports VizTracer only when its `VIZTRACER` activity starts;
`CUDA_PROFILER` capture does not require initializing that CPU tracer.
The `.nvtxt` sidecar retains the imported labels and boundaries.
The analysis command also accepts paired `--annotated-report` and
`--import-nvtxt` options to produce both outputs in one invocation. New analysis
JSON includes the source capture's SHA-256, checked before native annotation.

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
Native annotation uses NVIDIA's documented
[Import NVTXT](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#import-nvtxt)
merge command.


### Diagnostic CUDA Graph attribution

The diagnostic decode drivers use a separate capture collector; the regular
`ProfileSession` eager collector above still does not support capture. During
capture the diagnostic collector records each opted-in call's stable scope ID,
parent ID and CUDA graph node IDs. It creates no per-op CUDA Events. Replay has
no Python per-op hooks. Integer metadata, including routing IDs, is resolved
from retained graph tensors after an untimed replay, outside capture.

Offline analysis joins capture node IDs through Nsight's explicit
`CUDA_GRAPH_NODE_EVENTS.originalGraphNodeId` clone relation to replay activity.
Both clone lookup and activity lookup are isolated by process ID; runtime
`cudaGraphLaunch` correlation IDs select individual replays. Kernel names and
timestamp overlap are not used to assign work to an op. GPU activity sum,
interval union and first-to-last span are retained separately. Diagnostic MFU
and Estimated MBU use the span, including device gaps.

Mixed-compute diagnostic scopes use the same additive ideal-time definition
described under [hardware profile selection](#hardware-profile-selection-and-provenance).
Each layer counts descendant op costs once. Nodes outside opted-in
ops make the total explicitly partial; known contributions and missing reasons
remain visible. Estimated MBU is a logical boundary-IO model, not measured HBM
traffic. All-rank model labels sum local costs, divide by all selected GPUs'
combined peak, and use the earliest-to-latest associated device activity span.

The native report writer also accepts `nsight_graph_node_ids` diagnostic JSON
and groups imported lanes by rank. These diagnostic traces explain the
captured execution; node-level Nsight timing is not a low-interference speedup
measurement. A separate benchmark with detailed collection disabled is needed
for final performance comparisons.


## Runtime graph collection

`RuntimeUtilizationProfile` observes the existing serving path; it does not force
eager execution or change backend selection. `TOKENSPEED_TIMELINE_CONFIG` names a
JSON file with the existing run/output/hardware/layers/window/HBM policy fields.
Graph collection additionally requires explicit `cuda_runtime_library` and
`cupti_library` paths. CUPTI must be the same already-loaded instance used by the
external profiler. A selected eager step emits a separate
`rank-N.step-M.eager.json` through Python NVTX. Its integer snapshots become
CPU-readable at the existing result-copy event. Eager capture reserves one of
`metadata_slots` independently bounded sidecars. Exhaustion records an explicit
metadata drop before allocating the eager buffer. The ready identity is saved
at the event gate. The existing profile stop materializes the sidecar and
publishes its SHA256 and size after the measured window. Graph payloads still
drain through the live writer. Use `device_ready_sequence` for result readiness
order because eager rows can be written after later Graph rows. The same replay
index records any later ForwardId binding. The paired run manifest seals those sidecars with the
rank config and Graph artifacts. If Python NVTX is unavailable or the domain is
inactive, the step records an explicit drop reason and executes eagerly.

Eager grouped expert accounting uses the live unsharded token count and this
step's own route IDs. A packed multi-request forward can supply the total live
rows without a uniform per-request width. With 48 layers and a `472×12` int32
route matrix per layer, the route snapshots require 1,087,488 bytes before other
metadata. Set `max_metadata_bytes` to cover the requested observer data.

Mixed analysis projects the eager ranges with Nsight against the same SQLite
export used for Graph joins. Each eager range is matched by domain, signed
payload, process, thread and boundaries. The correlated CUDA activity row IDs,
span and correlation count must agree with Nsight's projection. A signed
`Profiler integer metadata D2H` range identifies each serving observer transfer.
The analyzer verifies the original copy row and byte count and excludes these
rows from Model, Layer, Op and scheduler model-work activity. Work spans can
still contain elapsed gaps. Post-Model sampling support is attributed to the
serving step separately. Graph steps retain node and clone attribution.
Scheduler overlap uses the captured activity union for either path. The explicitly selected offline cost provider
uses the eager sidecar's live tensors and branches with the uniquely owned
CUPTI runtime and GPU activity rows. Reviewed engine guards default to Graph.
A guard that independently qualifies eager may declare
`attribution_modes: ["eager_nvtx_gpu_projection"]`. The eager proof retains
the signed scope payload, launch resources, worker, CUDA device and context,
source-report and sidecar fingerprints. Eager compute without matching
reviewed evidence retains its useful FLOPs and N/A MFU. Sidecars from earlier runs are required
for eager attribution in those runs.
`max_metadata_bytes` bounds each capture slab and pinned slot;
`metadata_slots` bounds retained asynchronous results. `ranks` optionally selects
ranks. Each rank saves its normalized config contents/SHA256 and selected hardware
input snapshot before capture. `TOKENSPEED_PROFILE_RUN_ID`, when set, must agree.

The existing `Engine.start_profile(...)` forwards its activity/settings arguments
and returns the existing success status. Start it after service warmup, for
example with `activities=["CUDA_PROFILER"]` and the same `profile_id` as the run.
Only this request arms the relative `start_step`/`num_steps` metadata window;
warmup cannot consume it. `stop_profile()` disarms and flushes only CPU-ready
metadata before acknowledging. Use a unique run/output for another window. With
external Nsight, keep full-process node tracing (`--capture-range=none`) so startup
capture and clone evidence is retained. The start/stop API still controls the
metadata window. `enable_nvtx=False` can remain the normal service setting; the
collector emits its own necessary replay identity labels.

`ProfileSession.observe(observer)` is the shared public scope/integer observer
seam; standard module pre/post hooks provide decoder scopes. The runtime's direct
`ModelRunner.model.forward` call has an explicit root scope because this method
call bypasses PyTorch module hooks. Only actual graph capture is observed, after
warmups; replay performs no per-op Python calls. Each declared device integer
argument gets a stable D2D snapshot before its source storage can be reused.
Those observer graph nodes are listed separately and excluded from useful scope
costs. Their overhead remains part of the collected workload and enclosing span.
CPU integer arguments during capture remain unavailable, rather than freezing a
warmup/capture value as replay metadata.

After replay, same-stream nonblocking copies move the bounded slab to a free
pinned slot. A lease identifies `(slot,generation,step_id)` and cannot be reused
before persistence. The existing result-copy event covers these copies;
`ModelExecutionResult.sync()` retains its exact-once event wait and marks metadata
CPU-ready. A background CPU-only writer saves the binary slab and a small hashed
index. There is no device value read, synchronization or event query in the
replay loop or writer. Exhaustion, unsupported geometry, persistence errors and
unresolved results remain explicit. Even if writing the failure index also fails,
the writer acknowledges queued CPU work so stop does not deadlock; failed slots
remain owned and the manifest retains `writer_error`.

Artifacts are `rank-N.capture-C.json` (static op/layer descriptions, node scopes,
observer nodes, snapshot offsets), `rank-N.replays.jsonl` (exact per-step identity
and payload index), `rank-N.step-S.metadata.bin` (integer bytes), and
`rank-N.window.json` (`tokenspeed.profile_window/v1`, selected/ready/persisted/drop
counts, unresolved results and writer status). `device_results_ready` carries the
actual serving step, request IDs, pool slots and optional bound Rust ForwardId.
Late binding is a `forward_identity_bound` event with the same step identity.
Device readiness alone is `completion_identity_unavailable`; only the separate
true post-commit service evidence establishes committed token endpoints. An
unavailable ready row increments `dropped_metadata` only after it is written.
Unknown/index failures remain visible via `writer_error`. A measured decode
subset may exclude precisely identified disjoint drops; unknown drops cannot be
ignored, and whole-run coverage remains partial.

Offline code must decode and hash-verify each replay payload, replace that op's
`tensor_values`, and reevaluate trusted cost rules. Capture-time dynamic costs
are unavailable and must not be reused. Construct a `CostRegistry`, call
`profile_costs.register_compute_costs` and `register_collective_costs` with the
explicit HBM policy, then pass that registry to analysis. The generic rules
include `mla_decode.v2`, `mla_project_value.v2`, and `grouped_experts.v2`.
A trusted caller can additionally register model-specific CPU-safe cost rules
through an explicitly installed provider, for example
`my_model.profiling.register_costs(registry, hbm_policy=...)`. This cost entry
should not import model execution or load weights. No import path from a capture
is executed, and omitting model rules leaves generic coverage explicitly partial.

MLA validates actual query-axis or flattened request/query geometry, causal row
lengths and DCP1; cache IO uses a logical request union. Grouped expert costs keep
route instances separate from unique weights/scales, preserve useful intermediate
size declared before padding, and reject unsupported ownership/routing policies.
Missing rows/scales/constructor metadata stay unavailable. Engine normalization
requires the actual selected callable's owned contract or a reviewed selected-engine
proof bound after the exact native join. Without that proof, Torch bmm and private
FP8 expert apply retain useful algebra with an unknown engine; multiple leaves
also need a useful-FLOP partition. Ordinary B1 decode is the initial runtime scope;
padding, masked/uninitialized state, unsupported prefill graph and other missing
critical work remain explicit partial coverage. CPU serialization/math fixtures
cannot establish GPU numerical correctness, graph mapping, complete model costs
or collector overhead.


## Hardware profile selection and provenance

For the shipped paired serving flow, select a rank-local profile in the
timeline config's `hardware` field; see the [service TPOT pilot](../../examples/profiling/README.md).
There is no default GPU. Offline `--hardware PATH`, when supplied, must match
the captured profile. Programmatic callers of `HardwareProfile.from_selection`
can supply a path or `TOKENSPEED_PROFILE_HARDWARE`; that environment variable
does not replace the paired runtime's required timeline `hardware` field.

The [example profiles](../hardware-profiles/) describe theoretical per-device
NVIDIA product specifications. They do not establish kernel/model support or
measured sustainable throughput. Select the exact SKU, board form factor,
capacity, nominal maximum configurable TDP and full-device/MIG scope. A CUDA
name alone cannot establish all those fields; captures record observed identity,
verified fields and remaining unverified fields. Known device-name, capacity
or partition mismatches are errors. MIG peaks require a separately sourced
slice profile; a full-GPU peak is never scaled by a guessed fraction.

Version 2 retains the four numeric/profile-summary fields and adds `device`,
`units`, `compute_metadata`, `sources` and `bandwidth_source_id`. Compute keys
identify precision **and** engine (`bf16_tensor`, `fp32_simt`, etc.). Each compute
entry records the published value/unit/sparsity and its explicit conversion to
a dense FLOP/s peak. Sparse Tensor Core specifications may be halved only when
the source documents that relationship. FP32 SIMT and byte/s bandwidth are not
halved. Values use decimal units. Nominal TDP is not the current configured
power limit, and runtime clock/power observations never adjust theoretical peaks.

The existing `HardwareProfile.from_file()` JSON/YAML reader also accepts legacy
four-field profiles. Those have no structured SKU evidence; their identity audit
is explicitly `legacy_unverified`. They are not silently upgraded. New captures
save the complete selected UTF-8 file contents, path and SHA256 under
`hardware.input_snapshot`. All graph/eager metadata and offline reports retain
that snapshot. The launcher passes the selected absolute path and expected hash
through the worker environment; a changed file fails instead of silently using
a different denominator. Offline `--hardware` verifies the captured profile and
input hash, and rejects a different configuration. Historical captures without
file snapshots remain usable with their recorded profile; original file bytes
cannot be recovered from those numerical fields.

An op's storage dtype cannot establish its compute engine. Generic matrix cost
rules require an explicit `compute_type` implementation trait or recorded host
execution decision; missing evidence leaves MFU unavailable while keeping modeled
FLOPs and bytes. A rule tied to a known implementation may explicitly encode its
compute contract. Unknown peaks are never borrowed from another SKU or engine.

The shared eager/runtime-graph calculation uses
`mfu = mfu_total = sum(F_k / P_k) / T`, where `F_k` is useful FLOPs for a proven
precision/engine component, `P_k` its compatible rank-local dense peak, and `T`
the scope's associated GPU span in seconds, including gaps. Optional
`compute_components` must partition canonical useful FLOPs exactly. Contributions
are additive; no type-count or child-ratio averaging is performed. Positive
unknown components retain their FLOPs and missing reasons: the full MFU is N/A,
while `mfu_total_known` is explicitly partial. Zero model FLOPs need no compute
peak and do not imply GPU inactivity. Manual/scheduler diagnostics use the same
ideal-time convention with their versioned fields. Rank-zero metrics never
multiply a single GPU's peak by world size.


Keep the selected hardware file immutable during collection. The runtime saves
its contents and SHA256 in the capture; offline analysis checks that identity.
Use the shipped `examples/profiling/service_tpot.py pair` and
`tokenspeed_kernel.profile_nsys` flow with a new output directory for every
capture. Custom diagnostic launchers outside this repository need their own
recorded resource and hardware preflight; they are not prerequisites for this
public flow. Explicit precision names in v2 are
extensible (including future published formats); the schema checks their syntax
and matching engine key instead of interpreting them as storage dtypes or adding
op support. An unknown op compute key still has no utilization denominator.

## Selected leaf identity and versioned diagnostic costs

A diagnostic wrapper's `OpRecord.implementation` identifies that wrapper, not
necessarily the selected execution backend. An implementation may declare a
reviewed source contract with metadata-only `@profile_engine_contract(...)`.
The declaration belongs to the exact original callable; same-name replacement
and `functools.wraps` attribute copies do not transfer it. This declaration is
needed because `OpProfile` controls capture and registry specifications permit
same-name plugin replacement; neither proves an instruction engine. Resolution
stays internal to the existing recorder and adds no dispatch wrapper.
Dispatchers can call
`record_selected_kernel(name, solution, implementation=selected_callable)`
while an op is active. This public helper appends ordered `selected_kernel_names`,
`selected_solutions`, `selected_compute_types` and `selected_engine_contract_ids`
to that same op's host metadata. JSON exports these tuples as arrays. It neither
creates a nested op nor adds replay callbacks or device reads. Outside an active
op it does nothing. `profile_session_active()` lets diagnostic adapters leave
unprofiled calls on the original path.

The reserved `selected_compute_type` and `selected_engine_contract_id` summaries
are valid only for one selected leaf. A composite or repeated dispatch retains
all identities and clears the summaries: one engine cannot normalize all its
work without a separate FLOP partition. These fields do not replace an actual
caller argument named `compute_type`. The exact portable
`triton_mla_decode_with_kvcache` implementation uses FP32 multiply/sum for QK and
latent AV; its registered query-width-one contract can select `fp32_simt`.
Unproven leaves, including `torch.bmm` without a matched engine proof, remain unknown.
An FP8-weight MoE apply identity alone does not prove its private compiled
artifact's activation precision or instruction engine.

The task-owned `diagnostic_mla_decode.v2` rule requires the original logical
batch/width, actual query layout, request/query-position mapping and actual
visible lengths for every interface row. Full causal QK plus latent AV useful
FLOPs are `2*h*(D+R)*sum(visible_query_key_pairs)`. Query-axis rows contribute
`W*L-W*(W-1)/2`; flattened width-one rows already contain their individual
lengths and do not subtract another triangle. `logical_request_once.v2` reads
the cache-row union once per logical request across heads and queries, while
sequence lengths and touched page-table entries follow actual interface rows.
This is a stated logical sharing policy, not physical HBM traffic. Prologue KV
writes and separately recorded projections are not counted again. A projection
enclosed within the same op counts boundary weights/gate/output and excludes
internal latent-buffer IO.

The grouped expert v2 rule separates local route instances for useful FLOPs
`6*A_local*d_group*I_useful` from unique selected experts for weight/scale IO.
A constructor side table retains `I_useful` before padding; actual weight shapes
supply `I_allocated`. Install diagnostic adapters before model construction;
pre-existing modules without constructor evidence retain unavailable useful
geometry. FP8 scale bytes come from actual scale descriptions and block geometry.
Missing descriptions leave Estimated MBU unavailable while retaining calculable
FLOPs and a labeled incomplete logical-byte subtotal. Storage dtype only supplies
byte sizes and never establishes the execution engine.

The first diagnostic contract is full causal B1 width8, attention/dense TP8,
expert TP1/EP8 and DCP1, with live group-major rows, global expert IDs, trivial
contiguous ownership, `zero_expert_num=0`, no redundant placement/EPLB,
no all-to-all and finalized expert output. Unsupported policies or missing live
metadata remain unavailable; they are not filled from warmup, expected load
balance or another replay. Static metadata/identity attachment and CPU integer
checks do not implement or validate per-replay transport, GPU shapes, numerical
parity or complete Model/Layer coverage. Historical v1 reports retain their
original assumptions and values.
