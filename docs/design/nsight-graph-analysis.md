# Offline CUDA graph replay analysis

`tokenspeed_kernel.profile_nsys` consumes the runtime graph collector's actual
per-replay artifacts. It exports the completed Nsight report with the matching
`nsys` version and produces rank-local Model, Layer and Op metrics. Analysis
runs on CPU and does not initialize CUDA or import model execution.

```sh
python -m tokenspeed_kernel.profile_nsys \
  --nsys /path/to/nsys --report run.nsys-rep \
  --graph-replays run/rank-0.replays.jsonl \
  --graph-window run/rank-0.window.json \
  --hbm-policy logical_io_once --scheduler-mode disabled \
  --cost-provider builtin --output graph-metrics.json
```

The replay index supplies each capture path and binary payload path. Preserve
these original artifact locations. Files are fingerprinted; byte length, SHA256,
lease slot/generation, step identity, dtype, element count and snapshot bounds
must agree. Capture-time integer values are cleared. Each replay's lengths and
routing IDs are decoded and passed to the public cost registry again.
Useful costs describe that replayed model call, including speculative work later
rejected; they are not scaled by committed-token counts. Scheduled, device-ready
and authoritative-commit evidence remain separate.

The CLI registers the public compute and collective rules. Unknown model rules
remain unavailable. Trusted Python callers can construct a `CostRegistry`, call
`register_compute_costs` and `register_collective_costs`, register their model's
CPU-safe rules, and pass it explicitly to `analyze_graph_replays`. The analyzer
never imports serialized module or callable names. For example, a plugin may
provide `register_costs(registry, *, hbm_policy)` separately from its model
execution package.

Graph CLI calls must explicitly select `--cost-provider builtin` or an installed
`--cost-provider my_model.profiling:register_costs`. The latter registers the
model's CPU-safe rules alongside the builtin compute/collective rules. It runs
only the function explicitly named by the user; captures never choose imports.
Missing installations/dependencies are reported before exporting the trace.
The output records provider selection, HBM policy and source SHA256. Python
callers pass `registry=costs` to `finalize_graph_nsight`; explicit `registry=None`
selects builtin costs only. Optional `cost_provider_source` retains caller
provenance and never loads code.

For engine normalization, an explicit trusted Python provider can call
`register_selected_engine_resolver` with reviewed guard bytes and their expected
SHA256. Guards bind the actual source/API branch, boundary, hardware and complete
native role set after the exact graph join. Reused cache bytes need reviewed
artifact evidence; current generation needs matching source/include/compiler
and worker/run lineage. A source hash or storage dtype alone is insufficient.
Optional reviewed supplementary bytes/SHA must bind the exact run, rank, process
and capture, without rewriting raw captures. Engine-class evidence need not claim
historical module-byte identity; unresolved cases retain positive useful FLOPs
and an unavailable normalization. Neither captures nor run manifests select
executable imports.

The standard paired driver records a body/SHA observation before importing
Engine, then embeds it in the existing per-arm timeline. Runtime construction
and the sealed manifest check the same default-expanded config, run/case/arm
and executing driver source; capture and collector baseline/final retain the
same identity. Source/API/compiler bytes must predate that real observation,
remain unchanged, and generated cache artifacts must belong to its actual
worker/window lineage. Workers record their own raw PID/namespace/clock facts;
no process birthday is inferred by combining proc ticks with another clock.
Missing or inconsistent evidence leaves normalization unavailable.

GMMA include provenance retains the ordered actual roots and each probed path.
Existing regular candidates must have reviewed hashes; a typed, path-specific
ENOENT is a retained negative search result. Symlinks, unreadable/unknown errors,
changed candidates/order and unrepresented include environment segments reject.
Only fully checked CPATH/CPLUS_INCLUDE_PATH/C_INCLUDE_PATH segments are accounted
for by this closure; other compiler/FLAGS overrides still match reviewed values.

## Attribution and metric coverage

Each selected step must have one exact replay NVTX label on the recorded process
and thread. Its range must contain exactly one `cudaGraphLaunch`; the unique
runtime correlation ID selects the GPU activities. Same-process explicit
`originalGraphNodeId` clone relations map activities to this capture's CUPTI
node IDs. Node names, ordering and timestamp overlap never assign kernels to an
Op. Missing or ambiguous joins remain partial. Clone cycles or conflicting
parents are rejected.

Capture also caches `cudaGraphNodeGetType` for each exact CUPTI node ID.
Only proven CUDA Empty/WaitEvent/EventRecord nodes (enum 5/6/7) may lack an
independent GPU activity. They remain owned nodes with capture/clone evidence;
unknown, child-graph and GPU work nodes still require activity. One cached
per-capture type index supplies both engine-join and scope/native completeness;
actual type/activity contradictions are checked per replay. No type query runs
on replay. The enum/API are defined in the
[CUDA Runtime documentation](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-runtime-api/group__CUDART__TYPES.html).

Observer copy nodes are excluded. Recorded scope ancestry determines unique
cost membership; capture CPU timing does not determine ancestry. Every scope
reports its GPU span, activity duration sum and interval union. Span is the
utilization denominator, including gaps. The shared `scope_metrics` calculation
retains compute, HBM and communication cost semantics. Model and Layer scopes
with uncosted or missing GPU nodes remain partial, with explicit unknowns.
Cost coverage uses each scope's own captured nodes and costed descendants.
Nodes outside Model remain explicit whole-graph coverage unknowns; they do not
invalidate a complete Model scope. Named GPU support Ops remain in their actual
scope and may have zero model FLOPs without implying GPU inactivity.

A flushed window can contain both persisted steps and explicitly dropped
metadata. Valid payloads retain their per-step metrics after SHA256, identity,
lease and node checks; dropped steps remain unavailable and whole-run coverage
remains partial. Unexplained counts, writer errors, unresolved results or other
global integrity failures still invalidate the selected scopes.

For a structurally valid scope, MFU, HBM and communication availability are
independent: an unknown compute engine does not erase known MBU or communication
metrics, and unknown HBM does not erase known MFU. Required unknowns still leave
scope/run coverage partial and gate native verification. Invalid joins, timing,
lifecycle or cost coverage still invalidate the affected scope's metrics.

Quantities are per rank. They are not whole-group FLOPs or physical link traffic.
`algbw_bytes_per_second` and `busbw_bytes_per_second` use NCCL-tests normalized
traffic divided by the complete scope GPU span; a fused compute/communication
Op has the same denominator. Estimated HBM traffic follows the selected policy
and does not measure physical HBM traffic. Optional communication peak
utilization is displayed only when supplied by the shared calculation; otherwise
it is N/A. For a verified MNNVL one-shot multicast allocation on one NVSwitch
node, the reviewed payload model uses TX=S and remote RX=(P-1)S. Directional
rates and ratios use the whole fused Op GPU span and matching one-direction
capacity. `Est.CommBWUtil` is max(TX utilization, RX utilization); the payload
model and bottleneck direction accompany the value. This excludes unmodeled
wire/protocol overhead. Independent `CommRefEff` divides normalized bus BW by
its reference capacity; it is distinct from directional payload utilization.
Partial attribution clears rates, utilization ratios and warnings while keeping
static byte counts, capacities and model evidence in JSON. Model and Layer
communication utilization remains unavailable.

## Service proof

To emit the fixed `tokenspeed.native_coverage/v1` receipt, additionally provide
`--metadata instrumentation.json --model-config model-config.json
--native-coverage run/rank-0.native-coverage.json`.

The metadata is the service driver's verified instrumentation result, containing
all measured request ForwardIds, step/capture identities, artifact hashes and
expected scope IDs. The analyzer reads model depth from that exact frozen model
config. The receipt includes every measured ForwardId, actual layer and scope
coverage, source hashes and mandatory `critical_unknowns`. Missing any measured
forward, layer, scope, critical cost or node provenance prevents `verified`.
The requested decode set comes from those exact service ForwardIds and expected
layer/scope IDs, not every step in the collector window. Identified dropped
metadata can be excluded from that set only with verified service metadata,
matching window/index hashes and an exact disjoint dropped-step list. This
never excludes a selected critical unknown or a global integrity failure. The
receipt preserves whole-run status and unknowns alongside requested coverage.
Anonymous correlated nodes outside captured scopes remain critical for native
completeness, even when their source node IDs are known. Excluding a node from
Model's numerical denominator does not explain its semantic ownership. A named
support Op must be recorded by the actual execution and costed under its own
explicit scope; old captures are not relabeled to fill coverage gaps.
Device readiness and a persisted binary payload establish metadata readiness;
they do not establish native ownership. `synthetic_cpu_fixture` is an explicit
API evidence kind and can never produce verified GPU proof.

## Native report annotations

The offline writer requires the actual report target selected from the matching
`ImportNvtxt --cmd info -i REPORT` metadata. Supply its `Hw:Vm` ID explicitly;
there is no assumed target such as `0:0`.

```sh
python -m tokenspeed_kernel.profile_native_report \
  --report run.nsys-rep --metrics graph-metrics.json \
  --output annotated.nsys-rep --nsys /path/to/nsys \
  --import-nvtxt /path/to/ImportNvtxt --target-id HW:VM
```

The writer allocates fresh synthetic process IDs after inspecting existing
measurement and derived lanes. Communication Op labels put `Est.CommBWUtil`
immediately after the Op name, followed by the whole Op GPU span.
`Est.MBU(HBM)` independently compares estimated useful HBM bytes per second
with the HBM peak (3.35 TB/s for the H800 SXM profile); `Est.CommBWUtil` uses
the directional communication capacity (200 GB/s for the reviewed MNNVL path).
Both use the whole Op GPU span, including fused computation. Labels retain MFU,
commAlgBW, commBusBW and their provenance. Values without proof remain N/A,
with their reasons. Scopes without any correlated GPU interval are counted as omitted.
Known unmodeled communication is displayed as `Comm=N/A` with its reason under
the [cost-assumption contract](../hardware-profiles/communication.md#known-unmodeled-communication),
even when that Op's independent MFU or Estimated MBU is available.

After importing, the writer exports both reports and verifies all original
CUPTI and graph-clone columns exactly. Original NVTX measurements retain their
complete non-`eventId` multiset and referenced `StringIds`. Only the export-local
integer-primary-key `NVTX_EVENTS.eventId` may change; equivalent duplicate rows
retain complete audited locator sets. Existing derived ranges and every added
label/timestamp must also match. A new `.verification.json` records the audit.
The source report and metadata are never overwritten.
