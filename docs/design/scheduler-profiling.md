# Scheduler trace analysis

`tokenspeed_kernel.profile_scheduler.analyze_scheduler` is an offline,
standard-library-only reader of Nsight Systems SQLite exports. It reports
scheduled graph launch timing and the GPU activity between consecutive launches.
Select `selection_mode="legacy"` for a request NVTX range or an established
single-request capture; select `selection_mode="collector"` for collector step
identities and authoritative service metadata.
It does not run or modify the engine.

## Identity before timing

Select a worker by its recorded OS process ID. A unique request NVTX marker
selects CPU replay markers within that request. The request marker may belong
to the driver process; each replay and its enclosed `cudaGraphLaunch` runtime
call must share the exact `globalTid`. GPU activities join the runtime call by
`correlationId` and the full packed process key, including the namespace bits.
A reused launch correlation, ambiguous launch, incomplete range, or ambiguous
worker process fails analysis. Missing launch handling is an explicit
`missing_launch_policy`: `error` rejects missing identities, while
`report_unavailable` preserves the replay marker with unavailable launch/timing.
The latter reports marker and identified-launch counts separately, leaves total
scheduled count unavailable, and marks statistics partial. It never bridges
unknown markers to construct a consecutive pair.

Some exports have no request marker. Explicit whole-capture selection requires
source evidence that the export contains one request. Its first/last replay
CPU envelope is descriptive and is not a request wall-time measurement.

Optional capture JSONs identify model nodes using `process_id`, `rank`, and
`graph_scopes[].graph_node_ids`. They must come from the same run as the export.
The reader follows explicit `CUDA_GRAPH_NODE_EVENTS` clone ancestry and requires
a unique matching capture for each replay with GPU activity. It never assigns
nodes by kernel name, activity order, or temporal containment. Without capture
metadata, timing describes the entire correlated graph launch, and the model
boundary remains unavailable. Whole-launch timing is retained alongside model
timing when capture metadata is present.

## Timing and coverage

Each scheduled replay reports CPU start/end, runtime and NVTX row identities,
launch correlation, and these distinct GPU measures:

- Span: last selected activity end minus first selected activity start.
- Sum: sum of selected kernel, memcpy, and memset durations; concurrency may
  cause double counting.
- Union: wall time covered by at least one selected activity.

For consecutive scheduled replays, period is next GPU start minus previous GPU
start; signed gap is next GPU start minus previous GPU end. Negative gaps
represent overlapping spans. Replays without correlated GPU activities keep
unavailable timing; the reader does not bridge them to invent consecutive pairs.
Quantiles use linear interpolation over sorted observations.

Positive gaps include coverage from every captured process, device, context,
stream, and kernel/memcpy/memset table. This time intersection measures coverage
only; it does not establish op ownership, sampling/state-update/communication
semantics, rank identity, or causality. Device coverage is separate from the
wall-time union across all devices. Unknown ranks remain unknown rather than
being inferred from PID order or device number.

`no_captured_activity_ns` means no activity in the exported kernel/memcpy/memset
tables during that interval. It does not establish physical GPU idle. Missing
or unsupported activity tables and captured stream identities are reported.
Activities after the CPU request marker are retained if they correlate with a
selected launch. Diagnostic traces are timing explanations and must not replace
low-interference benchmark results.

## Completion and provenance

A graph launch records scheduling. Even a complete GPU activity span does not
prove that the scheduler committed a verify result. Optional completion evidence
contains request-level `completed_verify_count`, `output_token_count`, and a
`source`. The report preserves their difference from scheduled replay count,
while leaving every per-replay completion identity unavailable. A request with
64 scheduled replays, 63 completed verifies, and 64 output tokens therefore
still has 63 completed verifies. The aggregate difference cannot identify which
specific replay was in flight.

Input database, optional report, and capture files are hashed. Expected database
and report digests can be supplied for validation. The CLI also hashes the
completion JSON. For a reduced export, preserve its source report/database
hashes and table/row preservation receipt alongside the analyzer output; the
reduced database hash alone does not prove its relationship to the source.

## Offline CLI

Run the file directly when CUDA package bootstrap dependencies are unavailable:

```bash
python tokenspeed-kernel/python/tokenspeed_kernel/profile_scheduler.py \
  --selection-mode legacy --database trace.sqlite --process-id 123 \
  --request-label scheduler.trace_request --replay-label graph_replay \
  --missing-launch-policy error \
  --capture capture-0.json --capture capture-1.json \
  --completion completion.json --report trace.nsys-rep --output timing.json
```

Replace `--request-label` with `--whole-capture-source 'manifest/path and evidence'`
for an explicitly established single-request export without a marker. Never
reuse captures from a separate diagnostic run for an original performance trace.

## Inherited utilization annotations

Appending scheduler timing lanes to a previously annotated report preserves its
existing MFU/Est.MBU labels and timestamps. Preservation verifies that those
annotations survived the merge; it does not recalculate or validate their
analytical normalization. Historical reports may use legacy hardware peaks and
dtype-based compute-engine assumptions. A newer engine-aware YAML profile is
not selected retroactively. Such reports must carry provenance identifying the
inherited annotations as historical; the new scheduler gap timing is independent
of those utilization denominators.

## Collector selection and committed windows

Collector selection uses the same `analyze_scheduler` entry point and interval
calculations. Supply the SQLite export, explicit trusted registry (or None for builtin costs),
explicit missing-launch policy and
`service_metadata` path; the metadata's existing source references resolve the
ready index, window manifest and raw service observations. Each source digest is
verified. Per-step unique replay labels, recorded PID/TID, unique runtime
correlations and graph clone ancestry use the graph analyzer's exact native
index/join. No generic replay label, request NVTX range or whole-capture request
assumption is involved.

```python
result = analyze_scheduler(
    Path("trace.sqlite"), selection_mode="collector", registry=None,
    missing_launch_policy="report_unavailable",
    service_metadata=Path("1-metadata-coverage.json"),
)
```

The standalone CLI accepts the same selection with `--selection-mode collector
--service-metadata 1-metadata-coverage.json --missing-launch-policy
report_unavailable --cost-provider builtin --database trace.sqlite --output scheduler.json`.
Collector mode requires the installed CPU-safe `tokenspeed_kernel` modules; the
legacy standalone reader still needs only the standard library. Preserve the
original artifact paths referenced by the service metadata.

For an existing collector capture, one command exports and analyzes graph costs,
scheduler timing and service proof before deleting the temporary SQLite export:

```sh
python -m tokenspeed_kernel.profile_nsys \
  --nsys /path/to/nsys --report run.nsys-rep \
  --graph-replays run/1-graph/rank-0.replays.jsonl \
  --graph-window run/1-graph/rank-0.window.json \
  --metadata run/1-metadata-coverage.json --model-config run/model-config.json \
  --native-coverage native-coverage.json --hbm-policy logical_io_once \
  --scheduler-mode collector --cost-provider builtin --output graph-metrics.json
```

`finalize_graph_nsight(..., scheduler_mode="collector", registry=None)` embeds the result in
`scheduler_analysis`. Choose `scheduler_mode="disabled"` to omit it. The ready
index and window hashes must match the graph analysis inputs. Existing native
scope labels append the joined scheduled/GPU-observed/device-ready/authoritative
commit status and accepted-token intervals while preserving utilization and
communication bandwidth labels and all executed FLOPs.

Select the model's installed CPU-safe registration function with explicit
`--cost-provider my_model.profiling:register_costs` when builtin costs do not
cover it. Python callers supply their trusted `CostRegistry` as `registry`;
`None` explicitly selects builtin costs. The same registry object feeds graph
costs and scheduler semantic classification. Legacy scheduler callers explicitly
pass `registry=None`; that path does not import collector cost dependencies.
Serialized captures cannot select a
provider or cause an import. Provider source hashes accompany the result.

`steps` retains every recorded device-ready step, unavailable metadata and
scheduled replay markers without a ready receipt. Scheduling, GPU observation,
device readiness and authoritative service commit are independent fields.
Service commits require exact run/rank/request/ForwardId/step identities and a
request generation; late ForwardId bindings must agree. For ordinary AR,
accepted tokens occupy `(completion_tokens - len(output_ids), completion_tokens]`.
An executed tail without a matching commit remains
`executed_without_observed_token_commit`; absent observation cannot prove that
it was never committed. A scheduled step without a ready receipt retains unknown
request membership and cannot enter a committed request window.

`requests` contains each measured request's exact committed endpoints and
consecutive pairs. A count-2 through count-64 AR window has 63 endpoints and
62 pairs when all evidence is present. Pairs never cross requests, generations,
missing serving steps, unavailable GPU timing or missing accepted-token
intervals. Ready request-ID membership also lists extra executions for that
request, with generation explicitly unknown on uncommitted steps. Missing
endpoints or pairs make the window partial. `service_tpot_ns` uses authoritative
host commit elapsed time divided by accepted tokens between exact endpoints;
GPU start-to-start period remains a separate measurement.

`span_overlap_ns` intersects GPU envelopes; `gpu_activity_intersection_ns`
intersects the actual activity interval unions. The legacy `overlap_ns` is the
negative signed-gap magnitude and can overstate envelope intersection when one
span contains another. Semantic intervals use unique captured Op node ownership
and trusted cost rules. Pure communication activity N and compute activity C
use disjoint Op classifications; reports provide N intersection C and N outside
classified C. These are coverage measurements, not causal critical-path claims.
Fused compute/communication Op intervals are a separate class and cannot supply
a standalone communication duration. Unknown owners and costs remain explicit
unclassified coverage. No-captured-activity never establishes hardware idle.
