# Scheduler trace analysis

`tokenspeed_kernel.profile_scheduler.analyze_scheduler` is an offline,
standard-library-only reader of Nsight Systems SQLite exports. It reports
scheduled graph launch timing and the GPU activity between consecutive launches.
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
  --database trace.sqlite --process-id 123 \
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
