# Service TPOT pilot

The pilot first measures ordinary real-weight generation, natural run-to-run
noise, and the cost of observation. It does not set acceptance thresholds.
Keep the model/checkpoint manifest, input token ids, output length, topology,
Graph/overlap settings, warmup, request counts and token window identical.
Use the deployment's real checkpoint and request workload. A dummy load or
synthetic speculative acceptance setting is not an eligible baseline.

Create a frozen JSON case with these fields (replace deployment paths and
input ids with the actual frozen workload):

```json
{
  "engine_args": {
    "model": "/readonly/checkpoint",
    "attn_tp_size": 8,
    "stream_interval": 1,
    "enforce_eager": false,
    "disable_overlap_schedule": false
  },
  "sampling_params": {"temperature": 0, "ignore_eos": true, "max_new_tokens": 64},
  "input_ids": [1, 2, 3],
  "warmup_requests": 1,
  "measured_requests": 3,
  "sequence": ["baseline", "profiled", "baseline"],
  "cache_policy": "flush_before_each_request",
  "instrumentation_scope": "model_layer_op",
  "coverage_mode": "decode",
  "statistical_window": {"first_count": 2, "last_count": 64},
  "profile_environment": {
    "TOKENSPEED_TIMELINE_CONFIG": "/path/to/timeline.json"
  },
  "profile_command": ["nsys", "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none", "--cuda-graph-trace=node", "--capture-range=none", "--output={output}"]
}
```

Use actual request lengths and the existing topology configuration. The ids,
counts and sequence above illustrate the file format, not an approved workload
or an acceptance threshold. A `null` window uses the first and last *received*
new-token chunks. Explicit cumulative endpoints must both actually arrive;
there is no interpolation within a chunk. `ignore_eos` is an ordinary sampling
setting that holds requested output length fixed, not an acceptance override.

The normal flow has two commands after freezing the case and timeline config:

```sh
python examples/profiling/service_tpot.py pair --case /path/to/case.json --output-dir /path/to/pilot
python -m tokenspeed_kernel.profile_nsys \
  --run-manifest /path/to/pilot/run-1.*.json --rank 0 \
  --hbm-policy logical_io_once --scheduler-mode collector --cost-provider builtin \
  --output /path/to/pilot/metrics.json --annotated-report /path/to/pilot/annotated.nsys-rep \
  --import-nvtxt /path/to/ImportNvtxt --target-id HW:VM
```

The example sequence places the profiled arm at index 1. Each completed profiled
arm prints one `run-INDEX.SHA256.json` manifest. It records the exact source report,
client, case, frozen model config, service metadata, all collector ranks, capture
JSONs, binary payloads and eager sidecars with file sizes and SHA256s. When
the runtime emits communication and MNNVL allocation receipts, the manifest
also seals their exact bytes and binds each allocation to its rank's receipt. Keep those source locations
and bytes unchanged. The analyzer checks the manifest content hash, source hashes
and run/rank identities before loading any explicitly selected cost provider or
exporting the report. Select `--rank` when the manifest contains multiple ranks.
The existing separate source-file arguments remain available for older captures.

HBM estimates, scheduler analysis and trusted cost providers remain explicit CLI
choices; no capture or manifest can select Python imports. `builtin` registers
public cost rules. For additional model rules, install the model's CPU-safe
provider and explicitly use `--cost-provider package.profiling:register_costs`.
Use `--hbm-policy unavailable` to omit the logical HBM estimate and
`--scheduler-mode disabled` to omit scheduler analysis. Pair service proof and
its native-coverage receipt currently apply to rank 0; other ranks can be analyzed
with scheduler analysis disabled. Native annotation still requires the actual
`HW:VM` target reported by the matching `ImportNvtxt --cmd info -i REPORT` tool.
Supply `--nsys /path/to/nsys` when the matching CLI is not on PATH. Metrics,
annotated report and native-coverage outputs must be new files. The manifest flow
writes rank 0's native receipt beside its collector window; use the existing
`summarize` command afterward to update overhead eligibility from that receipt.

Before importing Engine in each profiled child, the driver prepares the public
`torch.cuda.nvtx.mark` binding. The first synchronous service start marker checks
that the installed Torch build provides NVTX before request warmup. Markers use
the callback thread and the same JSON text as the service log; preparation and
source hashing stay outside token callbacks. This imports Torch/NVTX only when
markers are enabled and launches no CUDA device work. Each arm retains its
Torch version and Python source fingerprints in `N-service-nvtx.json`.
The removed `TOKENSPEED_SERVICE_NVTX_LIBRARY` setting now raises a migration
error; no C compiler, NVTX headers or custom library are needed. A JSON-only
service case without a profiler command keeps JSON-only observation without
importing Torch/NVTX for markers. The optional
`tokenspeed-kernel[nvtx]` extra enables `--enable-nvtx` ranges and the separate
eager Model/Layer/Op timeline. Remove `library` and `nvtx_library` from older
timeline configs.

Run the pair driver outside an existing profiler session. Each arm starts a
fresh engine in a separate process with the same warmup, then shuts it down.
Only profiled children receive `profile_environment` and the frozen
`profile_command` prefix. The explicit `instrumentation_scope` is either
`model_layer_op` (requires the timeline config and node tracing) or
`service_only` (no timeline config). Service-only
arm differences cannot establish Model/Layer/Op collector overhead.

Full-process `--capture-range=none` remains supported. To omit weight loading
and global startup warmup from collection, use `--capture-range=cudaProfilerApi`
with `--capture-range-end=stop`, `--kill=none` and `--cuda-graph-trace=node`.
The driver requires explicit `--kill=none` when launching this collection mode
so that Nsight does not terminate the target when collection ends. The service
must finish its stop/flush reply and normal shutdown. This launch check does
not prevent offline analysis of older frozen cases. This requires
`enforce_eager=false`, decode capture including batch size 1, and explicit
timeline `ranks` including rank 0. On each selected rank the runtime starts
CUDA collection once, after the first eligible graph's four local warmups and
sampler reset, before its existing synchronize/barrier. No new collective is
introduced. Later graph warmups, probe release/rebind and final capture remain
inside that uninterrupted trace. An absent profile owner or an ineligible
capture does not call CUDA profiler APIs.

Before request warmup, the driver requires each selected rank's actual
`rank-N.collection.json` receipt and matching same-run/process capture report;
missing early startup aborts the arm. These receipts prove the runtime reached
the start boundary. They do not prove Nsight recorded the native graph events.
The Python graph object was constructed before this boundary; constructor-only
events are not guaranteed. On the actual run, verify capture, instantiate and
clone events and the exact node ancestry from the serving replay to the final
capture in the same report, retaining both probe and final capture metadata.

Every warmup and measured request calls the existing `Engine.flush_cache()`
before its timer begins and requires a successful all-replica reply. The pilot
currently accepts only `cache_policy=flush_before_each_request`; unknown policies
are rejected. This clears service prefix/KV cache, preserving loaded weights,
compilation caches and captured graphs. Actual received `cached_tokens` are
retained and must all be zero to verify this policy.

For profiled arms, after the real request warmup the driver calls
`Engine.start_profile(activities=["CUDA_PROFILER"], profile_id=run_id)` and checks
success. After the measured requests, `stop_profile()` must acknowledge the
CPU metadata writer flush before engine shutdown. No Torch CPU/GPU profiler is
started under Nsight. Full-process and early-start collection both retain graph
capture evidence for subsequent native verification; startup/warmup durations
are not part of the matched decode-token window. Use the case's explicit
cumulative decode-token endpoints; first-token/prefill evidence and client TTFT
remain separate.

The logical metadata window starts only at that service start after request
warmup. If startup already began CUDA collection, service start does not restart
it; the ordinary measured stop flushes metadata and ends CUDA collection.
Configured timelines support one window per engine. Repeated starts and
`profile_by_stage` return a failed control reply before starting a backend or
changing window state. An arming error disarms a partially armed window and
clears pending profiler state without stopping an existing startup collection;
retry requires a fresh engine/run. Continue serving or shut down that failed
run; do not treat it as a completed trace.

`case.json` is copied byte for byte. If configured, the original graph timeline
is retained as `timeline-template.json`. The driver sets each profiled arm's
`run_id` and `output_dir`, then appends its automatic `startup_observation`
before Engine import. Runtime config snapshots expand the shared optional
defaults; their hashes bind that actual config to the observation. These
per-arm copies are therefore not byte-identical to the template.
The real checkpoint's `config.json` is frozen as `model-config.json` for full
scope runs; each arm verifies it is unchanged. Expected layer IDs come from that
model's declared text `num_hidden_layers` (falling back to the released-checkpoint
`num_layers` alias), never just from layers which happened
to appear in a trace, and no model depth is hardcoded. Both spellings must be
positive integers and agree when both are present; conflicting declarations are
rejected. The raw frozen config bytes and SHA256 remain unchanged. The driver owns
`TOKENSPEED_PROFILE_RUN_ID` and
`TOKENSPEED_SERVICE_TRACE` (one JSONL per host pid). All referenced libraries,
checkpoint files and hardware-profile inputs must remain frozen/read-only.
The client JSONL contains raw chunks, cumulative token counts, independent
`perf_counter_ns` arrival timestamps, output-id hashes and actual endpoints.
Hashes are computed after the final chunk. Logical token sequences are
reconstructed only when the actual delta/cumulative count contract is
consistent; final logical token hashes establish parity even when chunk
coalescing differs. `summary.json` separates A-A noise and observed
B-minus-previous-A arm differences, with raw samples and no thresholds.
Comparisons with different received token endpoints remain unavailable.
`model_layer_op_overhead_ns` remains null unless token/cache parity, complete
measured metadata coverage and exact native attribution are all verified.

Export each completed report with `nsys export --type sqlite`, then align
against the corresponding profiled client's JSONL:

```sh
python examples/profiling/service_tpot.py align --client /path/to/pilot/1-profiled.jsonl \
  --nsight-sqlite /path/to/pilot/1-nsys.sqlite --boundary frontend_send_returned
```

The example sequence puts B at index 1; use `1-profiled.jsonl` and `1-nsys.sqlite`
for that sequence. `--service-pid` explicitly selects a rank process when comparing `commit` or
`output_ready` across a multi-rank trace. With no explicit selection, multiple
matching ranks are ambiguous and remain unknown. Exact joins require the same
run, RID, authoritative admission/generation, cumulative token endpoints and
process. The Rust pending `ForwardId` is passed with that exact dispatch to
Python; its exact `ModelExecutionResult.profile_identity` binds the same id.
No matching by batch order, kernel name, adjacent timestamp or replay order is
used. Missing Graph identity remains null; it is never inferred from a nearby
profiled step. Retraction epoch is null because this path does not yet expose
that authoritative scheduler value. A retraction is not a new admission.

The boundaries have different meanings:

* `commit`: observation immediately after a successful Rust token commit and
  finish/stop trimming, with actual kept token ids and cumulative counts.
* `output_ready`: that commit's actual prepared streaming output. A chunk can
  contain more than one token, and suppressed streaming commits have no output
  marker. Non-forward finishes lacking the exact identity remain unknown.
* `frontend_send_returned`: observation after the control-plane frontend send
  call returned. This is local enqueue completion, not network delivery or a
  client acknowledgement.
* Client arrival: when `Engine.generate(stream=True)` yields a chunk through
  the existing queue bridge. Generator exhaustion is not a token boundary.
* GPU execution/completion: the independent device graph/span evidence. A sum
  of kernel durations is not service elapsed TPOT.

All service callbacks run synchronously on the control thread and perform no
CUDA work or device synchronization. Acquiring the GIL and serializing markers
can delay observation after commit/send; `observer_delay_ns` records the Rust
callback-entry-to-Python lag plus per-row callback work measured before Python argument conversion and label
serialization. Remaining argument conversion/label encoding/native-call cost is observer overhead,
not an exact timestamp of the Rust state mutation. Nsight's own NVTX timestamps
are read from SQLite; client elapsed time is never copied into the trace.
A JSON-only run records `nvtx=false` and cannot supply Nsight timing evidence.

The report compares durations between the exact same received-token endpoints.
Its residual is the *change in endpoint delay*, not an absolute difference of
clock origins. Queueing, launch gaps, D2H, commit processing, frontend enqueue,
detokenization and client queue delivery remain distinct contributions. Neither
`generated_time` nor profiler `note_committed` is GPU/token-commit truth. Select
thresholds only after real pilot data establish alignment, noise and overhead.

CPU core checks:

```sh
PYTHONPATH=python python -m pytest --noconftest test/runtime/test_service_trace.py -q
```

The focused Rust `output::tests::service_observation_*` cases also require the
repository's native Rust/scheduler environment. CPU checks do not validate real
Graph replay, native NVTX injection, service/GPU timing, retraction behavior or
numerical parity on the deployment.


## Fixed coverage artifacts

After stop, the driver writes `{index}-metadata-coverage.json`
(`tokenspeed.metadata_coverage/v1`). It checks rank-0 `window.json` writer status,
zero unresolved metadata, exact identified drop records proven disjoint from
measured decode forwards, lifecycle acknowledgements, the frozen model
config, binary payload sizes/SHA256s and exact same-run true-commit
RID/generation/ForwardId/step identities for **every** token-producing forward
in the measured interval. Endpoint-only or warmup-only coverage is incomplete. `coverage_mode=decode`
uses explicit endpoints at count 2 or later. Any dropped metadata that cannot
be reconciled with exact unavailable ready records, or intersects a measured
forward, fails the gate. Known out-of-window prefill drops remain listed and
keep whole-run/prefill scope partial even when decode coverage is complete.
Capture scope/layer omissions and unavailable dynamic metadata remain incomplete.
This is metadata coverage, not GPU attribution.

The offline native analyzer consumes that artifact and produces
`{index}-graph/rank-0.native-coverage.json` (`tokenspeed.native_coverage/v1`). Its
fixed fields are `run_id`, `rank`, `scope=model_layer_op`, `coverage_mode=decode`,
`attribution=same_run_cuda_graph_node_ids`, `status`, `source_metadata_sha256`,
`source_nsight_report={path,sha256}`, and `requests` keyed by measured request
index. Each request carries exact `rid`, `generation`, `first_count`, `last_count`
and `forward_evidence` for the same complete forward set. Each forward has
`forward_id`, `step_id`, `capture_id`, actual `graph_node_ids`,
`scope_kinds=[model,layer,op]`, `layer_coverage` (`expected_ids`, `attributed_ids`,
`missing_ids`), `scope_coverage` (`expected_scope_ids`, `attributed_scope_ids`,
`missing_scope_ids`) and explicit `critical_unknowns`. Missing keys never default
to success. Expected layers/scopes must match the exact frozen config/capture;
label names alone cannot verify coverage. This receipt must come from actual
same-run node/clone attribution, not a manually asserted boolean.

Recompute the report after offline artifacts are ready, without regenerating:

```sh
python examples/profiling/service_tpot.py summarize --output-dir /path/to/pilot
```
