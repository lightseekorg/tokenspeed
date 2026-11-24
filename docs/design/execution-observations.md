# Request execution observations

Execution observations report actual target calls and fenced verification results.
They are optional diagnostics of the [ordinary event loop](event-loop.md), not
scheduler feedback, a numerical envelope, or proof that enabling graphs caused
any measured request to use one.

## Supported scope and activation

V1 supports GPU PP1, unsharded ordinary chains: no drafter (W1), DUMMY and DFLASH.
It observes pure prefill, decode and mixed batches, including ordinary and inline
breakable target prefill graphs. Packed verification, trees, harness drafters,
row-narrowing models, query sharding, and synthetic/simulated acceptance are
refused at the observation boundary. No fallback changes the selected execution.
This capability does not qualify a particular model, platform or numerical mode.

Set `TOKENSPEED_EXECUTION_EVENT_LOG=1` together with
`TOKENSPEED_CACHE_EVENT_LOG=1`, a public `TOKENSPEED_CACHE_EVENT_INSTANCE`, and
`--enable-async-logging`. Execution and cache records share one bounded native
publisher and engine identity. Invalid opt-in values fail startup. The CPU-only
`tokenspeed_engine._host.execution_observation_protocol()` reports
`tokenspeed-execution-observations-v1`; it initializes no model or device.

When off, the Python completion tuple remains its original nine elements,
wire snapshots are absent, and no observation GPU buffer is allocated.
When on, the executor owns int32 `[max_batch_size, 2]` scratch in the existing
CPU-backed `runtime_states` named region. It is batch scratch, not request-slot
state and not a retraction image. Every sampler invocation overwrites its rows.
Its stable addresses survive memory-saver release/resume like other runtime
buffers. No new RNG draw, collective or kernel selection is introduced.

## Producer and numerical meanings

The target call returns a receipt after the actual eager forward or prefill
replay. Decode replaces its capture-time receipt only after replay of the
selected target decode graph. A drafter's separate prefill/depth graphs do not
modify the target receipt. Breakable graph counts describe replay calls, not
per-kernel profiling: eager embeddings, breaks and logits remain part of that
route. Inline capture reports the selected request capacity, including padding.

Each completed request row reports:

| Field | Source and meaning |
| --- | --- |
| `phase` | Frozen native row partition: prefill or decode, independently of batch mode. |
| `path` | Actual target route: `none`, `eager`, `decode_graph`, `prefill_breakable`, or `prefill_inline`. A mixed decode row retains the shared target prefill/eager route. |
| `verify_width` (W) | Actual candidate layout consumed by the ordinary verifier, including its anchor; never configured draft steps or accepted length. Non-speculative decode still consumes a W1 candidate window. |
| `accepted_before_stop` (A) | Effective accepted output prefix after force-single/drafter finalization and before device stop clipping. Includes the residual/bonus token. Prefill uses sentinel -1. |
| `force_single` | The actual applied mask, read through the same helper that verification uses. An unused mask is not reported as applied. Prefill is false. |
| `kept_prefix` (K) | Existing landed `output_lengths`, after device stop/length clipping. Zero remains zero for late stopped rows. |
| `committed_tokens` (C) | Tokens added by native commit, after native grammar/terminal trimming; zero for discarded late output. |
| `num_extends`, `row_index`, `live_requests` | Frozen native batch geometry; distinguishes mixed rows and excludes padding. |

For decode, `1 <= A <= W`, `0 <= K <= A`, and `0 <= C <= K`. Applied force-single
requires A=1 but does not turn a W8 target execution into W1. A is deliberately
after force-single; it is not the raw unforced backend result. In an unforced
chain A-1 counts accepted draft tokens; K-1 need not, because stop clipping can
end inside the accepted drafts before the bonus token.

The sampler copies A and the applied force mask into the scratch *before*
`apply_stop_conditions` mutates lengths. These copies are recorded inside decode
graphs and refresh on every replay. The runner slices padded rows away. A fresh
per-result nonblocking CPU copy is queued before the existing output copy event;
`ModelExecutionResult.sync`/RoundFence remains the one completion gate. The native
buffer reader validates exact int32 `[live_requests, 2]` shape and copies into
owned vectors before the result's reuse lease retires. Control performs no new
device query. The zero-input no-forward arm uses explicit host sentinels.
Malformed optional diagnostics invalidate the frozen lease's evidence while
preserving the ordinary nine-field result and token commit. Ordinary result
decoding and completion-fence failures remain fatal.

## Admission ownership and closure

`OutputProcessor` owns an observation accumulator per native admission.
Dispatch freezes identity, slot, row phase and a checked per-admission ordinal
in an RAII lease moved with the pending forward. The lease retains only CPU
observation ownership. It does not hold GPU resources or delay slot retirement.

Commit observes every landed row, including open prefills and rows whose output
state has already ended. Reusing a public request ID creates a new admission;
late results stay attached to the old lease. Retraction preserves the admission
and all its ordinals and histograms while the slot may change. Dropping an
unfinished lease records loss; it never manufactures successful execution.

Records have `[execution-event] ` followed by one bounded JSON object:
`trace_started`, `request_started`, `target_round_dispatched`,
`target_round_completed`, `request_closed`, `observation_loss`, `trace_finished`.
The shared writer supplies protocol, instance, role, DP/physical rank,
`record_sequence`, `publication_order`, and cumulative `log_dropped_lines`.
Each protocol has its own contiguous sequence. Publication order is shared
with [cache observations](cache-observations.md), not a timestamp.

`request_closed` waits for output ownership and every pending observation lease;
it does not mean the scheduler released the slot. `trace_finished` waits for the
shared native context, including cache teardown. Require both protocols' start
and terminal records, zero loss, valid admission joins and all assigned rounds.
A retained owner, killed process, missing record or blocked/lossy logger does not
produce valid complete evidence. Records are at most 8192 bytes; publication
handoff is bounded to 4096 records. Identity is never truncated.

Capture, smoke replays, calibration and model warmup create no serving admission
records. Native startup requests have immutable `origin=startup`. An idle DP
rank has no request lease, even if it replays a graph. Dummy/padded rows are not
admissions. A fully cached zero-input arm is `path=none`. Pressure and harness
warmup requests are ordinary serving admissions; the caller's explicit measured
request manifest must exclude them without guessing from log times.

## Wire snapshots and consumers

`BatchTokenIDOutSlim.execution_observations` is one appended optional column,
empty when disabled or absent from an older producer; mixed frames use aligned
None entries. Both `OutputChunk` conversion paths and the Python chunk expose it.
`Completion` validates and replaces the latest cumulative snapshot instead of
adding counters again. Empty final chunks preserve it. Identity changes,
regressing counters/histograms, missing pre-stop data and unknown versions fail
validation before mutating completion output.

A version-1 snapshot contains engine/rank/admission identity and origin,
assigned/completed/missing round counts, `through_round`, committed-token total,
provisional validity, and these sparse cumulative histograms:

- `bins`: `(phase, path, W, K) -> count` (the decode part is H).
- `pre_stop_bins`: `(path, W, A, force_single) -> count` (decode-only H_pre_stop).

The two maps together have at most 4096 bins per admission. Each snapshot also
has a 256 KiB MessagePack byte budget, checked by the native snapshot boundary
and Python consumer. Byte overflow emits observation loss and sends a compact
invalid snapshot with `histograms_omitted=true`, empty histograms and unchanged
identity/counters. Omission remains explicit and sticky; Completion still accepts
ordinary token output. Invalid overlong identities yield no snapshot rather than
truncated identity. Bin or byte overflow invalidates evidence while preserving
normal token commit. U64 admission identities must
be decoded losslessly. A terminal client snapshot covers completed rounds through
that output; already-dispatched rounds can land later. It is not lifetime closure
or evidence from other physical ranks. Use the closed trace for those claims.

`python -m tokenspeed.runtime.entrypoints.execution_trace LOG ... --snapshots
MEASURED.json --ranks 0 --require decode:decode_graph:8` validates closed records
and reconciles both histograms with the supplied measured snapshots. Supply the
actual required physical-rank set; replicated records are not extra logical
rounds. `--restored` additionally requires a matching native restore completion
before that round's frozen dispatch and fenced completion in publication order.
The consumer joins the exact scheduled operation/generation and slot generation,
and requires the restored binding to remain current at dispatch. An older round
completed late or an orphan completion cannot establish post-restore execution.
Scheduled and completed records must also agree on typed frozen recovery
metadata: previous slot/generation, the destination's previous admission, and
committed/computed/prompt token counts. Completion-only physical copy counts
remain separate. No ordering between the frozen frontiers is inferred.
An empty local ACK can establish logical ordering, not positive cache copy;
`--copied-restore` additionally requires positive locally owned cache rows/bytes.
The slot-state image span alone does not count as bytes actually copied. Check
successful output and [numerical equality](numerics.md) separately.

## Validation and publication

CPU coverage includes native output commit/RAII lifetimes, mixed/open rows,
width versus pre-stop/kept/committed prefixes, force masks, buffer shape/dtype and
ownership, bounded histograms, logger loss/closure, both Rust client conversions,
native Python conversion and cumulative Completion validation. The frozen full
`output_pre_execution.hex` frame remains alongside the new execution fixture;
its absent observation is unavailable, not zero. A frozen old Python decoder
also reads the new frame's old columns while ignoring the added tail.

Real GPU tests live in `test/runtime/execution/test_execution_observations_cuda.py`
and `test/test_execution_observations.py`. They use normal CUDA availability
and vendor gating; the focused module is registered in `runtime-1gpu`. Resource
leases belong to launch orchestration, not a test skip flag. The serving module
requires qualified local model arguments in `TOKENSPEED_EXECUTION_TEST_ARGS`
(JSON argv), and a local `TOKENSPEED_EXECUTION_DRAFT_MODEL` for DFLASH. Additional
inline fixtures take `TOKENSPEED_EXECUTION_INLINE_TEST_ARGS` and
`TOKENSPEED_EXECUTION_INLINE_DRAFT_MODEL`, naming an inline-capable model/backend
and its compatible drafter. Each named route must actually appear; an ordinary
breakable replay cannot satisfy an inline case. DP and memory-saver fixtures take
`TOKENSPEED_EXECUTION_DP_TEST_ARGS` and
`TOKENSPEED_EXECUTION_MEMORY_TEST_ARGS`. They exercise actual eager/Graph routes,
mixed rows, padding/idle peers, pre-stop snapshots across replay and buffer reuse,
restore joins, observation-off/on token/logprob bits and memory release/resume.
The restore case additionally requires positive locally owned cache copy followed
by execution, beyond a logical empty ACK. Serving tests save both the closed log
and measured snapshot manifest for independent reconciliation.
The above-ladder case requires non-discarded, pure eager decode with an actual
live request count above the configured capture maximum on every required rank;
concurrent submissions alone do not establish that geometry. The memory-resume
case keeps the baseline and both resumed completions, requires their distinct
admissions and W1 decode Graph execution, and reconciles all three with the
closed native trace after shutdown.
The scratch release test requires zero ordinary Torch allocator segment bytes
and stable restored addresses over repeated cycles. Missing local model or draft
arguments fail an invoked serving qualification instead of silently skipping it.
Skips and CPU tests do not qualify GPU execution or numerical noninterference.
The [local asset recipe](../guides/execution-observation-qualification.md) provides
an offline synthetic W1/DUMMY W8 target and explains which cases need other assets.
Serving tests resolve the actual named numerical policy and both model-profile
gates; they do not alias a custom policy to `rl-bitwise` or change numerical
settings. Synthetic `auto` cases do not qualify another numerical envelope.

With the local fixtures and freshly built host/client bindings configured, the
direct GPU entry points are:

```sh
python -m pytest -q test/runtime/execution/test_execution_observations_cuda.py
python -m pytest -q test/test_execution_observations.py --junitxml=execution-results.xml --basetemp=execution-artifacts
```

Draft review may precede full GPU qualification. Before marking the feature ready,
run the actual serving matrix for W1 and DUMMY W8 with eager,
decode Graph and ordinary/inline prefill Graph targets. Require positive measured
request evidence for each intended cell, reconcile W/A/K/C and both histograms,
and compare output token/logprob bits with observation disabled. Then run DFLASH,
mixed, restore, idle-DP and release/resume cases with their matching local assets.
Retain closed native traces, completion snapshots and JUnit results; any required
case skipped, absent or failing its actual-route assertion leaves qualification
incomplete. Helper replay tests alone do not establish this end-to-end result.

The wire addition is backward compatible; the public Rust field additions are
source-breaking, including `OutputChunk` and the wire/control result structs.
They belong to the coordinated Rust `0.2.0` release; see the
[migration and publication policy](../guides/rust-engine-migration-0.2.md).
The Python wheel version and cancellation APIs are separate from this change.
