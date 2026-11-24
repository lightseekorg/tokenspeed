# Cache ownership and physical-copy observations

Cache observations are an opt-in diagnostic sideband of the ordinary scheduler
and cache executor. They do not change admission, retraction, copy ordering,
ACKs, numerical behavior, or replica agreement. See [the event loop](event-loop.md)
for feedback ownership and [the scheduler](scheduler.md#4-suspend-and-resume)
for recovery lifetimes.

## Activation and native capability

Set `TOKENSPEED_CACHE_EVENT_LOG=1`, supply `TOKENSPEED_CACHE_EVENT_INSTANCE`, and
select `--enable-async-logging`. The instance label must be unique for the traced
engine lifetime and contain 1–128 ASCII letters, digits, periods, underscores,
or hyphens. Use a public diagnostic label. Unset or `0` disables observations;
other values fail startup. Enabled observation with synchronous logging or an
invalid instance fails before model bootstrap; it does not override the logging
choice. Request IDs are diagnostic content too: use public request IDs when
sharing a trace.

Before model startup, call:

```python
from tokenspeed_engine import _host

assert _host.cache_event_protocol() == "tokenspeed-cache-events-v1"
```

The callable reads the linked C++ scheduler's protocol version. It does not
initialize a device or bootstrap a model. Ordinary client-only package imports
remain unchanged. A matching capability describes this protocol, not a hardware
or model qualification.

The native observer installs before startup requests and client admissions.
Its scheduler-owned lifetime includes release callbacks during teardown. The
host reads role, DP rank, and physical rank from the actual bootstrap config;
`engine_role="null"` is labeled `fused`, while `prefill` and `decode` keep their
names. No model or deployment identifiers are inferred for the instance label.

## Records and joins

Each record is one `[cache-event] ` marker followed by a JSON object. Records
use the existing host nonblocking logger's bounded queue, independently of the
ordinary tracing level filter. Serialization is bounded to 8192 bytes per
record, and request IDs to 256 UTF-8 bytes. The Python completed queue and the
host's CPU publication handoff each hold at most 4096 observations. Concurrent
producers never wait for each other: they try to enqueue, and one active
publisher assigns sequences and writes to the existing logger in queue order.
The handoff adds no exporter thread. Identity is never truncated.

Every record carries `protocol`, `instance_id`, `role`, `dp_rank`,
`physical_rank`, `scope`, `record_sequence`, and `log_dropped_lines`.

| Event | Kind | Scope | Meaning |
| --- | --- | --- | --- |
| `slot_assigned`, `slot_released` | `slot` | `scheduler_replica` | Actual native slot ownership and RAII release |
| `restore_scheduled` | `snapshot_restore` | `scheduler_replica` | Immutable operation created, request entered restoring |
| `prefix_load_scheduled` | `prefix_load` | `scheduler_replica` | Native prefix load scheduled with frozen ownership |
| `restore_completed` | `snapshot_restore` | `local_physical` | This physical rank completed its owned copy and slot-state import |
| `prefix_load_completed` | `prefix_load` | `local_physical` | This physical rank completed its owned prefix copy |

Operation joins use `(instance_id, role, dp_rank, physical_rank, kind,
op_generation, op_id)` and must also match `request_id`, `admission_id`, `slot`,
and `slot_generation`. Operation IDs may wrap; generation and kind are required.
The output processor allocates the admission identity before the data plane
learns the request. The same identity travels in the native request specification
and remains attached to its slot and cache operations throughout recovery.
The scheduler is replicated: mirrored scheduler records from different physical
ranks are not separate logical operations. One rank's copy completion is not
replica-wide completion. The existing ACK intersection remains authoritative.

With a Disk tier, prefetch fills Host before Device admission and owns no request
slot. Disk retirement alone emits neither a slot assignment nor a Device-copy
observation. The subsequent ordinary Host-to-Device prefix load carries the
admission and slot identity and completes through the same physical-copy fence.
Disk operation IDs are not prefix-load IDs and do not enter these joins.

Slot ownership additionally carries `slot_previous_admission_id`. Restore
records preserve `previous_slot`, `previous_slot_generation`,
`slot_previous_admission_id`, `restore_sequence`, `committed_tokens`,
`computed_tokens`, and `prompt_tokens`. The committed frontier is scheduler
accepted tokens at the quiescent snapshot, including the prompt; it is not
client-visible or stop-trimmed output. The computed frontier is separate and
can precede prompt end for a suspended prefill.

Completion counts `l2_rows`, `l2_bytes`, `pool_rows`, and `pool_bytes` describe
actual locally owned cache rows and field payload bytes after owner translation.
They are not planned counts or prefix-hit token metrics. `slot_state_image_bytes`
is the declared import image span, not a claim that every byte was copied by
prepared state exporters. Empty local ACKs have zero local copy counts and
cannot prove positive copy activity.

The host polls completed CPU dictionaries immediately after the ordinary cache
ACK poll. Observation polling performs no device queries. Pending or failed
copy/import operations produce no successful completion. A physical copy can
finish after cancellation because its ACK still retires pinned resources;
that record does not establish successful request resume. Consumers must join
the request's admission lifetime and successful output separately. Allocating a
slot to another admission likewise establishes ownership, not a GPU overwrite.

## Joining execution observations

[Execution observations](execution-observations.md) reuse this bounded publisher.
Records additionally carry `publication_order`, one native order shared between
the two protocols; each retains its own `record_sequence`. Cache completion is
published before its ACK returns to the loop. A completed execution qualifies
as post-restore only when its frozen dispatch follows the matched restore
completion in that shared order. Existing cache fields and C++ protocol remain
unchanged. Publication order alone is not proof of execution.

## Complete, loss-free evidence

`trace_started` begins each physical rank's stream at sequence zero.
`trace_finished` follows the final context owner's release, including native
slot teardown, and carries `lost_observations` and `evidence_valid`. On normal
host shutdown, the Python log guard reacquires the GIL after serving releases
its owners, draining deferred bootstrap/scheduler decrefs before the native log
guard flushes. It does not finalize Python or collect frozen cycles. A retained
native owner leaves the trace incomplete; process exit alone never synthesizes
a terminal record.

Consumers must require both endpoints, contiguous unique sequences through the
terminal record, zero loss counters, and valid operation/admission joins.
`observation_loss` records identify rejected native identities, invalid Python
payloads, oversized records, completed/publication-queue overflow, or log-queue
overflow, with `dropped_count` and `evidence_valid=false`. Every record carries
the logger's cumulative `log_dropped_lines`, including ordinary log lines lost
from its shared queue;
any such loss conservatively invalidates the trace.

A loss record or terminal summary can itself be lost when the output is full.
Sequence gaps or a missing terminal record therefore invalidate evidence even
if no loss record arrived. Truncated output, process termination, and bounded
shutdown drain can leave an incomplete trace. Observations do not block the
control thread to make diagnostics reliable, and an incomplete trace must never
be treated as proof of successful recovery.
