# Cache-copy completion observations

Cache diagnostics follow the existing cache operation ownership. Scheduler
metadata describes an admission and a scheduled operation. The Python executor
adds local physical-copy counts to the operation's existing completion ACK;
it does not infer completion from prefix-hit counters or timestamps.

The scheduler adapter freezes request/admission identity, operation generation
and slot/frontier metadata into immutable transfer objects before submission.
Opt-in `record_observations=True` requires those native fields. In particular,
observed restore identity comes from the captured native admission, not a later
lookup of a reusable request ID. The ordinary unobserved adapter is unchanged.

`TOKENSPEED_CACHE_EVENT_LOG=1` explicitly enables diagnostics at startup with
capacity4096; absent or0 disables them. The builder captures this once in plain
`DeviceSpecs`, passing the same choice to the executor and plan adapter.
`HostCacheExecutor` also accepts the explicitly configured observation capacity. For each observed prefix load or restore it counts
rows after the existing ownership translation and sums those rows' field payload
bytes. Counts are local to the physical owner; a replica that owns no prefix rows
acknowledges the operation as before and reports zero copied rows and bytes.
The slot-state image span is recorded separately from copied bytes: an exporter
with an unprepared image may import its marker without copying its state rows.

The immutable observations travel on the same `_Ack` as the operation. A prefix
load's final copy event and a restore's event after both Host tiers and slot-state
imports are the only completion witnesses. `CompletionQueue.pop_ready()` releases
the ACK and its observations together. No extra device synchronization, stream,
allocation, request-ID map or completion callback is added. Copy/import failures
queue no successful observation. Shutdown drops pending payloads and undrained
observations. A physical copy may finish after its request was cancelled; its ACK
must still release held resources. That physical completion does not claim the
admission resumed successfully.

After the normal ACK poll, `HostDataPlane.poll_cache_observations()` delegates
through `DeviceHandle` and drains already-completed
CPU dictionaries plus a dropped-record count. It issues no device queries or
work, serialization, logging, or admission lookup. The completed queue is bounded;
overflow and overlong request identities report loss without dropping scheduler
ACKs. Request identities are never truncated into ambiguous join keys. The host
must propagate loss rather than certify incomplete evidence.

Python reads no instance label; the host supplies and validates the explicit
public instance option. The control-plane owner adds engine, role, scheduler-replica and physical-rank
context, owns bounded nonblocking export, and distinguishes a local copy from
replica-wide ACK agreement. Scheduled and completed records join by operation
kind, wrap generation and operation ID within that context, then match the frozen
request/admission identity. Admission cancellation and actual client-visible
progress remain native owner facts. A restore's accepted-token frontier is not
client-delivery proof, and a slot allocation is not itself proof of a GPU write.

See [the event-loop boundary](event-loop.md) and
[the scheduler's restore ownership](scheduler.md#4-suspend-and-resume).

## Bounding debug injection

`--debug-force-retraction-max-per-request N` optionally limits successfully
imaged **forced** retractions per admission. Omission (`None`) keeps the legacy
unlimited behavior. A supplied value must be an integer in `[1, INT32_MAX]` and
requires a nonzero forcing interval and a retracting role with a snapshot pool.
Explicit zero is invalid; it is never treated as omission. Both positive decode
and negative mid-prefill intervals use the same bound.

Waits and refused/failed image attempts do not consume the bound. Reusing an
external request ID creates a new admission with a fresh counter. Ordinary
capacity-triggered recovery neither consumes nor consults the forced budget,
so a final scored tail must still be checked after the last actual restore.
Stopping debug injection is not itself proof of completed copies or output
progress. Python passes the exact optional value to the native scheduler; the
native request owns the counter and scheduling policy.
