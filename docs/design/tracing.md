# OTLP trace export

One span per request, assembled when the request leaves the processor. The
trace is what the request stats already measure: when it was received,
scheduled, prefilled, first-tokened and finished, its tokens, its cache hit
and how it ended. Nothing new is measured and the serving loop gains no work
it did not already have — `--enable-log-request-stats` and tracing read the
same `RequestTimings`, and either one switches the stamps on.

## The rules it obeys

- **Emission anchors to the request lifecycle, never the round.** The span
  is built at the same release points the finish log line is
  (`OutputProcessor`'s release paths, including `stream`'s in-place
  release), from the state the processor already holds. There is no
  per-round trace work.
- **The loop never blocks.** The span crosses to the export thread over a
  bounded channel (`try_send`); a full channel drops the span and counts
  it — the metrics relay's drop policy. The export thread batches, and a
  POST that fails logs and drops its batch; traces never queue behind an
  unreachable collector.
- **Sampling happens before the record is built.** `--trace-sample-ratio`
  keeps that share of requests; 0 (the default) keeps none and the
  publisher is not even constructed without an endpoint.
- **Ids derive from the request id.** The trace id is a 128-bit mix of the
  rid, so the same request resolves to the same trace id across ranks,
  logs and re-derivations; span ids mix the rid with a process counter so
  every span differs. There is no global tracing context to maintain.

## The export

OTLP/HTTP with a JSON body — every OTLP/HTTP receiver takes JSON, and the
body encodes with `serde_json` and ships with `ureq`, so the engine carries
no opentelemetry dependency stack. Integers are strings and ids hex, as
protobuf's JSON mapping writes 64-bit and bytes fields.

A request that names a continual-prompting session
(`session_params.id` on the wire) carries it as the GenAI `session.id`
attribute — every request stays its own trace, and the tracing UI threads
one conversation's requests under the session. Requests that name no
session carry no attribute; conversation branches
(`session_params.rid`) linking to the request they continue is a later
refinement.

Attributes follow the GenAI semantic conventions exactly
(`gen_ai.system` and `gen_ai.request.model` on the resource;
`gen_ai.response.id`, `gen_ai.usage.input_tokens` and
`gen_ai.usage.output_tokens`, and `gen_ai.response.finish_reasons` as an
array, on the span), so Langfuse, Datadog and the Jaeger UIs read them
natively; TokenSpeed-specific facts (cached prompt tokens, preemption
count and seconds) carry `tokenspeed.` prefixes. The span is a SERVER
span; an aborted or transfer-failed request carries the error status. The
lifecycle points ride as time-stamped events
(`tokenspeed.request.scheduled`, `…prefill_done`, `…first_token`)
between the span's start (received) and end (finished). On a decode role
`prefill_done` is the remote KV landing and `first_token` is the first local
decode output; the span's TTFT therefore includes the wait for a decode
execution permit, and `decode_process` / `tpot` start after it.

## What stays out

- **Context propagation.** The engine derives its own trace ids; a request
  joining an upstream trace (W3C `traceparent` from the agent framework
  through the gateway) needs a metadata field on the wire request and is a
  separate change.
- **Stage spans.** One span per request, not per round or per stage; PD
  roles each trace their own request lifetimes, as they each measure their
  own queue wait.

## Response timing metadata

The native output wire returns a typed `request_timing` record once, on a
request's terminal output. Each record covers one local engine request; PD
prefill and decode report separate lifetimes. It is independent of trace
sampling, metrics and request logging and adds no log records. Intermediate
outputs carry no timing record. Older wire producers decode with no timing.

`RequestTiming` is a map on the wire and in native Python chunk dictionaries.
The Rust client exposes the typed record; the Python wire schema and
`WireDriver.Completion` use the corresponding `msgspec` types. The existing
output fields remain the source of the request id, completion token count and
finish reason.

| Field | Meaning |
| --- | --- |
| `role`, `dp_rank`, `tp_rank` | Local engine role and host-configured DP/attention-TP ranks; unknown ranks are null. |
| `clock` | Shared `host_id`, `process_instance_id`, `clock_sync` and `clock_uncertainty_ns` for every boundary in this record. |
| `events.accepted` | Output-processor registration; null for requests rejected before registration. |
| `events.kv_ready` | First completed plain prefill output, or decode's agreed KV/bootstrap readiness; null on the prefill role or if readiness was never observed. |
| `events.finished` | Local output-processor completion, before transport and physical resource retirement. |
| `kv_output_tokens` | Committed output count at first readiness, or null without that boundary. |
| `e2e_latency_s` | Native backend latency: `max(finished_unix_seconds - created_time, 0)`, or null when the frontend supplied no positive `created_time`. |

Every boundary holds `ts_unix_ns` and `ts_mono_ns` as nanosecond decimal
strings. The monotonic value is null if the observation predates the process
clock origin. Subtract monotonic values only within the same process instance.
Wall-clock synchronization is `unknown` and its uncertainty is null unless
independently established; in particular, frontend `created_time` and engine
completion can use different hosts' clocks.

The transfer peer captures decode readiness after KV and the bootstrap token
have arrived and replica agreement completes, before publishing its completion
event. Retraction and later prefill commits preserve the first readiness
boundary and its token count. `complete_response` samples completion once,
constructs the terminal timing record and passes that same completion time to
the finish logger. Logging never constructs or mutates the response.
Cancellation paths without a terminal response return no timing record.

Consumers can derive backend proxy TTFT as readiness minus registration and
decode-path duration as completion minus readiness. Proxy TPOT divides that
duration by the containing output's `completion_tokens - kv_output_tokens`,
only when the denominator is positive and the required clocks/boundaries are
available. Missing measurements remain unavailable. These are backend proxies;
client receipt, GPU execution time and the existing native first-local-output
TTFT/TPOT use different boundaries.
