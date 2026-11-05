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
between the span's start (received) and end (finished).

## What stays out

- **Context propagation.** The engine derives its own trace ids; a request
  joining an upstream trace (W3C `traceparent` from the agent framework
  through the gateway) needs a metadata field on the wire request and is a
  separate change.
- **Stage spans.** One span per request, not per round or per stage; PD
  roles each trace their own request lifetimes, as they each measure their
  own queue wait.
