# Metric output plugins

Metric output plugins receive the observations already collected by TokenSpeed.
They adapt those observations to an external monitoring service and own their
delivery mechanism. They do not replace metric definitions, request timing, or
rank selection. Prometheus remains the default and retains its native aggregation
and `/metrics` endpoint.

## Discovery and selection

Publish an ordinary Python package with a runtime plugin entry point:

```toml
[project.entry-points."tokenspeed.plugins"]
example_metrics = "example_metrics:register"
```

TokenSpeed discovers installed entry points and calls `register` once per process.
Registration must only declare factories; it must not start threads or acquire
external resources. Install the package in the Python environment used
by every participating node. Installation makes an output available; selecting
its registered name enables it.

Add these arguments to the engine command:

```bash
--enable-metrics \
--metrics-reporters prometheus \
--metrics-reporters example_output \
--metrics-reporter-config /etc/tokenspeed/metrics.json
```

The configuration file maps external reporter names to plugin-specific settings:

```json
{
  "example_output": {
    "endpoint": "https://metrics.example.com/ingest",
    "queue_size": 10000
  }
}
```

An external output without a configuration entry receives an empty object.
Unknown selections, duplicate JSON keys, configuration for an unselected output,
and invalid plugin settings fail startup. Configure Prometheus with its existing
CLI flags, not this file. With metrics disabled, no output plugin resources are
created. Without an explicit reporter selection, `--enable-metrics` continues to
select Prometheus only. Selecting only external outputs does not start a
Prometheus endpoint.

## Register creation callbacks

The metrics API version is independent of the model plugin API. Check it before
registration, and declare the compatible TokenSpeed versions in the package's
dependencies. A package entry point can register more than one output name.

```python
def register():
    from tokenspeed.runtime.metrics.registry import (
        METRIC_REPORTER_API_VERSION,
        register_metric_reporter,
    )
    from example_metrics.adapter import create_sink, validate_config

    if METRIC_REPORTER_API_VERSION != 1:
        raise RuntimeError("Unsupported metric reporter API")

    register_metric_reporter(
        "example_output",
        validate=validate_config,
        create_sink=create_sink,
        start_node=None,
        override=False,
    )
```

| Callback | Contract |
| --- | --- |
| `validate(config)` | Validate and normalize settings without creating resources. Reject unsupported arguments. Return a JSON object. |
| `create_sink(producer_config)` | Create one process-local `MetricEventSink`, reused by its collectors. |
| `start_node(config)` | Optionally create launcher-owned resources shared by the node. Return a `MetricNodeSession` with `producer_config` and `close(timeout)`. |

Pass `start_node=None` explicitly when no shared resource is needed. In this case,
the normalized configuration becomes the producer configuration directly, and
`create_sink` can construct the plugin's SDK adapter in each producer process.
Only plain configuration crosses spawn; never return a live socket, thread, or
SDK client in `producer_config`. TokenSpeed validates and copies both configuration
results before passing them to another process.

Registering an existing name requires `override=True`. Failed plugin registration
rolls back its metric registrations along with its other runtime registrations.
The name `prometheus` is reserved. A selected plugin that fails registration or is
disabled through `TOKENSPEED_DISABLE_PLUGINS` causes an unknown-reporter startup
error rather than silently dropping the requested output.

## Event semantics

`MetricEvent`, `MetricEventSink`, `MetricNodeSession`, and `MetricConfig` are
defined in `tokenspeed.runtime.metrics.events`. A sink implements
`emit(event)` and `close(timeout)`. Neither method requires inheriting a base
class.

| Event field | Meaning |
| --- | --- |
| `timestamp_ns` | Observation wall-clock timestamp in Unix nanoseconds. |
| `metric` | Existing metric name, including the counter `_total` suffix. |
| `operation` | `inc`, `set`, or `observe`. |
| `value` | Counter delta, current gauge value, or histogram observation, in the metric's documented unit. |
| `labels` | Labels supplied by the collection point. |
| `weight` | Observation multiplicity; 1 for `inc` and `set`. |

Treat the event and its labels as read-only and retain a stable snapshot before
returning from `emit`. An `observe` event with value 0.01 and weight 3 contributes
three samples of 0.01: a count of 3 and a sum of 0.03. Preserve this weight when
adapting a destination SDK. It does not recover per-token timing that the
collector never measured.

Events are observations before reporter aggregation, not cumulative Prometheus
snapshots, request summaries, or request payload logs. Labels and collection
ranks retain their existing meanings; a plugin must not independently discard
every nonzero global rank. A headless engine has no frontend and therefore does
not generate frontend request metrics. See
[server observability](configuration/server.md#observability) for metric
definitions and engine roles.

For each metric and label set, plugins receive the first gauge value and every
value change. Repeated assignments of the same value are suppressed before event
construction and fan-out. If the value stays unchanged, the next assignment at
least one second after the last emitted event produces a fresh snapshot. The
interval uses a monotonic clock; event timestamps remain wall-clock observation
times. There is no timer emitting values when the producer stops updating, and
a long gap does not backfill snapshots. Plugins do not need to implement this
filtering themselves.

Gauge events describe state changes, not uniformly spaced samples. Carry the
last value forward between updates and use elapsed time for time averages; do
not sum successive gauge values. Counter increments and histogram observations
are never deduplicated. Prometheus continues to receive every update. Delivery
is best-effort: a later gauge snapshot can restore the current value but cannot
recover a lost transition.

## Asynchronous delivery and shutdown

Multiple collector threads can call `emit`. It must return promptly without
waiting for network I/O, disk I/O, serialization, or free queue capacity. Use a
bounded queue, perform delivery in the background, and report errors and dropped
events with rate-limited logs. Do not report output failures recursively through
the same metric channel.

TokenSpeed lazily creates one shared sink per selected output in each producer
process. A sink that raises during `emit` is disabled without stopping other
outputs; its resources are retained for shutdown. TokenSpeed does not rebuild a
failed sink on the observation path.

Stop accepting new events when `close` begins, and make repeated `close` calls safe.
Each process that reaches metrics cleanup drains its local sinks within a shared
time budget. The launcher closes its local sinks before shared node resources.
Metric reporters do not change worker termination: a forcibly terminated worker
may lose queued events without running `close`. Factories must clean up resources
they allocate before raising; TokenSpeed closes objects that were successfully
returned. An exception while closing one object does not skip the others.

These are cooperative contracts for trusted Python plugins. Catching exceptions
cannot prevent a plugin from blocking inside `emit` or ignoring its `close` timeout.
Delivery is best-effort unless a particular plugin documents stronger guarantees;
the runtime does not provide durable queues or exactly-once reporting.
