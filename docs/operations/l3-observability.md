# Observing the L3 cache tier

With `--enable-metrics`, the `prometheus` reporter and a
`--kvstore-storage-backend`, every engine rank exports the Mooncake Store L3
tier's metrics on the node's `/metrics` endpoint
([Prometheus metrics](../configuration/server.md#prometheus-metrics)). This
page explains what the series mean and how to alert on them;
[`examples/monitoring/tokenspeed-l3-alerts.yaml`](https://github.com/lightseekorg/tokenspeed/blob/main/examples/monitoring/tokenspeed-l3-alerts.yaml)
is a ready rule file. The design is in
[asynchronous L3](../design/async-l3-cache.md).

## How the values are sampled

The engine's event loop reads the executor's counters under its lock and
writes them out at most once a second, and whenever storage turns pending or
idle. Reading them calls neither the store nor the backend's health monitor,
so a scrape never takes the recovery probe and never waits for storage I/O.
L3 adds no HTTP endpoint. Because the event loop writes the values, a stalled
loop leaves the last ones in place: pair these alerts with the scrape's `up`
and the engine's request metrics.

## Labels and scope

Every L3 series carries `model_name`, `app_key`, `dp_rank`, `tp_rank` and
`engine_role`, plus at most two fixed labels: `kind` (`query`, `get`,
`put`), `state` (`queued`, `running`, `completed`), `event`, `result`,
`stage`, `resource`, `phase` or `reason`. No request ID, object key or other
unbounded value becomes a label.

The series come from two owners, and they aggregate differently:

| Owner | Series | Across attention-TP ranks |
| --- | --- | --- |
| The rank's executor: its own shard of the I/O | `tokenspeed:l3_io_*`, `tokenspeed:l3_backend_state`, `tokenspeed:l3_backend_epoch` | Each rank reports its own work and health; sum the counters for a replica total |
| The scheduler, mirrored on every TP rank | `tokenspeed:l3_scheduler_resources`, `tokenspeed:l3_preparation_expirations_total`, `tokenspeed:l3_pressure_bypasses_total`, `tokenspeed:l3_pressure_dropped_backup_keys_total`, `tokenspeed:l3_dropped_backup_keys_total` | Every rank reports the same value: use `max without (tp_rank)`, never `sum` |

`tokenspeed:l3_io_operations` counts this rank's executor jobs;
`tokenspeed:l3_scheduler_resources{resource="operations"}` counts the
scheduler's tickets, which also include Ready holds and quarantined
operations after the executor finished them.

## Operation age

`tokenspeed:l3_io_oldest_operation_age_seconds{kind, state}` is, for each
operation kind and state, the time since the oldest job now in that state
was submitted to this rank's executor, on its monotonic clock; it is 0 when
the state holds no job.

- `queued`: waiting for a worker.
- `running`: a store call in progress. A job whose request was cancelled
  stays here until the store call returns, because the call still uses its
  Host memory.
- `completed`: finished, including rejected and cancelled queued jobs, but
  not yet collected by the event loop. It returns to 0 once collected.

All three count from submission, not from entering the state, so the
largest `queued` or `running` age on a rank is the operation age it reports
to the scheduler's pressure gate (`operation_high_age_ms`). A running age
well past `wait_timeout_ms` means a store call that has not returned: its
request has already fallen back to recomputation, but the job still holds
its operation credit and Host references. A high `completed` age can also
come from a slow call that has just finished; it is not the time spent
waiting for collection. Check completed-job counts, request progress and
event-loop health before attributing persistent old completions to a polling
backlog.

The age gauge reports the oldest job only. The cumulative
`tokenspeed:l3_io_seconds_total{stage="queue"|"backend"}` gives mean times
through `rate()`, such as the mean store call:

```promql
sum without (kind, stage) (rate(tokenspeed:l3_io_seconds_total{stage="backend"}[5m]))
  / sum without (kind, event) (rate(tokenspeed:l3_io_events_total{event="backend_calls"}[5m]))
```

No L3 series gives latency percentiles.

## Bytes and hits

Three quantities are easy to confuse:

- `tokenspeed:l3_io_adapter_bytes_total` counts payload bytes the store
  adapter reported successful on this rank. It includes PUTs of objects that
  already existed and GETs whose request had detached. It is not network
  traffic; use the store's own transport metrics for that.
- `tokenspeed:prefix_l3_hit_tokens_total`, one series per DP rank, counts
  the tokens covered by keys that storage queries found, once per cache
  group. It shows what L3 could offer, not what admission reused.
- Prefix reuse that admission actually used is in
  `tokenspeed:prefix_host_hit_tokens_total`, which counts L2 and
  L3-supplied tokens together.

## Degraded, quarantined, and ready

These states are independent, and each needs a different response:

| Signal | Meaning | Recovery |
| --- | --- | --- |
| `tokenspeed:l3_backend_state` 1 (open) or 2 (half-open) | Store calls failed. The executor refuses jobs (`tokenspeed:l3_io_events_total{event="rejected"}`) for `recovery_interval_ms`, then lets one probe call through | Automatic, once a probe succeeds. A probe needs a job, so an idle rank stays open |
| `tokenspeed:l3_io_quarantined` 1, `tokenspeed:l3_scheduler_resources{resource="quarantined"}` above 0 | A store call could not prove it stopped touching Host memory. New L3 I/O stays off and those Host blocks stay held | Restart the engine process |
| `tokenspeed:l3_scheduler_resources{resource="pressure_gate_closed"}` 1 | Queue depth, queued bytes or operation age crossed a high watermark. New requests skip L3 preparation (`tokenspeed:l3_pressure_bypasses_total`) and backups are dropped | Automatic, once every observation is below its low watermark |

None of these is a readiness signal. Under `ts serve` the gateway (SMG)
decides whether an engine serves traffic; a degraded or quarantined L3 tier
can still serve from the Device and Host caches and by recomputing prefill
within the remaining capacity. Use gateway readiness and request health to
decide routing, rather than treating cache-tier health as serving readiness.
For quarantine, drain traffic and plan a process restart.

## Alerts

The [example rules](https://github.com/lightseekorg/tokenspeed/blob/main/examples/monitoring/tokenspeed-l3-alerts.yaml)
use only the series above. Their thresholds assume the default `async`
settings; scale them with your `wait_timeout_ms` and watermarks.

| Alert | Fires when | Severity |
| --- | --- | --- |
| `TokenSpeedL3OperationStuck` | A queued or running age stays above 30 s for 5 minutes | warning |
| `TokenSpeedL3CompletionsUndelivered` | A completed age stays above 30 s for 5 minutes | warning |
| `TokenSpeedL3Quarantined` | A rank or the scheduler reports quarantine for 1 minute | critical |
| `TokenSpeedL3BackendUnavailable` | The backend stays open while it rejects jobs, without quarantine, for 10 minutes | warning |
| `TokenSpeedL3PressureGateClosed` | The pressure gate stays closed while requests bypass L3, for 10 minutes | warning |

`ts-metrics`' tests check that every series, label value and aggregated
label these rules read is exported.

The Prometheus rule tests check firing delays, the age threshold, idle
traffic suppression, quarantine suppression and deduplication of mirrored
TP reports. Run them from the repository root with `promtool`:

```bash
promtool check rules examples/monitoring/tokenspeed-l3-alerts.yaml
promtool test rules examples/monitoring/tokenspeed-l3-alerts.test.yaml
```
