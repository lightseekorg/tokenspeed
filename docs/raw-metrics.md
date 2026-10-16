# Raw metrics JSONL

TokenSpeed can write metric observations to local JSONL files for an independent
consumer to forward to a monitoring system. The public interface is the file
format. A consumer reads the files directly from the same machine or a shared
volume; no TokenSpeed client library or HTTP endpoint is required.

JSONL is a built-in [metric output plugin](metrics-plugins.md). Other outputs can
be installed independently while sharing the same collection points and event
semantics.

## Enable the reporter

Add these arguments to the engine launch command:

```bash
--enable-metrics --metrics-reporters jsonl --metrics-jsonl-dir /var/log/tokenspeed/metrics
```

To also expose Prometheus metrics, select both reporters:

```bash
--enable-metrics \
  --metrics-reporters prometheus \
  --metrics-reporters jsonl \
  --metrics-jsonl-dir /var/log/tokenspeed/metrics
```

| Argument | Default | Meaning |
| --- | --- | --- |
| --metrics-jsonl-dir | Required for JSONL | Output directory owned exclusively by this instance's node-local writer. |
| --metrics-jsonl-max-size-mb | 256 | Target maximum shard size in MiB. |
| --metrics-jsonl-backup-count | 4 | Closed shards retained across restarts in this directory, in addition to the active shard. |
| --metrics-jsonl-queue-size | 10000 | Bounded pending sample count per sender. |
| --metrics-jsonl-flush-interval | 1.0 | Writer flush interval in seconds. |

The JSONL reporter uses the same collection points as the other reporters.
It does not change metric labels or turn scheduler statistics into per-request
measurements. A headless engine has no frontend and therefore does not generate
frontend request metrics. See [server observability](configuration/server.md#observability)
for metric definitions and engine roles.

## File format

Each node has one writer, with exclusive ownership of its output directory.
Instances and nodes using a shared volume must use different subdirectories.
Separate machines can use the same path on their own local filesystems. Readers
can access a writer's directory concurrently.

A writer has a unique stream ID, including after a restart. Its directory can
therefore contain retained files from earlier streams alongside the current one.
Files are named metrics-{stream_id}-{shard:06d}.jsonl. Shard numbers and sample
sequences start at zero. The writer assigns sequence numbers as samples reach it;
they increase across shards within a stream. Concurrent senders can arrive in a
different order from their observation timestamps. There is no total order
across streams.

Each complete UTF-8 line is one JSON object, terminated by a newline:

```json
{"schema_version":1,"stream_id":"example-stream","sequence":42,"timestamp_ns":1790000000000000000,"metric":"tokenspeed:inter_token_latency_seconds","operation":"observe","value":0.01,"labels":{"model_name":"example"},"weight":3}
```

| Field | Meaning |
| --- | --- |
| schema_version | Format version; currently 1. |
| stream_id | Writer stream identity, also present in the filename. |
| sequence | Monotonically increasing sample number within this stream. |
| timestamp_ns | Observation wall-clock timestamp in Unix nanoseconds. |
| metric | Metric name. |
| operation | inc for a counter increment, set for a gauge assignment, observe for a histogram observation. |
| value | Numeric increment, gauge value, or observed value, in the metric's documented unit. |
| labels | Labels attached to the collector call. |
| weight | Number of observations represented by this sample; 1 for inc and set. |

An observe sample with value 0.01 and weight 3 represents three observations of
0.01. Its histogram count contribution is 3 and its sum contribution is 0.03.
This preserves existing weighted observations, such as an inter-token interval
divided across several tokens. It cannot recover the individual token timings
that were not measured by the collector.

These are observations before reporter aggregation. They are not cumulative
Prometheus snapshots, request summaries, or request payload logs. A consumer can
aggregate the samples for its destination and use (stream_id, sequence) as the
record identity for deduplication. Counter increments and gauge assignments must
retain their different meanings; do not sum successive gauge values.

For each metric and label set, JSONL records the first gauge value and every
value change. Repeated assignments of the same value are suppressed before event
construction and enqueueing. If the value stays unchanged, the next assignment
at least one second after the last emitted sample records a fresh snapshot. The
interval uses a monotonic clock; sample timestamps remain wall-clock observation
times. There is no timer emitting stale values when the producer stops updating,
and a long gap does not backfill snapshots. The writer's flush interval controls
buffer flushing independently of this snapshot interval.

Gauge records are state updates, not uniformly spaced samples: consumers should
carry the last value forward between updates and use elapsed time to compute
time averages. All actual gauge changes are retained before transport; counter
increments and histogram observations are never deduplicated. Prometheus updates
are unaffected. Delivery remains best-effort as described below; a later gauge
snapshot can restore the current value but cannot recover a lost transition.

## Reference consumer

Run the standard-library example from a repository checkout:

```bash
python examples/metrics/read_jsonl.py \
  --directory /var/log/tokenspeed/metrics \
  --checkpoint /var/lib/metrics-consumer/checkpoint.json \
  --follow
```

The reader writes samples to stdout and warnings to stderr. Without --follow,
it reads the complete records available in the discovered files and exits.
With --follow, it repeatedly discovers new shards and streams. It maintains
independent cursors for each stream; timestamps do not imply a global ordering.

The checkpoint stores the directory, filename, byte offset and last delivered
sequence for each stream. It is replaced atomically after each successful
consumption. Give each consumer its own checkpoint and keep that checkpoint
outside the writer's log files. Do not run two readers against one checkpoint.
A missing checkpoint starts at the oldest retained shard. A malformed checkpoint
or one belonging to another directory is an error, rather than an implicit reset.

The CLI checkpoints after stdout has been written and flushed. This acknowledges
delivery to stdout, not ingestion by a downstream monitoring service. For reliable
forwarding, adapt the example's consume callback to return only after the
destination acknowledges the record and to raise on failure. A crash between
delivery and checkpoint replacement can repeat a record. Deduplicate at the
destination when repeated increments would affect results.

The example checkpoints every record for clarity. A production consumer may
checkpoint acknowledged batches to reduce I/O, accepting a larger replay window
after a crash. The example is not a delivery service or a high-throughput SDK.

## Rotation and failure behavior

Rotation and retention belong to the writer. It does not wait for consumers or
read their checkpoints before removing closed shards. Consumers must keep up
within the configured retention window; increasing the file size or retained
shard count gives a slower consumer more time.

The writer holds an exclusive flock on .metrics.lock for its lifetime. Once it
acquires the lock at startup, it includes existing shards with valid metric
filenames in retention, including streams from previous processes. The backup
count bounds all retained closed shards in the directory; the active shard is
counted separately. Files outside the metric filename format are not removed.
Restarting does not reset retention or require separate cleanup of older streams.

If another writer owns the lock, setup logs an error and disables the new writer
without changing existing metric files. Use a different output directory for
each active instance or node. The filesystem must support flock; sharing a
volume for collection does not permit sharing a writer directory.

The reader only delivers newline-terminated records. It waits for an incomplete
active tail to be completed. If a later shard exists for the same stream, the
earlier shard is closed: an incomplete tail there is reported and skipped so it
cannot block subsequent samples. A partial tail in an old stream does not block
another stream after a writer restart.

Missing checkpoint files, missing shard numbers and discontinuous sequences
produce warnings while the reader continues with retained records. Such warnings
cannot reconstruct lost data. An entirely expired stream that the reader has
never observed cannot be detected from the remaining files. A complete malformed
JSON line or unsupported schema stops the example without checkpointing that
record.

Sequence numbers are assigned by the writer, after the sender queues. A sender
queue can drop a sample without leaving a sequence gap in the files. Check the
sender's dropped_events and error_count logs for those failures; sequence checks
are not an end-to-end loss detector.

Non-finite values such as NaN and infinity cannot be encoded as standard JSON.
They are dropped individually with an error count; later valid samples continue.

The reporter prioritizes inference availability. Bounded sender queues, process
crashes, unflushed buffers, disk errors and retention can lose samples. A flush
does not promise power-loss durability, and consumer checkpoints are not an
exactly-once transaction with the destination. The files are a best-effort
observability feed, not an audit log.

Normal shutdown sends SIGTERM to workers, waits within a time budget, and kills
remaining workers before closing the launcher's sender and writer. This gives
worker senders a chance to drain while the writer remains available, with bounded
best-effort draining throughout. Forced termination, including SIGKILL, can leave
queued samples and a partial final line behind; normal shutdown also does not
guarantee lossless delivery.
