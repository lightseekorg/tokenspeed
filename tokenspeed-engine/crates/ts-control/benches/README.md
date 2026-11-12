# Rust engine CPU benchmark

`engine_cpu` measures the existing ingress, logprob output and frontend transport
paths with CPU metadata only. It runs real `EngineRequest`/`EngineOutput` codecs,
`ingress::admit`, `OutputProcessor`, `output_batch`, `EngineTransport`,
`EngineClient` and `RequestStream`. It uses the host's mimalloc allocator.

The workload supplies valid `ForwardBatchView` and `ForwardResult` values instead
of running a model or C++ scheduler. It does not measure GPU work, D2H copies,
Python result extraction, tokenization, HTTP/JSON/SSE handling, Python client
bindings, multi-rank collectives, or scheduler admission. Those boundaries need
separate measurements; a fast channel enqueue is not a completed output send.
Request logging, metrics, tracing and grammars are disabled in this workload.

## Build and run

Use the engine's normal native build prerequisites and a Python environment with
nanobind and tokenspeed-spdlog. `cargo bench` uses an optimized profile:

```bash
cargo bench -p tokenspeed-control --bench engine_cpu --no-run
# Run the executable path printed by Cargo, with all arguments explicit:
./engine_cpu SUITE TRANSPORT BATCH PROMPT STEPS LOGPROBS TOP_K PERIOD_US CONSUMER_DELAY_US CLIENT_THREADS WARMUPS REPEATS PHASE_TIMERS
```

| Argument | Meaning |
| --- | --- |
| `SUITE` | `stages` for synchronous component measurements; `pipeline` for genuine client/engine sockets |
| `TRANSPORT` | `none` for stages; `ipc` or loopback `tcp` for pipeline |
| `BATCH` | Simultaneously offered requests and rows in each result batch |
| `PROMPT` | Tokens per request; 128K means 131,072 and 1M means 1,048,576 |
| `STEPS` | Output tokens per request, one streamed chunk per step |
| `LOGPROBS` | `none`, `output`, or `prompt` (all prompt positions plus output logprobs) |
| `TOP_K` | 0–20 alternatives per scored position, subject to the production ingress restriction below |
| `PERIOD_US` | Pipeline result-ready cadence; 0 means run as fast as possible; must be 0 for stages |
| `CONSUMER_DELAY_US` | Pipeline consumer delay between chunks; 0 for immediate consumption and for stages |
| `CLIENT_THREADS` | Pipeline client Tokio worker count; must be 0 for stages |
| `WARMUPS` | Complete unrecorded trials |
| `REPEATS` | Complete recorded trials |
| `PHASE_TIMERS` | `on` or `off`; `off` disables component timers for paired pipeline calibration, preserving all work and correctness checks |

A new process should run one configuration. Pin stages to an otherwise idle CPU;
for pipeline use an explicit CPU set large enough for the control thread, the
engine's single I/O worker and the requested client workers. Record CPU topology,
source revision, binary hash, compiler, allocator, affinity and maximum RSS.
Do not compile or run other measurements concurrently.

Current production ingress only allows `top_logprobs_num > 0` when
`max_new_tokens <= 1`. Consequently streaming cases use selected-token logprobs,
and top-5/top-20 cases use one prefill output. Do not bypass ingress to manufacture
an unsupported streaming top-k result. Output-only requests follow the real
protocol, including its first-position placeholder for the last prompt token.

## Timers and interpretation

Each process emits JSON containing every recorded trial. Stage timers report wall
and current-thread CPU milliseconds. Pipeline also reports total process CPU
while requests and chunks are active. Thread-CPU clocks are read through rustix;
all benchmark Rust code remains safe.
Latency buffers are allocated before requests are released, and report JSON is
constructed after timing. `all_received_ms` ends at the last chunk's receipt;
`consumer_completed_ms` also includes final validation, disposal and task joins.
The `PHASE_TIMERS` switch calibrates component clocks only: latency timestamps
and constant-sized validation remain enabled in both runs.

`run_engine_cpu.py` runs the documented survey and selected repeat/calibration
matrices in separate processes, retaining raw samples and nonzero exit codes:

```bash
python run_engine_cpu.py --binary /path/to/engine_cpu --output results \
  --matrix full --stage-cpu 20 --pipeline-cpus 20-23 \
  --warmups 1 --repeats 3 --rounds 1 --round-offset 0 --timeout-seconds 900
```

The full survey covers 8K/32K/128K/256K/512K/1M at batches 16/128/512 for
output-only and complete-prompt logprobs. Top-k alternatives use a smaller,
explicit first-packet matrix; it is not the full Cartesian product. `anchors`
repeats representative configurations and `calibration` pairs component timers
on/off. Choose the CPU set for your machine; process affinity is inherited by
the benchmark's threads.

The stage suite measures request encoding, decoding, Rust admission and state
registration, logprob planning, commit, wire batch construction, encoding,
decoding, client chunk materialization and client chunk destruction. Fixture
construction and correctness checks are outside these stage timers. Returned
values are consumed, and every chunk's lengths, IDs, logprobs and finish status
are validated.

The pipeline performs the normal handshake before timing, then releases all
request tasks through a barrier. Every request has the same offer timestamp, so
waiting for a runtime worker remains part of ingress latency. The engine admits
all offered requests, then commits one batch per scheduled result-ready tick.
It does not wait for each batch to be consumed before producing the next.

Two chunk latency clocks are retained: actual model-result-ready to consumer,
and scheduled-result-ready to consumer. Producer lateness is reported separately.
Thus an overloaded producer cannot improve apparent latency simply by generating
fewer batches. All expected chunks must arrive, in order, before a trial ends.
Prompt logprobs must appear exactly once. The first large prompt-logprob payload
can delay later decode chunks; do not call those chunks steady-state throughput
until that backlog has drained.

`max_pending_output_rows` is the observed high-water mark of produced minus consumed rows, across all transport
and client queues combined. It is not the length of any particular internal
queue. Both transport and request-stream channels are currently unbounded. Slow
consumer cases quantify backlog and drain time rather than treating enqueue time
as network latency. A consumer checks a constant-sized sample of each chunk's
values and immediately drops it; its optional delay is explicit. The behavioral
tests check complete small prompt/top-k values and real IPC/TCP lifecycle cleanup.

Prefix arrays and the initial synthetic model-result arrays are built before the
pipeline timer. No model compute time is simulated. Long request encoding,
validation, prompt copies, logprob assembly and actual wire/client work are timed.
The C++ scheduler's token copy and planning are outside this benchmark.

Report the first chunk separately from subsequent streaming chunks. Compute
end-to-end totals and percentiles from matching sample records, never by adding
independent phase medians. Rows and chunks within a batch share queues and are
correlated; retain trial counts and raw distributions. Peak RSS includes the
fixture, native libraries, live state, queues and allocator-retained memory, so
it is not a pure engine-cache memory gauge.

Large full-prompt top-k cases can contain billions of alternatives. Increase
batch and prompt sizes gradually, publish the tested matrix, and keep untested
combinations explicit. Bounds on available memory do not establish a production
KV-cache capacity.
