# test/stress

Traffic-simulation harness for TokenSpeed. Emits structured JSONL events
and produces a summary with TTFT/TPOT/E2E percentiles, error taxonomy,
cancel-by-stage counts, RSS-growth tracking, output-quality audit findings,
and `/health` + `/health_generate` probe results (including status
transitions) correlated against the request timeline.

Run it as a module from the repo root: `python -m test.stress ...`.

## What "healthy" means here

A server can be *up* and *fast* and still wrong, so the harness checks state in
layers — each is a separate, independently-disable-able signal:

| Layer | Question | Where |
|-------|----------|-------|
| Liveness | Is the process answering `/health`? | `monitors/health.py` |
| Responsiveness | Do requests complete, and fast enough? | `client.py` → `metrics.py` |
| Resource | Is memory stable over time? | `monitors/rss.py` |
| Output correctness | Is the *response* actually good? | `audits/` (per-response) |
| Server-wide quality | Spec-decode acceptance, etc. | `monitors/metrics.py` (`/metrics`) |

The last two are the [output auditing](#output-auditing) component. Per-response
checks (gibberish, JSON validity, token/usage consistency) run in the client;
server-wide, time-windowed checks (acceptance length) scrape Prometheus.

## Requirements

- Python 3.9+ and `aiohttp` (already a TokenSpeed dependency — no extra install
  needed inside the project env).
- `jsonschema` is **optional**: the grammar workloads use it to validate that
  responses conform to their JSON Schema. Without it, schema checks degrade to
  JSON-parse checks only (everything else still runs).

## Layout

```
test/stress/
  __main__.py         # CLI: `python -m test.stress run ...`
  runner.py           # arrival processes + concurrency gating + circuit breaker
  client.py           # aiohttp /v1/chat/completions with streaming + cancel
  events.py           # event dataclasses + JSONL sink
  metrics.py          # percentile + summary aggregation
  launcher.py         # optional `--launch-cmd` server lifecycle manager
  monitors/
    health.py         # background /health + /health_generate poller
    rss.py            # background per-process RSS sampler (leak detection)
    metrics.py        # /metrics scraper (spec-decode acceptance, etc.)
  audits/             # pluggable per-response output checkers (see below)
  workloads/          # pluggable traffic generators (see table below)
  scripts/            # self-contained stress scenarios (server config + run)
```

## Scenarios & on-demand CI

A *scenario* is a self-contained script under `scripts/` that owns one
configuration end to end — it boots a server (via `--launch-cmd`), drives a
workload, and writes `events.jsonl` + `summary.txt` to `$OUT_DIR`. Tunables come
from the environment (`OUT_DIR`, `DURATION_S`, `MAX_CONCURRENCY`), so a scenario
runs the same by hand or in CI:

```bash
DURATION_S=120 bash test/stress/scripts/kimi-k2.5-nvfp4-reality-mix.sh
```

The `Stress Test` GitHub workflow (`.github/workflows/stress.yml`) is
config-agnostic: it just checks out a ref and runs the chosen scenario script,
then publishes the summary to the run page. It's **manual only** (Actions →
*Run workflow*) — heavyweight, not part of per-commit CI — and can stress any
branch/tag/commit (`ref`, default `main`). Add a scenario by dropping a new
script in `scripts/` and selecting it with the `scenario` input.

## Quick start

```bash
# Run against a local server for 60s, moderate concurrency.
python -m test.stress run \
    --base-url http://127.0.0.1:22345 \
    --model openai/gpt-oss-120b \
    --workload shared_prefix \
    --workload-arg num_prefixes=8 \
    --workload-arg prefix_chars=6000 \
    --max-concurrency 64 \
    --duration 60 \
    --out stress_out/shared_prefix_smoke

# Poisson arrivals, 20 req/s.
python -m test.stress run \
    --workload cancel_mix \
    --arrival poisson --rate 20 \
    --duration 120

# Bursty traffic: 50 requests every 5s.
python -m test.stress run \
    --workload long_context \
    --arrival bursty --burst-size 50 --burst-period 5 \
    --duration 300

# Re-summarize an existing run.
python -m test.stress summarize --events stress_out/.../events.jsonl

# Launch the server, run a workload, tear it down (one-shot).
python -m test.stress run \
    --launch-cmd 'bash run-mm25.sh' \
    --model togethercomputer/minimax-fp4-test \
    --workload shared_prefix \
    --max-concurrency 32 --duration 60
```

When `--launch-cmd` is passed, the harness:

1. Refuses to launch if something is already on `--base-url` (no silent collisions).
2. Spawns the command in its own process group, tees logs to
   `<out>/server.log`.
3. Polls `/health` until 200 (default 900s timeout; tune with
   `--launch-timeout`).
4. Runs the workload.
5. Sends SIGTERM to the whole process group, escalates to SIGKILL after
   30s. This matters: `tokenspeed serve` forks TP workers + a C++
   scheduler; a bare SIGTERM to the wrapper can orphan them.

## Arrival modes

`--arrival` controls *when* yielded requests fire; `--max-concurrency` caps
in-flight requests in every mode.

| Mode | Pacing | Key flags |
|------|--------|-----------|
| `constant` | As fast as the concurrency cap allows. | `--max-concurrency` |
| `poisson` | Exponential inter-arrivals at mean rate. | `--rate` |
| `bursty` | `burst-size` requests every `burst-period` seconds. | `--burst-size`, `--burst-period` |
| `sawtooth` | In-flight cap ramps linearly between `min`/`max` over a triangle period. Stresses continuous-batching transitions. | `--min-concurrency`, `--max-concurrency`, `--triangle-period` |
| `burst` | Dispatch `burst-size`, drain to zero, sleep `burst-gap`, repeat. Forces the scheduler through 0→N→0 cycles. | `--burst-size`, `--burst-gap` |

A **circuit breaker** trips open after `--breaker-threshold` consecutive request
errors and pauses dispatch for `--breaker-cool-s` seconds, so a dead server
can't turn `constant` arrival into a multi-million-error tight loop. Disable
with `--no-breaker` (not recommended).

## Workloads

| Name | Purpose |
|------|---------|
| `shared_prefix` | Fixed pool of long system prompts, short unique user turns. Exercises radix cache + hicache. |
| `long_context` | Large prompts + large `max_tokens`, tuned to stay under the retraction threshold. Stresses long prefill + decode. |
| `cancel_mix` | Streaming + non-streaming mix; random fraction cancelled at `queue` / `prefill` / `decode`. Stresses request-slot + KV-page release. |
| `retract_stress` | High-concurrency long generations that exceed the device KV pool, forcing retract → host writeback → host loadback cycles. |
| `reality_mix` | Realistic blended soak: grammar/non-grammar, short/medium/long prompts, streaming + cancels all at once. |
| `grammar_schema` | Rotates through valid JSON schemas (`response_format` + client-side `validate_schema`). Stresses the xgrammar + capturable-grammar pipeline. |
| `grammar_schema_diverse` | Many distinct schema keys to defeat the compile cache; stresses `GrammarManager` under sustained compile load. |
| `grammar_burst_drain` | Grammar requests paired with the `burst` arrival mode (0→N→0 admission/drain). |
| `grammar_cancel_mix` | Grammar requests crossed with mid-lifecycle cancels; exercises cancel-of-compiling-grammar abort paths. |
| `grammar_invalid_schema` | Malformed schemas that xgrammar should reject at compile time — must error cleanly, not crash/hang. |

List the registered names at runtime: `python -m test.stress run --help`
(they are the `--workload` choices). Add a workload by writing an async
generator under `workloads/`, decorating it with `@register("name")`, and
importing it in `workloads/__init__.py`.

## Health monitoring

By default a background poller hits `/health` and `/health_generate` once
per second and records:

- Per-probe `health_probe` events (status, latency, detail).
- `health_transition` events every time a path flips `ok` <-> `fail`.

In tokenspeed both paths currently share one handler (see
`python/tokenspeed/runtime/entrypoints/http_server.py:281`) — 200 means a
detokenizer heartbeat was observed within `HEALTH_CHECK_TIMEOUT`, 503
means it wasn't. We still poll both so that if the handler is later
split into cheap-liveness vs. deep-readiness, the monitor picks it up
for free.

Disable with `--no-health`; tune with `--health-interval` /
`--health-timeout`.

## RSS monitoring

A background sampler walks the server's process tree and records per-pid RSS,
so the summary can report total growth, slope (MB/min), peak, and the top
processes by growth — useful for catching leaks over long soaks.

It needs a root PID. When `--launch-cmd` is used the harness samples the
launched process group automatically; otherwise pass `--server-pid <N>`.
Disable with `--no-rss`; tune the cadence with `--rss-interval` (default 2s).

## Output auditing

The auditors answer "is the output *correct*", not just "did the server
respond". They split by the data they need:

**Per-response auditors** (`audits/`) — pure `(ResponseRecord) -> [Finding]`
functions run by the client on every completed response, emitting
`audit_finding` events. Built-ins:

| Auditor | Flags |
|---------|-------|
| `length_consistency` | `empty_content` (server billed N completion tokens but the response is blank — the special-token / empty-delta bug) and `low_visible_ratio`. Relies on the client tracking server-reported vs. client-observed tokens separately. |
| `json_schema` | Response doesn't parse as JSON, or violates the request's schema (grammar workloads). Subsumes the old inline schema check. |
| `degeneration` | Heuristic gibberish: runaway-whitespace loops, low lexical diversity / trigram repetition. **Warn-level** — no ground truth. |
| `finish_reason` | `finish_reason` outside the known-good set. |

Select with `--audit <name>` (repeatable; default: all), disable with
`--no-audit`. Add one by writing a `@register("name")` function in `audits/`.

**Hang / stall detection** — two distinct levels:

- **Per-request (warn)** — a single streaming request that stops producing
  tokens for `--stall-timeout` (default 20s; 0 disables) emits `request_stall`
  and ends as a `stall` error. Scope is one request (the engine may have lost
  track of it); the rest of the fleet is unaffected, so it does not fail the run
  on its own.
- **Global (fatal)** — if requests are **in decode** (have produced a first
  token, not finished) yet none yields another token for `--global-stall-timeout`
  (default 20s; 0 disables), that's a server-wide decode wedge. It emits
  `global_stall`, **aborts the run immediately** (cancelling wedged requests
  instead of waiting out `--request-timeout`), and exits nonzero. Gating on
  in-decode requests keeps the short window from false-firing on benign
  all-prefill lulls.

Non-streaming hangs are caught by `--request-timeout` (600s).

### Severity levels

Every finding carries a severity that decides what it does to the run:

| Severity | Effect | Examples |
|----------|--------|----------|
| `warn` | Logged + in the final report. | per-request `request_stall`, `degeneration`, `finish_reason`, `low_visible_ratio`, `spec_acceptance` |
| `error` | Logged + in the report; counts toward `--fail-on-audit`. | `empty_content`, `json_schema` |
| `fatal` | Run is **cut off immediately**, report still produced, exit code 2. | `global_stall` |

**Server-wide metrics** (`monitors/metrics.py`) — scrapes `/metrics` and
derives spec-decode `accept_len = Δaccepted/Δdrafts + 1` over each window,
emitting `metrics_probe` and an `audit_finding` (`spec_acceptance`) when it
stays below `--accept-len-min` (default 1.1) for consecutive windows. Disable
with `--no-spec-metrics`. Silent when spec decode is off (counters absent).

**Gating (opt-in).** Findings are recorded by default. To make a run fail,
pass `--fail-on-audit CHECK=N` (repeatable) — the process exits nonzero if that
check's count exceeds `N`. `CHECK` is an auditor name or `stall`. Example:

```bash
python -m test.stress run --workload reality_mix --duration 600 \
    --fail-on-audit length_consistency=0 \
    --fail-on-audit stall=0 \
    --fail-on-audit spec_acceptance=0
```

## Fault injection (future work)

Fault injection is intended to become a first-class harness **component**, not
something the operator drives by hand. The plan: a `faults/` module of
registered fault scenarios (mirroring the `workloads/` registry) that the runner
schedules automatically during a run — e.g. `--fault sigstop_scheduler@30s`,
`--fault kill_tp_worker@60s` — and records as structured events so the summary
can correlate each injected fault against the `/health` transitions and request
timeline it caused. This is **not implemented yet**.

Until then, inject faults manually: run a workload in one terminal, and in
another:

- `SIGSTOP` the scheduler process → expect `/health` to flip to `fail`.
- Kill one TP worker → NCCL hang; `/health` should flip.
- Exhaust KV with long-context + zero retraction budget → detokenizer
  goes silent; `/health` should flip.
- Start the server and poll `/health` during load; expect 503 until the
  engine reaches `ServerStatus.Up`.

The summary prints every health transition with its timestamp, so you can
eyeball whether the signal led, lagged, or missed each incident — the same
correlation the planned component will produce automatically.

## Event schema

Each line of `events.jsonl` is `{"kind", "ts", "data": {...}}`. Kinds:

- `run_started`, `run_finished`
- `request_submitted`, `request_first_token`, `request_completed`,
  `request_cancelled`, `request_error`, `request_stall`
- `global_stall` (fatal server-wide decode wedge; aborts the run)
- `audit_finding` (per-response and server-wide quality findings)
- `health_probe`, `health_transition`
- `rss_probe`, `metrics_probe`
- `breaker_open`
- `request_invalid_schema` (legacy; superseded by `audit_finding`)

Add your own consumer by reading the file line-by-line — there is no
shared in-memory schema to import.
