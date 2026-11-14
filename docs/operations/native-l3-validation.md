# Native Mooncake L3 serving validation

The cross-instance model gate is implemented in
`test/runtime/test_l3_serving_e2e.py` and **pending GPU validation**. Its
current scope is one fused attention-TP rank, a local TCP Mooncake Store,
and greedy decoding without speculation. The example configuration is a
starting point, not a qualified model result.

The gate complements the executor's live-master CPU tests and
`test/runtime/test_l3_flat_kv_e2e.py`, which tests D2H/PUT/GET/H2D byte
preservation. Passing those component tests does not establish model
serving correctness.

## Required evidence

Each configured case starts a new master and an independent segment
provider, then runs three inference engines sequentially:

1. The **writer** serves a prefix containing a random codeword. Filler is
   tokenized and cut to an exact prefill-chunk boundary; decoding must
   confirm that the codeword survived. The writer must report no cached
   tokens, positive successful PUT keys and adapter bytes, no failed or
   dropped backups, and zero outstanding PUT jobs, scheduler operations,
   backup candidates and quarantine. Evidence must remain stable across
   several metrics exports before the writer stops.
2. The fresh **reader's first request** appends a question to that prefix.
   No earlier request warms its Device/Host cache. The request must report
   `cached_tokens` exactly equal to the prefix length, alongside positive
   successful GET key/byte deltas and no failed GET keys. The
   `tokenspeed:prefix_l3_hit_tokens_total` delta is supporting **QUERY-hit**
   evidence: it counts keys found before GET and admission, per cache group.
   It is not an admitted-token counter.
3. A second, disjoint request on the reader is the **negative control**.
   It must report zero cached tokens, GET successes and QUERY-hit deltas,
   with no quarantine.
4. With the provider and master stopped, an **oracle** engine recomputes
   the probe with prefix caching and Host KVStore disabled. Both the
   server's `--enable-output-logprobs` flag and the request's
   `return_logprob=True` are set. Reader and oracle continuations must
   satisfy the configured token-mismatch and finite logprob envelopes.
   Both decoded answers must contain the random codeword; the control's
   answer must not. Agreement between answers that ignore the prefix fails.

Every mandatory gauge must be present; an absent gauge cannot prove zero
outstanding work. Counters may be absent before their first increment.
NaN and infinite evidence or logprobs fail. PUT adapter bytes describe
successful submitted payload, not measured wire traffic.

The independent provider outlives the writer. Inference workers mount no
storage segments (`global_segment_size=0`). The gate never calls
`flush_cache` between writer and reader because that deletes remote objects.

Chunk alignment and a one-chunk suffix are intended to align the reader's
extend with the oracle's final prefill chunk. Numerical equivalence still
needs real runs for each model, kernel and layout. Dense, MLA, hybrid-state,
quantized and speculative configurations are not collectively validated by
this implementation or its CPU tests.

## Run on a GPU host

Install TokenSpeed with its Rust engine host, `torch`, `transformers`, and
the required `tokenspeed-mooncake` wheel. The wheel supplies
`libmooncake_store.so` and `mooncake_master`. Unset
`TOKENSPEED_MOONCAKE_STORE_LIBRARY`: this gate rejects overrides so that
engines and provider use the same wheel library. Make each checkpoint
accessible. The test owns its master and provider; no existing store is required.

Copy and edit `test/runtime/l3_serving_e2e_config.example.json`, then run:

```bash
L3_SERVING_E2E_CONFIG=/path/to/gate.json \
    python3 -m pytest test/runtime/test_l3_serving_e2e.py -v -s
```

Only an **unset** environment variable skips the test. Once set, even a
blank value, invalid config, unavailable dependency, missing CUDA device,
missing native library or model-loading failure fails the requested gate.
Retain the configuration, checkpoint revision, engine logs, hardware and
software versions with any claimed result. No GPU serving run is recorded
by this change.

## Configuration

All fields in the example are explicit. Unknown fields are rejected.

| Fields | Meaning |
| --- | --- |
| `schema_version`, `tenant_prefix` | Version `1`; a lowercase prefix extended with a random tenant suffix per case. |
| `mooncake` | Shared provider/engine connection. `protocol` must be `tcp`. |
| `provider.segment_bytes` | Independent storage capacity, at least 16 MiB; size it for the chosen prefix and model. |
| `timeouts` | Engine startup, generation, evidence settling and metrics-poll intervals in seconds; all must be finite and positive. |
| `async_overrides` | Engine L3 options; `wait_timeout_ms` and `max_prefetch_tokens` are required explicitly. The whole aligned prefix must fit the prefetch cap. |
| `cases[].name`, `model_path`, `dtype` | Unique case name, accessible checkpoint, and supported floating-point dtype. |
| `extra_engine_args` | Model/backend arguments shared by all engines. Managed flags and aliases, parallel topology overrides and speculative settings are rejected. |
| `page_granularity_tokens`, `chunked_prefill_tokens` | Operator-supplied model page geometry and a chunk size divisible by it. The runtime's actual geometry still needs checking for each model. |
| `min_prefix_tokens` | Rounded up to a whole chunk, then enforced by cutting token IDs without padding. The codeword must survive decoding. |
| `max_model_len`, `max_num_seqs`, `max_prefill_tokens` | Engine limits; prefix/suffix/output budgets must fit and the prefill budget must cover a chunk. |
| `gpu_memory_utilization`, `kvstore_ratio`, `seed` | Engine/Host sizing and deterministic seed. |
| `writer_max_new_tokens`, `suffix_max_new_tokens`, `min_output_tokens` | Writer and probe generation budgets; minimum oracle output length rejects vacuous comparisons. |
| `max_token_mismatches`, `logprob_atol` | Predeclared numerical envelope. Start with zero token mismatches; do not loosen thresholds just to pass a failed run. |
| `min_put_success_keys`, `min_put_success_bytes` | Writer evidence floors. |
| `min_get_success_keys`, `min_get_success_bytes` | Reader GET-delta floors. |
| `min_l3_query_hit_tokens` | QUERY-hit floor; multiple cache groups can count the same token span more than once. Admission is checked separately through exact `cached_tokens`. |

## Failure and cleanup

Missing PUT evidence or pending backups prevent the writer from finishing.
Missing GET evidence, partial reuse or a QUERY miss fails the restore.
An output mismatch can reflect incorrect restored state or different
numerics; investigate the actual backend and shape before interpreting it.
A missing codeword makes the output comparison inconclusive and fails.

Generation uses a daemon-thread deadline so a stuck request cannot keep
Python alive at exit. Cleanup then stops its engine through `WireDriver`'s
owned launcher process group. The provider logs to a temporary file, avoiding
an unread pipe, and receives SIGTERM followed by SIGKILL after its grace
period. Its stop is idempotent; cleanup diagnostics are attached to an
existing failure instead of replacing it. Final OS process reaping is not
an absolute wall-clock guarantee. Repeated-signal and whole-job cleanup
qualification remains separate from these provider contract checks.

## Remaining acceptance work

Real GPU model results, MLA/hybrid layout equivalence, multi-rank and
PD-role restores, RDMA, deterministic abort-during-L3-wait, sustained load,
latency and release/canary qualification remain open. A passing opt-in model
case would establish only that recorded case and environment.

The CPU workflow runs `test/test_l3_serving_gate.py` with only pytest. It
checks configuration, command lines, evidence semantics, output/codeword
comparisons, alignment, real child-process cleanup, and the unset/invalid
entrypoint behavior. It does not execute a model or native store.
