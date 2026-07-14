# Server Parameters

This page documents the parameters operators usually set directly. TokenSpeed
uses familiar serving parameter names where the semantics match and keeps
TokenSpeed-specific knobs for runtime features with different meaning.

For a compact compatibility table, see
[Compatible Parameters](./compatible-parameters.md).

## Model Loading

| Parameter | Purpose |
| --- | --- |
| positional `model` | Model path or Hugging Face repo ID. |
| `--model` | Equivalent to positional `model`. |
| `--tokenizer` | Tokenizer path when it differs from the model path. |
| `--tokenizer-mode` | Select tokenizer behavior. `auto` uses fast tokenizers and model-specific hooks when available. |
| `--skip-tokenizer-init` | Skip tokenizer initialization for input-ID-only serving paths. |
| `--load-format` | Weight loading format: `auto`, `pt`, `safetensors`, `npcache`, `dummy`, or `extensible`. |
| `--trust-remote-code` | Allow custom model code from the model repository. |
| `--revision` | Model branch, tag, or commit. |
| `--download-dir` | Hugging Face download/cache directory. |
| `--hf-overrides` | JSON overrides for model configuration values. |

## Precision And Quantization

| Parameter | Purpose |
| --- | --- |
| `--dtype` | Model weight and activation dtype. `auto` follows model metadata. |
| `--kv-cache-dtype` | KV cache dtype. Lower precision reduces KV memory and may require scaling factors. |
| `--kv-cache-quant-method` | KV cache quantization method. |
| `--quantization` | Weight quantization mode such as `fp8`, `nvfp4`, `w8a8_fp8`, or `compressed-tensors`. |
| `--quantization-param-path` | JSON file for KV cache scaling factors, commonly needed with FP8 KV cache. |

## API Surface

| Parameter | Purpose |
| --- | --- |
| `--host` | HTTP bind host. |
| `--port` | HTTP bind port. |
| `--served-model-name` | Model name returned by the OpenAI-compatible API. |
| `--api-key` | API key required by the server. |
| `--chat-template` | Built-in chat template name or template file path (handled by the smg gateway). |
| `--stream-interval` | Streaming buffer interval in generated tokens. Smaller values stream more frequently. |
| `--stream-output` | Return generated text as disjoint streaming segments. |

## Scheduler And Memory

| Parameter | Purpose |
| --- | --- |
| `--max-model-len` | Maximum sequence length. If omitted, TokenSpeed uses the model config. |
| `--gpu-memory-utilization` | Fraction of GPU memory used for model weights and KV cache. Lower it to leave headroom. |
| `--max-num-seqs` | Maximum number of active sequences the scheduler may process concurrently. |
| `--chunked-prefill-size` | Token budget the scheduler may issue in one iteration. Defaults to `8192`. Set `-1` to disable chunked prefill. |
| `--max-prefill-tokens` | Prefill token budget used when chunked prefill is disabled. Defaults to `8192`. |
| `--max-total-tokens` | Override the automatically calculated token pool size. |
| `--block-size` | KV cache block size. |
| `--enable-prefix-caching` / `--no-enable-prefix-caching` | Enable or disable prefix cache reuse. |
| `--enforce-eager` | Disable CUDA graph execution. |
| `--max-cudagraph-capture-size` | Largest batch size to capture with CUDA graphs. |
| `--cudagraph-capture-sizes` | Explicit CUDA graph capture sizes. |

`--chunked-prefill-size` is intentionally separate from
`--max-num-batched-tokens`: in TokenSpeed it is the scheduler's per-iteration
issue budget, while `--max-total-tokens` controls the global token pool.

## Parallelism

| Parameter | Purpose |
| --- | --- |
| `--tensor-parallel-size`, `--tp` | Familiar alias for setting attention tensor parallel size. |
| `--attn-tp-size` | Tensor parallel size for attention. |
| `--dense-tp-size` | Tensor parallel size for dense layers. |
| `--moe-tp-size` | Tensor parallel size for MoE layers. |
| `--data-parallel-size` | Number of data-parallel replicas. |
| `--enable-expert-parallel` | Set expert parallelism across the selected world size. |
| `--expert-parallel-size`, `--ep-size` | Explicit expert parallel size. |
| `--world-size` | Total worker process count across all nodes. |
| `--nprocs-per-node` | Worker process count per node. |
| `--nnodes` | Number of nodes. |
| `--node-rank` | Rank of the current node. |
| `--dist-init-addr` | Distributed initialization address. |

Use `--tensor-parallel-size` for simple launches. Use the
TokenSpeed-specific split knobs when attention, dense, and MoE layers need
different process groups.

## Backend Selection

| Parameter | Purpose |
| --- | --- |
| `--attention-backend` | Attention kernel backend. Common values include `mha`, `fa3`, `fa4`, `triton`, `flashinfer`, `trtllm_mla`, and `tokenspeed_mla`. |
| `--drafter-attention-backend` | Attention backend for speculative decoding drafter model. |
| `--moe-backend` | MoE backend. |
| `--draft-moe-backend` | MoE backend for the speculative decoding draft model. |
| `--all2all-backend` | MoE all-to-all backend. |
| `--deepep-mode` | DeepEP mode: `auto`, `normal`, or `low_latency`. |
| `--sampling-backend` | Sampling backend: `greedy`, `flashinfer`, or `flashinfer_full`. |

Set backend choices explicitly in production. `auto` is useful for bring-up, but
explicit values make benchmark comparisons and regressions easier to reason
about.

When `--dp-sampling` is enabled, the logits processor owns the per-forward
logits layout decision and carries the resulting plan to the sampling backend
with the logits output.

## Reasoning And Tool Calling

| Parameter | Purpose |
| --- | --- |
| `--reasoning-parser` | Parser for extracting reasoning content from model outputs (handled by the smg gateway). |
| `--tool-call-parser` | Parser for OpenAI-compatible tool-call payloads (handled by the smg gateway). |
| `--enable-custom-logit-processor` | Allow custom logit processors. Keep disabled unless the deployment needs it. |

Common reasoning parser values include `kimi_k25`, `base`, `qwen3`,
`deepseek_r1`, and `deepseek_v31`. Common tool-call parser values include
`kimik2`, `qwen`, `deepseek_v4`, `json`, and `passthrough`. The parser names
are validated by the SMG gateway, so use
the values accepted by the bundled `tokenspeed-smg` package.

## Speculative Decoding

| Parameter | Purpose |
| --- | --- |
| `--speculative-config` | JSON speculative decoding configuration. |
| `--speculative-algorithm` | Speculative algorithm, such as `EAGLE3`, `MTP`, or `DFLASH`. |
| `--speculative-draft-model-path` | Draft model path or repo ID. |
| `--speculative-draft-model-quantization` | Draft model quantization. Defaults to `unquant`. |
| `--speculative-num-steps` | Number of draft model steps. Defaults to `3`. |
| `--speculative-num-draft-tokens` | Number of draft tokens. Defaults to `--speculative-num-steps + 1`. |
| `--speculative-eagle-topk` | EAGLE top-k. Defaults to `1`. |
| `--eagle3-layers-to-capture` | EAGLE3 layers to capture. |

Prefer `--speculative-config` for recipe-style launches because it keeps method,
draft model, and token count together.

## Observability

| Parameter | Purpose |
| --- | --- |
| `--log-level` | Runtime log level. |
| `--log-level-http` | HTTP server log level. Defaults to `--log-level` when unset. |
| `--enable-log-requests` | Log request metadata and optionally payloads. |
| `--log-requests-level` | Request logging verbosity. |
| `--enable-log-request-stats` | Log a one-line per-request performance summary on finish/abort (see below). |
| `--enable-metrics` | Enable metrics reporting. |
| `--metrics-reporters` | Metrics reporter, such as `prometheus`. |
| `--decode-log-interval` | Decode batch log interval. |
| `--enable-cache-report` | Include cached-token counts in OpenAI-compatible usage details. |
| `--kv-events-config` | JSON config for KV cache mutation events (`KVEventsConfig`). Keys: `enable_kv_cache_events`, `publisher`, `endpoint`, `replay_endpoint`, `buffer_steps`, `hwm`, `max_queue_size`, `topic`, `wire_format`, `backend_id`, `tenant_id`, `model_name`, `publish_medium`, `publish_tiers`, `hash_mode`. See [KV Cache Events](#kv-cache-events). |

### Per-Request Stats

`--enable-log-request-stats` enriches the scheduler's per-request finish line for
latency/throughput debugging. When set, the `Req: <rid> Finish! ...` line carries
a Python-object repr (`RequestStats(...)`) instead of the default
`Accept_num_tokens_avg` value (which it subsumes as `acc_len`). Every field is
derived from host-side timestamps and counters already available in the
scheduler — it adds **no GPU sync** and so no engine slowdown. Example:

```
Req: chatcmpl-019ef6b7 Finish! RequestStats(status='finished', reason='stop', prompt_tokens=28684, cache_tokens=832, output_tokens=33, cache_hit_rate=0.029, queue_ms=13.8, prefill_ms=15.8, ttft_ms=42.1, total_ms=58.0, preempt_ms=0.0, preempt_count=0, decode_tps=210.4, acc_len=None, acc_rate=None, recv_ts=1782255696.726, commit_ts=1782255696.74, finish_ts=1782255696.784)
```

| Field | Meaning |
| --- | --- |
| `status` / `reason` | `finished` vs `aborted`; finish-reason type (`stop`/`length`/`abort`). |
| `prompt_tokens` / `cache_tokens` / `output_tokens` | Prompt tokens, prefix-cache-hit tokens, generated tokens. |
| `cache_hit_rate` | `cache_tokens / prompt_tokens` (0–1). |
| `queue_ms` | Received → first scheduled into a forward batch. |
| `prefill_ms` | Scheduled → prefill complete. |
| `ttft_ms` | Received → first output token (always ≥ `prefill_ms`; it also spans the queue). |
| `total_ms` | Received → finished/aborted. |
| `preempt_ms` / `preempt_count` | Wall-clock this request's decode was delayed by prefilling other requests, and the number of such interruptions. Host-side best-effort. |
| `decode_tps` | Decode throughput (generated tokens / decode window). |
| `acc_len` / `acc_rate` | Spec-decode acceptance length and rate (`None` when speculative decoding is off). |
| `recv_ts` / `commit_ts` / `finish_ts` | Absolute epoch timestamps for received / scheduled / finished. |

### KV Cache Events

KV cache events publish reusable prefix-cache mutations from the live C++
scheduler path (device/GPU and host/CPU tiers) and, when enabled, Mooncake L3
(disk) backup/clear paths. Block hash lineage is cached on prefix-cache nodes,
so publishing a stored block uses the parent node's cached hash instead of
rebuilding the full ancestor prefix.

`--kv-events-config` accepts a JSON object with these `KVEventsConfig` keys:

| Key | Default | Purpose |
| --- | --- | --- |
| `enable_kv_cache_events` | `false` | Publish scheduler KV cache mutation events. |
| `publisher` | `zmq` when enabled, else `null` | Event publisher (`zmq` or `null`). |
| `endpoint` | `tcp://*:5557` | ZMQ PUB bind address. |
| `replay_endpoint` | unset | Optional ZMQ endpoint for event replay. |
| `buffer_steps` | `10000` | Steps retained for the replay endpoint. |
| `hwm` | `100000` | ZMQ high water mark (events drop if consumers lag). |
| `max_queue_size` | `100000` | Max queued events waiting to publish. |
| `topic` | `""` | ZMQ topic subscribers filter on. |
| `wire_format` | `legacy` | `legacy` (Dynamo-compat) or `rfc1527` envelopes. |
| `backend_id` | env / hostname | Worker id shared across all tier publishes. |
| `tenant_id` | `default` | Multi-tenant indexer isolation. |
| `model_name` | unset | Model name in RFC #1527 envelopes when set. |
| `publish_medium` | `true` | Include `medium` in RFC #1527 envelopes. |
| `publish_tiers` | `["gpu"]` | Tiers to publish: `gpu`, `cpu`, `disk`. |
| `hash_mode` | `fnv` | Block hash mode (`fnv` or `xxh3`). |

Minimal example:

```bash
--kv-events-config '{"enable_kv_cache_events":true,"publisher":"zmq","endpoint":"tcp://*:5557","topic":"kv-events"}'
```

Recommended Mooncake multi-tier deployment:

```bash
--kv-events-config '{
  "enable_kv_cache_events": true,
  "publisher": "zmq",
  "endpoint": "tcp://*:5557",
  "replay_endpoint": "tcp://*:5558",
  "buffer_steps": 10000,
  "hwm": 100000,
  "max_queue_size": 100000,
  "topic": "kv-events",
  "wire_format": "rfc1527",
  "backend_id": "ts-worker-0",
  "tenant_id": "default",
  "model_name": "Qwen3-8B",
  "publish_medium": true,
  "publish_tiers": ["gpu", "cpu", "disk"],
  "hash_mode": "xxh3"
}'
```

The ZMQ publisher sends three frames: topic bytes, an 8-byte big-endian sequence
number, and a msgpack payload. The payload is an array-like `KVEventBatch`
(`[ts, events, attn_dp_rank]`). Individual events are tagged **maps** (not
positional arrays), which Dynamo's ZMQ relay accepts:

```python
[timestamp, [{"type": "BlockStored", "block_hashes": [...], "parent_block_hash": ..., "token_ids": [...], "block_size": N}], attn_dp_rank]
[timestamp, [{"type": "BlockRemoved", "block_hashes": [...]}], attn_dp_rank]
```

Unset optional fields are omitted via msgspec `omit_defaults`. With
`wire_format=rfc1527`, RFC #1527 envelope fields (`backend_id`, `medium`,
`dp_rank`, `model_name`, `tenant_id`, `event_id`) may also appear on events.
`medium` is derived from the scheduler event `tier` when present
(`device`/`0` → `gpu`, `host`/`1` → `cpu`); older events without `tier` keep
the publisher fallback (`gpu` for device-path publishes). Disk-path publishes
set `medium="disk"` explicitly. `event_id` is assigned monotonically per stream
keyed by `(model_name, block_size, backend_id, medium, dp_rank)`, starting at
`0`. The default `wire_format=legacy` leaves those fields unset so payloads
stay Dynamo-compatible.

#### Multi-tier deduplication (gpu / cpu / disk)

Each tier publish is **independent**: when the same logical block is present on
GPU, host, and Mooncake at once, TokenSpeed emits a separate event per tier
(distinct `medium`). Indexers that follow [Dynamo PR
#8912](https://github.com/ai-dynamo/dynamo/pull/8912) **aggregate those events
cumulatively** as per-instance tier counts — they must not open duplicate
radix entries for the same worker. That contract requires a single shared
`backend_id` across all tier publish paths for one worker (set once on
`KVEventsConfig` / `TOKENSPEED_KV_EVENTS_BACKEND_ID` and applied by
`apply_envelope` for every medium). Do not configure different `backend_id`
values for GPU, host, and disk on the same worker.

`publish_tiers` (default `["gpu"]`) selects which storage tiers to publish.
Include `"disk"` (e.g. `["gpu","cpu","disk"]`) to emit `medium="disk"`
`BlockStored` events when Mooncake L3 backup succeeds, and
`AllBlocksCleared` when the Mooncake store is cleared (`remove_all`).
GPU and host events for the same token page share scheduler `block_hashes`.
Disk hashes are an interim mapping from Mooncake SHA256 hex storage keys via
XXH3-64 seed 1337 (empty `token_ids`) until `BackUpOp` carries token page
spans, so disk hashes may intentionally differ from GPU/CPU hashes until that
plumbing lands; identity for multi-tier aggregation still relies on the shared
`backend_id`.

#### Dynamo standalone indexer (end-to-end)

To validate multi-tier KV events against NVIDIA Dynamo's standalone indexer
(`python -m dynamo.indexer`), run TokenSpeed with RFC #1527 envelopes and all
three publish tiers, then point the indexer at the worker's ZMQ PUB endpoint.
Dynamo is an external dependency and is **not** required in TokenSpeed CI —
see `test/integration/test_kv_events_dynamo_indexer.py` for the manual
checklist.

```bash
# TokenSpeed worker
--kv-events-config '{
  "enable_kv_cache_events":true,
  "publisher":"zmq",
  "endpoint":"tcp://*:5557",
  "replay_endpoint":"tcp://*:5558",
  "topic":"kv-events",
  "wire_format":"rfc1527",
  "backend_id":"ts-worker-0",
  "tenant_id":"default",
  "model_name":"YOUR_MODEL",
  "publish_medium":true,
  "publish_tiers":["gpu","cpu","disk"],
  "hash_mode":"xxh3"
}'
--kvstore-storage-backend mooncake
--kvstore-storage-backend-extra-config '{"master_server_address":"HOST:PORT"}'

# Dynamo indexer (separate process)
python -m dynamo.indexer --zmq-endpoint tcp://ts-worker-0:5557
```

Register the worker (or pass `--workers` / `--zmq-endpoint` at indexer startup
per your Dynamo version), drive traffic that stores prefixes on GPU, host, and
Mooncake, then query overlap:

```bash
curl -X POST http://localhost:8090/query \
  -H 'Content-Type: application/json' \
  -d '{"token_ids": [/* prompt tokens */], "model_name": "YOUR_MODEL"}'
```

##### Expected `/query` per-instance tier breakdown

[Dynamo PR #8912](https://github.com/ai-dynamo/dynamo/pull/8912) aligns the
standalone indexer's HTTP API with [Mooncake RFC
#1403](https://github.com/kvcache-ai/Mooncake/issues/1403). Successful
multi-tier publish yields an `instances` map keyed by instance id. Example
shape (counts are **matched tokens** = block overlap × block size):

```json
{
  "scores": {"1": {"0": 32}},
  "frequencies": [1],
  "instances": {
    "1": {
      "longest_matched": 48,
      "gpu": 32,
      "dp": {"0": 32},
      "cpu": 48,
      "disk": 48
    }
  }
}
```

| Field | Meaning |
| --- | --- |
| `gpu` | Tokens matched on the device tier (longest device-tier prefix across DP ranks). |
| `dp` | Per-`dp_rank` device-tier match counts. |
| `cpu` | Cumulative through device → host-pinned (includes everything in `gpu` plus host extension). |
| `disk` | Cumulative through device → host → disk/external. |
| `longest_matched` | `max(gpu, cpu, disk)` — best prefix length for gateway ranking. |

Under a natural offload pipeline (device → host → disk), tier counts are
**cumulative**: `gpu ≤ cpu ≤ disk` for every instance. Legacy `scores` remain
equal to each instance's per-`dp_rank` `gpu` count. Because TokenSpeed emits
one event per tier with a shared `backend_id`, the indexer aggregates those
events into a single instance entry rather than duplicate radix workers.

##### Follow-up: `/v1/tokenize` with chat `messages`

TokenSpeed does **not** yet expose `/v1/tokenize` that accepts ChatCompletion-style
`messages` (HTTP `/v1/*` is proxied to the SMG gateway today). Routers that need
template-identical token IDs for indexer `/query` should track adding that API
following the [SGLang PR #23981](https://github.com/sgl-project/sglang/pull/23981)
pattern; it requires SMG/gateway changes outside the KV-events Mooncake Store
scope. Until then, pass pre-tokenized `input_ids` / `token_ids`, or apply the
engine's chat template offline before querying the indexer.

#### Mooncake master publisher relay (scaffold)

For DaemonSet / decoupled cache topologies, Mooncake master can publish RFC
#1527 KV events (PR
[#2214](https://github.com/kvcache-ai/Mooncake/pull/2214)). TokenSpeed accepts
an optional nested `kv_events` object inside
`--kvstore-storage-backend-extra-config`:

```json
{
  "master_server_address": "mooncake-master:50051",
  "kv_events": {
    "source": "engine|master|both",
    "master_subscribe_endpoint": "tcp://mooncake-master:6000",
    "backend_id": "node-3-cache-daemon"
  }
}
```

- `source=engine` (default): engine publishes L3/disk events (current
  behavior). Master subscribe is not attempted.
- `source=master`: skip engine L3 disk publish; attempt master SUB relay.
- `source=both`: keep engine L3 publish and also attempt master SUB.

Master relay is a **safe scaffold** pending Mooncake master publisher
availability in the installed SDK: if the subscribe endpoint is missing or the
event API is unavailable, TokenSpeed logs a warning and stays idle
(fail-open) without crashing the engine. A production ZMQ poll loop will land
once the master publisher is ready.

With attention data parallelism, each attention DP rank publishes on an offset
port from the configured endpoint.

## TokenSpeed-Specific Runtime Knobs

These parameters are TokenSpeed-specific. They expose runtime
features directly:

- `--max-total-tokens`
- `--max-prefill-tokens`
- `--chunked-prefill-size`
- `--attn-tp-size`
- `--dense-tp-size`
- `--moe-tp-size`
- `--kvstore-*`
- `--enable-mla-l1-5-cache`
- `--kv-events-config`
- `--mla-chunk-multiplier`
- `--disaggregation-*`
- `--comm-fusion-max-num-tokens`
- `--enable-allreduce-fusion`
