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
| `--use-modelscope` / `--no-use-modelscope` | Select ModelScope instead of Hugging Face for remote model and tokenizer downloads. Disabled by default. |
| `--model-redirect-path` | Explicit JSON or whitespace-delimited mapping from model IDs to local paths. |

Hub selection and model redirection are request-independent server arguments;
TokenSpeed does not infer either setting from the process environment.

## Precision And Quantization

| Parameter | Purpose |
| --- | --- |
| `--dtype` | Model weight and activation dtype. `auto` follows model metadata. |
| `--kv-cache-dtype` | KV cache dtype. Lower precision reduces KV memory and may require scaling factors. |
| `--kv-cache-quant-method` | KV cache quantization method. |
| `--quantization` | Weight quantization mode such as `fp8`, `nvfp4`, `w8a8_fp8`, or `compressed-tensors`. |
| `--quantization-param-path` | JSON file for KV cache scaling factors, commonly needed with FP8 KV cache. |
| `--mamba-ssm-dtype` | Recurrent Mamba SSM-state dtype: `float32` (default) or `bfloat16`. |

Mamba state precision is an explicit CLI setting and has no environment-variable
alias.

For MiniMax-M3 release runs, cache dtype, quantization, GPU placement, and
workspace behavior are CLI/code configuration. Do not use
`FLASHINFER_WORKSPACE_SIZE`, visible-device masks, TF32 override variables, or
TokenSpeed feature environment variables. The mixed BF16-Q/E4M3-KV path owns a
stable 512 MiB planning workspace internally. CI preflight fails when those
inherited configuration channels or a persistent TokenSpeed-kernel override
file are present.

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
| `--disable-logo` | Suppress the `ts serve` startup banner in non-interactive launchers. |

## Scheduler And Memory

| Parameter | Purpose |
| --- | --- |
| `--max-model-len` | Maximum sequence length. If omitted, TokenSpeed uses the model config. |
| `--allow-overwrite-longer-context-len` | Allow `--max-model-len` to exceed the model-derived context length, with a warning. Disabled by default. |
| `--gpu-memory-utilization` | Fraction of GPU memory used for model weights and KV cache. Lower it to leave headroom. |
| `--max-num-seqs` | Maximum number of active sequences the scheduler may process concurrently. |
| `--chunked-prefill-size` | Token budget the scheduler may issue in one iteration. Defaults to `8192`. Set `-1` to disable chunked prefill. |
| `--max-prefill-tokens` | Prefill token budget used when chunked prefill is disabled. Defaults to `8192`. |
| `--max-total-tokens` | Override the automatically calculated token pool size. |
| `--block-size` | KV cache block size. |
| `--enable-prefix-caching` / `--no-enable-prefix-caching` | Enable or disable prefix cache reuse. |
| `--enforce-eager` | Disable language-model CUDA Graph execution. An explicitly enabled multimodal encoder graph remains independent. |
| `--max-cudagraph-capture-size` | Largest batch size to capture with CUDA graphs. |
| `--cudagraph-capture-sizes` | Explicit CUDA graph capture sizes. |

`--chunked-prefill-size` is intentionally separate from
`--max-num-batched-tokens`: in TokenSpeed it is the scheduler's per-iteration
issue budget, while `--max-total-tokens` controls the global token pool.
Use `--max-total-tokens` when a small deterministic token pool is needed for a
test; there is no separate CI-only KV-size environment override.

Longer-context override is an explicit CLI setting with no environment-variable
alias. Enable it only when the checkpoint is known to support the requested
length; an invalid override can produce incorrect output or CUDA errors.

## Parallelism

| Parameter | Purpose |
| --- | --- |
| `--tensor-parallel-size`, `--tp` | Familiar alias for setting attention tensor parallel size. |
| `--attn-tp-size` | Tensor parallel size for attention. |
| `--attn-cp-size` | Context parallel size for attention. Defaults to 1 after parallelism resolution. |
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
| `--base-gpu-id` | First local GPU index assigned to the server. |
| `--gpu-id-step` | Distance between assigned local GPU indices. For example, base `0` and step `2` selects `0,2,4,...`. |
| `--enable-numa-aware-worker-affinity` / `--no-enable-numa-aware-worker-affinity` | Enable or disable NVIDIA worker pinning to the GPU-local CPU set. Enabled by default. |

Use `--tensor-parallel-size` for simple launches. Use the
TokenSpeed-specific split knobs when attention, dense, and MoE layers need
different process groups.

Parallel topology is configured through CLI arguments; TokenSpeed does not
read `ENABLE_CP` from the process environment.

## Backend Selection

| Parameter | Purpose |
| --- | --- |
| `--attention-backend` | Attention kernel backend. Common values include `mha`, `fa3`, `fa4`, `triton`, `flashinfer`, `trtllm_mla`, and `tokenspeed_mla`. |
| `--drafter-attention-backend` | Attention backend for speculative decoding drafter model. |
| `--moe-backend` | MoE backend. |
| `--draft-moe-backend` | MoE backend for the speculative decoding draft model. |
| `--all2all-backend` | MoE all-to-all backend. |
| `--deepep-mode` | DeepEP mode: `auto`, `normal`, or `low_latency`. |
| `--sampling-backend` | Sampling backend: `greedy`, `flashinfer`, `flashinfer_full`, `triton`, or `triton_full`. |

Set backend choices explicitly in production. `auto` is useful for bring-up, but
explicit values make benchmark comparisons and regressions easier to reason
about.

FlashInfer's generic workspace reservation is a stable runtime constant.
Kernel-selection experiments use the explicit `override=` argument,
`kernel_override(...)` context manager, or
`load_config_overrides("/path/to/overrides.yaml")`. TokenSpeed-kernel does not
read per-op override environment variables or automatically load a per-user
override file.

## Multimodal Execution

| Parameter | Purpose |
| --- | --- |
| `--mm-attention-backend` | Attention backend used inside supported multimodal encoders, such as `fa3`, `fa4`, or `triton_attn`. |
| `--mm-skip-compute-hash` | Replace multimodal content hashes with random per-item IDs, disabling content deduplication and content-aware prefix reuse. Disabled by default. |
| `--enable-mm-encoder-cuda-graph` / `--no-enable-mm-encoder-cuda-graph` | Capture supported multimodal encoders during startup and replay matching input budgets. Disabled by default. |
| `--mm-encoder-cudagraph-max-metadata-sequences-per-batch` | Explicit maximum number of attention-metadata sequences captured per encoder batch. Supported models derive it from the token budget when omitted. |
| `--language-model-only` | Disable the active multimodal model path and reject multimodal requests. |

Encoder CUDA Graph enablement is fail-closed: startup fails if the active model
does not provide an encoder graph adapter or the selected multimodal attention
backend cannot be captured. A supported model may still execute inputs outside
its captured budget range eagerly. This graph is separate from the
language-model graph controlled by `--enforce-eager` and the language-model
capture-size options.

Encoder-graph enablement and metadata sizing are CLI settings. They do not
have environment-variable aliases.

MiniMax-M3 captures image budgets `[16, 32, 64, 128, 256, 512, 1024, 2048,
2304]` at startup. A fixed-candidate TP4 validation captured all nine budgets
on every rank and replayed differently sized single- and multi-image requests
without request-time recapture. This is real encoder execution capture; it is
not merely a graph wrapper around an eager tower.

Multimodal hash policy is also an explicit CLI setting with no
environment-variable alias. Leave content hashing enabled unless intentionally
diagnosing deduplication or prefix-reuse behavior.

Multimodal modalities remain model-specific. MiniMax-M3 currently supports
image items only; video items are rejected.

When `--dp-sampling` is enabled, the logits processor owns the per-forward
logits layout decision and carries the resulting plan to the sampling backend
with the logits output.

## PD and EPD Transport

PD/EPD transport tuning is part of the server argument snapshot. None of these
settings has an environment-variable alias.

| Parameter | Purpose |
| --- | --- |
| `--disaggregation-queue-size` | Number of room-affine transfer queues. Defaults to `4`. |
| `--disaggregation-thread-pool-size` | Total transfer worker threads. When omitted, TokenSpeed derives a bounded value from the available CPUs. |
| `--disaggregation-bootstrap-timeout` | Bootstrap registration timeout in seconds. Defaults to `120`. |
| `--disaggregation-waiting-timeout` | Completed-transfer wait timeout in seconds. Defaults to `300`. |
| `--disaggregation-failed-session-ttl` | Failed-session quarantine in seconds. Defaults to `30`; `0` disables quarantine. |
| `--disaggregation-heartbeat-interval` | Decode-side prefill heartbeat interval in seconds. Defaults to `5`. |
| `--disaggregation-heartbeat-max-failures` | Consecutive heartbeat failures before affected requests fail. Defaults to `2`. |
| `--pd-layerwise-debug` / `--no-pd-layerwise-debug` | Enable additional layerwise transfer consistency checks. Disabled by default. |
| `--pd-prefill-metadata-wait-log-interval` | Debug-log interval while waiting for prefill metadata. Defaults to `5` seconds. |
| `--epd-encode-ring-slots` | Pre-registered encode bounce-buffer slots. Defaults to `64`. |
| `--epd-encode-ring-slot-mb` | Capacity of each encode bounce-buffer slot in MiB. Defaults to `256`. |
| `--epd-encode-embedding-cache-mb` | Per-encode-process VRAM embedding-cache budget in MiB. Defaults to `4096`. |
| `--epd-encode-embedding-cache-dram-mb` | Per-encode-process host embedding-cache budget in MiB. Defaults to `0` (disabled). |
| `--epd-recv-pool-slots` | Lifetime-registered prefill receive slots. Defaults to `16`; `0` selects per-request buffers. |
| `--epd-recv-pool-slot-mb` | Capacity of each prefill receive slot in MiB. Defaults to `256`; `0` also disables the pool. |
| `--epd-embedding-shard` / `--no-epd-embedding-shard` | Shard image-embedding rows across prefill attention-TP ranks. Enabled by default. |

The encode ring and receive pool reserve their configured capacities per
process. Size them for the largest post-merge image embedding and multiply the
memory budget by the number of co-located TP ranks. At the defaults, each
encode rank reserves a 16 GiB main ring and a 4 GiB L1 embedding cache; each
prefill rank reserves a 4 GiB receive pool. A model with a deepstack embedding
path may allocate a second encode ring.

## SMG Process Integration

SMG launch behavior is also explicit server configuration: use
`--grpc-max-message-bytes`, `--skip-grpc-warmup`,
`--health-check-timeout`, `--log-mm-tensor-data`, `--enable-log-mm-timing`,
`--unlink-mm-shm-after-read`, `--epd-pixel-shm`, and
`--epd-ingest-offloop` (including each Boolean option's `--no-...` form).
These settings are propagated to the SMG TokenSpeed gRPC adapter without
feature environment variables.

Shared multimodal RDMA is configured with `--mm-pixel-rdma`,
`--mm-rdma-slot-bytes`, `--mm-rdma-landing-slots`,
`--mm-rdma-landing-wait-seconds`, and
`--mm-rdma-read-timeout-seconds`. Metadata sending is tri-state:
omit `--mm-rdma-send-metadata` for automatic behavior, or use its positive or
`--no-mm-rdma-send-metadata` form to force the choice.

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
| `--logging-config-path` | Explicit path to a JSON `logging.dictConfig` document. |
| `--enable-log-requests` | Log request metadata and optionally payloads. |
| `--log-requests-level` | Request logging verbosity. |
| `--enable-log-request-stats` | Log a one-line per-request performance summary on finish/abort (see below). |
| `--enable-log-mm-timing` | Log detailed multimodal SHM, encoder, embedding, and forward timings. |
| `--enable-metrics` | Enable metrics reporting. |
| `--metrics-reporters` | Metrics reporter, such as `prometheus`. |
| `--decode-log-interval` | Decode batch log interval. |
| `--enable-cache-report` | Include cached-token counts in OpenAI-compatible usage details. |
| `--kv-events-config` | JSON config for KV cache mutation events. Set `enable_kv_cache_events` and a publisher such as `zmq` to publish device prefix-cache stores and removals. |

`--enable-log-mm-timing` is disabled by default and is intended for diagnostics.
It is an explicit per-server setting; environment variables do not enable it.
The encoder timing path may synchronize CUDA, so leave it disabled for normal
throughput measurements.

Model execution always enters `torch.inference_mode`; this is a stable runtime
policy rather than an operator-controlled environment switch. The detokenizer
also uses a stable 65,536-request state capacity.

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

KV cache events publish reusable device prefix-cache mutations from the live
C++ scheduler path. Host/L2 loadback events are not published by this initial
stream. Block hash lineage is cached on prefix-cache nodes, so publishing a
stored block uses the parent node's cached hash instead of rebuilding the full
ancestor prefix.

Example:

```bash
--kv-events-config '{"enable_kv_cache_events":true,"publisher":"zmq","endpoint":"tcp://*:5557","topic":"kv-events"}'
```

The ZMQ publisher sends three frames: topic bytes, an 8-byte big-endian sequence
number, and a msgpack payload. The payload is an array-like `KVEventBatch`:

```python
[timestamp, [["BlockStored", [block_hash], parent_hash, token_ids, block_size]], attn_dp_rank]
[timestamp, [["BlockRemoved", [block_hash]]], attn_dp_rank]
```

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
- `--enable-mm-encoder-cuda-graph`
- `--kv-events-config`
- `--mla-chunk-multiplier`
- `--disaggregation-*`
- `--epd-*`
- `--comm-fusion-max-num-tokens`
- `--enable-allreduce-fusion`
