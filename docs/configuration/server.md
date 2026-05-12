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
| `--chat-template` | Built-in chat template name or template file path. |
| `--completion-template` | Completion template for code-completion style serving. |
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
| `--attention-backend` | Attention kernel backend. Common values include `trtllm_mla`, `tokenspeed_mla`, `fa3`, and `mha`. |
| `--drafter-attention-backend` | Attention backend for speculative decoding drafter model. |
| `--moe-backend` | MoE backend. |
| `--draft-moe-backend` | MoE backend for the speculative decoding draft model. |
| `--all2all-backend` | MoE all-to-all backend. |
| `--deepep-mode` | DeepEP mode: `auto`, `normal`, or `low_latency`. |
| `--sampling-backend` | Sampling backend: `greedy`, `flashinfer`, or `flashinfer_full`. |

Set backend choices explicitly in production. `auto` is useful for bring-up, but
explicit values make benchmark comparisons and regressions easier to reason
about.

## Reasoning And Tool Calling

| Parameter | Purpose |
| --- | --- |
| `--reasoning-parser` | Parser for extracting reasoning content from model outputs. |
| `--tool-call-parser` | Parser for OpenAI-compatible tool-call payloads. |
| `--tool-server` | Built-in demo tool server. |
| `--enable-custom-logit-processor` | Allow custom logit processors. Keep disabled unless the deployment needs it. |
| `--think-end-token` | End marker for thinking models. |

Common parser values include `kimi_k2` and `gpt-oss`.

## Speculative Decoding

| Parameter | Purpose |
| --- | --- |
| `--speculative-config` | JSON speculative decoding configuration. |
| `--speculative-algorithm` | Speculative algorithm, such as `EAGLE3` or `MTP`. |
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
| `--enable-metrics` | Enable metrics reporting. |
| `--metrics-reporters` | Metrics reporter, such as `prometheus`. |
| `--decode-log-interval` | Decode batch log interval. |
| `--enable-cache-report` | Include cached-token counts in OpenAI-compatible usage details. |
| `--kv-events-config` | JSON config for KV cache mutation events. Set `enable_kv_cache_events` and a publisher such as `zmq` to publish device prefix-cache stores and removals. |

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
- `--kv-events-config`
- `--mla-chunk-multiplier`
- `--disaggregation-*`
- `--comm-fusion-max-num-tokens`
- `--enable-allreduce-fusion`
