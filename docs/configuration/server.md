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
| `--load-format` | Weight loading format: `auto`, `pt`, `safetensors`, `instanttensor`, `npcache`, `dummy`, or `extensible`. See [InstantTensor](/guides/instanttensor) for the accelerated NVIDIA loader. |
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
| `--api-key` | SMG gateway API key for authorization with upstream workers. |
| `--chat-template` | Built-in chat template name or template file path (handled by the smg gateway). |
| `--stream-interval` | Streaming buffer interval in generated tokens. Smaller values stream more frequently. |
| `--stream-output` | Return generated text as disjoint streaming segments. |
| `--weight-version` | Initial model-weight version stamped into generation metadata. Defaults to `default`. |

### Weight Version Metadata

Every generation response includes the current version in
`meta_info["weight_version"]`. RL trainers can use this value to identify the
policy version that produced a sample.

The SGLang-compatible `update_weights_from_distributed`,
`update_weights_from_tensor`, and `update_weights_from_disk` requests accept an
optional `weight_version`. The version changes only after the update succeeds.
`Engine.update_weights_from_distributed` requires `weight_version`; pass
`None` to keep the current value on an intermediate update. Flushed L3
updates must pass a caller-supplied identity so independent checkpoints
cannot share a minted successor. When L3 is on, a
new `weight_version` requires `flush_cache=True`; intermediate updates may
pass `None` until the last call flushes.

Use `GET /get_weight_version` to read the current value,
and `GET /model_info` to read the model path and version together.
When L3 storage is disabled, `POST /update_weight_version` with
`{"new_version": "..."}` sets the value directly. With L3 enabled, this endpoint
returns HTTP 400 without changing the version: changing only frontend metadata
would leave the cache namespace on the old checkpoint. Use
`POST /update_weights_from_distributed` with an explicit `weight_version` and
`flush_cache=True` to coordinate the weight load and cache namespace change.

### Slime RL Compatibility

TokenSpeed exposes the SGLang HTTP surface used by slime. The supported path is
an externally launched TokenSpeed rollout engine on separate GPUs, using full
NCCL weight updates and TokenSpeed data-parallel size 1:

- rollout: `POST /generate`, `POST /abort_request`, `GET /v1/loads`, and
  `GET /health_generate`;
- update coordination: `POST /pause_generation`,
  `POST /continue_generation`, and `GET /flush_cache`;
- NCCL weight sync: `POST /init_weights_update_group`,
  `POST /update_weights_from_distributed`, and
  `POST /destroy_weights_update_group`;
- memory control: `POST /release_memory_occupation` and
  `POST /resume_memory_occupation`.

Use TokenSpeed's control-server address, not its OpenAI gateway address, as the
external rollout-engine address. Real rollout log probabilities require
`--enable-output-logprobs`.

The following slime paths are not yet supported end to end:

- colocated CUDA-IPC updates through `update_weights_from_tensor`;
- quantized-update hooks `post_process_weights` and `weights_checker`;
- disk-delta `pull_weights`;
- slime's retained top-p token set (`rollout_top_p != 1.0`). Use
  `--rollout-top-p 1.0` until TokenSpeed returns that metadata;
- rollout routing replay (`--use-rollout-routing-replay`).

The HTTP route for `update_weights_from_tensor` remains for SGLang clients, but
TokenSpeed's scheduler does not yet implement its CUDA-IPC receive path. Use the
distributed update mode until that implementation is added.

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
| `--enable-prefix-caching` / `--disable-prefix-caching` | Enable or disable prefix cache reuse. |
| `--enforce-eager` | Disable device-graph execution (CUDA Graph on CUDA, ACL Graph on NPU). |
| `--disable-prefill-graph` | Keep prefill eager while leaving decode device graphs enabled. |
| `--disable-kda-prefill-graph` | Disable KDA prefill CUDA graphs while retaining ordinary prefill and decode graph settings. Enabled by default for supported `cutedsl_kda` prefill attention when prefill graphs are enabled. |
| `--disable-cudagraph-memory-reserve` | Size the KV cache from free memory instead of reserving what the device graphs will cost. |
| `--max-cudagraph-capture-size` | Largest decode batch size to capture as a device graph. |
| `--cudagraph-capture-sizes` | Explicit decode batch sizes to capture as device graphs. |
| `--prefill-graph-capture-token-sizes` | Total input-token capacities per forward, summed across the batch. Shorter inputs are padded. |
| `--prefill-graph-capture-batch-sizes` | Request capacities for inline KDA prefill capture. Replay selects the smallest compatible capacity that fits the batch. |

For pure prefill, token capacities count newly computed tokens, not cached
prefixes or each request's full sequence length. Two requests extending by
868 and 869 tokens use the 2048-token bucket and request capacity 2 when
configured. A smaller batch can reuse that capture if there is room for its
dummy request slots. These settings do not replace the scheduler's
`--max-num-seqs` limit.

`--prefill-graph-capture-sizes` remains a compatibility alias for
`--prefill-graph-capture-token-sizes`; specify only one spelling per command.
Both populate the existing `prefill_graph_capture_sizes` Python field.
Unset token sizes use the existing default ladder; unset batch sizes use the
minimum request count that fits each token bucket within the model context.

`--chunked-prefill-size` is intentionally separate from
`--max-num-batched-tokens`: in TokenSpeed it is the scheduler's per-iteration
issue budget, while `--max-total-tokens` controls the global token pool.

## Parallelism

| Parameter | Purpose |
| --- | --- |
| `--tensor-parallel-size`, `--tp` | Familiar alias for setting attention tensor parallel size. |
| `--attn-tp-size` | Tensor parallel size for attention. |
| `--dense-tp-size` | Tensor parallel size for dense layers. Defaults to the attention replica width (attn TP x CP): the full world without DP attention, one replica with it. |
| `--moe-tp-size` | Tensor parallel size for MoE layers. |
| `--data-parallel-size` | Number of data-parallel replicas. |
| `--mm-encoder-tp-mode` | Multimodal encoder parallelism: `weights` shards encoder weights with attention TP; `data` uses TP1 whole-item DP and currently requires aggregate serving and no attention context parallelism. |
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
| `--attention-backend` | Attention kernel backend. Common values include `mha`, `fa3`, `fa4`, `triton`, `flashinfer`, `trtllm_mla`, and `tokenspeed_mla`. Names are checked against the backend registry at startup, after plugins load, so an installed plugin's backends are accepted too. |
| `--drafter-attention-backend` | Attention backend for speculative decoding drafter model; accepts the same names as `--attention-backend`. |
| `--moe-backend` | MoE backend. |
| `--moe-mxfp4-fp8-activation` | Opt-in: run MXFP4 routed experts with FP8 activations. On Hopper this is the FlashInfer cutlass W4A8 MoE (faster than the default W4A16 kernel, a few percent of extra error on expert outputs). Applies to every MXFP4 expert layer, target and draft; startup fails where the selected MoE backend has no FP8-activation kernel for a layer, when a model's routed experts are not MXFP4, when the model pins another activation precision (Kimi-K3 on Hopper Marlin), or when the layer's SwiGLU is unclamped (the W4A8 FC2 scale relies on the clamp). |
| `--draft-moe-backend` | MoE backend for the speculative decoding draft model. |
| `--all2all-backend` | MoE all-to-all backend. |
| `--deepep-mode` | DeepEP mode: `auto`, `normal`, or `low_latency`. |
| `--sampling-backend` | Sampling backend: `greedy`, `flashinfer`, `flashinfer_full`, `triton`, or `triton_full`. |

Set backend choices explicitly in production. `auto` is useful for bring-up, but
explicit values make benchmark comparisons and regressions easier to reason
about.

LongCat-Flash computes top-k routing in the model to handle its zero experts.
Its MoE layers require a backend that accepts precomputed expert IDs and weights,
and request `swiglu` for their gated SiLU activation. These requirements apply to
both unquantized and block-FP8 expert layers, including when selecting
`--moe-backend flashinfer_trtllm` on Blackwell.

When `--dp-sampling` is enabled, the logits processor owns the per-forward
logits layout decision and carries the resulting plan to the sampling backend
with the logits output.

## Reasoning And Tool Calling

| Parameter | Purpose |
| --- | --- |
| `--reasoning-parser` | Parser for extracting reasoning content from model outputs (handled by the smg gateway). |
| `--tool-call-parser` | Parser for OpenAI-compatible tool-call payloads (handled by the smg gateway). |

Common reasoning parser values include `kimi_k25`, `base`, `qwen3`,
`deepseek_r1`, and `deepseek_v31`. Common tool-call parser values include
`kimik2`, `qwen`, `deepseek_v4`, `json`, and `passthrough`. The parser names
are validated by the SMG gateway, so use
the values accepted by the bundled `tokenspeed-smg` package.

## Speculative Decoding

| Parameter | Purpose |
| --- | --- |
| `--speculative-config` | JSON speculative decoding configuration. |
| `--speculative-algorithm` | Speculative algorithm, such as `EAGLE3`, `MTP`, `DFLASH`, or `DSPARK`. |
| `--speculative-draft-model-path` | Draft model path or repo ID. |
| `--speculative-draft-model-quantization` | Draft model quantization. Defaults to `unquant`. |
| `--speculative-num-steps` | Number of draft model steps. Defaults to `3`. |
| `--speculative-num-draft-tokens` | Number of draft tokens. Defaults to `--speculative-num-steps + 1`. |
| `--speculative-eagle-topk` | EAGLE top-k. Defaults to `1`. |
| `--eagle3-layers-to-capture` | EAGLE3 layers to capture. |

Prefer `--speculative-config` for recipe-style launches because it keeps method,
draft model, and token count together.

`DFLASH` and `DSPARK` are block drafters: one draft forward proposes a whole
block instead of one token per step, so their two token counts are coupled.
`--speculative-num-draft-tokens` is the verify width -- one anchor row plus one
row per drafted token -- and `--speculative-num-steps` must be one less. The
draft checkpoint's `block_size` fixes both, and a mismatch is rejected at
startup rather than silently drafting a wrong-width block. The two families
spell that `block_size` differently:

- DSpark checkpoints store the drafted token count, so `block_size`
  (`dspark_block_size` on same-checkpoint DSpark) equals
  `--speculative-num-steps`. `block_size: 8` wants `--speculative-num-steps 8
  --speculative-num-draft-tokens 9`.
- DFlash and DFlash2 checkpoints store the verify width, so `block_size` equals
  `--speculative-num-steps + 1`. `block_size: 8` wants
  `--speculative-num-steps 7 --speculative-num-draft-tokens 8`.

A checkpoint that declares no `block_size` leaves both flags as given.

A checkpoint whose architecture is `DFlash2DraftModel` uses the same `DFLASH`
launch method. TokenSpeed selects its grouped-convolution and candidate-selector
runtime from the checkpoint architecture; no separate algorithm flag is needed.
Draft proposals greedily follow the selector's transition-conditioned path,
walked by one Triton kernel per verify step. A request's `temperature`,
`top_k` and `top_p` are applied by the target's verification step, never by
the proposal, so the served distribution is the target's whatever the drafter
proposed.

A block drafter writes its KV at the target's cache locations, so it shares the
target's page table: `--block-size` is a target-side choice and the draft
follows it. Any sliding window the draft checkpoint declares is an attention
mask applied by the draft's own layers, never a cache-retention policy of its
own. Only the backends that forward that mask to their kernels can serve such a
draft: `mla` and `tokenspeed_mla` (`gluon` on AMD) for MLA drafts, and
`mha`/`fa3`/`fa4`/`triton`/`flashinfer`/`trtllm_mha` for GQA drafts. Any other
`--drafter-attention-backend` is rejected at startup rather than quietly
widening the draft's attention to the full history.

## Observability

| Parameter | Purpose |
| --- | --- |
| `--log-level` | Runtime log level. |
| `--enable-log-requests` | Log request metadata and optionally payloads. |
| `--log-requests-level` | Request logging verbosity. |
| `--enable-log-request-stats` | Log a one-line per-request performance summary on finish/abort (see below). |
| `--enable-metrics` | Enable metrics reporting. |
| `--metrics-reporters` | Metrics reporter, such as `prometheus`. |
| `--decode-log-interval` | Decode batch log interval. |
| `--kv-events-config` | JSON config for KV cache mutation events. Set `enable_kv_cache_events` and a publisher such as `zmq` to publish device prefix-cache stores and removals. |

Every `--decode-log-interval` decode rounds the scheduler's representative rank
prints one `Decode batch.` line: `#running-req`, `avg_seq_len` (the mean of
prompt plus generated tokens over the running requests, so a step's attention
cost can be read alongside its batch size), device page usage, the generation
throughput accumulated since the previous line, `avg_accept_len` /
`accept_rate` under speculative decoding, and `#queue-req`. Every field is a
host-side scheduler counter; the line adds no GPU synchronization.

`#queue-req` counts requests admitted to the scheduler but not yet running.
On a PD engine that includes requests still bootstrapping with the peer —
on the prefill role, waiting for the decode side to allocate their KV pages —
which the `#req-state(bootstrap/prefill/remote-prefill/decode/pd-pinned)`
suffix also lists on its own. The Prometheus waiting gauge and the router
load snapshot report the scheduler's narrower waiting count, without the
bootstrapping share.

Set `TOKENSPEED_LOG_SPEC_ACCEPT_LENGTHS=1` to log each speculative verify
step's committed widths and accepted draft-token counts. This reads the
already-synchronized CPU result and does not add a GPU synchronization, but it
is intentionally verbose and should only be enabled while debugging. For
decode-only batches it also logs the anchor, draft candidates, target verify
tokens, and their position-wise matches.

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
- `--kv-events-config`
- `--mla-chunk-multiplier`
- `--disaggregation-*`
- `--comm-fusion-max-num-tokens`
- `--enable-allreduce-fusion`

### Host L2 and Mooncake Store L3

Host KVStore (`--kvstore-ratio` / `--kvstore-size`) is a compact pinned
buffer under GPU cache (flat KV). `--kvstore-storage-backend mooncake`
adds Mooncake Store as L3 under that buffer:

```
GPU Device KV (L1)
  ↕ D2H / H2D
Host pinned buffer (L2 / flat KV)
  ↕ batch_put_from / batch_get_into
Mooncake Store (L3)
```

Each packed Host CacheBlock is one Mooncake object, keyed as
`{tsl3v1-<sha256>}_{content_hash}|g{group}|o{page_offset}|r{tp_rank}|c{cp_rank}`.
The hashed prefix includes the loaded checkpoint (`--model`, the resolved
immutable revision or a local fingerprint of selected weights, metadata,
and local `*.py` including imported package subdirectories and
directory symlinks Python follows on import — never an inherited
config `_commit_hash` or a 40-hex folder name outside a Hugging Face hub
`(models|datasets|spaces)--*/snapshots/<commit>` cache path with a sibling
`refs` directory (a directory merely named `snapshots` is fingerprinted)
— plus `--load-format` so a directory that contains
more than one weight encoding cannot share objects across loaders
(`sharded_state` combines every rank's local files matching the
configured shard pattern, default `model-rank-*-part-*`, not only rank
0's; `npcache` fingerprints the NumPy cache when present; `extensible`
also hashes `--ext-yaml` and the `ext_def_file` `ExtensibleLM` imports
with the same cwd-relative `os.path.abspath` resolution as the loader,
plus that module's transitive local helpers, including on a Hugging Face
hub snapshot whose commit does not cover those files; the path is parsed
without PyYAML for quoted keys, spaces around `:`, and a document-level
flow mapping), and
`--weight-version`), `--hf-overrides` (the effective
HF text-config delta: `rope_theta`, `rope_scaling`, and other architecture
fields), the packed Host layout (field payloads, not GPU-capacity
device arena offsets), the
cache-quantization config (including `quantization_param_path` scale-file
bytes and `--speculative-draft-model-quantization` when a draft pool is
present), the pipeline stage, the context-parallel
width (`cp_size`), any
speculative draft checkpoint, `--skip-softmax-threshold` (nonzero
changes attention output and therefore downstream cached K/V), the
resolved EAGLE3 capture-layer list (`--eagle3-layers-to-capture` or the
draft config's `eagle_aux_hidden_state_layer_ids`; empty when EAGLE3 is
off), and
`L3_RUNTIME_COMPAT` (bumped when
built-in model code, RoPE, or a cache-producing kernel changes KV for
the same checkpoint and layout). Live weight updates flush Device/Host
before the GPU load, then rebuild that prefix. A requested `flush_cache`
must succeed first: in-flight Host writebacks cause `ClearCache` to
reject. Weight-update `flush_cache` and standalone `/flush_cache`
first MAX-reduce flush intent across attention DP so every DP worker
enters the same collectives, then MIN-reduce a non-mutating
`can_clear_cache` probe across cache-owning
ranks (attention TP, then CP, then PP) and then across attention DP
before any rank clears. Exists, prefetch, and `WriteBackDone` stay
TP/CP/PP because DP ranks hold different sequences; flush includes DP
because object keys omit DP rank. Remote L3 deletion is the next
replica-then-DP phase: it
returns success/failure instead of raising, is MIN-reduced, and only
then does `ClearCache` destroy Device/Host. A rank whose writebacks have
drained cannot rotate L3 or drop local indexes while a peer still
rejects or while Mooncake `remove_by_regex` failed on another rank. The
frontend ANDs every DP worker's `/flush_cache` reply. A
split flush would leave mirrored
schedulers with different prefix indexes. The weight-update RPC then
fails so the caller retries instead of serving new weights against the
previous checkpoint or entering NCCL weight broadcasts alone. A
`batch_exists` hit is not a lease: if
`batch_get_into` misses after Admit, the runtime unregisters the key,
skips publishing empty Host pages, and retracts the batch snapshot-less
so the next admit recomputes those tokens. A short Mooncake read (fewer
bytes than the requested page) is a miss, not a success. Failed `batch_get_into` pages
stay unread so a later `batch_exists` hit cannot re-register them and
retry the same prefetch; only replica-converged misses are blacklisted.
Replica admission MIN-reduces local readability (exists and not unread).
A later Host backup forgets an unread entry only when it created a
missing object; a create-only skip of an unreadable object keeps the
blacklist. The unread set is bounded to Host CacheBlock capacity (LCM
parents times each group's `cache_blocks_per_lcm_block`).
A backend exception or malformed result is a
local miss so every replica rank still enters the MIN-reduce. Clients
are not failed.
L2 write-back ACKs use the same replica groups: `WriteBackDone` is
emitted only after every cache-owning rank holds the completion, so an
ENABLE_CP worker cannot publish Host while a CP peer's Mooncake put is
still in flight. A truncated `batch_is_exist` reply is a failed put, not
an implicit success.
Supplying a new
`weight_version` with `flush_cache=False` is rejected when L3 is on so
stale Device/Host KV and in-flight D2H copies cannot be treated as the
new checkpoint. Flushed L3 updates require an explicit `weight_version`;
minting `{current}-uN` would let independent checkpoints collide.
A successful Engine update stamps that version into
frontend `server_args`.
Context-parallel workers (`ENABLE_CP`) share
`attn_tp_rank == 0` and are distinguished by `c{cp_rank}` plus `cp_size`
in the hashed namespace. Without PP, only `cp_rank==0` owns the request
socket and load reporting, and `recv_reqs` broadcasts across CP so exists
MIN is rank-identical. GQA with TP above the KV-head count assigns
different heads to the same `r{tp_rank}`, so `attn_tp_size` (resolved
`mapping.attn.tp_size`) is also in the namespace. Resolved target and draft
attention backends, including the full-attention sub-backend of a hybrid model,
are isolated too: different implementations can produce different downstream
KV even with identical cache layouts. This namespace extension intentionally
starts a cold L3 cache instead of reusing objects written without backend identity.
`global_segment_size` is split across
attention-TP × context-parallel × pipeline-parallel ranks so the
mounted total matches the configured size. Use the resolved mapping
(`mapping.attn.tp_size` and `mapping.attn.cp_size`), not `--attn-tp-size`
alone: `ENABLE_CP` with an omitted `--attn-tp-size` infers `cp_size = N`
and `tp_size = 1`. L3 requires Host L2 (do not pass `--disable-kvstore`).
Pass Mooncake client settings as JSON
in `--kvstore-storage-backend-extra-config`, for example:

```json
{
  "master_server_address": "10.0.0.1:50051",
  "local_hostname": "localhost",
  "metadata_server": "P2PHANDSHAKE",
  "global_segment_size": "16gb",
  "protocol": "tcp"
}
```

Constructing `MooncakeKvStore` requires `extra_config`; pass `None` to
use `MOONCAKE_MASTER` / `MOONCAKE_CLIENT` and the other env defaults.
Queued requests that can take a batch slot and Device pages this round
re-probe L3 immediately before admission so a hit that waited for capacity
cannot keep a deleted or evicted object as a Host hit. A full decode batch
or exhausted Device pool does not rehash the rest of the wait queue.
`--kvstore-storage-backend memory` is an in-process dict for tests only.
CI exercises that Mooncake-compatible contract end-to-end (scheduler
prefetch after `register_storage_keys` / Host eviction, and a CUDA
D2H → store → Host wipe → prefetch → H2D round trip). A separate
ubuntu job boots `mooncake_master` and runs
`test/test_l3_mooncake_master.py` against the real TCP client
(`P2PHANDSHAKE`). Reuse an already-running master with
`MOONCAKE_MASTER=host:port`.
Mooncake Store is the offload backend; PD KV transfer still uses the
separate Mooncake TransferEngine (`--disaggregation-transfer-backend`).
