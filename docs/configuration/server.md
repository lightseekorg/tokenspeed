# Server parameters

This page documents the parameters operators usually set directly. TokenSpeed
uses familiar serving parameter names where the semantics match and keeps
TokenSpeed-specific knobs for runtime features with different meaning.

For a compact compatibility table, see
[Compatible parameters](./compatible-parameters.md).

## Model loading

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

### Checkpoint prefetch and TP shards

Safetensors loading prefetches checkpoints into the OS page cache. Ranks on
the same node divide background reads in sorted shard order. Every node
prefetches its own copy. Each reader keeps the full consumption order within
the existing window of min(40 GiB, 25% of available host memory), including
shards assigned to peers. Reads remain asynchronous: a consumer does not
wait for another rank and can demand-page an unfinished peer shard. Models
with rank-dependent weight-name filters prefetch independently. Use
`--disable-weight-loader-prefetch-checkpoints` to disable prefetch or
`--weight-loader-prefetch-num-threads` to set reader concurrency per rank.

The programmatic `LoadConfig(load_format="sharded_state")` loader reads only
the current global rank's files,
named `model-rank-{rank}-part-{part}.safetensors` by default. These are
post-processed runtime state dictionaries, not ordinary Hugging Face shards.
Reload with the same model configuration, parallel mapping, quantization,
and runtime weight layout. The loader constructs and post-processes the
model before copying the saved state into it. Check compatibility for the
model and quantization in use. Keep model configuration/tokenizer
files with the checkpoint. Pass a custom filename pattern through
`LoadConfig.model_loader_extra_config`, for example
`{"pattern": "model-rank-{rank}-part-{part}.safetensors"}`. The serving CLI does
not expose this loader or its extra configuration.

## Precision and quantization

| Parameter | Purpose |
| --- | --- |
| `--dtype` | Model weight and activation dtype. `auto` follows model metadata. |
| `--kv-cache-dtype` | KV cache dtype. Lower precision reduces KV memory. |
| `--kv-cache-quant-method` | KV cache quantization method. |
| `--quantization` | Weight quantization mode such as `fp8`, `nvfp4`, `w8a8_fp8`, or `compressed-tensors`. |
| `--quantization-param-path` | JSON file of FP8 KV cache scaling factors, read only under an FP8 KV cache. KV caches run unscaled, so every factor must be 1.0, as must any KV-cache scale the checkpoint carries. |

## Numerics

`--numerics` names the numerical contract a deployment promises. The design
and the per-model verification gate are in `docs/design/numerics.md`. Every
switch an envelope folds is also available individually under `auto`.

| Parameter | Purpose |
| --- | --- |
| `--numerics {auto,rl-bitwise}` | `auto` keeps every performance default. `rl-bitwise` folds the determinism switches (`--batch-invariant-collectives`, `--disable-autotune`, `--disable-tf32`, `--disable-pdl`, no fused all-reduce, `--moe-backend aok`, `--sampling-stream per-request`, `--dsa-slot-order sorted`) so tokens and logprobs are bitwise identical across runs and batch compositions, and the trainer-operation-order switches below so a teacher-forced pass reproduces the RL trainer's logprobs. Needs MoE TP 1 and a vocabulary that is a multiple of 32768; a model serves it only once its profile lists it. |
| `--batch-invariant-collectives` | One association order per reduction, independent of the batch: a 2-D bf16 all-reduce on a multicast-reachable group runs as the NVLS in-switch reduction issued by one fixed rank (verified bitwise at startup), every other reduction as NCCL data movement plus a fixed-rank-order fp32 fold; gathers keep the multicast kernels. Folded in by `rl-bitwise`. |
| `--force-deterministic-rsag` | NCCL and the fold only: no symmetric-memory path (multicast gathers, in-switch all-reduce, the trtllm/Triton all-reduce tiers, distributed argmax). Not folded in by `rl-bitwise`; the escape when the startup self-check refuses the switch. |
| `--sampling-stream {batch,per-request}` | `per-request` draws every non-greedy row from a stream keyed by the request's seed and position only, so a request samples the same tokens alone and inside any batch. Folded in by `rl-bitwise`. |
| `--yarn-ramp-mask-device {cuda,cpu}` | Device that computes the `deepseek_yarn` RoPE inverse frequencies (position frequencies, both divisions and the YaRN linear ramp mask) before the table is moved to the model device once; the trainer builds it on the host. Folded to `cpu` by `rl-bitwise`. |
| `--mla-lora-scale {folded,runtime}` | Where LongCat-style MLA applies its `sqrt(hidden / lora_rank)` norm scales: folded into the norm weights at load, or multiplied at runtime after `q_b_proj` / `kv_a_layernorm` as the trainer does. Folded to `runtime` by `rl-bitwise`. |
| `--layer-boundary-norm {fused,unfused}` | `unfused` materializes `hidden + residual` in bf16 before the norm that opens each physical layer and before the final norm, instead of the fused add+norm kernel; also vetoes all-reduce+norm fusion. Folded to `unfused` by `rl-bitwise`. |
| `--router-topk {fused,torch}` | Correction-bias MoE routing: the fused CUDA kernel, or fp32 `torch.softmax` + `torch.topk(probs + bias)` with PyTorch tie order and `-1` zero-expert ids. Folded to `torch` by `rl-bitwise`. |
| `--logprob-order {torch,megatron}` | Order of the selected-token log-softmax: `torch.log_softmax`, or Megatron's vocab-parallel cross-entropy order over fixed 32768-wide vocab blocks; output and prompt (input) logprobs share it. Changes logprobs only. Folded to `megatron` by `rl-bitwise`. |
| `--dsa-slot-order {selection,sorted}` | The order a token's selected KV rows are reduced in by the sparse (DSA) attention: as the top-k leaf emitted them (`selection`), or in ascending position order (`sorted`: one reduction order per selected set, invariant across batch compositions, runs and engines -- not ascending slot order, which follows the page placement). The top-k leaf emits the order and the core keeps it, so both must declare the `slot_order` trait (the `aok` leaves); a silent kernel is refused under `sorted`. Folded to `sorted` by `rl-bitwise`. |
| `--moe-combine-order {rank,slot}` | How a token's routed-expert contributions meet across the MoE TP-EP group. `rank`: the MoE kernel returns this rank's partial and the host sums the partials, adding LongCat's identity zero-expert residual once around the reduction. `slot`: the MoE kernel folds the token's top-k slots in fp32 slot order across the EP group itself, residual included, as the trainer's grouped MLP does, and the host reduces nothing; needs MoE TP 1 and a kernel declaring `combine_order` with `slot` (the `aok` leaf), and vetoes all-reduce+norm fusion. Folded to `slot` by `rl-bitwise`. |

## API surface

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
| `--rl-control-host` | Bind host for the in-engine RL control app. Defaults to `--host`. |
| `--rl-control-api-key` | Bearer token required on every RL control route. Unset leaves the app open, which is what slime expects by default. |
| `--model-update-config` | JSON object handed to the Model Updater SDK for `POST /update_weights_from_mooncake`. Requires the three flags below; see [Mooncake weight updates](#mooncake-weight-updates). |
| `--model-update-sdk-module` | Import path of the Model Updater SDK module. Imported in the scheduler process on the first Mooncake update, not at startup. Required with `--model-update-config`. |
| `--model-update-engine-type` | SDK `EngineType` member name for this engine, resolved as `EngineType[value.upper()]`. Required with `--model-update-config`. |
| `--model-update-draft-weights` | `retain` or `refresh`: whether a Mooncake update also streams the speculative draft model's weights. Required with `--model-update-config`. |

### Weight version metadata

Every generation response includes the current version in
`meta_info["weight_version"]`. RL trainers can use this value to identify the
policy version that produced a sample.

The SGLang-compatible `update_weights_from_distributed`,
`update_weights_from_mooncake`, `update_weights_from_tensor`, and
`update_weights_from_disk` requests accept an
optional `weight_version`. The version changes only after the update succeeds.
`Engine.update_weights_from_distributed` requires `weight_version`. Pass
`None` to keep the current value on an intermediate update. Flushed L3
updates must pass a caller-supplied identity so independent checkpoints
cannot share a minted successor. When L3 is on, a
new `weight_version` requires `flush_cache=True`. Intermediate updates may
pass `None` until the last call flushes.

Use `GET /get_weight_version` to read the current value,
and `GET /model_info` to read the model path and version together.
When L3 storage is disabled, `POST /update_weight_version` with
`{"new_version": "..."}` sets the value directly. With L3 enabled, this endpoint
returns HTTP 400 without changing the version: changing only frontend metadata
would leave the cache namespace on the old checkpoint. Use
`POST /update_weights_from_distributed` with an explicit `weight_version` and
`flush_cache=True` to coordinate the weight load and cache namespace change.

### Logprobs

| Parameter | Purpose |
| --- | --- |
| `--enable-output-logprobs` | Gate for every logprob request. Off by default; the sampler gathers logprobs only when on, and a request asking for them on a server without the flag is rejected. |
| `--input-logprob-chunk-tokens` | Prompt rows pushed through the LM head per chunk when a request asks for prompt (input) logprobs. Defaults to `256`. A sizing knob only: the transient per-chunk cost is about `chunk × vocab × (2 + 2 + 4)` bytes (the bf16 logits shard, the bf16 TP-gathered logits, the fp32 log-softmax), so the default costs about 256 MiB at a 128K vocabulary, and the value never changes a result (log-softmax is row-local). |

Two request dialects share one compute path. The vLLM dialect
(`sampling_params.logprobs`) returns the sampled tokens' logprobs under
`meta_info["logprobs"]`. The SGLang dialect (`/generate` with
`return_logprob=true`) returns `meta_info["output_token_logprobs"]` as
`(logprob, token_id, text|null)` triples and, with `logprob_start_len`,
prompt (input) logprobs under `meta_info["input_token_logprobs"]`:

- `logprob_start_len=-1` (the default) returns the single entry
  `[(null, input_ids[-1], text)]` and computes nothing extra.
- `logprob_start_len=s` with `0 <= s < len(input_ids)` returns
  `len(input_ids) - s` entries: `(null, input_ids[s], text)` first, then the
  logprob of each following prompt token given its prefix. `s >= len(input_ids)`
  is a 400.
- `return_text_in_logprobs` fills the text field; `logprob_format` selects
  `"vllm"`, `"sglang"`, or `"both"`.

The engine computes prompt logprobs with the same fp32 log-softmax as the
output logprobs (see `docs/design/numerics.md`), accumulates them across
chunked-prefill chunks, and ships them once, on the first frame after the
prompt finished. A request that returns them from position `s` skips the
prefix cache for positions `>= s`, so the engine recomputes those positions
and always has logits for them. Positions before `s` reuse the cache as
usual. Mixed prefill/decode batches (`--enable-mixed-batch`) are supported.
A NaN or infinite prompt logprob terminates the request with a
`NumericalError` exactly like NaN logits on a sampled token
(`--enable-nan-detection`). The response never carries a non-finite value.

The ingress refuses with a 400 any request with `logprob_start_len >= 0`
that asks for at least one prompt logprob when the engine cannot score every
prompt position: models that narrow their prefill rows (DeepSeek V4.1's CED
decoder keeps only each prompt's last window for the LM head). The scheduler
reports this capability at startup and the frontend checks it before
admitting the request, so the data plane never has to. The ingress also
refuses multimodal prompts (their media positions carry content-hash ids,
not tokens) and any prompt whose client-supplied `input_ids` fall outside
the vocabulary. `logprob_start_len=-1` is always accepted.

Under pipeline parallelism (`--pipeline-parallel-size > 1`) the last stage scores the prompt
rows and the commit path carries both logprob vectors to the other stages
with the sampled tokens. Under query context parallelism
(`--prefill-context-parallel-size N`) the prompt rows of a chunk live on the
rank whose shard holds them. Because the LM head is vocab-sharded over the
same ranks, the ranks gather the planned rows' activations to the group and
every rank scores the whole plan, so each rank's
`--input-logprob-chunk-tokens` chunks cover the chunk's planned rows exactly
as without sharding (the per-chunk transient of the row above is the same),
and the result is identical on every rank. The ranks gather the sampled rows
only after scoring the prompt rows.

In a disaggregated deployment the prefill node and the decode node each
return their own frames, exactly as SGLang's do: the prefill node's finished
frame carries `meta_info.input_token_logprobs` (and the bootstrap token under
`output_token_logprobs`), while the decode node's frames carry
`output_token_logprobs` only and never `input_token_logprobs`. The prefill
node forwards the bootstrap token's logprob to the decode node, so the decode
node's `output_token_logprobs` covers every generated token. Merging the two
into one response is the router's job -- an SGLang `mini_lb`-style router or
the Dynamo compatibility frontend joins the prefill node's prompt logprobs
with the decode node's output, just as it does for SGLang. The direct msgpack
scheduler drive (SMG) carries sampled-token logprobs only: it refuses a
`logprob_start_len` that would produce prompt logprobs rather than compute
and drop them.

`top_logprobs_num > 0` and `token_ids_logprob` are not supported yet.

This runtime requires a `tokenspeed-scheduler` build that has
`RequestSpec.max_cached_prefix_tokens` (the admission-probe bound; see
`docs/design/scheduler.md` §1): every admission sets it, so an older scheduler
fails at the first request, not only at the first prompt-logprob request.

### Slime RL compatibility

TokenSpeed exposes the SGLang HTTP surface used by slime. The supported path is
an externally launched TokenSpeed rollout engine on separate GPUs, using full
NCCL weight updates. Attention data parallelism of any size is supported:
each weight op fans out to every DP worker and the frontend ANDs the
replies, and the scheduler completes an op only in a round where every DP
rank holds it (see [Weight updates under attention DP](#weight-updates-under-attention-dp)).

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

These slime paths are not yet supported end to end:

- colocated CUDA-IPC updates through `update_weights_from_tensor`;
- quantized-update hooks `post_process_weights` and `weights_checker`;
- disk-delta `pull_weights`;
- slime's retained top-p token set (`rollout_top_p != 1.0`). Use
  `--rollout-top-p 1.0` until TokenSpeed returns that metadata;
- rollout routing replay (`--use-rollout-routing-replay`).

`POST /update_weights_from_disk` and `POST /update_weights_from_tensor` stay on
the router for slime-compatible clients, but answer `501 Not Implemented` with
`{"success": false, "message": "..."}` before anything reaches the scheduler:
TokenSpeed's scheduler implements neither the disk load path nor the CUDA-IPC
receive path and would answer such a request with `success=false` ("not
supported on this engine"). Use `POST /update_weights_from_distributed` or the
Mooncake update described below. For the same reason the engine advertises
`rl.update_from = "distributed,mooncake"`, so a gateway never routes a disk
or tensor update here.

### Weight updates under attention DP

With `--data-parallel-size > 1`, the frontend sends
`init_weights_update_group`, `update_weights_from_distributed`,
`update_weights_from_mooncake`, and `destroy_weights_update_group` to every
attention-DP worker and ANDs the replies (joining distinct messages with
` | `). Each scheduler queues the op and completes it only in a round where
every DP rank holds the same kind of op at the head of its queue, decided on
the per-round DP all-reduce that already carries flush intent. One op
completes per round. TokenSpeed then MIN-reduces the device result across
the replica before it publishes the L3 weight version, so a failure on one
rank fails the update everywhere. A rank whose peer never receives the op
waits indefinitely, as with `/flush_cache`. The design rationale is in
`docs/design/event-loop.md`.

Only the two loads take the frontend's model-update writer lock (the lock
keeps generation out while the load rewrites parameters).
`init_weights_update_group` and `destroy_weights_update_group` rewrite
nothing and do not wait for in-flight generation, so the trainer's
rendezvous is not held up by long requests.

### Mooncake weight updates

`POST /update_weights_from_mooncake` loads one committed checkpoint version
that the RL trainer published to a Mooncake weight store through the Model
Updater SDK. Body: `{"version": int, "flush_cache": bool = true,
"weight_version": str | null}`. A missing or non-integer `version` is a 400.
The `flush_cache` wire default mirrors the reference engine's (FluentLLM's)
API so its trainer clients work unchanged. Every scheduler process reads its
own shard with its global rank as the SDK reader rank, on the forward
thread, ordered against forwards like the distributed update. The reply
arrives only after every worker finished its read, so the control server
proxies this route with a longer inactivity timeout (3600 s) than the other
RL routes.

Start the server with the four `--model-update-*` flags (table above): the
engine imports the SDK module lazily on the first update, and a missing
module fails that update with a clear message rather than failing startup.
`--model-update-draft-weights retain` updates the target model only.
`refresh` streams the target and the speculative draft model (every pipeline
stage that holds draft weights). Both policies notify the drafter afterwards
like the distributed update does.

`flush_cache` and `weight_version` follow the distributed update's rules
with one default: a flushed Mooncake update publishes `weight_version =
str(version)` when none is given (an explicit value wins). An unflushed
update keeps the current namespace unless one is given, and with L3 storage
a new `weight_version` still requires `flush_cache=true`. A successful update
stamps the version into generation metadata.

Trainer-side contract:

- Pause dispatch at the router before calling and resume after the reply.
  The scheduler's control thread blocks for the duration of the SDK read,
  so load reporting, PD transfer polling, and health responses stall on
  every worker. The frontend's writer lock only drains requests already
  admitted on this engine.
- While PD transfers or Host write-backs are in flight on any replica rank,
  the engine rejects `flush_cache=true` (and skips the load). Retry after
  they drain, or send intermediate updates with `flush_cache=false` and
  flush on the last one.
- Model update session: the SDK streams `(name, tensor)` pairs into each
  model's `load_weights` in many partial calls. The runtime brackets the
  models in a weight-update session (`begin_weight_update` /
  `end_weight_update` on `BaseCausalLM`) so a model derives its post-load
  state once, after the last chunk: `BaseCausalLM` defers every
  `post_load_weights` call made while the session is active and runs it once
  at the end, so a model's loader needs no session awareness of its own. The
  update rewrites the absorbed MLA `w_kc`/`w_vc` and the KDA conv banks in
  their existing storage (captured CUDA graphs keep valid addresses; a
  geometry change is an error). In-place one-shot transforms such as the
  LoRA norm scale fold apply only to the parameters this update reloaded,
  and fused parameters assembled from several checkpoint tensors (the NextN
  drafts' `q_a_proj`/`kv_a_proj_with_mqa`, GLM's FP8 indexer `wk` weight and
  scale) may straddle chunks. When the session ends, it rejects an update
  that streamed one half without the other. The session also screens every
  chunk for KV-cache scales other than one (the runtime writes and reads KV
  caches at unit scale): it loads such an update to completion and then
  rejects it, like the distributed update's. Models outside `BaseCausalLM`
  take no session hooks.
  The distributed update uses the same session.
- The engine does not roll back a failed update (`success: false`): the SDK
  may already have rewritten part of the parameters on some ranks, so the
  engine may be serving a mix of old and new weights, and replicas may
  disagree. The engine keeps the current weight version. Re-issue the update
  (a successful retry streams the whole checkpoint and restores consistency)
  or restart the engine before resuming dispatch. The same holds for the
  distributed update.

### Driving TokenSpeed from an external gateway

A gateway that fronts several engines (for example SMG with `--enable-rl`)
talks to this control app directly. The `ts serve` sidecar is not involved.
Launch the engine with `--rl-control-port <port>` and
`--rl-control-host <address the gateway can reach>` (the default binds
localhost only). Unless the network is trusted, set `--rl-control-api-key`:
an open control app on a routable host accepts weight updates from anyone who
can connect. The engine puts the resulting control URL and its capabilities
(`rl.control_url`, `rl.pause_modes`, `rl.update_from`, ...) into its server
info. SMG reads them when it registers the gRPC worker, so the gateway side
needs no configuration. The routes keep slime's expectations:
`POST /pause_generation` accepts `{"mode": "wait"|"abort"|"keep"}` (default
`wait`), and `/flush_cache` answers on both GET and POST. Routes with optional
bodies accept an omitted body, but malformed or non-object JSON answers `400`
before any control operation runs. The engine does not advertise wildcard
bind addresses (`0.0.0.0` or `::`) as control URLs. Use a concrete address
for gateway discovery.

## Scheduler and memory

| Parameter | Purpose |
| --- | --- |
| `--max-model-len` | Maximum sequence length. If omitted, TokenSpeed uses the model config. |
| `--gpu-memory-utilization` | Fraction of GPU memory used for model weights and KV cache. Lower it to leave headroom. |
| `--max-num-seqs` | Maximum number of active sequences the scheduler may process concurrently. |
| `--chunked-prefill-size` | Token budget the scheduler may issue in one iteration; it also bounds the multimodal placeholder tokens one encoder call produces (an item larger than that runs alone). Defaults to `8192`. Set `-1` to disable chunked prefill. |
| `--max-prefill-tokens` | Prefill token budget used when chunked prefill is disabled. Defaults to `8192`. |
| `--max-total-tokens` | Override the automatically calculated token pool size. |
| `--prefix-granularity` | Scheduler prefix granularity in tokens — the identity boundary of cache reuse (`--block-size` is a deprecated alias). |
| `--enable-prefix-caching` / `--disable-prefix-caching` | Enable or disable prefix cache reuse. |
| `--enforce-eager` | Disable device-graph execution (CUDA Graph on CUDA, ACL Graph on NPU). |
| `--disable-prefill-graph` | Keep prefill eager while leaving decode device graphs enabled. |
| `--disable-kda-prefill-graph` | Disable KDA prefill CUDA graphs while retaining ordinary prefill and decode graph settings. Enabled by default for supported `cutedsl_kda` prefill attention when prefill graphs are enabled. |
| `--disable-cudagraph-memory-reserve` | Size the KV cache from free memory instead of reserving what the device graphs will cost and, on CUDA, what startup keeps resident. |
| `--max-cudagraph-capture-size` | Largest decode batch size to capture as a device graph. |
| `--cudagraph-capture-sizes` | Explicit decode batch sizes to capture as device graphs. |
| `--prefill-graph-capture-token-sizes` | Total input-token capacities per forward, summed across the batch. Shorter inputs are padded. |
| `--prefill-graph-capture-batch-sizes` | Request capacities for inline KDA prefill capture. Replay selects the smallest compatible capacity that fits the batch. |
| `--retraction-snapshot-host-gb` | Explicit size, in gigabytes, of the per-rank pinned Host pool that holds a retraction image's tail (a retracted request's unaligned tail pages, its blocks of groups Host L2 never holds, and its slot state; the hash-complete blocks go to Host L2 and stay pinned there until the request is restored; with `--disable-kvstore` the pool holds whole images). Wins over `--retraction-snapshot-ratio` when set; `0` (the default) means not set, like `--kvstore-size` against `--kvstore-ratio`. A request a capacity retraction suspends is imaged here and resumes exactly where it stopped once the image is copied back; nothing is recomputed. When no resident's image fits -- the pool or its rows are exhausted -- the scheduler aborts the newest resident instead (finish `err_type` 524, see "Retraction snapshot pool" below), so the pool bounds how much can be suspended at once, never how long an admission waits. Fused and decode roles only; the prefill and encode roles never retract and ignore it with a log. Allowed under `--kv-parallel-size > 1`. |
| `--retraction-snapshot-ratio` | Size of the retraction snapshot pool as a multiple of this rank's Device KV capacity (the base `--kvstore-ratio` scales too), used when `--retraction-snapshot-host-gb` is not set. Unset (the default) derives the pool at device build: 0.1 of the Device KV without the Host KVStore (whole images, kept cheap to pin; `1` makes every resident suspendable), else the image tails of `--retraction-snapshot-max-requests` requests (table below). An explicit `0` is the one way to run **without a pool**: nothing can be imaged, so every capacity block aborts its victim (the newest resident). |
| `--retraction-snapshot-max-requests` | Slot-state image rows of the retraction snapshot pool, i.e. the most requests suspended at once, and the request count the derived pool size is multiplied by. `0` (the default) derives to `--max-num-seqs / attention-DP size`, which covers every resident request of the rank; refused with `--retraction-snapshot-ratio 0`, which has no rows to size. When no row is free the scheduler aborts a resident rather than waiting. |
| `--debug-force-retraction-interval` | **Test only.** Every `N` scheduler plans retract the oldest quiescent decoding request (`N > 0`), or with `-N` the one prefilling request between its chunks, without capacity pressure, so a test run exercises the suspend/restore path on every request (the bitwise continuation oracle of [Numerics](../design/numerics.md)). Refused with `--retraction-snapshot-ratio 0` (no pool to image into). `0` (the default) is off; never set it in serving. |

For pure prefill, token capacities count newly computed tokens, not cached
prefixes or each request's full sequence length. Two requests extending by
868 and 869 tokens use the 2048-token bucket and request capacity 2 when
configured. A smaller batch can reuse that capture if there is room for its
dummy request slots. These settings do not replace the scheduler's
`--max-num-seqs` limit.

`--prefill-graph-capture-sizes` remains a compatibility alias for
`--prefill-graph-capture-token-sizes`. Specify only one spelling per command.
Both populate the existing `prefill_graph_capture_sizes` Python field.
Unset token sizes use the existing default ladder. Unset batch sizes use the
minimum request count that fits each token bucket within the model context.

`--chunked-prefill-size` is intentionally separate from
`--max-num-batched-tokens`: in TokenSpeed it is the scheduler's per-iteration
issue budget, while `--max-total-tokens` controls the global token pool.

## Parallelism

| Parameter | Purpose |
| --- | --- |
| `--tensor-parallel-size`, `--tp` | Familiar alias for setting attention tensor parallel size. |
| `--attn-tp-size` | Tensor parallel size for attention. |
| `--kv-parallel-size` | KV parallelism (KVP): shard full-history KV pages (MLA/DSA latent and index-K, DeepSeek V4 compressed KV) cyclically over a consecutive subgroup of attention TP; must divide `--attn-tp-size`. Each rank then stores one shard of every request's pages, so the KV capacity per GPU grows by that factor and the DSA indexer scores only owned pages; decode attends the owned pages and merges the partials by LSE (DCP, `docs/serving/parallelism.md`). Allowed on aggregated engines and with `--disaggregation-mode prefill` (every rank of the subgroup sends its owned pages to an unsharded decode). Not supported yet: the decode role; speculative decoding on any ordinary MLA/DSA model (the recipe refuses to shard a cache holding a draft group, whichever dense kernel runs it -- only the DeepSeek V4 and Kimi K3 recipes shard with a draft, and FlashMLA/GPU DSA reject speculation in their DCP decode outright); and L3 storage (`--kvstore-storage-backend`: a page-cyclic sharded group has no owner-stable L3 key). The Host KVStore and the retraction snapshot pool are supported: the scheduler allocates every Host block in its Device block's residue class and each rank copies the blocks it owns. |
| `--attn-head-tp-size` | Shard the MLA head projections (`q_b_proj`, `kv_b_proj`, `o_proj`) by heads over this many contiguous ranks that hold different rows; the attention exchanges heads for tokens around its core. Over attention-DP ranks (needs attention TP 1, attention DP and `--disaggregation-mode decode`; each rank keeps its own KV) the layout serves decode rows only, so it sets `--disable-prefill-graph` and tunes on a decode step; it admits any `max_new_tokens`, because a retracted request resumes by restore rather than by a recovery prefill. Over the query shards of a prefill engine (must equal `--prefill-context-parallel-size`) the extend rows run the absorbed sparse prefill through the exchange and none of the decode-only rules apply. Defaults to the ranks holding the same rows: the attention TP size, or 1 (head-replicated) under `--prefill-context-parallel-size`. See [Parallelism](../serving/parallelism.md#decode-side-tp-layouts-under-attention-dp). |
| `--lm-head-tp-size` | Vocab-shard the LM head over this many contiguous ranks. Under attention DP the default 1 replicates it; a wider group gathers the ranks' rows before the logits GEMM and transposes the shards back. Without attention DP it must equal the attention TP size. Not combinable with `--dp-sampling`; under attention DP, requests asking for prompt logprobs (`logprob_start_len`) are refused. |
| `--tp-batch-invariant` | `none` (default), `attn`, or `attn+dense`: make the head-sharded `o_proj` and the dense `down_proj` column-parallel on hidden (all-gather of the reduction dim, full-K GEMM, all-to-all back to own rows) so no cross-rank sum remains outside MoE and the bits equal a TP1 full-K GEMM. `attn` needs `--attn-head-tp-size` > 1; `attn+dense` also needs a dense TP group wider than attention TP (so not under `--prefill-context-parallel-size`, whose dense group is 1 or the attention TP width; only `attn` applies there); both need unquantized `o_proj` / `down_proj`, judged on the checkpoint's resolved quantization (a quantized checkpoint passes when its `disable_quant_module` excludes `self_attn` and, for `attn+dense`, `dense_mlp` / `mlps`). |
| `--prefill-context-parallel-size` | Query context parallelism on the PD prefill role: shard every extend forward's rows over the attention TP group, rank `r` computing a contiguous slice of the chunk against the gathered KV history of its requests (the KV write, index-K write, sampled rows and prompt-logprob rows are gathered across the group; the scheduler, cache allocation and PD transfer are unchanged, and `--chunked-prefill-size` keeps counting the whole chunk). Must equal `--attn-tp-size`; requires `--disaggregation-mode prefill`, `--disable-prefill-graph`, attention DP 1, no `--enable-mixed-batch`, a DSA-family attention backend with a bf16 KV cache (the gathered write stores native latent rows), `--dense-tp-size` and the MoE TP×EP group each 1 or the attention TP width, and `--kv-parallel-size` 1 or equal to it (the sharded-page combination inherits KVP's rules: Host KVStore and snapshot pool allowed, L3 refused). The attention weights are head-replicated unless `--attn-head-tp-size` equals it, which shards them over the shard group. 1 (default) is off. |
| `--dense-tp-size` | Tensor parallel size for dense layers. Defaults to the attention TP width: the full world without DP attention, one replica with it. |
| `--moe-tp-size` | Tensor parallel size for MoE layers. |
| `--data-parallel-size` | Number of data-parallel replicas. |
| `--mm-encoder-tp-mode` | Multimodal encoder parallelism: `weights` shards encoder weights with attention TP; `data` uses TP1 whole-item DP and currently requires aggregate serving or the prefill role. |
| `--enable-expert-parallel` | Set expert parallelism across the selected world size. |
| `--expert-parallel-size`, `--ep-size` | Explicit expert parallel size. |
| `--pipeline-parallel-size` | Pipeline stages for prefill chunk pipelining. Requires `--disaggregation-mode prefill`; forces eager execution; every per-layer parallelism resolves inside one stage's world. |
| `--pp-layer-partition` | Explicit per-stage layer counts, front to back (`"24,24,24,21"`); one entry per stage, summing to the model's layer count. Default: even split with the remainder on the front stages. |
| `--world-size` | Total worker process count across all nodes. |
| `--nprocs-per-node` | Worker process count per node. |
| `--nnodes` | Number of nodes. |
| `--node-rank` | Rank of the current node. |
| `--dist-init-addr` | Distributed initialization address. |
| `--emulate-rank-zero` | Run only global rank 0 of the configured layout on one GPU, with local stand-ins for its collectives. For single-GPU performance work; outputs are not meaningful. See [Emulating rank 0 on one GPU](../serving/parallelism.md#emulating-rank-0-on-one-gpu). |

Use `--tensor-parallel-size` for simple launches. When attention, dense,
and MoE layers need different process groups, use the TokenSpeed-specific
split knobs.

### Expert placement

| Parameter | Purpose |
| --- | --- |
| `--ep-num-redundant-experts` | Extra physical expert slots per MoE layer for replicas of hot experts (`P = E + R`, must divide over the EP size; needs `ep_size > 1`). Default 0. |
| `--init-expert-location` | `trivial` (default). Otherwise the form is decided in order: inline JSON when the value starts with `{`, a directory of per-rank `*.expert-load.pt` records (merged), an existing `.pt`/`.json` file, else a glob over record files (merged). A `logical_count` `[layers, experts]` load record derives the placement with the EPLB algorithm, a `physical_to_logical_map` `[layers, slots]` pins one exactly. |
| `--ep-dispatch-algorithm` | How routing picks among an expert's replicas; required with any of the flags above or below. `static_with_zero_expert` for models with zero experts (LongCat), `static` otherwise; `dynamic`/`dynamic_with_zero_expert`/`fake` draw at random (refused under `--numerics rl-bitwise` and on replicated-input EP). |
| `--eplb-algorithm` | `auto` (default), `deepseek` or `deepseek_hierarchical`. |
| `--expert-distribution-recorder-mode` | `stat` (the only mode): count the routes to every physical expert so the `EXPERT_LOAD` profile activity (`/start_profile` ... `/stop_profile`) can write each rank's load record and `--enable-eplb` can rebalance from the counters. |
| `--enable-eplb` | Online expert rebalancing: every `--eplb-rebalance-num-iterations` forwards the routing load since the previous snapshot is rebalanced with the EPLB algorithm and the expert weights move between slots. Requires `--expert-distribution-recorder-mode stat` and a static `--ep-dispatch-algorithm`, both explicit, and `ep_size > 1`; `--ep-num-redundant-experts 0` is allowed (permutation only). `POST /rebalance_experts` starts one rebalance now. Under a `--numerics` envelope it rides the placement-independent slot-order MoE combine the envelope folds in (`--moe-combine-order slot`). |
| `--eplb-rebalance-num-iterations` | Forwards between two load snapshots; required with `--enable-eplb`, `> 0`. |
| `--eplb-rebalance-layers-per-chunk` | MoE layers whose experts move in one scheduling round; required with `--enable-eplb`, `1..num MoE layers`. Fewer layers per chunk bound the per-round stall. |

All of these apply only to models that opt in to expert placement
(LongCat-Flash); other models refuse them at startup. See
[static expert placement](../serving/parallelism.md#static-expert-placement-with-redundant-experts)
for the record → place → route flow and the two dispatch flavours, and
[dynamic expert rebalancing](../serving/parallelism.md#dynamic-expert-rebalancing)
for the online variant.

## Backend selection

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

A LongCat layer runs two dense MLPs and one MoE off the same attention output.
Its rows follow the dense comm pattern. When the MoE pattern differs
(attention TP equal to the dense TP but not to the MoE TP x EP width, as under
attention DP with `--enable-expert-parallel`), the layer re-gathers the MoE
output into the dense layout. TokenSpeed rejects `--enable-allreduce-fusion`
for that layout.

When `--dp-sampling` is enabled, the logits processor owns the per-forward
logits layout decision and carries the resulting plan to the sampling backend
with the logits output.

## Reasoning and tool calling

| Parameter | Purpose |
| --- | --- |
| `--reasoning-parser` | Parser for extracting reasoning content from model outputs (handled by the smg gateway). |
| `--tool-call-parser` | Parser for OpenAI-compatible tool-call payloads (handled by the smg gateway). |

Common reasoning parser values include `kimi_k25`, `base`, `qwen3`,
`deepseek_r1`, and `deepseek_v31`. Common tool-call parser values include
`kimik2`, `qwen`, `deepseek_v4`, `json`, and `passthrough`. The SMG gateway
validates the parser names, so use the values accepted by the bundled
`tokenspeed-smg` package.

## Speculative decoding

| Parameter | Purpose |
| --- | --- |
| `--speculative-config` | JSON speculative decoding configuration. |
| `--speculative-algorithm` | Speculative algorithm, such as `EAGLE3`, `MTP`, `DFLASH`, or `DSPARK`. |
| `--speculative-draft-model-path` | Draft model path or repo ID. |
| `--speculative-draft-model-quantization` | Draft model quantization. Defaults to `unquant`. |
| `--speculative-num-steps` | Number of draft model steps. Defaults to `3`. |
| `--speculative-num-draft-tokens` | Number of draft tokens. Defaults to `--speculative-num-steps + 1`; required for draft trees. |
| `--speculative-eagle-topk` | Children each draft node expands to per step. Defaults to `1` (a chain); above 1 the draft is a tree. |
| `--enable-speculative-sampling` | Draft-prob rejection sampling for the chain drafters (see below). Off by default. |
| `--spec-reject-draft-prob-threshold` | With `--enable-speculative-sampling`, recorded draft probabilities above this value mark a request with no proposal yet and always reject. Defaults to `2.0`; must lie within `[1.0, 2**20]`. |
| `--eagle3-layers-to-capture` | EAGLE3 layers to capture. |
| `--disable-replay-ssm` | Stage every verify position's recurrent state instead of replaying the accepted tokens. ReplaySSM is on by default for supported Qwen GDN and Nemotron-H Mamba2 targets; `--enable-replay-ssm` is accepted as a deprecated no-op. |

Prefer `--speculative-config` for recipe-style launches because it keeps method,
draft model, and token count together.

`EAGLE3` and `MTP` drafts are chains by default: `--speculative-num-draft-tokens`
must equal `--speculative-num-steps + 1`. With `--speculative-eagle-topk` above 1
they draft a tree instead, and `--speculative-num-draft-tokens` is its node
budget (root included). Set it explicitly: topk 1..16, steps 1..10,
`(steps - 1) * topk` lane slots within the node budget, and at most 64 nodes. Trees need the `trtllm`
attention backends (`trtllm_mla` or `tokenspeed_mla` for MLA models) and the `greedy` or `triton` sampling backend; see
[draft-tree speculation](../design/tree-speculation.md) for the full scope.

`MTP` serves two head shapes under one flag. An Eagle-like head (one MTP
layer chained on its own hidden, such as DeepSeek NextN) runs the Eagle chain.
A multi-depth head (one distinct depth layer per draft step over the same
window, such as Inkling, or an out-of-tree draft registered for the multi-depth
drafter) runs every depth `0..--speculative-num-steps-1` each round, so the
draft checkpoint needs at least that many depths. Both shapes run under
attention data parallelism (idle ranks mirror the depth loop with empty
forwards) and with PD layerwise transfer
(`--disaggregation-layerwise-interval`), where the draft's per-depth cache
planes become ready together after the drafter's run. Known PD limitation
of the multi-depth head: the PD transfer does not carry the drafter's
cross-round stash (the last `k-1` committed tokens and their target hiddens
per request) with the KV, so for up to `k-1` decode rounds after a request
lands on the decode node the draft rewrites prompt-tail draft-KV positions
from an unfilled stash. Draft acceptance may dip for those rounds.
Verification stays exact. Shipping the stash with the bootstrap payload is a
planned follow-up.

### Draft-prob rejection sampling

By default the chain drafters (`EAGLE3`, `MTP`) propose the argmax of their
logits and the verifier runs the target-only rule: it accepts draft `x` with
probability `p(x)`, and a rejection samples the target with `x` removed. The
served distribution is the target's `p` whatever the drafter proposed, so the
verifier needs no draft distribution. `--enable-speculative-sampling`
switches to the standard rule: each draft step samples its token from
the drafter's own distribution `q = softmax(draft logits / T)` at the
request's `temperature`
(greedy requests keep the argmax and a one-hot `q`), records `q`, and the next
round's verify accepts with `coin * q(x) < p(x)` and resamples from
`norm(relu(p - q))`. Both rules serve `p`. The draft-prob rule accepts
`1 - TV(p, q)` of the drafts, which is markedly higher than `p(argmax q)` when
requests sample at temperature. Greedy requests behave identically under both
rules. `top_k`, `top_p`, `min_p`, penalties and `logit_bias` stay on the
verifier's side. `q` only follows the temperature.

A request admitted has no
recorded `q` for its first chain: its rows hold a sentinel above
`--spec-reject-draft-prob-threshold`, which rejects at the first draft and
samples the first token from the full target. A request a capacity
retraction suspends keeps its `q`: the recorded distributions travel in its
slot-state image and come back with the restore, so its first verify after
resuming accepts drafts exactly as the uninterrupted run would. The
sentinel is written as
`threshold + 1.0` in fp32, hence the range: below `1.0` a real probability
would read as the sentinel, and the cap keeps the `+ 1.0` representable.
Under PD disaggregation the prefill node's candidates land the same way, so
the decode node's first verify of a landed request accepts nothing.
TokenSpeed refuses the flag on the prefill role (`--disaggregation-mode
prefill`): that role never verifies a chain and its candidates ship without
`q`, so it would only allocate the distribution buffer. Pass it to the decode
role only.

Requirements: `--speculative-algorithm EAGLE3` or `MTP` (block drafters
`DFLASH`/`DSPARK` propose a whole block greedily), `--speculative-eagle-topk 1`,
and `--sampling-backend flashinfer` or `flashinfer_full` (`greedy` verifies by
exact match, the Triton backends by target-sampled exact match; neither reads
`q`). The drafter switches its draft model's fused TP-sharded argmax off so
the full-vocab logits reach it, which adds the logits all-gather to every
draft step under tensor parallelism.

Known limits, kept as in the reference engine for now: a landed PD request's
first verify always rejects its shipped candidates (sentinel rows) rather than
verifying them target-only, and the verifier gathers the full `[bs, N, vocab]`
block of recorded rows per step instead of only the entries the accept test
reads. A draft step whose logits give no finite distribution (all NaN, or an
overflow) proposes a junk token and records the sentinel for that row, so
verify rejects the token and samples from the full target. It never raises a
device error.

Memory: the recorded distributions take
`(max_num_seqs + 2) x num_draft_tokens x vocab_size x 4` bytes
(`--speculative-num-draft-tokens` fp32 rows per request-pool slot), plus a
batch-ordered gather buffer of `max_num_seqs x num_draft_tokens x vocab_size x
4` bytes on the verifier. 80 requests at 4 draft tokens over a 129K
vocabulary cost about 330 MB in total. On CUDA, when the CUDA-graph memory
reserve is on, the runtime charges both to it as startup residue, out of the
KV-cache budget. Otherwise they come out of the `--gpu-memory-utilization`
headroom.

`DFLASH` and `DSPARK` are block drafters: one draft forward proposes a whole
block instead of one token per step, so their two token counts are coupled.
`--speculative-num-draft-tokens` is the verify width -- one anchor row plus one
row per drafted token -- and `--speculative-num-steps` must be one less. The
draft checkpoint's `block_size` fixes both, and startup rejects a mismatch
rather than silently drafting a wrong-width block. The two families
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
runtime from the checkpoint architecture, so you need no separate algorithm
flag. Draft proposals greedily follow the selector's transition-conditioned
path. One Triton kernel per verify step walks it. The target's verification
step applies a request's `temperature`, `top_k` and `top_p`, never the
proposal, so the served distribution is the target's whatever the drafter
proposed.

On a prefill server with `--pipeline-parallel-size > 1`, TokenSpeed accepts
speculation for `MTP` and `DSPARK` only. The drafter runs on the last stage,
the only stage that samples. It writes the candidate block the remote decode
carries to the decode server, which verifies it as usual. `DSPARK` also
produces its draft context across stages and keeps requiring attention CP = 1
and matching dense/attention TP groups. An `MTP` (NextN) draft reads only the
last stage's final hidden states: the other stages build and load no draft
model at all, and the NextN checkpoint must ship its `embed_tokens` weight
because the target embedding lives on the first stage. `DFLASH` and `EAGLE3`
read target taps from several stages, and TokenSpeed rejects them on a
pipeline. The runtime decides layerwise transfer
(`--disaggregation-layerwise-interval`) per stage: stages before the last own
no draft cache and always allow it. The last stage allows it exactly when the
same drafter would on a single-stage server (the `Mtp` and EAGLE-style
drafters enqueue every depth's KV write inside their run, so they finalize
layerwise; startup still rejects a drafter class without that guarantee
there).

A block drafter writes its KV at the target's cache locations, so it shares the
target's page table: `--prefix-granularity` is a target-side choice and the draft
follows it. Any sliding window the draft checkpoint declares is an attention
mask that the draft's own layers apply, never a cache-retention policy of its
own. Only the backends that forward that mask to their kernels can serve such a
draft: `mla` and `tokenspeed_mla` (`gluon` on AMD) for MLA drafts, and
`mha`/`fa3`/`fa4`/`triton`/`flashinfer`/`trtllm_mha` for GQA drafts. Startup
rejects any other `--drafter-attention-backend` rather than quietly widening
the draft's attention to the full history.

## Observability

| Parameter | Purpose |
| --- | --- |
| `--log-level` | Runtime log level. |
| `--enable-log-requests` | Log request metadata and optionally payloads. On by default; `--no-enable-log-requests` disables. |
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
host-side scheduler counter. The line adds no GPU synchronization.

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
is intentionally verbose. Enable it only while debugging. For
decode-only batches it also logs the anchor, draft candidates, target verify
tokens, and their position-wise matches.

### Per-request stats

`--enable-log-request-stats` enriches the scheduler's per-request finish line for
latency/throughput debugging. When set, the `Req: <rid> Finish! ...` line carries
a Python-object repr (`RequestStats(...)`) instead of the default
`Accept_num_tokens_avg` value (which it subsumes as `acc_len`). The scheduler
derives every field from host-side timestamps and counters it already has —
the line adds **no GPU sync** and so no engine slowdown. Example:

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

### KV cache events

KV cache events publish reusable device prefix-cache mutations from the live
C++ scheduler path. This initial stream does not publish Host/L2 loadback
events. The scheduler caches block hash lineage on prefix-cache nodes, so
publishing a stored block uses the parent node's cached hash instead of
rebuilding the full ancestor prefix.

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

## TokenSpeed-specific runtime knobs

These parameters are TokenSpeed-specific. They expose runtime
features directly:

- `--max-total-tokens`
- `--max-prefill-tokens`
- `--chunked-prefill-size`
- `--attn-tp-size`
- `--attn-head-tp-size`
- `--lm-head-tp-size`
- `--tp-batch-invariant`
- `--dense-tp-size`
- `--moe-tp-size`
- `--kvstore-*`
- `--retraction-snapshot-*`
- `--kv-events-config`
- `--mla-chunk-multiplier`
- `--disaggregation-*`
- `--comm-fusion-max-num-tokens`
- `--enable-allreduce-fusion`

### Retraction snapshot pool

A capacity retraction suspends a resident request instead of recomputing
it: the request's cache pages and its per-slot Device state are imaged to
pinned Host memory, and once Device pages are free again the image is copied
back and the request continues exactly where it stopped -- a decoding
request decodes, a mid-prompt request runs its next chunk
(`docs/design/scheduler.md` section 4). The image is split: the
hash-complete prefix pages go to Host L2 as a stream-ordered write-back and
stay pinned there until the restore; the unaligned tail, the blocks of
groups L2 never holds (a replayable sliding-window group, a state group's
live block) and the slot-state blob go to the request-private pool
(`docs/design/cache-concepts.md`, "Retraction image"). The two buffers are
separate allocations of one Host cache executor and both count against its
Host-memory headroom check. The pool is independent of the KVStore: with
`--disable-kvstore` there is no L2 leg and the pool holds whole images; with
L2 it holds the tails plus the slot-state blob.

**Sizing.** The pool is on by default on the roles that retract and sized
like the Host KVStore, in LCM blocks of the rank's transfer layout
(`python/tokenspeed/runtime/cache/l2/sizing.py`); the slot-state arena
(`--retraction-snapshot-max-requests` x blob bytes) is allocated on top.
The first rule that applies wins:

| Arguments | Pool size | Request cap (`--retraction-snapshot-max-requests` when `0`) |
|---|---|---|
| `--retraction-snapshot-host-gb G` (G > 0) | `G` GB, whole LCM blocks | `--max-num-seqs / attention-DP size` |
| `--retraction-snapshot-ratio R` (R > 0) | `R` x this rank's Device LCM blocks | same |
| `--retraction-snapshot-ratio 0` | **no pool**: a capacity block aborts its victim | `0` (an explicit cap is refused) |
| neither, `--disable-kvstore` | 0.1 x this rank's Device LCM blocks (whole images) | same as above |
| neither, Host KVStore on | request cap x one image tail | same as above |

One image tail, per cache group, is one page for a group that publishes
to Host L2 (its unaligned last page; a state group's live block) and a
request's worst-case pages at the context limit for a group that never
publishes (a replayable sliding-window group), folded to LCM blocks by the
scheduler's `CapacityModel`, so the pool is counted the way the scheduler
later claims it. Without the KVStore a whole image is the request's every
page, so the default is a tenth of the Device KV rather than all of it: an
on-by-default pool must stay cheap to pin (the Device KV once is hundreds of
gigabytes of pinned Host memory for a small model on a large GPU), and it
still suspends a few residents; an image that does not fit aborts its
victim. A deployment that wants every resident suspendable passes
`--retraction-snapshot-ratio 1`, or enables the KVStore, whose tails are
cheap. A size or ratio that holds no whole LCM block is refused,
not rounded down to none. Each engine logs one line at startup with the
resolved pool (GB and LCM blocks, which rule sized it), the request cap
and arena, and whether Host L2 takes the hash-complete pages; without a
pool the line says that capacity blocks abort. The prefill and encode
roles never retract: their pool arguments are ignored with a log.

**When the image does not fit.** Host accounting stays optimistic -- no
admission gate charges a request's worst-case image -- and retraction never
waits on Host capacity or falls back to recomputation. The capacity policy
ranks only victims whose image fits (a free slot-state row and pool room for
the tail; hash-complete pages ride Host L2, whose shortfall falls back to
the pool). When no resident fits, the scheduler **aborts the newest
retractable resident** in the same round and grants its pages to the
blocked request. The client sees the finish `{"type": "abort", "err_type":
524, "message": "Request aborted by the scheduler: <detail>"}`
(`ABORT_CODE.CapacityAbort`; the detail names the shortfall -- "snapshot
pool cannot hold the image" or "no blob slot free
(max_retracted_requests=N)" -- and the arg to raise), distinguishable from
a NaN abort (523) or a transfer failure (521), so a caller that can retry --
an RL rollout, a batch job -- resubmits; the engine counts them in
`tokenspeed:num_capacity_aborted_requests`. The two args therefore size how
much work may be suspended at once, and exceeding them costs an abort of
one request, never latency for everyone else. With no pool
(`--retraction-snapshot-ratio 0`) nothing can be imaged, so a
capacity-blocked round aborts the newest resident instead of imaging
anyone. `--debug-force-retraction-interval` never aborts: a forced
retraction whose image does not fit is refused and logged. The scheduler's
victim choice is in `docs/design/scheduler.md` section 2.

The ops ride the round's `DeviceHandle.execute` with the L2 write-backs and
load-backs (`docs/design/event-loop.md`), their ACKs the same cache poll, and
the pool arms the completion hooks like the L2 tier does. Both Host tiers are
allowed under `--kv-parallel-size > 1`: the scheduler allocates
every Host block in its Device block's residue class and each rank copies the
blocks it owns; L3 storage stays refused there. The weight-update cache flush
is refused while a request is suspended with an image, so an image of
old-weight KV is never restored under new weights.
`--debug-force-retraction-interval` exercises the path without pressure in
tests (see the flag above).

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
`{tsl3v1-<sha256>}_{content_hash}|g{group}|o{page_offset}|r{tp_rank}|c0`
(the trailing `c0` is the retired context-parallel shard id, kept literal so
objects written before its removal stay addressable).
The hashed prefix includes the loaded checkpoint (`--model`, the resolved
immutable revision or a local fingerprint of selected weights, metadata,
and local `*.py` including imported package subdirectories and
directory symlinks Python follows on import — never an inherited
config `_commit_hash` or a 40-hex folder name outside a Hugging Face hub
`(models|datasets|spaces)--*/snapshots/<commit>` cache path with a sibling
`refs` directory (the fingerprint includes a directory merely named
`snapshots`) — plus `--load-format` so a directory that contains
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
present), the pipeline stage, any
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
TP/CP/PP because DP ranks hold different sequences. Flush includes DP
because object keys omit DP rank. Remote L3 deletion is the next
replica-then-DP phase: it
returns success/failure instead of raising, the ranks MIN-reduce the
result, and only then does `ClearCache` destroy Device/Host. A rank
whose writebacks have drained cannot rotate L3 or drop local indexes
while a peer still
rejects or while Mooncake `remove_by_regex` failed on another rank. The
frontend ANDs every DP worker's `/flush_cache` reply. A
split flush would leave mirrored
schedulers with different prefix indexes. The weight-update RPC then
fails so the caller retries instead of serving new weights against the
previous checkpoint or entering NCCL weight broadcasts alone. A
`batch_exists` hit is not a lease: the L3 pages are fetched into Host
before the request is admitted (below), and whatever is no longer there
lands short. A short Mooncake read (fewer bytes than the requested
page) is a miss, not a success. A backend exception or malformed result is a
local miss so every replica rank still enters the MIN-reduce. Clients are
not failed.
L2 write-back ACKs use the same replica groups: `WriteBackDone` is
emitted only after every cache-owning rank holds the completion, so a
worker cannot publish Host while a replica peer's Mooncake put is still in
flight. A truncated `batch_is_exist` reply is a failed put, not
an implicit success.
When L3 is on, the engine rejects supplying a new `weight_version` with
`flush_cache=False`, so the runtime cannot treat stale Device/Host KV
and in-flight D2H copies as the new checkpoint. Flushed L3 updates
require an explicit `weight_version`. Minting `{current}-uN` would let
independent checkpoints collide.
A successful Engine update stamps that version into
frontend `server_args`.
GQA with TP above the KV-head count assigns
different heads to the same `r{tp_rank}`, so `attn_tp_size` (resolved
`mapping.attn.tp_size`) is also in the namespace. The namespace also isolates
resolved target and draft attention backends, including the full-attention
sub-backend of a hybrid model: different implementations can produce
different downstream KV even with identical cache layouts. This
namespace extension intentionally starts a cold L3 cache instead of
reusing objects written without backend identity.
TokenSpeed splits `global_segment_size` across
attention-TP × pipeline-parallel ranks so the mounted total matches the
configured size. Use the resolved `mapping.attn.tp_size`, not
`--attn-tp-size` alone. L3 requires Host L2 (do not pass `--disable-kvstore`)
and an unsharded KV cache (not `--kv-parallel-size > 1`).
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

Constructing `MooncakeKvStore` requires `extra_config`. Pass `None` to
use `MOONCAKE_MASTER` / `MOONCAKE_CLIENT` and the other env defaults.

**Prefetch before admission.** An L3 hit is never loaded under a running
forward. When a waiting request's registered L3 keys past its Device/Host
hit span at least `--kvstore-prefetch-min-pages` prefix pages, the
scheduler allocates Host blocks for them, emits a prefetch op and holds the
request in a pre-admission state that owns those Host blocks and nothing
else -- no Device page, no batch slot -- while later requests may be
admitted past it. The Host cache executor's prefetch lane (a CPU thread)
fills the pages in prefix order, `--kvstore-prefetch-batch-pages` per
`batch_get_into`, and stops at the first missing page or at the deadline
`--kvstore-prefetch-timeout-base-s + --kvstore-prefetch-timeout-per-page-s
x pages`, counted from when the op starts on the lane and checked between
batches only (a `batch_get_into` already issued runs to completion -- the
store API has no timeout -- so one batch may overrun the deadline, and a
hung store holds the lane and every later prefetch behind it); the replica
agrees (a MIN across TP, CP and PP) on the pages
landed, the scheduler publishes exactly that prefix in Host L2 and forgets
the rest of the keys, and the request's admission is then an ordinary Host
hit whose Host-to-Device load overlaps its first chunk layer by layer.
Fewer free Host blocks than the hit truncate the prefetch; below the minimum
there is none and the pages are computed -- admission never waits on Host
room. The cost: an L3 hit waits for its prefetch before it is admitted (one
plan of latency at least), holding no Device pages meanwhile; the gain: no
forward is ever skipped or recomputed because of L3. The decode role of a PD
deployment never prefetches (its admission probes Device only). The four
knobs are required with `--kvstore-storage-backend` and refused without it:

| Argument | Description |
| --- | --- |
| `--kvstore-prefetch-min-pages` | The shortest L3-only prefix, in prefix pages, worth a prefetch; a shorter hit is computed. The reference engine's `prefetch_threshold`. |
| `--kvstore-prefetch-timeout-base-s` | The prefetch deadline's base term in seconds (positive). The deadline says when the lane stops starting batches; it does not interrupt a batch in flight. |
| `--kvstore-prefetch-timeout-per-page-s` | The deadline's per-page term in seconds (0 for a fixed deadline). |
| `--kvstore-prefetch-batch-pages` | Prefix pages per `batch_get_into` of a prefetch; the fetch stops at the first batch with a missing page (the reference engine uses 128). |

`--kvstore-storage-backend memory` is an in-process dict for tests only.
CI exercises that Mooncake-compatible contract end-to-end (scheduler
prefetch after `register_storage_keys` / Host eviction, and a CUDA
D2H → store → Host wipe → prefetch → H2D round trip). A separate
ubuntu job boots `mooncake_master` and runs
`test/test_l3_mooncake_master.py` against the real TCP client
(`P2PHANDSHAKE`). Reuse an already-running master with
`MOONCAKE_MASTER=host:port`.
Mooncake Store is the offload backend. PD KV transfer still uses the
separate Mooncake TransferEngine (`--disaggregation-transfer-backend`).
