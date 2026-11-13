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

## Sparse KV offloading

`--kv-offload-config` accepts a JSON object with four required fields:
`layers` (unique target attention layer IDs), `hot_tokens` (a positive power of
two), `host_gb` (a finite positive budget in GiB), and `overlap` (an explicit
boolean enabling cross-layer prefetch). Omitting the argument disables offload.
The model cache recipe validates supported fields, selection capacity, and
execution constraints; unsupported recipes reject the configuration.

Model-specific behavior stays in `--hf-overrides`. For a plugin supporting a
window ring, set its `kv_offload_window_ring` option explicitly. Do not put
`window_ring` in the runtime object or use the legacy `hf_overrides.kv_offload`
entry; both are rejected. This option does not enable MTP, CUDA Graph, scheduler
overlap, or change numerical settings implicitly.

Offloading uses capacity aborts instead of snapshot recovery. Its snapshot
pool is disabled automatically; positive snapshot sizes, snapshot request
caps, and `--debug-force-retraction-interval` are rejected. Under decode
memory pressure the scheduler releases the victim's history blocks and
request slot and reports `CapacityAbort` to the client. A client that wants
to continue must submit a new request. The fixed hot allocation persists;
its request partition is invalidated before another PD owner uses it.

## Precision and quantization

| Parameter | Purpose |
| --- | --- |
| `--dtype` | Model weight and activation dtype. `auto` follows model metadata. |
| `--kv-cache-dtype` | KV cache dtype. Lower precision reduces KV memory. |
| `--kv-cache-quant-method` | KV cache quantization method. |
| `--mamba-ssm-dtype` | Stored linear-attention recurrent state dtype (GDN and KDA): `float32` (default) or `bfloat16`. Scans still accumulate in FP32; `bfloat16` halves state memory. |
| `--quantization` | Weight quantization mode such as `fp8`, `nvfp4`, `w8a8_fp8`, or `compressed-tensors`. |
| `--quantization-param-path` | JSON file of FP8 KV cache scaling factors, read only under an FP8 KV cache. KV caches run unscaled, so every factor must be 1.0, as must any KV-cache scale the checkpoint carries. |

## Numerics

`--numerics` names the numerical contract a deployment promises. The design
and the per-model verification gate are in `docs/design/numerics.md`. Every
switch an envelope folds is also available individually under `auto`.

| Parameter | Purpose |
| --- | --- |
| `--numerics {auto,rl-bitwise}` | `auto` uses each option's default. `rl-bitwise` folds the determinism switches (`--batch-invariant-collectives`, `--disable-autotune`, `--disable-tf32`, `--disable-pdl`, no fused all-reduce, batch-invariant MoE planning, `--sampling-stream per-request`, `--dsa-slot-order selection`) so tokens and logprobs are bitwise identical across runs and batch compositions, and the trainer-operation-order switches below so a teacher-forced pass reproduces the RL trainer's logprobs. Needs MoE TP 1 and a vocabulary that is a multiple of 32768; a model serves it only once its profile lists it. |
| `--disable-autotune` | Disable startup FlashInfer tactic profiling. Matching caches still load; CuTe warmup continues except under pipeline parallelism. PrimTS skips extra group enumeration. |
| `--batch-invariant-collectives` | One association order per reduction, independent of the batch: every reduction runs as NCCL data movement plus a fixed-rank-order fp32 fold (an all-reduce costs world_size x the traffic); gathers keep the multicast kernels. Folded in by `rl-bitwise`. |
| `--force-deterministic-rsag` | NCCL and the fold only: no symmetric-memory path (multicast gathers, the trtllm/Triton all-reduce tiers, distributed argmax). Not folded in by `rl-bitwise`, under which it only moves the gathers to NCCL. |
| `--sampling-stream {batch,per-request}` | `per-request` draws every non-greedy row from a stream keyed by the request's seed and position only, so a request samples the same tokens alone and inside any batch. Folded in by `rl-bitwise`. |
| `--yarn-ramp-mask-device {cuda,cpu}` | Device that computes the `deepseek_yarn` linear ramp mask. Defaults to `cuda`, independently of the frequency device. Folded to `cpu` by `rl-bitwise`. |
| `--yarn-frequency-device {cuda,cpu}` | Device that computes YaRN powers, divisions and interpolation. Defaults to `cuda`, independently of the ramp device. Folded to `cuda` by `rl-bitwise`, overriding an explicit `cpu`; paired with its CPU ramp. |
| `--mla-lora-scale {folded,runtime}` | Where LongCat-style MLA applies its `sqrt(hidden / lora_rank)` norm scales: folded into the norm weights at load, or multiplied at runtime after `q_b_proj` / `kv_a_layernorm` as the trainer does. Folded to `runtime` by `rl-bitwise`. |
| `--layer-boundary-norm {fused,unfused}` | `unfused` materializes `hidden + residual` in bf16 before the norm that opens each physical layer and before the final norm, instead of the fused add+norm kernel; also vetoes all-reduce+norm fusion. Folded to `unfused` by `rl-bitwise`. |
| `--router-topk {fused,torch}` | Correction-bias MoE routing: the fused CUDA kernel, or fp32 `torch.softmax` + `torch.topk(probs + bias)` with PyTorch tie order and `-1` zero-expert ids. Folded to `torch` by `rl-bitwise`. |
| `--logprob-order {torch,blockwise,blockwise-direct,blockwise-16384,blockwise-direct-16384}` | Order of the selected-token log-softmax: `torch.log_softmax`, or fixed-width blockwise reduction over 32768-wide vocab blocks; output and prompt (input) logprobs share it. `blockwise` returns negative cross-entropy; `blockwise-direct` uses direct subtraction. The `-16384` variants select 16384-wide blocks independently of serving TP. Changes logprobs only. `rl-bitwise` folds to `blockwise`, overriding explicit `torch` and preserving an explicit blockwise form and block width. |
| `--dsa-slot-order {selection,sorted}` | The order a token's selected KV rows are reduced in by the sparse (DSA) attention: as the top-k leaf emitted them (`selection`), or in ascending position order (`sorted`: one reduction order per selected set, invariant across batch compositions, runs and engines -- not ascending slot order, which follows the page placement). The top-k leaf emits the order and the core keeps it, so both must declare the `slot_order` trait (registered batch-invariant leaves); a silent kernel is refused under `sorted`. Defaults to `selection`; `rl-bitwise` fixes it to `selection`, overriding an explicit `sorted`. Both top-k and attention leaves must still be batch-invariant under `rl-bitwise`. |
| `--moe-combine-order {rank,slot}` | How a token's routed-expert contributions meet across the MoE TP-EP group. `rank`: the MoE kernel returns this rank's partial and the host sums the partials, adding LongCat's identity zero-expert residual once around the reduction. `slot`: the MoE kernel folds the token's top-k slots in fp32 slot order across the EP group itself, residual included, as the trainer's grouped MLP does, and the host reduces nothing; needs MoE TP 1 and a kernel declaring `combine_order` with `slot` (a registered batch-invariant leaf), and vetoes all-reduce+norm fusion. Folded to `slot` by `rl-bitwise`. |

## API surface

| Parameter | Purpose |
| --- | --- |
| `--host` | HTTP bind host. |
| `--port` | HTTP bind port. |
| `--served-model-name` | The model's name in metric labels. Under `ts serve`, requests may also name the model by it; the OpenAI-compatible API lists the model path. |
| `--api-key` | SMG gateway API key for authorization with upstream workers. |
| `--chat-template` | Built-in chat template name or template file path (handled by the smg gateway). |
| `--stream-interval` | Streaming buffer interval in generated tokens. Smaller values stream more frequently. |
| `--weight-version` | Initial model-weight version, the L3 cache's namespace (see [Weight versions](#weight-versions)). Defaults to `default`. |
| `--model-update-config` | JSON object handed to the Model Updater SDK for the Mooncake weight update. Requires the three flags below; see [Mooncake weight updates](#mooncake-weight-updates). |
| `--model-update-sdk-module` | Import path of the Model Updater SDK module. Imported in the scheduler process on the first Mooncake update, not at startup. Required with `--model-update-config`. |
| `--model-update-engine-type` | SDK `EngineType` member name for this engine, resolved as `EngineType[value.upper()]`. Required with `--model-update-config`. |
| `--model-update-draft-weights` | `retain` or `refresh`: whether a Mooncake update also streams the speculative draft model's weights. Required with `--model-update-config`. |

### Weight versions

The weight loads (see [RL weight updates](#rl-weight-updates)) take an
optional `weight_version`. With L3 storage on, it names the namespace that new
pages publish under (`--weight-version` names the first one), and it changes
only after the load succeeds on every rank. A new `weight_version` requires
`flush_cache=True`, and a flushed load must name its version, so independent
checkpoints cannot share a namespace. Intermediate loads pass `None` to keep
the current one until the last load flushes. Without L3 storage the engine
keeps no version.

### Logprobs

| Parameter | Purpose |
| --- | --- |
| `--enable-output-logprobs` | Gate for every logprob request. Off by default; the sampler gathers logprobs only when on, and a request asking for them on a server without the flag is rejected. |
| `--input-logprob-chunk-tokens` | Prompt rows pushed through the LM head per chunk when a request asks for prompt (input) logprobs. Defaults to `256`. A sizing knob only: the transient per-chunk cost is about `chunk × vocab × (2 + 2 + 4)` bytes (the bf16 logits shard, the bf16 TP-gathered logits, the fp32 log-softmax), so the default costs about 256 MiB at a 128K vocabulary, and the value never changes a result (log-softmax is row-local). |

A request asks for logprobs with `return_logprob`. Its output frames carry
the sampled tokens' logprobs with their ids (`output_token_logprobs_val` and
`_idx`) and, with `logprob_start_len`, prompt (input) logprobs
(`input_token_logprobs_val` and `_idx`):

- `logprob_start_len=-1` (the default) returns the single entry
  `(null, input_ids[-1])` and computes nothing extra.
- `logprob_start_len=s` with `0 <= s < len(input_ids)` returns
  `len(input_ids) - s` entries: `(null, input_ids[s])` first, then the
  logprob of each following prompt token given its prefix. `s >= len(input_ids)`
  is a 400.

`ts serve`'s gateway (SMG's direct lane) carries the sampled tokens'
logprobs, the prompt logprobs and the top candidates the engine computes;
which of its routes renders them is the gateway's (`/generate` returns the
prompt logprobs, chat completions the candidates).

The engine computes prompt logprobs with the same fp32 log-softmax as the
output logprobs (see `docs/design/numerics.md`), accumulates them across
chunked-prefill chunks, and ships them once, on the first frame after the
prompt finished. A request that returns them from position `s` skips the
prefix cache for positions `>= s`, so the engine recomputes those positions
and always has logits for them. Positions before `s` reuse the cache as
usual. Mixed prefill/decode batches (`--enable-mixed-batch`) are supported
where the selected parallelism configuration permits them.
A NaN or infinite prompt logprob terminates the request with a
`NumericalError` exactly like NaN logits on a sampled token
(`--enable-nan-detection`). The response never carries a non-finite value.

The ingress refuses with a 400 any request with `logprob_start_len >= 0`
that asks for at least one prompt logprob when the engine cannot score every
prompt position: models that narrow their prefill rows (DeepSeek V4.1's CED
decoder keeps only each prompt's last window for the LM head). The data plane
reports this capability at startup and the host's ingress checks it before
admitting the request, so the data plane never has to. The ingress also
refuses multimodal prompts (their media positions carry content-hash ids,
not tokens) and any prompt whose client-supplied `input_ids` fall outside
the vocabulary. `logprob_start_len=-1` is always accepted.

Under pipeline parallelism (`--pipeline-parallel-size > 1`) the last stage scores the prompt
rows and the commit path carries both logprob vectors to the other stages
with the sampled tokens. Under query context parallelism
(`--query-context-parallel-size N`) the prompt rows of a chunk live on the
rank whose shard holds them. Because the LM head is vocab-sharded over the
same ranks, the ranks gather the planned rows' activations to the group and
every rank scores the whole plan, so each rank's
`--input-logprob-chunk-tokens` chunks cover the chunk's planned rows exactly
as without sharding (the per-chunk transient of the row above is the same),
and the result is identical on every rank. The ranks gather the sampled rows
only after scoring the prompt rows.

In a disaggregated deployment the prefill node and the decode node each
return their own frames, exactly as SGLang's do: the prefill node's finished
frame carries `input_token_logprobs` (and the bootstrap token under
`output_token_logprobs`), while the decode node's frames carry
`output_token_logprobs` only and never `input_token_logprobs`. The prefill
node forwards the bootstrap token's logprob to the decode node, so the decode
node's `output_token_logprobs` covers every generated token. Merging the two
into one response is the router's job: it joins the prefill node's prompt
logprobs with the decode node's output, just as it does for SGLang.

`top_logprobs_num=k` (1 to 20) adds each position's `k` most likely tokens,
most likely first, as `(logprob, token_id)` pairs: `input_top_logprobs`
beside the prompt logprobs (`null` for the first position, like its value)
and `output_top_logprobs` beside the sampled token. They come from the same fp32 log-softmax as the
values. The engine computes them where the forward runs eagerly, so the
ingress takes `top_logprobs_num` only for a request whose every token is
sampled in its prefill (`max_new_tokens` 0 or 1, as an `echo` scoring request
sends) and with `--logprob-order torch`; a longer generation, pipeline
parallelism and disaggregated prefill and decode get a 400.
`token_ids_logprob` is not supported yet.

This runtime requires a `tokenspeed-scheduler` build that has
`RequestSpec.max_cached_prefix_tokens` (the admission-probe bound; see
`docs/design/scheduler.md` §1): every admission sets it, so an older scheduler
fails at the first request, not only at the first prompt-logprob request.

### RL weight updates

An RL trainer's operations are control operations on the engine's wire
(`zmq_wire.CONTROL_REQUESTS`). The frontend that drives the engine sends each
to every engine of the replica and gets every engine's reply, through the
engine's client library (`ts-client`, or its Python binding,
`tokenspeed_engine.FrontendListener`):

- update coordination: `PauseSchedulerReqInput`, `ResumeSchedulerReqInput`,
  and `FlushCacheReqInput`;
- NCCL weight sync: `InitWeightsUpdateGroupReqInput`,
  `UpdateWeightsFromDistributedReqInput`, and
  `DestroyWeightsUpdateGroupReqInput`;
- checkpoint loads: `UpdateWeightsFromMooncakeReqInput` and
  `UpdateWeightFromDiskReqInput`;
- memory control: `ReleaseMemoryOccupationReqInput` and
  `ResumeMemoryOccupationReqInput`.

TokenSpeed serves no HTTP route for them. `ts serve`'s gateway (SMG's direct
lane) serves them under its `/v1/rl` routes (`--enable-rl`), with the
SGLang-compatible route names and bodies: each POST route is a control call
of the same name on every engine of the worker, its JSON body the call's
argument, read with the route's field names and defaults
(`ts-control/src/calls.rs`): `pause_generation`, `continue_generation`,
`init_weights_update_group`, `destroy_weights_update_group`,
`update_weights_from_distributed`, `update_weights_from_disk`,
`update_weights_from_mooncake`, `release_memory_occupation`,
`resume_memory_occupation`, `rebalance_experts` and `flush_cache`. The GET
routes, such as `get_weight_version`, have no call, and a response's
`weight_version` stays the one the engine registered with. Attention data
parallelism of any size is
supported: the scheduler completes an operation only in a round where every
DP rank holds it (see
[Weight updates under attention DP](#weight-updates-under-attention-dp)).
Real rollout log probabilities require `--enable-output-logprobs`.

#### Releasing GPU memory (`--enable-memory-saver`)

Set `--weight-memory-policy cpu_backup|discard` explicitly whenever
`--enable-memory-saver` is enabled. The host Python package, native host and
kernel package must be built from matching sources; an older native host is
rejected before device initialization. See [weight memory policies](../serving/weight-memory.md)
for the update contract and validation procedure.

`ReleaseMemoryOccupationReqInput` frees GPU memory between rollouts so a
trainer colocated on the same GPUs can use it;
`ResumeMemoryOccupationReqInput` maps it back. Both take optional `tags`; a
release without tags frees every region the engine has, a resume without tags
restores whatever is released:

| Tag | Memory | On resume |
| --- | --- | --- |
| `kv_cache` | the cache arena's device regions | zeroed; a PD engine registers it with the transfer engine again |
| `weights` | model and draft parameters | CPU backup, or remapped and invalid until a complete update; frozen draft allocations retain CPU backup |
| `cuda_graph` | the decode CUDA graphs' memory pool | discarded; replays rewrite it |
| `nccl` | NCCL communicator buffers (`ProcessGroupNCCL.suspend`); requires a PyTorch build with suspend support and NCCL 2.29.7 or later | resumed in place |

A release waits until the scheduler is drained, and on a disaggregated
(`--disaggregation-mode prefill|decode`) engine also until no request is
bootstrapping, no pages are pinned for a transfer and the transfer peer holds
no sender or receiver. While a release is draining or memory is released, a
disaggregated engine finishes new requests with an abort. Weight updates are
refused while `weights` or `nccl` is released, so the RL order is: release
everything, train, resume `["nccl", "cuda_graph", "weights"]` (without `nccl`
on an older NCCL), push the new
weights (`UpdateWeightsFromMooncakeReqInput`), resume `["kv_cache"]`. With `discard`,
KV must be released before the weight update and cannot resume until every rank
has loaded and committed the complete version. Under either policy, a weight
load that fails after it started requires a restart; an update that any rank
refuses before loading runs on no rank and can be retried. A bare resume of all
tags immediately after discard is therefore invalid. Sparse KV
offload keeps its pinned Host history mapped across a release and zeroes it on
resume. Other device memory (input buffers, communication workspaces such as
symmetric memory or DeepEP buffers, the prefill CUDA graphs) stays resident.

With attention DP, every rank must stop admissions and drain before any rank
releases memory. Send the release to every DP engine concurrently; a release
itself stops admissions, so a separate pause is unnecessary. A peer explicitly
paused with `wait` or `abort` may also satisfy the drain. An idle engine that
has not paused prevents release. After a partial or staggered resume, no DP
rank forwards until every region released on every rank has been restored.

Queued weight operations finish before a release frees memory. If a PD engine
is paused with buffered requests, release is refused: resume and drain those
requests first. While any region is released, a pause is refused; resume the
memory first.

With memory saver enabled, startup skips the disposable CUDA-graph memory
probe and captures the serving graphs once. The installed memory saver cannot
destroy a tagged probe pool and then capture again. Graph memory is therefore
not reserved by that probe; leave headroom in `--gpu-memory-utilization`.

`UpdateWeightFromDiskReqInput` reloads the weights in place from a checkpoint
directory (`model_path`, and `load_format`, which defaults to the server's).
It takes the same scheduler path as the distributed and
Mooncake updates: the `weight_version` checks, the attention-DP same-round gate,
a cache flush before the load, and the replica-wide result. The reply counts
the checkpoint tensors actually read, and a checkpoint that yields none is a
failure. Load formats that do not read checkpoint files (for example `dummy`)
are refused. The checkpoint is streamed the way the startup load streams it,
so a model that reads tables from its checkpoint directory (DeepSeek-V4.1's
Engram tables) reads them from the new one.

The request is refused, with nothing written, when the startup loader
transformed the model's weights after loading them, because a checkpoint holds
those weights untransformed: quantized checkpoints (FP8, MXFP4, NVFP4,
compressed-tensors), and MoE layers whose kernel repacks the experts, which
includes the FlashInfer TRT-LLM and CUTLASS kernels for unquantized experts.
Restart the engine with the new checkpoint instead. Unquantized dense models,
and unquantized MoE on a kernel that keeps the checkpoint layout (the Triton
kernel, or the Gluon kernel on gfx950), reload in place.

For a safetensors checkpoint the shard headers are read before the first
write. A shard shorter than its header declares (for example, one still being
written) or a tensor whose shape differs from the same name in the checkpoint
the model holds fails the request with the weights unchanged. A failure after
the load has started says that the worker's weights may now mix the two
checkpoints: reload a complete checkpoint or restart the engine before relying
on its output. The weight version does not advance on any failure.

`UpdateWeightsFromTensorReqInput` (colocated CUDA-IPC updates) fails before
anything reaches the scheduler: TokenSpeed's scheduler does not implement the
CUDA-IPC receive path. Use the distributed or disk update, or the Mooncake
update described below.

### Weight updates under attention DP

With `--data-parallel-size > 1`, the frontend sends every weight operation
to every attention-DP engine and combines their replies.
Each scheduler queues the op and completes it only in a round where every DP
rank holds the same kind of op at the head of its queue, decided on the
per-round DP all-reduce that already carries flush intent. One op completes
per round. TokenSpeed then MIN-reduces the device result across the replica
before it publishes the L3 weight version, so a failure on one rank fails the
update everywhere. A rank whose peer never receives the op waits
indefinitely, as with a cache flush. The design rationale is in
`docs/design/event-loop.md`.

The engine does not wait for running requests before a load: a request that
runs across one generates under both weights. Pause first (see the
trainer-side contract below) when that matters.

### Mooncake weight updates

Install a Model Updater SDK provider in every scheduler environment and select
its import path with `--model-update-sdk-module`. TokenSpeed supplies the update
session and control-plane coordination; the provider owns store access, model
support and publication metadata. The standalone WeightStore receiver is supplied
by an external plugin rather than bundled with the runtime. Follow that
provider's documentation for its module path, dependencies and configuration.

`UpdateWeightsFromMooncakeReqInput` loads one committed checkpoint version
that the RL trainer published to a Mooncake weight store through the Model
Updater SDK. Fields: `version` (an integer), `flush_cache` (default true),
`weight_version` (optional), and `snapshot` (optional SDK publication metadata
passed unchanged to the provider). Every scheduler process reads its own shard
with its global rank as the SDK reader rank, on the forward thread, ordered
against forwards like the distributed update. The reply arrives only after
every worker finished its read.

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

- Pause dispatch at the router, or the engine (`PauseSchedulerReqInput`),
  before calling and resume after the reply: requests that run across the
  load generate under both weights. The scheduler's control thread blocks
  for the duration of the SDK read, so load reporting, PD transfer polling,
  and health responses stall on every worker.
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

## Scheduler and memory

| Parameter | Purpose |
| --- | --- |
| `--max-model-len` | Maximum sequence length. If omitted, TokenSpeed uses the model config. |
| `--gpu-memory-utilization` | Fraction of GPU memory used for model weights and KV cache. Lower it to leave headroom. |
| `--max-num-seqs` | Maximum number of active sequences the scheduler may process concurrently. |
| `--chunked-prefill-size` | Token budget the scheduler may issue in one iteration; it also bounds the multimodal placeholder tokens one encoder call produces (an item larger than that runs alone). Defaults to `8192`. Set `-1` to disable chunked prefill. |
| `--prefill-chunk-yield-tokens` | Prefill servers: the room, in tokens, a prefill chunk that cannot finish its prompt makes for bootstrapped prompts waiting behind it that fit whole (counted past their prefix hit), so they run complete in the same forward instead of waiting for the long prompt's chunk to pass. The chunk ends on an earlier page and gives up fewer than N plus one page, only while such prompts wait and only what they take, so the long prompt keeps about half of every round or more. `0` (default) is off; at most half of `--chunked-prefill-size`; a decode server ignores it ([scheduler §1.1](../design/scheduler.md#11-head-of-line-an-incomplete-prefill-holds-the-queue)). |
| `--max-prefill-tokens` | Prefill token budget used when chunked prefill is disabled. Defaults to `8192`. |
| `--enable-short-prefill-first` | Prefill and aggregated servers: rank new local prefills by remaining work after Device/Host prefix reuse, including replay work, with stable ties and oldest-request protection after eight successful bypasses. Resident chunks keep priority. Off by default, preserving submission order ([scheduler §1.1](../design/scheduler.md#11-head-of-line-an-incomplete-prefill-holds-the-queue)). |
| `--prefill-decode-interval` | After a round that runs a prefill chunk, run this many decode rounds before the next prefill round, unless no decode can run. Every chunk starts a new interval, but no prompt is held back more than this many decode rounds in total, so TTFT grows by at most this many decode steps and running decodes stall less often. Aggregated serving only. `0` (default) is off; nonzero values are mutually exclusive with the [DP prefill gate](../design/prefill-delay.md) ([scheduler §3.3](../design/scheduler.md#33-fused--one-engine-everything-local)). |
| `--prefill-attempt-interval` | ALL attempt interval, in scheduling rounds, for the [DP prefill gate](../design/prefill-delay.md): every rank with decode work must also have eligible prefill work to open the window. Set together with `--prefill-any-attempt-interval`, with `1 <= ALL <= ANY`; leaving both unset (default) disables the gate. Aggregated (fused) serving only (`--disaggregation-mode null`); mutually exclusive with nonzero `--prefill-decode-interval`. |
| `--prefill-any-attempt-interval` | ANY attempt interval, in scheduling rounds, for the [DP prefill gate](../design/prefill-delay.md): one eligible rank is enough to open the window. Must be set together with `--prefill-attempt-interval` and be at least its value (`1 <= ALL <= ANY`). This is an admission opportunity, not a hard TTFT deadline. Aggregated (fused) serving only (`--disaggregation-mode null`); mutually exclusive with nonzero `--prefill-decode-interval`. |
| `--max-total-tokens` | Override the automatically calculated token pool size. |
| `--enable-state-checkpoint-in-forward` | Hybrid (recurrent-state) models: the prefill forward that spans the last prefix-page boundary before a long prompt's replay tail (normally the final token) also writes the recurrent state there, so a later request that resends the prompt or diverges from it after that boundary resumes there. No extra prefill round; that forward holds one more state block. The recurrent-state checkpoints, and their KV tails, that a next turn re-rendering the end of the previous prompt and its reply cannot resume at are then kept out of the prefix cache. On prefill and fused servers this is the prompt end when it lies inside the page that boundary starts and the request did not resume past it; a prompt end published at the request's finish, as on a prefill server, is kept if the state at the boundary has left the cache by then. On fused servers it is also every finish past the prompt end, including that of a prompt too short to have such a boundary. A next turn that repeats the previous reply verbatim therefore resumes at an earlier checkpoint and recomputes the reply. Requires prefix caching; a decode server ignores it; startup refuses it unless every State group is GDN (FlashInfer prefill, a prefix granularity that is a multiple of its 64-token scan chunk) or Qwen4-Exp PLE. With prefill graphs, the GDN layers of a forward that writes one stay captured; every captured shape shares one float32 scan scratch of (largest prefill-graph token capacity / prefix granularity) recurrent states. When a request publishes an in-forward checkpoint past the one it resumed from, which its own predecessor wrote past an earlier one, that earlier checkpoint -- two turns back on the conversation -- is evicted early from the device and host caches, and no new store or eviction write-back copies it to host, unless another request also resumed from it, or acquired or republished it after the conversation's previous turn resumed there, so multi-turn traffic keeps more live prefixes cached. On a prefill server, a prompt too short for the in-forward checkpoint (by default, one no longer than the chunk budget) whose end lies inside a prefix page also leaves only its full pages in the prefix cache, with no recurrent state and no KV tail at its end; a longer prompt that repeats it whole resumes at an earlier checkpoint instead, a page-aligned prompt end keeps its state, and a fused server keeps short prompt ends. Off by default ([scheduler §1.2](../design/scheduler.md#12-state-groups-two-slots-natural-checkpoints)). |
| `--enable-prefix-caching` / `--disable-prefix-caching` | Enable or disable prefix cache reuse. |
| `--enforce-eager` | Disable device-graph execution (CUDA Graph on CUDA, ACL Graph on NPU). |
| `--disable-prefill-graph` | Keep prefill eager while leaving decode device graphs enabled. |
| `--disable-kda-prefill-graph` | Disable KDA prefill CUDA graphs while retaining ordinary prefill and decode graph settings. Enabled by default for supported `cutedsl_kda` prefill attention when prefill graphs are enabled. |
| `--disable-cudagraph-memory-reserve` | Size the KV cache from free memory instead of reserving what the device graphs will cost and, on CUDA, what startup keeps resident. |
| `--max-cudagraph-capture-size` | Largest decode batch size to capture as a device graph. |
| `--cudagraph-capture-sizes` | Explicit decode batch sizes to capture as device graphs. |
| `--prefill-graph-capture-token-sizes` | Total input-token capacities per forward, summed across the batch. Shorter inputs are padded. |
| `--prefill-graph-capture-batch-sizes` | Request capacities for inline recurrent-layer (KDA, Mamba2, GDN) prefill capture. Replay selects the smallest compatible capacity that fits the batch. With `--enable-drafter-prefill-graph` these request counts also select the drafter's step-0 and decode-step graphs, which replay only at the exact count (unset: one request per bucket, or the fewest that fit a bucket longer than the context length). |
| `--enable-drafter-prefill-graph` | Also replay the speculative drafter's work inside a prefill forward from CUDA graphs: its extend step per prefill token bucket at each exact request count of `--prefill-graph-capture-batch-sizes` (unset: the fewest requests the bucket fits in the context length, one whenever the context length is at least the bucket), its decode steps per request count, attention at eager breaks. Only pure-extend forwards at a captured request count replay; mixed forwards and other request counts draft eagerly. Needs the prefill graph on one rank (no TP, EP, PP or attention DP) and a chain EAGLE-style drafter (no `--speculative-eagle-topk` > 1, no `--enable-speculative-sampling`) whose draft model declares it (Qwen4-Exp MTP without `index_share_for_mtp_iteration`) and reads no request-token history; startup refuses other launches. The graphs' memory is part of the CUDA-graph reserve; their static input buffers (the target's hidden rows at the largest bucket) come out of the `--gpu-memory-utilization` headroom. Off by default ([design](../design/unified_path.md#drafter-prefill-graphs)). |
| `--retraction-snapshot-host-gb` | Explicit size, in gigabytes, of the per-rank pinned Host pool that holds a retraction image's tail (a retracted request's blocks of groups Host L2 never holds, a state group's unpublished input block, and its slot state; the published blocks, the partial last page and the live state block included, go to Host L2 and stay pinned there until the request is restored; with `--disable-kvstore` the pool holds whole images). Wins over `--retraction-snapshot-ratio` when set; `0` (the default) means not set, like `--kvstore-size` against `--kvstore-ratio`. A request a capacity retraction suspends is imaged here and resumes exactly where it stopped once the image is copied back; nothing is recomputed. When no resident's image fits -- the pool or its rows are exhausted -- the scheduler aborts the newest resident instead (finish `err_type` 524, see "Retraction snapshot pool" below), so the pool bounds how much can be suspended at once, never how long an admission waits. Fused and decode roles only; the prefill and encode roles never retract and ignore it with a log. Allowed under `--kv-parallel-size > 1`. |
| `--retraction-snapshot-ratio` | Size of the retraction snapshot pool as a multiple of this rank's Device KV capacity (the base `--kvstore-ratio` scales too), used when `--retraction-snapshot-host-gb` is not set. Unset (the default) derives the pool at device build: 0.1 of the Device KV without the Host KVStore (whole images, kept cheap to pin; `1` makes every resident suspendable), else the image tails of `--retraction-snapshot-max-requests` requests (table below). An explicit `0` is the one way to run **without a pool**: nothing can be imaged, so every capacity block aborts its victim (the newest resident). |
| `--retraction-snapshot-max-requests` | Slot-state image rows of the retraction snapshot pool, that is, the most requests suspended at once, and the request count the derived pool size is multiplied by. `0` (the default) derives to `--max-num-seqs / attention-DP size`, which covers every resident request of the rank; refused with `--retraction-snapshot-ratio 0`, which has no rows to size. When no row is free the scheduler aborts a resident rather than waiting. |
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
| `--kv-parallel-size` | KV parallelism (KVP): shard full-history KV pages (MLA/DSA latent and index-K, DeepSeek V4.1 global KV) cyclically over a consecutive subgroup of attention TP; must divide `--attn-tp-size`. Each rank then stores one shard of every request's pages, so the KV capacity per GPU grows by that factor and the DSA indexer scores only owned pages. Aggregated and PD decode engines require matching `--decode-context-parallel-size`. PD sides shard their caches independently (a transfer edge copies only the pages both its prefill rank and decode rank own); the prefill role writes them with or without `--query-context-parallel-size`. GPU MLA/DSA speculative decoding is supported within each backend's device, dtype and draft-attention capabilities: the recipe shards the draft group like the target's, FlashMLA verification uses per-query local causal lengths and GPU DSA uses causal Top-K selection; non-causal DSA block drafts remain unsupported. The Host KVStore, retraction snapshot pool and L3 storage (`--kvstore-storage-backend`) use the same cyclic owner translation. L3 keys name a payload replica so an allocation may restore on a different owner; startup rejects head-partitioned fields in page-sharded groups. For Device/Host copies, the scheduler allocates every Host block in its Device block's residue class and each rank copies the blocks it owns. |
| `--decode-context-parallel-size` | Decode context parallelism (DCP) in aggregated or PD decode engines: every rank attends the KV pages it owns under `--kv-parallel-size` and the partials are merged across the KVP group by their log-sum-exp (`docs/serving/parallelism.md`). Requires `--disaggregation-mode null` (the default) or `decode` and must equal `--kv-parallel-size`, which requires it in turn on those roles: a sharded cache has no other decode, so the launch names the compute explicitly. Aggregated DCP rejects `--enable-mixed-batch`. The backends key the DCP path on the mapping's KVP size, which the drafter's decode-shaped steps on a sharded prefill engine take as well. 1 (default) is off. |
| `--attn-head-tp-size` | Shard the MLA head projections (`q_b_proj`, `kv_b_proj`, `o_proj`) by heads over this many contiguous ranks that hold different rows; the attention exchanges heads for tokens around its core. Over attention-DP ranks (needs attention TP 1, attention DP and `--disaggregation-mode decode`; each rank keeps its own KV) the layout serves decode rows only, so it sets `--disable-prefill-graph` and tunes on a decode step; it admits any `max_new_tokens`, because a retracted request resumes by restore rather than by a recovery prefill. Over the query shards of a prefill engine (must equal `--query-context-parallel-size`) the extend rows run the absorbed sparse prefill through the exchange and none of the decode-only rules apply. Defaults to the ranks holding the same rows: the attention TP size, or 1 (head-replicated) under `--query-context-parallel-size`. See [Parallelism](../serving/parallelism.md#decode-side-tp-layouts-under-attention-dp). |
| `--lm-head-tp-size` | Vocab-shard the LM head over this many contiguous ranks. Under attention DP the default 1 replicates it; a wider group gathers distinct DP groups' rows before the logits GEMM and transposes the vocabulary shards back. Attention TP greater than one is supported; LM-head and attention TP widths must tile each other. Without attention DP it must equal the attention TP size. Not combinable with `--dp-sampling`; under attention DP, requests asking for prompt logprobs (`logprob_start_len`) are refused. |
| `--tp-batch-invariant` | `none` (default), `attn`, or `attn+dense`: make the head-sharded `o_proj` and the dense `down_proj` column-parallel on hidden (all-gather of the reduction dim, full-K GEMM, all-to-all back to own rows) so no cross-rank sum remains outside MoE and the bits equal a TP1 full-K GEMM. `attn` needs `--attn-head-tp-size` > 1; `attn+dense` also needs a dense TP group wider than attention TP (so not under `--query-context-parallel-size`, whose dense group is 1 or the attention TP width; only `attn` applies there); both need unquantized `o_proj` / `down_proj`, judged on the checkpoint's resolved quantization (a quantized checkpoint passes when its `disable_quant_module` excludes `self_attn` and, for `attn+dense`, `dense_mlp` / `mlps`). |
| `--head-tp-exchange-max-rows` | Rows per rank the head-TP Lamport exchanges are prepared for (the absorbed query, the latent attention output, and, under `--tp-batch-invariant`, the `o_proj` rows). The symmetric-memory scratch is allocated before CUDA-graph capture; larger forwards fall back to the generic all-to-all. Only used with `--attn-head-tp-size` > 1. 256 (default). |
| `--query-context-parallel-size` | Query context parallelism (QCP) on the PD prefill role: shard every extend forward's rows over the attention TP group, each request split into `2N` near-equal segments with rank `r` computing segments `r` and `2N - 1 - r` against the gathered KV history (the KV write, index-K write, sampled rows and prompt-logprob rows are gathered across the group; the scheduler, cache allocation and PD transfer are unchanged, and `--chunked-prefill-size` keeps counting the whole chunk). Requires `--disaggregation-mode prefill` and must equal both `--attn-tp-size` and `--kv-parallel-size` (the shard is a layout over the sharded pages, so it inherits KVP's rules: Host KVStore, snapshot pool and L3 use the same owner translation); attention DP 1, no `--enable-mixed-batch`, `--dense-tp-size` and the MoE TP×EP group each 1 or the attention TP width; GPU DSA attention with a bf16 KV cache (the gathered write stores native latent rows), checked where the attention config is built. The prefill stays eager -- the GPU DSA backend declares no prefill graph under a shard -- without `--disable-prefill-graph` on the launch. The attention weights are head-replicated unless `--attn-head-tp-size` equals it, which shards them over the shard group. 1 (default) is off. |
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
| `--eplb-algorithm` | `auto` (default), `deepseek`, or `deepseek_hierarchical`. |
| `--expert-distribution-recorder-mode` | `stat` (the only mode): count the routes to every physical expert so the `EXPERT_LOAD` profile activity can write each rank's load record when its profile stops and `--enable-eplb` can rebalance from the counters. |
| `--enable-eplb` | Online expert rebalancing: every `--eplb-rebalance-num-iterations` forwards the routing load since the previous snapshot is rebalanced with the EPLB algorithm and the expert weights move between slots. Requires `--expert-distribution-recorder-mode stat` and a static `--ep-dispatch-algorithm`, both explicit, and `ep_size > 1`; `--ep-num-redundant-experts 0` is allowed (permutation only). The `RebalanceExpertsReqInput` control operation starts one rebalance now. Under a `--numerics` envelope it rides the placement-independent slot-order MoE combine the envelope folds in (`--moe-combine-order slot`). |
| `--eplb-rebalance-num-iterations` | Forwards between two load snapshots; required with `--enable-eplb`, `> 0`. |
| `--eplb-rebalance-layers-per-chunk` | MoE layers whose experts move in one scheduling round; required with `--enable-eplb`, `1..num MoE layers`. Fewer layers per chunk bound the per-round stall. |

All of these apply only to models that opt in to expert placement
(LongCat-Flash); other models refuse them at startup. See
[static expert placement](../serving/parallelism.md#static-expert-placement-with-redundant-experts)
for the record → place → route flow and the two dispatch flavors, and
[dynamic expert rebalancing](../serving/parallelism.md#dynamic-expert-rebalancing)
for the online variant.

## Backend selection

| Parameter | Purpose |
| --- | --- |
| `--attention-backend` | Attention kernel backend. Common values include `mha`, `fa3`, `fa4`, `triton`, `flashinfer`, `trtllm_mla`, and `tokenspeed_mla`. Names are checked against the backend registry at startup, after plugins load, so an installed plugin's backends are accepted too. On SM103 (B300/GB300), `fa4` ranks below every other attention kernel: naming it here or in `--drafter-attention-backend` selects it, and the default selection is unchanged. |
| `--drafter-attention-backend` | Attention backend for the speculative decoding drafter model; accepts the same names as `--attention-backend`. |
| `--enable-qsa-prefill-bf16-scratch` | Opt-in for QSA sparse-attention models (Qwen4-Exp) with `--kv-cache-dtype fp8_e4m3` (or `fp8`) on SM100 to SM107. A target prefill call of at least 2048 query rows copies the FP8 KV pages it can read into a BF16 scratch and attends there with the CuTe QSA kernel's BF16 path instead of the FP8 prefill kernel (FlashInfer FA2 by default, or the CuTe kernel's FP8 path under a kernel override); smaller or mixed calls, calls whose context exceeds the scratch or 128 cache tokens per query row, decode, verify and drafts keep the FP8 path, and the cache stays FP8. The scratch holds one maximum-length context plus one `--chunked-prefill-size` chunk of pages, with that chunk's remapped slots and attention output; it is reserved from the KV cache budget before the pool is sized, so the pool shrinks by its size (logged at startup). Startup fails for a model without QSA layers, another KV cache dtype (`mxfp8` included), `--chunked-prefill-size` below 2048 or disabled, or a GPU without the CuTe QSA kernel. The decode role, which runs no prefill, reserves no scratch. Part of the L3 storage identity when set. Off by default. |
| `--qwen4-exp-flat-kv` | Experimental Qwen4-Exp FlatKV layout. Packs full-attention K/V fields into one plane and translates scheduler block IDs to contiguous atom-page IDs before attention, without a KV gather or attention-kernel changes. Requires PP=1 and uniform K/V head geometry; recurrent, PLE and QSA side caches keep their existing groups and plane policy. Off by default and included in the L3 storage identity when enabled. |
| `--moe-backend` | MoE backend. |
| `--moe-cutlass-fused-finalize` / `--no-moe-cutlass-fused-finalize` | GEMM2 atomic expert combine for unquantized FP16/BF16 MoE on `flashinfer_cutlass`, for both target and draft models. On by default and forced off by `--numerics rl-bitwise`. The fused reduction order is nondeterministic. |
| `--moe-mxfp4-fp8-activation` | Opt-in: run MXFP4 routed experts with FP8 activations. On Hopper this is the FlashInfer cutlass W4A8 MoE (faster than the default W4A16 kernel, a few percent of extra error on expert outputs). Applies to every MXFP4 expert layer, target and draft; startup fails where the selected MoE backend has no FP8-activation kernel for a layer, when a model's routed experts are not MXFP4, when the model pins another activation precision (Kimi-K3 on Hopper Marlin), or when the layer's SwiGLU is unclamped (the W4A8 FC2 scale relies on the clamp). |
| `--draft-moe-backend` | MoE backend for the speculative decoding draft model. |
| `--all2all-backend` | MoE all-to-all backend. |
| `--deepep-mode` | DeepEP mode: `auto`, `normal`, or `low_latency`. `auto` uses `low_latency` for decode batches and `normal` for prefill batches. `normal` reads its per-expert receive counts on the host, so it forces eager execution and disables CUDA graphs. |
| `--sampling-backend` | Sampling backend: `greedy`, `flashinfer`, `flashinfer_full`, `triton`, `triton_full`, or `sonic` (vendored [sonic-sampler](https://github.com/tachyontrails/sonic-sampler) kernels: one fused Triton pipeline per step; a request with no `top_k`, `top_p` or `min_p` samples its whole vocabulary (each vocabulary block keeps its best logit plus the request's Gumbel noise), the others select among the top-128 candidates; bf16 logits ranked at bf16 resolution, fp32 logits sampled exactly; under TP with a vocab-parallel LM head each rank selects its vocabulary shard's top-128 and only those are gathered, instead of whole logit rows; kernel configs from a table measured on Blackwell, no startup tuning; see the `--sampling-backend` help for its memory cost). |
| `--disable-deepseek-v41-token-blacklist` | Allow DeepSeek V4.1 to generate BOS, PAD, and image control/placeholder token IDs `0, 2, 129262..129264`; these are masked by default. |

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
| `--reasoning-parser` | Parser for extracting reasoning content from model outputs (handled by the smg gateway). The engine also uses its single-token block markers to emit cumulative `reasoning_tokens` in output batches, independently of `--grammar-backend`; parsers without known markers report zero. |
| `--tool-call-parser` | Parser for OpenAI-compatible tool-call payloads (handled by the smg gateway). |

Common reasoning parser values include `kimi_k25`, `base`, `qwen3`,
`deepseek_r1`, `deepseek_v31`, and `deepseek_v41`. Common tool-call parser values include
`kimik2`, `qwen`, `deepseek_v4`, `json`, and `passthrough`. The SMG gateway
validates the parser names, so use the values accepted by the bundled
`tokenspeed-smg` package.

When neither parser is named and the model family has no parser defaults,
`ts serve` passes `--reasoning-parser passthrough` to the gateway, unless the
model's `tokenizer_config.json` declares a `response_template` and no other
`--tokenizer` is given: `ts serve` then passes `--parser symphony`,
`--reasoning-parser hf`, and `--tool-call-parser hf` so the bundled
`tokenspeed-smg` gateway selects both parsers from that template. Naming any
parser option keeps its existing behavior.

## Speculative decoding

| Parameter | Purpose |
| --- | --- |
| `--speculative-config` | JSON speculative decoding configuration. |
| `--speculative-algorithm` | Speculative algorithm, such as `EAGLE3`, `EAGLE4`, `MTP`, `DFLASH`, `DSPARK`, or `DUMMY`. `DUMMY` generates deterministic candidates without a draft checkpoint; it tests target multi-token verification, not draft-model quality. Set `--speculative-num-draft-tokens` above 1 and do not pass a draft model path. |
| `--speculative-draft-model-path` | Draft model path or repo ID. |
| `--speculative-draft-model-revision` | Branch, tag, or commit of a draft in its own repo. Defaults to the repo's default branch. A draft inside the target checkpoint uses `--revision`. |
| `--speculative-draft-model-quantization` | Draft model quantization. Defaults to `unquant`. |
| `--speculative-num-steps` | Number of draft model steps. Defaults to `3`. |
| `--speculative-num-draft-tokens` | Number of draft tokens. Defaults to `--speculative-num-steps + 1`; required for draft trees. |
| `--synthetic-acceptance-length` | Benchmark-only mean emitted tokens per speculative verification, including the guaranteed target token. Unset disables it. |
| `--speculative-eagle-topk` | Children each draft node expands to per step. Defaults to `1` (a chain); above 1 the draft is a tree. |
| `--enable-speculative-sampling` | Draft-prob rejection sampling for the chain drafters, EAGLE-Harness trees and chains, and `DFLASH` with a DFlash2 draft (see below). Off by default. |
| `--speculative-rejection-sample-method` | Verify rule, named as vLLM's `rejection_sample_method`: `standard` (default, token-wise), `block` (block verification; see [Block verification](#block-verification-experimental)) or `synthetic` (set by `--synthetic-acceptance-length`). |
| `--spec-reject-draft-prob-threshold` | With `--enable-speculative-sampling`, recorded draft probabilities above this value mark a request with no proposal yet and always reject. Defaults to `2.0`; must lie within `[1.0, 2**20]`. No effect on `DFLASH`. |
| `--eagle3-layers-to-capture` | Target layers whose hidden states feed an `EAGLE3` or `EAGLE4` draft. A target whose forward returns no layer taps (for example Qwen3, Qwen3-MoE or DeepSeek V4.1) refuses these drafts and `DFLASH` at startup. |
| `--speculative-exec-mode` | `EAGLE4` drafting (see below): `-1` (default) static chains and trees, `0` EAGLE-Harness dynamic chains and trees, `1` the same for evaluation runs, `2` measures the EAGLE-Harness cost data. |
| `--speculative-data-path` | EAGLE-Harness run directory: the cost data exec modes 0 and 1 read, which exec mode 2 writes. |
| `--speculative-compute-intensity` | EAGLE-Harness policy tier: `high_intensity` (default) or `low_intensity` for trees, `chain_intensity` for chains (`--speculative-eagle-topk 1`). |
| `--speculative-chain-width-table` | EAGLE-Harness chains in exec modes 0 and 1: `MIN_BS:WIDTH,...` fixes the verify width of rounds with at least `MIN_BS` live requests. `off` (default) lets the confidence rule decide every round. |
| `--disable-replay-ssm` | Stage every verify position's recurrent state instead of replaying the accepted tokens. ReplaySSM is on by default for supported Qwen GDN and Nemotron-H Mamba2 targets; `--enable-replay-ssm` is accepted as a deprecated no-op. |

Prefer `--speculative-config` for recipe-style launches because it keeps method,
draft model, and token count together.

### Synthetic acceptance length

The greedy and sonic sampling backends accept it, and the flashinfer backend
only over sampled drafts (`--enable-speculative-sampling`). Speculative
decoding must be enabled, and `--dp-sampling`, `--enable-output-logprobs`,
and draft trees are rejected at startup, as are the other sampling backends.
On the greedy backend, token selection is always argmax, so stochastic sampling
and penalties do not apply there; use unconstrained greedy requests
(`temperature: 0`) with it.

Use `--synthetic-acceptance-length L` to benchmark speculative execution at a
fixed mean acceptance length (AL). AL includes the guaranteed target token,
so `1 <= L <= --speculative-num-draft-tokens` (the verify width). AL `1`
rejects every draft; AL equal to the verify width accepts every draft. A
fractional AL emits either `floor(L)` or `ceil(L)` tokens per verification,
with probability `L - floor(L)` of the larger length. For example, AL `2.6`
accepts one draft token on 40% of steps and two on 60%, plus a target token.

The draft and target models and the normal verification kernel still execute.
The verification result is then overridden with synthetic acceptance before
committing tokens and updating request state, preserving verification overhead.
The override copies the forced draft prefix and selects the existing target
argmax at the cutoff. It adds no second verification or target-resampling pass.
**Do not use generated text for correctness or accuracy evaluation.** Request
termination can truncate the last verification's output.

Either spelling below configures three draft slots (verify width four) and
AL `2.6`:

```bash
--sampling-backend greedy --speculative-algorithm MTP --speculative-num-steps 3 --synthetic-acceptance-length 2.6
--sampling-backend greedy --speculative-config '{"method":"mtp","num_speculative_tokens":3,"synthetic_acceptance_length":2.6}'
```

Conflicting values between the flag and JSON configuration are rejected.
The value must be finite and speculative decoding must be enabled.
It selects the `synthetic` rejection method. The greedy and sonic backends
accept it; the flashinfer backend only over sampled drafts
(`--enable-speculative-sampling`); see
[Block verification](#block-verification-experimental). On sonic the token at
the cutoff is the target's draw there, taken from the candidates the verify
kernel already selected: a greedy request's argmax, otherwise the draw sonic
makes past a fully accepted window, with the same per-request noise.
For block drafters with implicit widths, the upper bound is checked against
the checkpoint's resolved verify width when the sampler is initialized.

`EAGLE3` and `MTP` drafts are chains by default: `--speculative-num-draft-tokens`
must equal `--speculative-num-steps + 1`. With `--speculative-eagle-topk` above 1
they draft a tree instead, and `--speculative-num-draft-tokens` is its node
budget (root included). Set it explicitly: topk 1..16, steps 1..10,
`(steps - 1) * topk` lane slots within the node budget, and at most 64 nodes. Trees need the `trtllm`
attention backends (`trtllm_mla` or `tokenspeed_mla` for MLA models) and the `greedy`, `triton`, `flashinfer` or `sonic` sampling backend; see
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

### EAGLE-4

`EAGLE4` serves an EAGLE-4 draft, whose candidate head scores a short list of
candidate tokens per draft row instead of the whole vocabulary. With the
default `--speculative-exec-mode -1` it drafts static chains and trees under
the `EAGLE3` rules above.

The draft config selects its layer's structure: `attn_norm` merges the
embedding and hidden inputs through `eh_proj`, `use_rope: false` attends
without rotary embeddings, and `enable_moe` with `moe_architecture:
nemotron_latent_relu2` routes Nemotron-H's latent MoE (without expert
parallelism). Loading refuses a draft config key or checkpoint tensor the
model does not build.

With exec mode `0`, EAGLE-Harness sizes each round on the device from
measured cost data (or, for chains, from the width table where it applies):
how deep the draft grows and how many nodes the target verifies. Measure the cost data with the exec mode `2` driver
(`python -m tokenspeed.runtime.execution.drafter.eagle_harness.measure_run --help`),
then serve with `--speculative-data-path` set to its run directory and the
same draft shape, TP size, attention backends and state dtype. Exec mode
`1` serves evaluation clients that write their batch size to
`<data path>/config/benchmark_config.json`: it runs like exec mode `0`,
except that a tree round far smaller than that batch drafts only the root's
children and verifies narrowly. Harness rounds, chains included,
run on the draft-tree path and share its requirements, with these limits:

- Trees: `--speculative-eagle-topk` 2, 4 or 8 with `high_intensity` or
  `low_intensity`, at most 32 requests per rank.
- Chains: `--speculative-eagle-topk 1` with `chain_intensity`, at most 15
  steps and 16 draft tokens, and at most 512 requests per rank (256 when
  measuring). The
  confidence rule decides rounds of up to 256 requests (the cost data's
  range); `--speculative-chain-width-table` must cover larger rounds. To
  calibrate a table, `TOKENSPEED_EAGLE_HARNESS_ROUND_STATS=N` logs rank 0's
  round counts per batch size every `N` rounds.
- `--speculative-num-steps` 2 to 32 (64 when measuring), with
  `(steps - 1) * topk` at most 128 and `1 + topk + (steps - 1) * topk^2` at
  most 1024.
- `--enable-speculative-sampling` samples harness trees and chains (not
  when measuring) and needs `--sampling-backend flashinfer`, which accepts
  them by tree rejection sampling. Static `EAGLE4` drafts do not take it.
  Its sampling kernels are NVIDIA-only: elsewhere (AMD) harness rounds
  draft greedy trees and chains, and the drafter refuses sampling at
  startup.

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
rules. `top_k`, `top_p`, `min_p`, penalties, and `logit_bias` stay on the
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

Requirements: `--speculative-algorithm EAGLE3` or `MTP` (`DSPARK` proposes a
whole block greedily outside [Block verification](#block-verification-experimental);
`DFLASH` is covered below), `--speculative-eagle-topk 1`,
and `--sampling-backend flashinfer`, `flashinfer_full`, or `sonic` (`sonic`
reads `q` at its top-128 target candidates; `greedy` verifies by exact match,
the Triton backends by target-sampled exact match; neither reads the dense
`q`). The drafter switches its draft model's fused TP-sharded argmax off so
the full-vocab logits reach it, which adds the logits all-gather to every
draft step under tensor parallelism. With `sonic` sampling vocabulary shards
(the server logs `Sampled rows stay vocab-sharded`) and an MTP or EAGLE3 draft
head spanning the target vocabulary, the draft steps stay sharded too (`Draft
steps stay vocab-sharded`): each rank samples its shard of the draft logits,
the ranks exchange a few values per request instead of the logits, every rank
draws the same token, and each records `q` over its own vocabulary columns;
its verify candidates carry that `q` through their gather.

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

#### DFLASH with a DFlash2 draft

`DFLASH` takes the flag with a DFlash2 draft (a checkpoint with candidate
selector weights) and `--sampling-backend triton` only; `DFLASH` with any other
backend is refused at startup, and a base DFlash draft (no selector) is refused
when the server loads it. The DFlash2 walk picks one of the selector's `K`
candidates per draft step. Under the flag a sampling request draws that
candidate from `q = softmax(scores / T)` over the `K` edges out of the
previous pick, with noise keyed by the request's seed, the round and the drawn
token id, so every TP rank draws the same token. Greedy requests keep the best
edge. Requests with top-k disabled and `top_p < 1` also keep it (see the
routes below), as does a step whose scores are not finite. None of these
three records a proposal. The walk records `q` and the candidate ids per step,
which is `K` entries instead of a vocabulary row. Verify accepts draft `x` with
probability `min(1, p'(x) / q(x))` and on rejection samples from
`norm(max(p' - q, 0))`, where `p'` is the target distribution after the
request's temperature, top-k and top-p, the one the Triton backend serves.

- Memory: `q` reuses the draft-tree q tables (`RuntimeStates.future_draft_q`),
  `(max_num_seqs + 2) x (num_draft_tokens - 1) x K x 8` bytes plus two int32
  rows of `num_draft_tokens` per slot. That is under 1 MB at 512 slots, 8
  draft tokens and `K = 16`. No dense per-vocab rows, no gather buffer, and
  the draft head keeps its shard-local top-K.
- No proposal: a fresh admission, a PD landing and a forced single-token
  verify record no `q`. Their drafts are checked by the target-only rule,
  which keeps a draft exactly when it equals the target's own draw at that
  position. So a landed request's first verify can still accept drafts,
  unlike the step drafters' sentinel, which always rejects.
  `--spec-reject-draft-prob-threshold` has no effect here.
- Routes: the Triton backend picks one sampling route per batch. The
  rejection rule runs on batches with no filter (top-k disabled, `top_p = 1`)
  and on batches where every request has a finite top-k, with any `top_p`.
  Under the flag those top-k batches always use the two-stage top-k kernel,
  at every row count. Batches with top-p only (top-k disabled, `top_p < 1`)
  and mixed batches (for example greedy next to no-filter requests) have no
  exact `p'` and verify every row by the target-only rule. There a sampled
  draft is kept with probability `sum(p * q)`, which can be below the greedy
  draft's `p(argmax q)`. Keep sampling parameters uniform across a batch for
  the best acceptance.

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
step applies a request's `temperature`, `top_k`, and `top_p`, never the
proposal, so the served distribution is the target's whatever the drafter
proposed.
`DFlash2BeamDraftModel`, Beam's drafter, is a DFlash2 drafter as well.

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
target's page table: the prefix granularity is the target cache recipe's and
the draft follows it. Any sliding window the draft checkpoint declares is an
attention mask that the draft's own layers apply, never a cache-retention policy
of its own. Only the backends that forward that mask to their kernels can serve such a
draft: `mla` and `tokenspeed_mla` (`gluon` on AMD) for MLA drafts, and
`mha`/`fa3`/`fa4`/`triton`/`flashinfer`/`trtllm_mha` for GQA drafts. Startup
rejects any other `--drafter-attention-backend` rather than quietly widening
the draft's attention to the full history.

Use `--attention-backend mha` and `--drafter-attention-backend mha` for
standard or grouped-query attention in the target and draft, respectively.

## PD transport

Mooncake KV transfer selects `ascend_direct` on Ascend NPU and `rdma` on
GPU platforms. Install a Mooncake build that supports the selected transport
and provides the TokenSpeed C entry points in `libtokenspeed_transfer_engine.so`,
including `ts_mooncake_write_pages` for page transfers.
Both Prefill and Decode engines select their transport from the detected
platform. Startup settings pass this protocol to the Rust transfer peer,
which keeps it when recreating a session after a memory resume.

Upgrade Prefill and Decode nodes together: cache registrations now carry a
separate cache identity in a ninth frame, and requests name that identity.
Older peers without this field are incompatible. Each Decode cache resume
gets a new identity even if its transport IP and port are reused. The sender
refreshes cached remote memory keys on address reuse without changing the
public address used by EPD.

## Observability

| Parameter | Purpose |
| --- | --- |
| `--log-level` | Engine logging threshold for Python, the Rust host and the C++ scheduler. Python `critical`/`fatal` maps to Rust `error`, its highest severity. Explicit `TS_HOST_LOG` and `SPDLOG_LEVEL` settings override their native writers. |
| `--enable-async-logging` | Write Rust host and default Python root logs on background threads. Defaults to off; `--no-enable-async-logging` explicitly selects synchronous output. |
| `--ttft-trace-sampling-percentage` | Percentage of PD rooms to trace through their first engine output (0–100, default 0). |
| `--enable-log-request-stats`, `--enable-request-time-stats-logging` | Log a one-line per-request performance summary on finish/abort (see below). |
| `--enable-metrics` | Enable the selected metrics reporters. |
| `--metrics-reporters` | Select `prometheus` or an installed metric output; repeat for multiple. Defaults to `prometheus`. |
| `--metrics-reporter-config` | JSON file mapping external reporter names to their settings. See [metric output plugins](../metrics-plugins.md). |
| `--metrics-port` | Port for the engine's `/metrics` HTTP endpoint. Defaults to an automatically allocated free port (logged at startup). |
| `--metrics-host` | Bind host for the `/metrics` endpoint. Defaults to `--host`. |
| `--decode-log-interval` | Decode batch log interval. |
| `--kv-events-config` | JSON config for KV cache mutation events. Set `enable_kv_cache_events` and a publisher such as `zmq` to publish device prefix-cache stores and removals. |
| `--otlp-traces-endpoint` | OTLP/HTTP traces endpoint (a collector's `/v1/traces`, Jaeger, Langfuse). Each finished request exports one span built from its lifecycle timestamps. |
| `--trace-sample-ratio` | The share of requests the OTLP endpoint traces (0..1). 0, the default, disables tracing. |

Every `--decode-log-interval` decode rounds the scheduler's representative rank
prints one `Decode batch.` line: `#running-req`, `avg_seq_len` (the mean of
prompt plus generated tokens over the running requests, so a step's attention
cost can be read alongside its batch size), device page usage, the generation
throughput accumulated since the previous line, `avg_accept_len` /
`accept_rate` under speculative decoding, and `#queue-req`. Every field is a
host-side scheduler counter. The line adds no GPU synchronization.

Logging is synchronous by default. With `--enable-async-logging`, the Rust
host's `tracing` output and the default Python root logger each use a queue
of at most 8,192 records. A background thread owns writes and flushes, so
an occasional slow write to a distributed file system or collector does
not hold the thread emitting the log. When a queue fills, new records are
dropped without writing a synchronous overflow warning. This removes output
stalls from the serving thread; it does not increase the sink's throughput.

The host explicitly drains Python logs on both normal and error returns,
before notifying the launcher of a failure; it does not rely on Python
interpreter finalization. Python cleanup waits up to five seconds, and the
Rust appender also performs a bounded, best-effort drain on exit. A sink
that remains blocked through shutdown can lose pending records.

Colorful Python loggers retain their direct handlers. Custom handlers remain
under `TOKENSPEED_LOGGING_CONFIG_PATH` control: combining that variable with
`--enable-async-logging` is rejected. Configure asynchronous handlers in the
custom file, or use the default configuration with this flag.

`#queue-req` counts requests admitted to the scheduler but not yet running.
On a PD engine that includes requests still bootstrapping with the peer —
on the prefill role, waiting for the decode side to allocate their KV pages —
which the `#req-state(bootstrap/prefill/remote-prefill/decode/pd-pinned)`
suffix also lists on its own. The Prometheus waiting gauge and the router
load snapshot report the scheduler's narrower waiting count, without the
bootstrapping share.

`tokenspeed:request_queue_time_seconds` is the Prometheus per-request
queue-wait histogram (`--enable-metrics`): the wall time from a request's
submission to the scheduler's waiting queue to its first dispatch into a
forward batch, observed once per request. The enqueue point is submission —
frontend and IPC wait are not included, and neither are the waits
that precede submission: a structured-output request's grammar-compile wait
(readiness gates submission), a pause admission hold, and an EPD embedding
wait all stay out of the sample. A request preempted by capacity retraction
re-enters the queue without recording a second sample. Under prefill/decode
disaggregation each node records its own leg: the prefill node's sample spans
its queue admission to the prefill batch, the decode node's its own admission
to the decode batch. Bucket boundaries span 0.3 s to 7680 s.

Set `TOKENSPEED_LOG_SPEC_ACCEPT_LENGTHS=1` to log each speculative verify
step's committed widths and accepted draft-token counts. This reads the
already-synchronized CPU result and does not add a GPU synchronization, but it
is intentionally verbose. Enable it only while debugging. For
decode-only batches it also logs the anchor, draft candidates, target verify
tokens, and their position-wise matches.

Set `TOKENSPEED_VALIDATE_PAGE_IDS=1` to check every page ID in each forward's
block tables against its cache group's page range before the forward runs.
The structural checks (cache groups, dtype, shape, rows) always run; the scan
reads every ID of every scheduled row, which at long contexts costs as much as
the rest of the forward thread's table packing, so it is off by default.
Enable it while debugging a page-table fault. The model executor reads it once
at startup. Prefill-graph capture scans the tables it builds either way.

To send observations to another monitoring system, install and select a
[metric output plugin](../metrics-plugins.md). Selecting only external outputs
does not start a Prometheus endpoint.

### Prometheus metrics

With `--enable-metrics` and the `prometheus` reporter, every engine rank on a
node writes its metrics to a node-local `PROMETHEUS_MULTIPROC_DIR`, and the
node's launcher process aggregates and serves them as `GET /metrics` in
Prometheus text format. Scrape it at
`http://<metrics-host>:<metrics-port>/metrics`; the bound address is logged at
startup. Scrape traffic shares nothing with the request path — a bind failure
degrades to an error log and serving continues.

Where the endpoint lives depends on the launch mode:

- **Single node / `ts serve`**: the engine's launcher process serves
  `/metrics` on `--metrics-port` (an allocated free port if unset). `ts serve` pins this
  port itself and prints `engine metrics available at http://...` once ready.
  The gateway's own router metrics are separate: smg serves them on its
  `--prometheus-port`, which `ts serve` also allocates per launch.
- **Multi-node**: every node's launcher serves its own node's `/metrics` the
  same way. Scrape every node — the multiprocess directory is node-local, so
  no single endpoint covers the whole cluster.

### Request metrics

The engine exports the scheduler's view of a request: the queue wait above
(`tokenspeed:request_queue_time_seconds`), the running, waiting, and inflight
gauges, and the preemption and abort counters. Latency from the client's side
(time to first token, inter-token latency, end-to-end latency) and the
per-request finish counts are the frontend's: under `ts serve`, SMG's router
metrics on its `--prometheus-port`.

`tokenspeed:num_requests_running` counts the rank's running batch: local
prefill forwards, prompts whose prefill is done (about to decode on a fused
rank, or awaiting the PD transfer on a prefill rank), and decode rows. On a
decode rank that is the decode batch, bounded by the rank's
`max_num_seqs / attention-DP size`; it excludes `RemotePrefilling` requests
waiting for the prefill peer's KV.
`tokenspeed:num_requests_inflight` counts every request the rank admitted and
has not finished, including the scheduler's waiting and retracted requests and
the PD states the batch excludes — a decode rank's `RemotePrefilling` and
`PdTransferring` requests wait for the prefill node's KV and take no batch row.
On a PD decode rank the inflight gauge therefore exceeds the running gauge while
prompts wait for transfer; use `num_requests_running` for batch saturation and
`num_requests_inflight` for how much work the rank holds.

The decode role splits that PD wait into its own gauges.
`tokenspeed:num_decode_kv_wait_reqs` counts `RemotePrefilling` requests: the
prefill peer is working, this rank already holds their destination pages, and
nothing here is schedulable until the bootstrap token arrives.
`tokenspeed:num_decode_prealloc_queue_reqs` counts the earlier wait for the peer
to connect. `tokenspeed:num_decode_transfer_queue_reqs` counts requests whose
pages a transfer still pins, including requests that already ended (aborted or
failed) and are only waiting for the transport to report stopped; it is a
resource count, not a queue depth. The transfer gauge is a superset of
`num_decode_kv_wait_reqs`: every waiting request's destination pages are pinned
by its inbound transfer, so do not add the two. Use `num_decode_kv_wait_reqs`
for "waiting for KV", and `num_requests_inflight` for the rank's total held
requests.
`tokenspeed:num_requests_waiting` is the scheduler's narrower waiting queue,
without the PD bootstrapping share (see `#queue-req` above). All of these carry
the `dp_rank` label and are per scheduler rank.

The scheduler process exports `tokenspeed:num_preemptions_total`: requests
preempted by a capacity retraction — suspended
with their pages imaged to the Host snapshot pool (they resume by restore;
nothing is recomputed) to free Device pages for a blocked admission. One
increment per retraction victim, counted from the plan's snapshot-store ops;
forced retractions from `--debug-force-retraction-interval` count too. A
retraction whose victim cannot be imaged aborts it instead — those are
`tokenspeed:num_capacity_aborted_requests`, never preemptions. A forced
retraction's victim refused after its pages were published is aborted and
counts there too.

### OTLP tracing

`--otlp-traces-endpoint` exports one span per request to an OTLP/HTTP
receiver, assembled at the request's release from the same timestamps the
stats line prints — the serving loop never blocks on it, a full export
queue drops spans and counts them, and `--trace-sample-ratio` bounds the
share. A request that names a continual-prompting session
(`session_params.id`) carries it as the `session.id` attribute, so a
tracing UI groups one conversation's requests under the session. Trace ids derive from the request id, so a span, its `Req: … Finish!`
line and its Prometheus samples cross-reference. The span carries the GenAI
attribute conventions (`gen_ai.request.model`, prompt and completion tokens,
finish reason) plus the cached prompt tokens and preemption counts, with the
lifecycle points as time-stamped events. See
[the tracing design](../design/tracing.md).

### Request lifecycle logs

Search the engine logs for a request ID to follow it across roles. For example,
`rg -F 'chatcmpl-019ef6b7' prefill.log decode.log` finds both sides of a
disaggregated request when the frontend uses the same ID. The PD events also
carry `bootstrap_room` to correlate peers that use different request IDs.

| Event | When it appears | Fields besides `rid` |
| --- | --- | --- |
| `ts_request_accepted` | The engine registers the request. | Role, session ID when present, input tokens and effective maximum new tokens |
| `ts_prefill_kv_transfer_queued` | A prefill rank queues the final KV chunk for a data transfer. | Bootstrap room, rank, token range `[start_idx, end_idx)` and block count across cache groups |
| `ts_decode_kv_transfer_success` | The decode replica confirms receipt of KV and the bootstrap token. | Bootstrap room, rank, cached tokens and output token count |
| `ts_decode_kv_transfer_failed` | The decode replica confirms transfer failure. | Bootstrap room (also retained as `room` for compatibility), rank and failure reason when available |
| `ts_request_finished` | The engine releases the request or refuses it at admission. | Role, prompt, cached and completion token counts, finish and abort reasons, `e2e_latency_s` when the frontend supplied a creation time, and optional `request_stats` |

These events are enabled for every request at the default INFO log level, with
WARN for a failed transfer. They are independent of the request-stats flag and
TTFT sampling percentage.
Request admission and completion use the same logging ranks as the per-request
stats. Each PD role records its own completion time. Ingress refusals have a
finish event only. A duplicate-ID refusal logs a finish event for the rejected
submission with an abort reason; the original request with that ID remains live
and emits its own finish event later.

Logging uses synchronous writers by default. `--enable-async-logging` moves
Rust and Python root logger writes to bounded background queues; C++ scheduler
logs still write synchronously. Formatting and queueing cost CPU time, and full
asynchronous queues can drop records.

For a first successful prefill admission, set `TOKENSPEED_PREFIX_CACHE_DIAGNOSTICS=1`
to include the Device and Host contributions to the prefix hit in
`[PrefixCacheDiagnostic]`. Their sum is the resumed token count.
`host_gain_tokens` measures the extra reusable prefix beyond a device-only match.
At DEBUG, the native client logs the chosen engine and routing policy by request
ID. Scheduler diagnostics (`SPDLOG_LEVEL=debug`) report restore and decode
preallocation waits, and link L3 prefetch queries, reads and timeouts to the request ID.

### Sampled PD first-output logs

Set `--ttft-trace-sampling-percentage 1` on both roles to select approximately
1% of rooms. The selection uses the CRC32 of the decimal bootstrap room,
modulo 1000, with a threshold of ten times the percentage. A frontend owner
records a selected request once, even when several token batches stream out.
Pair the JSON records using `endpoint` and `room` within the same time window.

| INFO record | Boundary | Measurements |
| --- | --- | --- |
| `TTFTBootstrap` | The prefill replica agrees the room is bootstrapped | Engine registration to bootstrap agreement |
| `TTFTBreakdown` on Prefill | First engine output or terminal release | Bootstrap, admission to first dispatch after bootstrap, prompt execution, transfer observation and output preparation |
| `TTFTBreakdown` on Decode | First engine output or terminal release | Bootstrap, KV landing after bootstrap, and landing to output |

Prefill holds its first output until the KV transfer completes. Its normal
breakdown therefore includes `transfer_observe` and `output_pack`; producing
the bootstrap token alone does not emit the breakdown.

Schema `ts-engine-v1` measures local `phases_ms` and `engine_elapsed_ms` on a
monotonic clock. `first_dispatch_ms` is relative to engine registration.
`total_ms` starts at the frontend creation timestamp when supplied and ends
when the engine prepares the output. Unreached phases and an unavailable
creation time are JSON null. Tokenization, detokenization and delivery to the
HTTP client belong to the gateway and fall outside this measurement.

### Per-request stats

`--enable-log-request-stats` adds a `request_stats` field to the existing
`ts_request_finished` event for registered requests. There is one finish log
record per completion. The field retains the `Req: <rid> Finish! RequestStats(...)`
text, including the speculative acceptance average as `acc_len`. With the flag
off, the finish event still appears without detailed stats. The scheduler derives
the stats from host-side timestamps and counters, adding no GPU synchronization;
formatting and writing the log still have a CPU cost. Example `request_stats` value:

```
Req: chatcmpl-019ef6b7 Finish! RequestStats(status='finished', reason='stop', prompt_tokens=28684, cache_tokens=832, output_tokens=33, cache_hit_rate=0.029, queue_ms=13.8, prefill_ms=15.8, decode_wait_ms=0.0, ttft_ms=42.1, total_ms=58.0, preempt_ms=0.0, preempt_count=0, decode_tps=210.4, acc_len=None, acc_rate=None, recv_ts=1782255696.726, commit_ts=1782255696.74, finish_ts=1782255696.784)
```

| Field | Meaning |
| --- | --- |
| `status` / `reason` | `finished` vs `aborted`; finish-reason type (`stop`/`length`/`abort`). |
| `prompt_tokens` / `cache_tokens` / `output_tokens` | Prompt tokens, prefix-cache-hit tokens, generated tokens. |
| `cache_hit_rate` | `cache_tokens / prompt_tokens` (0–1). |
| `queue_ms` | Received → first scheduled into a forward batch. |
| `prefill_ms` | Scheduled → prefill complete. Always `0` on a decode node, whose prompt is prefilled on the prefill node. |
| `decode_wait_ms` | Decode nodes only: KV (with the bootstrap token) landed → first decode batch dispatch. `0` elsewhere. |
| `ttft_ms` | Received → first output token. On a decode node the first token is the prefill node's bootstrap token, so `ttft_ms` spans received → KV landed. |
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

Host L2 entries publish no events. Under `--kvstore-write-policy write_back`,
a block the GPU evicts is reported `BlockRemoved` even when it is written
back to the Host.

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

### PD KV transfer metrics

In a PD-disaggregated deployment (`--disaggregation-mode prefill` /
`decode`), both engine roles record KV cache transfer metrics when metrics
are enabled (`--enable-metrics --metrics-reporters prometheus`). All three
carry the `model_name` and `app_key` labels:

| Metric | Type | Meaning |
| --- | --- | --- |
| `tokenspeed:kv_transfer_latency_seconds` | Histogram | Wall time of one KV transfer send to a peer, recorded on the prefill node (one sample per peer per transferred chunk — per layerwise interval in layerwise mode, except an interval with nothing for that peer; buckets span 1 ms to 60 s). |
| `tokenspeed:kv_transfer_timeouts_total` | Counter | Transfers that timed out, with a bounded `stage` label: `bootstrapping` (prefill never received the cache manifest; `TOKENSPEED_DISAGGREGATION_BOOTSTRAP_TIMEOUT`) or `waiting_for_input` (decode never received the transfer-done signal; `TOKENSPEED_DISAGGREGATION_WAITING_TIMEOUT`). |
| `tokenspeed:kv_transfer_failures_total` | Counter | Transfers aborted on failure: send failures on the prefill node, and rooms failed on prefill-node loss (heartbeat) on the decode node. Timeouts are counted separately. |

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
- `--head-tp-exchange-max-rows`
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
retraction publishes the victim's completed prefix like a finish, its
partial last page and its live state block included, and those blocks go to
Host L2 as a stream-ordered write-back and stay pinned there until the
restore; the blocks of groups L2 never holds (a replayable sliding-window
group), a state group's input block when nothing published it (an ordinary
chunk end's output) and the slot-state blob go to the request-private pool
(`docs/design/cache-concepts.md`, "Retraction image"). The two buffers are
separate allocations of one Host cache executor and both count against its
Host-memory headroom check. The pool is independent of the KVStore: with
`--disable-kvstore` there is no L2 leg and the pool holds whole images; with
L2 it holds the unpublished blocks plus the slot-state blob.

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

One image tail, per cache group, is one block for a group that publishes
to Host L2 (a state group's unpublished input block; a history group's
margin for a published page L2 could not take) and a request's worst-case
pages at the context limit for a group that never publishes (a replayable
sliding-window group), folded to LCM blocks by the scheduler's
`CapacityModel`, so the pool is counted the way the scheduler later claims
it. Without the KVStore a whole image is the request's every
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
the tail and for the hash-complete pages Host L2 does not take: which ones
it takes is planned before anything is evicted for the image, under either
write policy). When no resident fits, the scheduler **aborts the newest
retractable resident** in the same round and grants its pages to the
blocked request. The client sees the finish `{"type": "abort", "err_type":
524, "message": "Request aborted by the scheduler: <detail>"}`
(`ABORT_CODE.CapacityAbort`; the detail names the shortfall -- "snapshot
pool cannot hold the image", "Host L2 cache cannot take the image's
published pages" or "no blob slot free (max_retracted_requests=N)" -- and
the argument to raise), distinguishable from a NaN abort (523) or a transfer
failure (521), so a caller that can retry -- an RL rollout, a batch job --
resubmits; the engine counts them in
`tokenspeed:num_capacity_aborted_requests`. The two arguments therefore size
how much work may be suspended at once (with Host L2, so does
`--kvstore-size`, which takes the hash-complete pages), and exceeding them
costs an abort of one request, never latency for everyone else. With no pool
(`--retraction-snapshot-ratio 0`) nothing can be imaged, so a
capacity-blocked round aborts the newest resident instead of imaging
anyone. A forced retraction (`--debug-force-retraction-interval`) whose
image does not fit is refused and logged, and the request runs on; only a
request whose image is refused after its pages were published is aborted as
above. The scheduler's victim choice is in `docs/design/scheduler.md` section
2.

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

**When a lone request cannot progress.** On the fused role the scheduler's
no-progress abort fails a request like the image-fit abort above (524), with
the detail "no progress for 4096 consecutive plans ...", counted in
`tokenspeed:num_capacity_aborted_requests` too: a request no
other request holds KV pages beside, whose admission needs Device pages that
neither eviction nor lone-request compaction can free, or whose restore does
not fit even when it claims no cached page
([scheduler §1.4](../design/scheduler.md#14-what-bounds-a-single-request)),
is aborted after 4096 consecutive scheduler plans without progress (the
scheduler warns about such a stall from its 1024th plan on) instead of
waiting for good. On the prefill and decode roles of a disaggregated
deployment such a request waits behind the warning instead: a scheduler
abort would not conclude the PD transfer it has registered. A prefill
request past its first admission is not counted as stalled at all, as its
pages stay pinned for the transfer.

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

Host L2 and the retraction snapshot pool pack each group's payload independently
of Device strides. A shared Host parent fits the largest single group payload,
and each group fits as many whole children in it as possible. When that
payload is a recurrent-state checkpoint, or a replayable sliding-window page
that Host L2 never stores, the KV pages of every parent leave a gap smaller
than one page, so the parent may instead round up to a whole number of pages
of the KV group that stores the most Host bytes per token: those pages then
fill it without a gap, but every parent of the largest payload gains the same
bytes as slack. The parent rounds up only when that lowers the Host bytes per
cached token, assuming one State checkpoint per 8,192 cached tokens (a
request publishes only a few, so a prefix cache of long prompts holds far
fewer). Empty parents are available to every group, including recurrent
State; no group has reserved capacity.
Mirrored ranks must have the same Host packing; startup refuses local layouts
that would give the same Host block ID different meanings across ranks.

**Write policy.** `--kvstore-write-policy` chooses when pages of the GPU
prefix cache are copied to Host L2:

- `write_through` (the default): on the fused and prefill roles, every page
  the GPU caches is also copied to the Host, which keeps it after the GPU
  evicts the page. The decode role's Host L2 holds retraction images only.
- `write_back`: nothing is copied when a page is cached. When the GPU
  evicts cached pages to make room for a request, or to resume a retracted
  one, it copies them to the Host in the same scheduling step, and they can
  be reused from there once the copy completes.

`write_back` lets the two tiers cache more together: the GPU and the Host
hold mostly different pages, up to about the sum of their capacities less
pages in use or in transit, where under `write_through` the Host spends its
room on a copy of what the GPU holds. Eviction trims a conversation from its
end, so the GPU keeps its beginning and the Host the rest, and a later
prompt resumes from both. The cost: a step that evicts must finish copying
the evicted pages to the Host before its forward pass can write over them,
so the copy adds to that step's time.

`write_back` runs on the fused and prefill roles, not with
`--disaggregation-mode decode`. Both cached and replayable sliding-window
groups are supported. It requires Host L2,
`--pipeline-parallel-size 1`, `--query-context-parallel-size 1` and no
`--kvstore-storage-backend`, and a model with at most 64 cache groups;
startup refuses it otherwise, the group limit once the model's cache groups are
built.

Under `write_back`:

- Clearing only the GPU cache, as `POST /release_memory_occupation` does
  when it releases `kv_cache`, loses what only the GPU holds. The Host pages
  that continue a prefix only the GPU held stay in Host L2: they serve hits
  again once a request recomputes that prefix on the GPU, and otherwise leave
  only through Host LRU eviction. `/flush_cache` clears both tiers under
  either policy.
- Not everything the GPU evicts reaches the Host: what a request releases
  itself, what nothing can resume from any more, and pages evicted together
  that the Host cannot fit as a whole are dropped.
- [KV cache events](#kv-cache-events) keep describing the GPU cache only,
  so a router reading them does not see the Host's share of the reusable
  cache, which `write_back` makes larger.
- Under attention TP, every rank copies its own part of what the engine
  evicts to its own Host L2 (under `--kv-parallel-size`, the pages it owns),
  and an evicted page can be reused from the Host once every rank's copy has
  completed. Where every rank holds whole pages (an MLA cache without
  `--kv-parallel-size`), the ranks copy the same bytes: attention TP then
  multiplies the copy traffic without letting the Host hold more pages.

Elsewhere on this page, "write-back" means any copy from the GPU to Host L2,
whichever policy is set, not only the copies `write_back` makes. How the
scheduler implements the two policies is in
[cache concepts](../design/cache-concepts.md#the-coordinator-layer-csrccachecoordinator)
and, for `write_back`,
[scheduler §3.1](../design/scheduler.md#31-p--prefill-worker) and, on the
fused role,
[§3.3](../design/scheduler.md#33-fused--one-engine-everything-local).

**Sizing.** `--kvstore-size` (gigabytes per rank) wins over
`--kvstore-ratio` (a multiple of this rank's Device LCM block count, 2 by
default). Host parents use compact payload geometry, so the ratio counts
parents rather than allocated bytes. Use `--kvstore-size` to request a byte
budget. A parent rounded up to whole KV pages (above) is less than one KV page
larger than the largest payload, a large share when it holds few pages: with
Qwen3.8-Flash-Next and FP8 KV it grows 4.3% at attention TP 1 and 2, 16% at
TP 4 and 39% at TP 8, where the recurrent state keeps splitting across the
ranks but the two KV heads stop at TP 2. Sizes that count parents (the
ratios and the derived retraction snapshot pool) pin that much more Host
memory; the gigabyte sizes keep their byte budget.
Every rank of a node pins its Host buffers at the same point of startup, so
each rank checks the Host L2 and the retraction snapshot pool against an
equal share of the Host memory its node has available, less a 10 GiB
headroom per rank. Available means the smaller of the host's free memory and
what the process's memory cgroup still allows (its limit, `memory.max` or
`memory.high` on cgroup v2, less its usage with page cache counted as free;
the tightest limit over its cgroup and the ones above it, found through
`/proc/self/mountinfo`), so a container or a Slurm step is sized against its
own limit; the sizing warning and the refusal name which one was used.
Each buffer counts at the size its configured pinned
allocator reserves. When the CUDA/ROCm allocator supports it, the launcher sets
`pinned_max_round_threshold_mb:1024` as the default before starting the ranks.
Allocations up to 1 GiB round to a power of two; larger buffers use their
requested size.
This avoids reserving 274.88 GB for a 140 GB L2 buffer. Allocators without this
option keep their native defaults. The capability probe has a 60-second timeout.
Probe failures stop startup. Explicit thresholds and other allocator options
are preserved. To override the default, set
`PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:<MiB>` before launch;
the legacy `PYTORCH_CUDA_ALLOC_CONF` and `PYTORCH_HIP_ALLOC_CONF` variables
take precedence, in that order. The check reads the effective allocator
settings. A `pinned_max_cached_size_mb` limit also disables rounding above
its threshold and releases those blocks when unused; it limits individual
block size, not total cached bytes. Device backends without allocator
snapshots retain the conservative power-of-two estimate.

A ratio size larger than a rank's share is capped to the largest whole-LCM
buffer that fits under those allocator settings beside its snapshot pool,
with one size agreed across ranks. Each rank logs the requested and capped
size. An explicit
`--kvstore-size` is never shrunk: one that does not fit is refused at
startup, as is a share without room for a single LCM block. Every rank
must also resolve the same page counts: startup compares them across the
ranks and refuses a difference on every rank. A size in gigabytes is a
different page count wherever the ranks' LCM blocks differ in bytes, as
they usually do on pipeline stages, which store different layers, so with
`--pipeline-parallel-size > 1` size Host L2 with `--kvstore-ratio` rather
than `--kvstore-size`.

**Requirements and deployment.** L3 requires Host L2 (omit
`--disable-kvstore`) and currently supports PP=1. Cyclic KV-page sharding
(`--kv-parallel-size`) is supported when the payload is replicated within
each owner subgroup; head-partitioned payloads in a sharded group are
rejected at startup. Existing model and parallelism restrictions still apply.
Use a scheduler built from the matching runtime source.

Deploy a Mooncake master and storage providers before starting inference.
The master holds metadata; providers hold the cached bytes. Providers backed
by CPU DRAM need no GPU memory. Set `global_segment_size` to `0` in inference
workers to use those providers without mounting storage in the workers.
Reuse after inference restarts requires the master and providers to retain
the objects. `--kvstore-size` sizes each rank's Host L2, not the shared L3
capacity. Account separately for provider segments and client staging memory.
A positive `global_segment_size` mounts storage in the inference workers;
TokenSpeed divides it across attention-TP × PP ranks, using the resolved
`mapping.attn.tp_size`, so their total matches the configured size.

Pass client settings as JSON in
`--kvstore-storage-backend-extra-config`. For example, a client using a
separate DRAM provider over RDMA:

```json
{
  "master_server_address": "store.example:50051",
  "local_hostname": "10.0.0.2",
  "metadata_server": "P2PHANDSHAKE",
  "protocol": "rdma",
  "global_segment_size": 0,
  "tenant_id": "example-model",
  "async": {
    "max_operations": 32,
    "max_keys_per_operation": 256,
    "wait_timeout_ms": 2000,
    "ready_timeout_ms": 1000,
    "max_prefetch_tokens": 8192,
    "query_workers": 2,
    "read_workers": 2,
    "write_workers": 1
  }
}
```

Set `local_hostname` to this node's reachable address. For RDMA, validate
connectivity between all clients and providers and set `device_name` when a
specific HCA is required. The default protocol is `tcp`; choose `rdma`
explicitly. Without JSON connection settings, the client reads
`MOONCAKE_MASTER` / `MOONCAKE_CLIENT` and the other Mooncake environment
settings. `global_segment_size` defaults to 4 GiB, so explicitly set `0` for
clients that must not mount storage. The default tenant is `default`.

**Asynchronous preparation before admission.** Queries and GETs hold no
Device pages or request-pool slots. Once a GET has retired on every owner,
rows with full-length success on every owner become ordinary Host entries.
The group matcher chooses the reusable prefix, which may be shorter than
requested. A soft deadline permits local computation while native I/O keeps
its Host references until retirement. A result detached by timeout or
cancellation cannot publish data. Unknown native retirement quarantines the
buffers and disables new L3 work until process restart; valid L2 hits and
ordinary computation can continue. D2H publication and Device-source release
wait for replica copy agreement, independently of L3 PUT. The PD prefill role
can prepare L3 hits; decode receives its missing cache from the prefill peer.

The `async` object tunes this single execution path. Unspecified settings use
the following defaults; unknown options are rejected:

| Setting | Default | Meaning |
| --- | --- | --- |
| `max_operations` | `32` | Outstanding operations, including undelivered completions, Ready holds and quarantine |
| `max_keys_per_operation` | `256` | Key limit per query, GET or PUT |
| `max_host_blocks` | `max(1, min(256, host_blocks // 2))` | Logical Host references across operations, Ready holds and quarantine; `host_blocks` is the usable Host LCM-parent count |
| `metadata_entries` | `max(1, host_blocks * 4)` | Limit for negative-key metadata and for backup candidates |
| `min_prefetch_tokens` | Model prefix granularity | Minimum extension beyond the local hit worth preparing |
| `max_prefetch_tokens` | `8192` | Maximum extension beyond the local hit, rather than a prompt-length limit |
| `wait_timeout_ms` | `2000` | Shared soft wait budget for query and GET |
| `ready_timeout_ms` | `1000` | Hold time for completed Host results awaiting ordinary admission |
| `query_workers` / `read_workers` / `write_workers` | `2` / `2` / `1` | Independent executor worker counts |
| `failure_threshold` / `recovery_interval_ms` | `1` / `30000` | Backend-health failure threshold and recovery-probe interval |
| `queue_high_operations` / `queue_low_operations` | `min(8, max_operations)` / high divided by two, rounded down | Queue-count admission watermarks |
| `queue_high_bytes` / `queue_low_bytes` | `268435456` / `134217728` | Queued payload-byte watermarks (256 / 128 MiB) |
| `operation_high_age_ms` / `operation_low_age_ms` | `500` / `250` | Oldest queued/running operation age watermarks |

Prefetch length is also constrained by keys, Host references and group
geometry: Full history, SWA windows and State checkpoints must fit together.
Raising `max_prefetch_tokens` alone does not guarantee a longer restore.
`max_host_blocks` is a logical-reference budget, not bytes or a per-worker
quota, and cannot exceed the usable Host-parent count. PUTs may use at most
half of the operation and Host credits, rounded down, to leave room for reads.

Any high watermark closes new L3 read/write admission; all observations must
reach their low watermarks before it reopens. The gate bypasses new requests'
L3 preparation and drops nonowning backup candidates. Already submitted I/O
keeps its lifetime and deadlines; completions, ordinary forward work and L2
publication continue. Pressure admission and backend health are independent.
See [asynchronous L3](../design/async-l3-cache.md) for queue and ownership
invariants. The four legacy `--kvstore-prefetch-*` arguments are removed.

**Object compatibility and maintenance.** Each packed Host CacheBlock is
one object, keyed as
`{tsl3-<sha256>}_{content_hash}|g{group}|o{page_offset}|r{object_rank}|c0`.
For replicated groups, `object_rank` is the attention-TP rank; for cyclic
page groups it is `tp_rank // shard_count`, independent of which cyclic owner
allocated the block. The literal `c0` is a reserved format field.
The namespace isolates loaded target/draft checkpoints, explicit weight
versions, packed Host layout, parallel geometry (including ordered group shard
counts for page-sharded layouts) and cache-producing runtime
choices. Different compatibility epochs start cold caches. See
[cache concepts](../design/cache-concepts.md) for the identity contract.

Host eviction leaves shared objects intact. Full cache clear and weight
updates require replica-agreed idle state, including storage operations,
Ready holds and quarantine. Remote namespace purge is still a synchronous
administrative RPC; preparation soft deadlines do not bound it. Failed
preflight or purge prevents local clearing. With L3 enabled, weight updates
require an explicit `weight_version` and a successful flush; the
metadata-only `/update_weight_version` endpoint is rejected. Independent
inference jobs sharing a namespace are outside this flush agreement.

Omitting `--kvstore-storage-backend` creates no L3 executor, workers or SDK
client. `--kvstore-storage-backend memory` selects an in-process test store;
it is separate from Mooncake's distributed DRAM mode. The engine host loads
the Mooncake store client from the `tokenspeed-mooncake` wheel's
`libmooncake_store.so` (the store's C ABI);
`TOKENSPEED_MOONCAKE_STORE_LIBRARY` names another build. CPU tests cover the
scheduler, executor and replica agreement; GPU tests cover packed-byte
D2H/PUT/GET/H2D round trips. See the
[validation scope](../design/async-l3-cache.md#initial-scope-and-validation)
for native Mooncake test requirements.

Mooncake Store is the offload backend. PD KV transfer still uses the
separate Mooncake TransferEngine (`--disaggregation-transfer-backend`).
On a host whose address is IPv6, the PD transfer engine sets `MC_USE_IPV6=1`
(unless it is already set) so Mooncake's handshake uses IPv6. The Mooncake L3
store client reads `MC_USE_IPV6` on its own when the store starts, so export
`MC_USE_IPV6=1` yourself when the L3 store runs on an IPv6 host.

### Block verification (experimental)

`--speculative-rejection-sample-method block --enable-speculative-sampling --sampling-backend flashinfer`
verifies a sampled draft chain by block verification instead of token-wise
rejection. The names follow vLLM: `--enable-speculative-sampling` is
`draft_sample_method=probabilistic`, and `--speculative-config` also accepts
the `draft_sample_method` and `rejection_sample_method` keys. It supports `DSPARK` (probabilistic Markov drafting) and `EAGLE3`
with `--speculative-eagle-topk 1`; block verification is defined on a single
chain, so draft trees are refused. EAGLE3 already samples its chain under
`--enable-speculative-sampling`, and its hot-token `q` is recorded over the full
vocabulary with zeros outside the draft vocabulary. The feature is disabled by
default; without it, the greedy DSpark path and the token-wise EAGLE3 verifier
are unchanged. The initial support scope is aggregate TP, including DCP, without
attention DP, DP sampling, pipeline parallelism or PD disaggregation. Other
combinations fail at startup.

Draft probabilities follow the request temperature; greedy requests retain
one-hot proposals. Target filtering and grammar masks still apply before
verification. The verifier uses clipped prefix likelihood ratios, selects the
longest accepted prefix, and samples a prefix-scaled positive residual (or a
target bonus token). It is not a single whole-block likelihood-ratio test.
The output uses the existing accepted-length and state rollback contract.

The existing per-request proposal RNG and verifier coin streams are reused.
A stochastic output difference from greedy decoding is expected; validate the
output distribution and accuracy, rather than requiring identical token IDs.

Replacing `block` with `--synthetic-acceptance-length L` (the `synthetic`
method) gives a benchmark-only mode, the equivalent of vLLM's probabilistic
draft sampling with synthetic rejection; for `DSPARK` it also turns on sampled
Markov drafting.
The drafter still samples: the full-vocabulary draft logits (and DSpark's Markov
steps) and the recorded fp32 draft probabilities all stay on the path. Verify
gathers those probabilities and runs the standard draft-probability chain rule
for its cost; the synthetic length then replaces its outcome, as in the greedy
synthetic mode above. The block verifier itself is not run, because synthetic
rejection replaces it. The token at the cutoff is drawn from the target
probabilities at that position (the argmax for greedy requests); this costs one
full-vocabulary pass per request. Do not use generated text for correctness or
accuracy evaluation; validate the algorithm and its natural acceptance length
without this flag. Full-vocabulary draft gathering, probability storage, and
verification reductions all add work; measure their cost and natural acceptance
length before drawing performance conclusions.
