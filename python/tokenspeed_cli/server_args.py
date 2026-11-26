# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Declare server options using only standard-library dependencies."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

# ``--dsa-slot-order``: the order the sparse attention cores reduce a token's
# selected KV slots in — ``selection``: as the top-k leaf emitted them (what
# every core does); ``sorted``: ascending token position, so the reduction is
# batch-invariant whenever the selected set is; served only by cores
# declaring the ``slot_order`` trait (registered batch-invariant leaves). Mirrors
# ``tokenspeed_kernel``'s ``attention.dsa.SLOT_ORDERS``.
DSA_SLOT_ORDERS = ("selection", "sorted")

# ``--layer-boundary-norm``: the norm at each physical layer boundary — the
# fused add+norm kernel, or a bf16 ``hidden + residual`` materialized first and
# a standalone RMSNorm, as the trainer does.
LAYER_BOUNDARY_NORMS = ("fused", "unfused")

# ``--logprob-order``: the selected-token log-softmax behind every returned
# logprob — ``torch.log_softmax``, or fixed-width blockwise reduction
# over ``BLOCKWISE_VOCAB_BLOCK``-wide vocab blocks. Logprobs only.
BLOCKWISE_VOCAB_BLOCK = 32768

# Block width describes the trainer's reduction, independently of serving TP.
BLOCKWISE_LOGPROB_ORDERS = {
    "blockwise": (BLOCKWISE_VOCAB_BLOCK, "negative-cross-entropy"),
    "blockwise-direct": (BLOCKWISE_VOCAB_BLOCK, "direct"),
    "blockwise-16384": (16384, "negative-cross-entropy"),
    "blockwise-direct-16384": (16384, "direct"),
}

LOGPROB_ORDERS = ("torch", *BLOCKWISE_LOGPROB_ORDERS)

# ``--mla-lora-scale``: where LongCat-style MLA applies its sqrt(hidden /
# lora_rank) norm scales — folded into the q_a/kv_a layernorm weights at load,
# or multiplied at runtime after q_b_proj / kv_a_layernorm as the trainer does.
MLA_LORA_SCALES = ("folded", "runtime")

# ``--moe-combine-order``: how a token's routed-expert contributions meet
# across the MoE TP-EP group — ``rank``: per-rank partials summed by the
# host's all-reduce / reduce-scatter, the identity zero-expert residual added
# around it; ``slot``: the MoE leaf folds the top-k slots in fp32 slot order
# across the EP group itself, residual included, as the trainer's grouped MLP
# does, and the host reduces nothing. Mirrors ``tokenspeed_kernel``'s
# ``moe.COMBINE_ORDERS``.
MOE_COMBINE_ORDERS = ("rank", "slot")

# ``--router-topk``: correction-bias MoE routing — the fused CUDA kernel, or
# fp32 ``torch.softmax`` + ``torch.topk(probs + bias)`` in PyTorch tie order
# with ``-1`` zero-expert ids, as the trainer does.
ROUTER_TOPKS = ("fused", "torch")

# ``--sampling-stream``: which random stream the FlashInfer backends' sampled
# (non-greedy) rows draw from. ``batch`` is flashinfer's Philox stream keyed by
# the batch row; ``per-request`` is the Gumbel-max pool route keyed by
# (request seed, position) and position-keyed chain verify coins, which the
# bitwise envelope requires.
SAMPLING_STREAMS = ("batch", "per-request")

YARN_FREQUENCY_DEVICES = ("cuda", "cpu")

# ``--yarn-ramp-mask-device`` selects the deepseek_yarn linear ramp device;
# ``--yarn-frequency-device`` selects powers, divisions and interpolation.
# Under auto, both devices default independently to CUDA.
# rl-bitwise pins a CPU ramp and CUDA frequency construction.
YARN_RAMP_MASK_DEVICES = ("cuda", "cpu")

EP_DISPATCH_ALGORITHMS: tuple[str, ...] = (
    "static",
    "dynamic",
    "fake",
    "static_with_zero_expert",
    "dynamic_with_zero_expert",
)

# Checkpoint architectures with decode monokernels (--decode-monokernel). A
# model adds its architecture here with its monokernel seam.
DECODE_MONOKERNEL_ARCHITECTURES = (
    "Inkling2ForConditionalGeneration",
    "Inkling2LoopForConditionalGeneration",
    "MuseSparkForConditionalGeneration",
    "Qwen4ExpForCausalLM",
    "Qwen4ExpForConditionalGeneration",
)

# Draft architectures with decode monokernels, and the target architectures
# whose drafts they serve. Under --decode-monokernel such a draft of such a
# target takes the switch too; any other draft (a Qwen3.8 target's DFlash2
# draft, a Muse Spark target's other drafts) keeps its regular layers.
DRAFT_MONOKERNEL_TARGETS = {
    "DFlash2DraftModel": ("MuseSparkForConditionalGeneration",),
    # The MTP draft of a Qwen3.8 target (hf_transformers_utils renames the
    # draft's architecture to it).
    "Qwen4ExpForCausalLMNextN": (
        "Qwen4ExpForCausalLM",
        "Qwen4ExpForConditionalGeneration",
    ),
}

# Muse Spark's --owned-attention-output choices.
OWNED_ATTENTION_OUTPUTS = ("reduce-scatter", "all-to-all")

# Muse Spark's --prefill-collective-payload choices (the Iris state's
# MUSE_PAYLOADS; a test checks they match).
PREFILL_COLLECTIVE_PAYLOADS = ("bf16", "int8-block16")

# Verify rules, named as vLLM's rejection_sample_method.
REJECTION_SAMPLE_METHODS = ("standard", "block", "synthetic")


def nullable_str(val: str) -> str | None:
    """Treat an empty CLI string or the spelling None as unset."""
    if not val or val == "None":
        return None
    return val


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def _uint16(value: str) -> int:
    """argparse type for values packed as a two-byte identity (``<H``)."""
    index = int(value)
    if not 0 <= index <= 0xFFFF:
        raise argparse.ArgumentTypeError(f"value must be in [0, 65535], got {value}")
    return index


def _nonempty_str(value: str) -> str:
    """argparse type for string flags that must not be empty."""
    if not value.strip():
        raise argparse.ArgumentTypeError("value must be a non-empty string")
    return value


@dataclass
class ServerArgsOptions:
    """Declare the fields and defaults shared by the launcher and engine."""

    # Model and tokenizer
    model: str
    tokenizer: str | None = None
    tokenizer_mode: str = "auto"
    load_format: str = "auto"
    trust_remote_code: bool = True
    dtype: str = "auto"
    kv_cache_dtype: str = "auto"
    kv_cache_quant_method: str = "none"
    quantization: str | None = None
    quantization_param_path: nullable_str = None
    max_model_len: int | None = None
    device: str = "cuda"
    served_model_name: str | None = None
    revision: str | None = None
    language_model_only: bool = False

    # The frontend handshake endpoint dialed by each engine rank: the
    # frontend binds the handshake/input/output sockets of the msgpack wire.
    data_parallel_address: str = "127.0.0.1"
    # Default chosen to avoid the 20000-29999 range used for derived
    # per-worker handshake ports and common rendezvous defaults of
    # co-located services.
    data_parallel_rpc_port: int = 30500
    zmq_engine_index: int = 0

    # Engine host and base port for derived control-plane endpoints.
    host: str = "127.0.0.1"
    port: int = 8000

    # Memory and scheduling
    gpu_memory_utilization: float | None = None
    max_num_seqs: int | None = None
    # Additional resident requests per attention-DP rank; forwards still use
    # max_num_seqs. Omission adds one rank-local batch on decode; zero disables
    # the extra capacity.
    disaggregation_decode_extra_slots: int | None = None
    max_total_tokens: int | None = None
    chunked_prefill_size: int | None = None
    max_prefill_tokens: int = 8192
    # Optional bound on summed per-request KV pages in a local prefill batch.
    chunk_prefill_kv_size: int | None = None
    # Opt-in, prefill servers: the most tokens a prefill chunk that cannot
    # finish its prompt gives up, page-aligned, so that bootstrapped prompts
    # waiting behind it that fit whole run in the same forward
    # (docs/design/scheduler.md section 1.1). 0 is off; at most half of
    # chunked_prefill_size; a decode server ignores it.
    prefill_chunk_yield_tokens: int = 0
    enable_mixed_batch: bool = False
    # Decode rounds owed to the running decodes after each prefill round; 0 is
    # off (docs/design/scheduler.md §3.3).
    prefill_decode_interval: int = 0
    # The same, owed even while an overdue prompt suspends the interval.
    prefill_decode_min_interval: int = 0
    # The most back-to-back prefill rounds an overdue prompt may suspend the
    # interval for; 0 keeps the unbounded suspension.
    max_consecutive_prefill_rounds: int = 0
    # The interval is owed only while at least this many requests are ready
    # to decode; below it --max-consecutive-prefill-rounds still ends a run of
    # prefill rounds that hold a decode back. 0 always owes the interval.
    prefill_interval_min_decoding: int = 0
    enable_short_prefill_first: bool = False
    # The fused engine's prefill-delay gate (docs/design/prefill-delay.md):
    # prefill work is held for a window of decode rounds and then batched, so
    # a scattered prompt does not interrupt every rank's decode. Unset keeps
    # the engine planning prefill every round.
    # Rounds between ALL attempt points: every rank that would decode also
    # has prefill work, so the group's prefills land in the same rounds.
    prefill_attempt_interval: int | None = None
    # Rounds between ANY attempt points: one rank's backlog opens the gate,
    # bounding how long a prefill waits. Must be >= prefill_attempt_interval.
    prefill_any_attempt_interval: int | None = None
    # One prefill round per window: the round after a prefill round is not
    # an ALL point while any rank decodes, so a backlog does not drain in
    # consecutive prefill rounds. Requires prefill_attempt_interval.
    prefill_single_round: bool = False
    # Opt-in: the prefill forward that spans a long prompt's last prefix-page
    # boundary before its replay tail (normally the final token) also writes
    # the recurrent state there, where hybrid models publish a checkpoint;
    # no chunk is cut for it. The prompt end that checkpoint covers and a
    # fused finish past the prompt end then publish no state and no KV tail:
    # a next turn that re-renders the end of the prompt and its reply cannot
    # resume there. A finish also demotes the in-forward checkpoint its
    # conversation published two turns back, which both cache tiers then evict
    # early. On a prefill server a prompt too short for the in-forward
    # checkpoint whose end lies inside a page publishes no state and no KV tail
    # there either (docs/design/scheduler.md section 1.2).
    enable_state_checkpoint_in_forward: bool = False
    # Hybrid models: a State checkpoint a later request's checkpoint extends
    # ranks, under cache pressure, as if last used this many new requests ago. None
    # is the scheduler's default, 16 times the per-rank batch size.
    checkpoint_retention_requests: int | None = None
    # Checkpoints within this many later requests of the newest on their
    # branch rank as it does. None is the scheduler's default, 0.
    checkpoint_retention_leaf_distance: int | None = None
    # JSON on the CLI, normalized once to generic runtime options.
    kv_offload_config: str | dict | None = None
    # special kv cache
    mamba_ssm_dtype: str = "float32"

    # Other runtime options
    stream_interval: int = 1
    seed: int | None = None
    distributed_timeout_seconds: int | None = None
    download_dir: str | None = None
    # Used for customizing extensible models
    ext_yaml: str | None = None
    base_gpu_id: int = 0
    gpu_id_step: int = 1

    # Logging
    log_level: str = "info"
    enable_async_logging: bool = False
    enable_log_request_stats: bool = False
    ttft_trace_sampling_percentage: int = 0
    enable_metrics: bool = False
    decode_log_interval: int = 40
    metrics_reporters: list[str] | None = None
    metrics_reporter_config: str | None = None
    # Port for the engine's Prometheus /metrics HTTP endpoint, served from the
    # engine's main process when --enable-metrics is on. None auto-allocates a
    # free port (logged at startup).
    metrics_port: int | None = None
    # Bind host for the /metrics endpoint. None binds the engine's --host.
    metrics_host: str | None = None
    app_key: str | None = None

    # Cache events
    kv_events_config: str | None = None

    # OTLP tracing: an OTLP/HTTP traces endpoint (a collector's /v1/traces,
    # Jaeger, Langfuse) and the share of requests it samples. 0 disables.
    otlp_traces_endpoint: str | None = None
    trace_sample_ratio: float = 0.0
    trace_provider: str | None = None

    # Version identifier for the model weights: the L3 cache's namespace,
    # which a flushed weight update that names a new version replaces.
    weight_version: str = "default"
    # Model Updater SDK (the ``UpdateWeightsFromMooncake`` control operation).
    # The config is an opaque JSON object handed to the SDK; the other three
    # are required with it and must stay unset without it (validated in
    # ``validate``).
    model_update_config: str | None = None
    # Import path of the SDK module exposing ``make_model_updater``,
    # ``ModelUpdaterConfig``, ``EngineType``, ``MooncakeWeightStore``,
    # ``FluentLlmEngineConfig`` and ``FluentLlmModelUpdateInitConfig``.
    model_update_sdk_module: str | None = None
    # ``EngineType`` member name the SDK resolves this engine as.
    model_update_engine_type: str | None = None
    # Whether an update also streams the speculative draft model's weights.
    model_update_draft_weights: Literal["retain", "refresh"] | None = None

    # Data parallelism
    data_parallel_size: int | None = None
    # Pipeline parallelism (prefill-only): number of stages. Only supported on
    # PD-disaggregated prefill servers; see resolve_disaggregation.
    pipeline_parallel_size: int = 1
    # Optional explicit per-stage layer counts, front to back (e.g.
    # "8,11,11,8"). Default None = even split, remainder to the front. Use to
    # lighten the embed (first) and lm_head (last) stages.
    pp_layer_partition: tuple[int, ...] | None = None
    load_watch_interval: float = 0.02

    # Expert parallelism
    ep_size: int = 1
    init_expert_location: str = "trivial"
    ep_num_redundant_experts: int = 0
    ep_dispatch_algorithm: (
        Literal[
            "static",
            "dynamic",
            "fake",
            "static_with_zero_expert",
            "dynamic_with_zero_expert",
        ]
        | None
    ) = None
    eplb_algorithm: str = "auto"
    # 'stat': int64 route counters per physical expert, read by the
    # EXPERT_LOAD profile activity and by --enable-eplb.
    expert_distribution_recorder_mode: Literal["stat"] | None = None
    # Online expert rebalancing; the two knobs below are required with it.
    enable_eplb: bool = False
    eplb_rebalance_num_iterations: int | None = None
    eplb_rebalance_layers_per_chunk: int | None = None

    # Dense GEMM selection is independent of routed-expert kernels.
    dense_gemm_backend: str = "auto"
    # Online W8A8 for Qwen3.5/Qwen4-Exp attention projections only.
    enable_fp8_attn_proj: bool = False

    # MoE backend
    moe_backend: str = "auto"
    draft_moe_backend: str | None = None
    # GEMM2 atomic expert combine for unquantized FlashInfer CUTLASS MoE.
    moe_cutlass_fused_finalize: bool = True
    # Opt-in: run MXFP4 routed experts with FP8 activations (FlashInfer cutlass
    # W4A8 on Hopper). Off keeps the checkpoint's BF16 activation contract.
    moe_mxfp4_fp8_activation: bool = False
    all2all_backend: str = "none"
    deepep_mode: Literal["auto", "normal", "low_latency"] = "auto"
    disable_flashinfer_cutlass_moe_fp4_allgather: bool = False

    # KVStore
    enable_kvstore: bool = False
    kvstore_ratio: float = 2.0
    kvstore_size: int = 0
    kvstore_host_headroom_gib: int = 10
    kvstore_release_checkpoint_cache: bool = False
    kvstore_io_backend: str = "kernel"
    # When Device prefix-cache pages are copied to Host L2 (--kvstore-write-policy).
    kvstore_write_policy: str = "write_through"
    # Optional L3 storage beneath the compact Host cache.
    kvstore_storage_backend: str | None = None
    kvstore_storage_backend_extra_config: str | None = None
    # A paged L2 pool shared by every prefill rank, in place of their Host L2
    # (--shared-l2-config, a JSON object).
    shared_l2_config: str | None = None
    # Peer prefix sharing (docs/rfcs/0004-peer-prefix-sharing.md): the port
    # the node's replica leaders share for external control messages;
    # None turns it off.
    peer_prefix_port: int | None = None
    # The host other replicas and the router reach this replica's endpoint at;
    # its name is host:port. None takes this host's detected address.
    peer_prefix_host: str | None = None
    peer_prefix_answer_timeout_ms: int = 50
    peer_prefix_min_fetch_tokens: int = 256
    # Where a peer prefix lands: "device" (the request's admitted Device
    # pages) or "host" (Host blocks before its admission). Required with
    # peer sharing; no default.
    peer_prefix_landing_tier: str | None = None
    # The Disk tier: the node's local SSD below Host L2
    # (docs/rfcs/0006-disk-tier.md). One directory per drive; None turns it
    # off. The I/O backend and the write budget have no default.
    disk_tier_paths: list[str] | None = None
    disk_tier_size_gb: float = 0.0
    disk_tier_io: str | None = None
    disk_tier_write_mbps: int | None = None
    disk_tier_config: str | None = None
    # Retraction snapshot pool (pinned Host, per rank) and its slot-state
    # arena rows: a retracted request is suspended with its image and
    # resumed by restore; a victim whose image does not fit is aborted
    # instead. Sized like the KVStore: an explicit size in gigabytes wins
    # (0 = not set), else a ratio of this rank's Device KV capacity (an
    # explicit 0 disables the pool: every capacity block then aborts its
    # victim), else derived at device build -- 0.1 of the Device KV without
    # L2, the tails of ``retraction_snapshot_max_requests`` images with L2
    # (cache/l2/sizing.py). When 0, the request cap derives from rank-local
    # resident capacity, including decode extra slots (resolve_retraction_snapshot_options).
    retraction_snapshot_host_gb: float = 0.0
    retraction_snapshot_ratio: float | None = None
    retraction_snapshot_max_requests: int = 0
    # Test knob: every |N| plans the scheduler retracts the oldest quiescent
    # Decoding (N > 0) or Prefilling (N < 0) request without capacity
    # pressure, to exercise the suspend/restore path. 0 (off) in serving.
    debug_force_retraction_interval: int = 0
    # None preserves unlimited debug injection; a supplied cap is positive.
    # Capacity-triggered retractions do not consume this per-admission budget.
    debug_force_retraction_max_per_request: int | None = None

    # Multi-node distributed serving. ``None`` means "not given by the user",
    # which is what lets the launcher environment fill them in.
    dist_init_addr: str | None = None
    nnodes: int | None = None
    node_rank: int | None = None

    # Hugging Face model config overrides in JSON
    hf_overrides: str = "{}"

    # Kernel backend
    attention_backend: str | None = None
    min_kv_split: int | None = None
    kda_backend: str = "auto"
    # Records a request keeps per KDA layer between dense-state folds,
    # including the verify window; None keeps max(16, draft tokens).
    kda_history_capacity: int | None = None
    drafter_attention_backend: str | None = None
    # BLASST skip-softmax sparsity, gluon MHA prefill only (gfx950). 0.0
    # (default) is exact dense attention; see --skip-softmax-threshold help.
    skip_softmax_threshold: float = 0.0
    # Opt-in: large QSA prefill calls over an FP8 KV cache attend a BF16 copy
    # of their pages, reserved from the KV cache budget; see the flag's help.
    enable_qsa_prefill_bf16_scratch: bool = False
    enable_ple_hot_cache: bool = False
    # Pack Qwen4-Exp target and draft cache fields into one physical plane.
    qwen4_exp_flat_kv: bool = True
    sampling_backend: str | None = None
    # Random stream of the non-greedy rows of the FlashInfer sampling backends:
    # "batch" keys flashinfer's Philox stream by the batch row, so a request's
    # draw depends on its co-batch; "per-request" keys it by (request seed,
    # position) through the Gumbel-max pool kernels and keys chain verify
    # coins by the request seed and verify round's first position. See
    # docs/design/numerics.md, sampling.deterministic.
    sampling_stream: str = "batch"
    dp_sampling: bool = False
    dp_sampling_min_bs: int | None = None
    attention_use_fp4_indexer_cache: bool | None = None
    use_trtllm_ragged_deepseek_prefill: bool | None = None

    # KV parallelism: shard the full-history KV pages page-cyclically over a
    # consecutive subgroup of attention TP (1 = off).
    kv_parallel_size: int = 1
    # Decode context parallelism in aggregated or PD decode engines: attend owned
    # pages and merge the partials across the KVP group (1 = off; equals
    # kv_parallel_size when on). Names the compute; the backends key the DCP
    # path on the mapping's kvp_size, which the drafter's steps on a sharded
    # prefill engine take as well.
    decode_context_parallel_size: int = 1
    # Query context parallelism on the PD prefill role: shard every extend
    # forward's rows over the attention TP group (1 = off; equals
    # kv_parallel_size when on).
    query_context_parallel_size: int = 1
    disable_deepseek_v41_token_blacklist: bool = False
    deepseek_v41_attention_query_swa_min_tokens: int = 4096
    deepseek_v41_attention_query_min_tokens: int = 1024
    deepseek_v41_attention_query_parallel_size: int = 1
    deepseek_v41_indexer_query_parallel_size: int = 1
    deepseek_v4_mega_moe_max_num_tokens: int = 0
    deepseek_v4_indexer_prefill_max_logits_mb: int = 512
    # DeepSeek V4.1 Engram host tables. Off by default (GPU-sharded).
    engram_host_table: bool = False
    engram_host_table_dir: str | None = None
    engram_host_table_layout: str = "auto"
    engram_host_table_alloc: str = "auto"
    engram_host_table_numa: str = "interleave"
    # Run each decoder layer's small decode steps as one persistent
    # monokernel launch (DECODE_MONOKERNEL_ARCHITECTURES). Off by default.
    decode_monokernel: bool = False
    # A directory for each rank's decode monokernel stuck-poll record file
    # (None: the records stay in pinned memory).
    monokernel_stuck_record_dir: str | None = None
    # A variable-loop Inkling2 checkpoint's loop passes (None: the
    # checkpoint's num_passes), and whether all passes of a core layer share
    # one KV cache (False: one per pass, the checkpoint's layout).
    inkling2_loop_num_passes: int | None = None
    inkling2_loop_share_cache: bool = False

    # Grammar backend. None means the checkpoint default has not been resolved.
    grammar_backend: str | None = None
    # Model-declared constrained decoding policy, reported through server info.
    constrained_decoding: Literal["requested", "always"] | None = None
    # Lets the host defer json_schema grammars past the model's reasoning
    # channel (``reasoning_structural_tag.reasoning_grammar``).
    reasoning_parser: str | None = None
    grammar_compile_timeout_secs: float = 30.0
    grammar_compile_max_retries: int = 2
    disable_any_whitespace: bool = False
    # Force the synchronous eager grammar fallback even on CUDA. Useful
    # for parity-testing against the captured-grammar path (output should
    # match; throughput will be lower since the sync stalls every step).
    disable_capturable_grammar: bool = False

    # Speculative decoding
    draft_model_path_use_base: bool | None = False
    speculative_config: str | None = None
    speculative_algorithm: str | None = None
    speculative_draft_model_path: str | None = None
    speculative_draft_model_revision: str | None = None
    speculative_draft_model_quantization: str | None = "unquant"
    speculative_num_steps: int = 3
    speculative_eagle_topk: int = 1
    speculative_num_draft_tokens: int | None = None
    enable_dsv41_adaptive_dspark: bool = False
    dsv41_dspark_min_batch_size: int = 32
    synthetic_acceptance_length: float | None = None
    # Standard (draft-prob) rejection sampling for the chain drafters: the
    # drafter samples each step from its own distribution q and records it,
    # verify accepts with coin * q(x) < p(x). Off: the target-only rule
    # (accept with probability p(x) whatever the proposal). Both serve the
    # target distribution; this one trades a sampled proposal for a higher
    # acceptance rate under sampling temperatures.
    enable_speculative_sampling: bool = False
    # Verify rule, named as vLLM's rejection_sample_method. "standard" is the
    # existing token-wise rule; "block" is block verification of a sampled
    # DSpark or EAGLE3 topk=1 chain; "synthetic" forces benchmark acceptance
    # lengths (set by --synthetic-acceptance-length).
    speculative_rejection_sample_method: Literal["standard", "block", "synthetic"] = (
        "standard"
    )
    # Recorded draft probabilities above this value mark a slot with no
    # proposal (fresh admission, PD landing): always reject. Sentinel rows
    # are written as threshold + 1, so it must lie within
    # [SPEC_REJECT_DRAFT_PROB_THRESHOLD_MIN, SPEC_REJECT_DRAFT_PROB_THRESHOLD_MAX].
    spec_reject_draft_prob_threshold: float = 2.0
    enable_replay_ssm: bool = True
    enable_deferred_gdn_state: bool = False
    eagle3_layers_to_capture: str | None = None
    # EAGLE-Harness dynamic draft trees for EAGLE4 (see the flags' help).
    speculative_exec_mode: int = -1
    speculative_data_path: str | None = None
    speculative_compute_intensity: str = "high_intensity"
    # EAGLE-Harness chains: the verify width per live batch size ("MIN_BS:WIDTH,...", or "off").
    speculative_chain_width_table: str = "off"
    # Logprob support flags — all OFF by default. Enabling extends the
    # captured CUDA-graph footprint; requests asking for logprobs on a
    # server started without the matching flag will receive empty logprobs.
    enable_output_logprobs: bool = False
    # Optional diagnostics over the same distribution as output logprobs.
    enable_output_cumulative_probs: bool = False
    # Sizing knob for prompt (input) logprobs: prompt rows pushed through the
    # LM head per chunk, bounding the transient [rows, vocab] logits
    # (chunk x vocab x (2 + 2 + 4) bytes: bf16 shard, bf16 gathered, fp32
    # log-softmax).
    input_logprob_chunk_tokens: int = 256

    # Runtime options
    disable_pdl: bool = False
    enable_prefix_caching: bool = True
    disable_kvstore: bool = False
    enforce_eager: bool = False
    enable_dp_local_cuda_graph: bool = False
    disable_cuda_graph_padding: bool = False
    disable_autotune: bool = False
    enable_startup_requests: bool = False
    enable_cudagraph_gc: bool = False
    disable_nccl_nvls: bool = False
    disable_symm_mem: bool = False
    disable_overlap_schedule: bool = False
    disable_tf32: bool = False
    force_deterministic_rsag: bool = False
    batch_invariant_collectives: bool = False
    disable_sampling_tp_sync: bool = False
    # Numerics envelope: "auto" keeps every performance default; "rl-bitwise"
    # asks for bitwise run-to-run and batch-composition invariance and the
    # trainer's operation order, and folds the determinism and trainer-order
    # switches below (resolve_numerics). Each folded switch can still be set
    # individually; the umbrella only ever tightens.
    numerics: str = "auto"
    # Muse Spark: the payload encoding of the owned-row prefill collectives
    # (the QKVG all-to-all and the attention and expert reduce-scatters) in
    # extend forwards. "bf16" transfers the rows as produced. "int8-block16"
    # transfers INT8 codes with an FP32 scale per 16 elements, decoded to BF16
    # by the receiver before the collective's arithmetic: 1.25 bytes an
    # element instead of 2, and a numerics change. Decode and verify forwards
    # stay BF16.
    prefill_collective_payload: str = "bf16"
    # Muse Spark: how owned-row prefill (extend) forwards return attention
    # outputs to the token owners. "reduce-scatter": every rank projects its
    # heads of all rows and the partial sums are reduce-scattered.
    # "all-to-all": each owner receives its rows of every rank's heads and
    # projects them through the whole output projection, gathered on every
    # rank after loading (128 MiB a layer, about 7.75 GiB a rank over 62
    # layers); fewer bytes, a different summation order. Refused where Iris
    # owned rows do not run.
    owned_attention_output: str = "reduce-scatter"
    # Muse Spark's dense FP8 projections on gfx950 (docs/configuration/server.md).
    dense_fp8_small_m: str = "none"
    # Trainer-operation-order switches (docs/design/numerics.md,
    # alignment.trainer). Each keeps the engine's own form by default and is
    # folded to the trainer's form by --numerics rl-bitwise.
    # Device that computes the deepseek_yarn linear ramp mask.
    yarn_ramp_mask_device: str = "cuda"
    # Device that computes deepseek_yarn powers, divisions and interpolation.
    yarn_frequency_device: str = "cuda"
    # Where LongCat-style MLA applies its sqrt(hidden / lora_rank) norm scales:
    # folded into the q_a/kv_a layernorm weights at load, or multiplied at
    # runtime after q_b_proj / kv_a_layernorm as the trainer does.
    mla_lora_scale: str = "folded"
    # The norm at each physical layer boundary (a layer's first norm, the
    # final norm): the fused add+norm kernel, or a bf16 `hidden + residual`
    # materialized first as the trainer does.
    layer_boundary_norm: str = "fused"
    # Correction-bias MoE routing: the fused CUDA kernel, or fp32 torch.softmax
    # + torch.topk(probs + bias) in PyTorch tie order as the trainer does.
    router_topk: str = "fused"
    # Order of the selected-token log-softmax: torch.log_softmax, or
    # Megatron's vocab-parallel cross-entropy over fixed 32768-wide vocab
    # blocks. Changes the reported logprobs only, never the sampled tokens.
    logprob_order: str = "torch"
    # How a token's routed-expert contributions meet across the MoE TP-EP
    # group: per-rank partials summed by the host (rank), or folded in fp32
    # slot order inside the MoE leaf as the trainer does (slot).
    moe_combine_order: str = "rank"
    # The order the sparse attention cores reduce a token's selected KV slots
    # in: as the top-k leaf emitted them (selection), or ascending (sorted,
    # batch-invariant whenever the selected set is).
    dsa_slot_order: str = "selection"
    # Storage of Kimi-K3.1's DSA indexer key plane; None keeps its default.
    # Other models refuse a set value.
    dsa_index_k_format: str | None = None
    low_latency_max_num_tokens_per_gpu: int = 256
    max_cudagraph_capture_size: int | None = None
    disable_prefill_graph: bool | None = False
    disable_kda_prefill_graph: bool = False
    # Breakable prefill graph bucket cap: None = auto min(8192, chunk); 0 disables.
    prefill_graph_max_tokens: int | None = None
    # Explicit prefill bucket list; unset = the relative-stride ladder (see get_prefill_token_buckets).
    prefill_graph_capture_sizes: list[int] | None = None
    # Request capacities for inline attention; unset keeps the minimum per bucket.
    prefill_graph_capture_batch_sizes: list[int] | None = None
    cudagraph_capture_sizes: list[int] | None = None
    enable_nan_detection: bool = False
    enable_nvtx: bool = False
    weight_loader_prefetch_checkpoints: bool = True
    weight_loader_prefetch_num_threads: int = 8
    enable_memory_saver: bool = False
    # Dev: permit the reload_forward control call; nothing reloads until it arrives.
    allow_reload_forward: bool = False
    weight_memory_policy: Literal["cpu_backup", "discard"] | None = None
    disable_cudagraph_memory_reserve: bool = False
    mla_disable_ragged: bool = False

    # parallel strategy
    nprocs_per_node: int | None = None
    world_size: int | None = None
    attn_tp_size: int | None = None
    # Decode-side layouts under attention DP: head-shard the MLA head
    # projections / vocab-shard the LM head over contiguous DP ranks, and
    # make the sharded o_proj / dense down_proj column-parallel on hidden so
    # no cross-rank reduction remains outside MoE (TP batch invariance).
    attn_head_tp_size: int | None = None
    # Rows per rank the head-TP Lamport exchanges are prepared for. The
    # scratch is allocated before graph capture, so larger forwards fall
    # back to the generic all-to-all instead of growing it at runtime.
    head_tp_exchange_max_rows: int = 256
    ple_table_layout: str = "global_sharded"
    lm_head_tp_size: int | None = None
    tp_batch_invariant: Literal["none", "attn", "attn+dense"] = "none"
    dense_tp_size: int | None = None
    moe_tp_size: int | None = None
    moe_weight_sharing_size: int | None = None
    # Keep the field order while the runtime supplies the concrete Mapping type.
    mapping: object | None = None
    emulate_rank_zero: bool = False

    mla_chunk_multiplier: int = 4
    mm_attention_backend: str | None = None
    mm_encoder_tp_mode: Literal["weights", "data"] = "weights"

    # For PD/EPD disaggregation: "null", "prefill", "decode", or "encode" (vision-tower-only).
    enable_external_encoder: bool = False
    disaggregation_mode: str = "null"
    # Prefill role only; None resolves to "early" there (resolve_disaggregation).
    disaggregation_decode_allocation_policy: str | None = None
    disaggregation_bootstrap_port: int = 8998
    disaggregation_transfer_backend: str = "mooncake"
    disaggregation_ib_device: str | None = None
    disaggregation_layerwise_interval: int = 1
    pdlb_url: str | None = None

    # For communication + norm fusion
    comm_fusion_max_num_tokens: int = 2048
    enable_allreduce_fusion: bool = False

    enable_expert_parallel: bool = False


def add_server_cli_args(
    parser: argparse.ArgumentParser, *, numerics_envelopes: Sequence[str] | None = None
) -> None:
    """Register the flags and the runtime-provided numerics choices, if supplied."""
    parser.allow_abbrev = False

    # Model and port args
    parser.add_argument(
        "model_path",
        nargs="?",
        metavar="model",
        default=None,
        help="The model name or path (positional argument). Equivalent to --model.",
    )
    parser.add_argument(
        "--model",
        "--model-path",
        metavar="MODEL",
        type=str,
        default=None,
        help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--tokenizer",
        metavar="TOKENIZER",
        type=str,
        default=ServerArgsOptions.tokenizer,
        help="The path of the tokenizer.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=ServerArgsOptions.host,
        help="The host of the server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=ServerArgsOptions.port,
        help="The port of the server.",
    )
    parser.add_argument(
        "--tokenizer-mode",
        type=str,
        default=ServerArgsOptions.tokenizer_mode,
        choices=["auto", "slow"],
        help="Tokenizer mode. 'auto' will use the fast "
        "tokenizer and model-specific tokenizer hooks if available, "
        "'slow' will always use the slow tokenizer.",
    )
    parser.add_argument(
        "--language-model-only",
        action="store_true",
        default=ServerArgsOptions.language_model_only,
        help="Skip vision/audio encoders on a multimodal checkpoint and "
        "run text-only. Multimodal requests are rejected.",
    )
    parser.add_argument(
        "--data-parallel-address",
        type=_nonempty_str,
        default=ServerArgsOptions.data_parallel_address,
        help="Host of the frontend-bound handshake ROUTER the engine "
        "ranks connect to (default "
        f"{ServerArgsOptions.data_parallel_address}).",
    )
    parser.add_argument(
        "--data-parallel-rpc-port",
        type=_uint16,
        default=ServerArgsOptions.data_parallel_rpc_port,
        help="Port of the frontend-bound handshake ROUTER (default "
        f"{ServerArgsOptions.data_parallel_rpc_port}, outside the smg frontend's "
        "derived-port band 20000..=29999). `smg serve --backend "
        "tokenspeed --connection-mode zmq` passes an explicit port "
        "derived from the worker's ipc:// URL instead.",
    )
    parser.add_argument(
        "--zmq-engine-index",
        type=_uint16,
        default=ServerArgsOptions.zmq_engine_index,
        help="This engine's index on the msgpack wire; used as the two-byte "
        "little-endian ZMQ routing identity SMG addresses it by "
        "(0..65535).",
    )
    parser.add_argument("--ext-yaml", type=str, default=None)
    parser.add_argument(
        "--load-format",
        type=str,
        default=ServerArgsOptions.load_format,
        choices=[
            "auto",
            "pt",
            "safetensors",
            "instanttensor",
            "npcache",
            "dummy",
            "extensible",
        ],
        help="The format of the model weights to load. "
        '"auto" will try to load the weights in the safetensors format '
        "and fall back to the pytorch bin format if safetensors format "
        "is not available. "
        '"pt" will load the weights in the pytorch bin format. '
        '"safetensors" will load the weights in the safetensors format. '
        '"instanttensor" accelerates safetensors loading on NVIDIA GPUs '
        "via distributed loading, pipelined prefetching, and direct I/O "
        "(with optional GPUDirect Storage support). "
        '"npcache" will load the weights in pytorch format and store '
        "a numpy cache to speed up the loading. "
        '"dummy" will initialize the weights with random values.',
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether or not to allow for custom models defined on the Hub in their own modeling files.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=ServerArgsOptions.dtype,
        choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
        help="Data type for model weights and activations.\n\n"
        '* "auto" will use FP16 precision for FP32 and FP16 models, and '
        "BF16 precision for BF16 models.\n"
        '* "half" for FP16. Recommended for AWQ quantization.\n'
        '* "float16" is the same as "half".\n'
        '* "bfloat16" for a balance between precision and range.\n'
        '* "float" is shorthand for FP32 precision.\n'
        '* "float32" for FP32 precision.',
    )
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        default=ServerArgsOptions.kv_cache_dtype,
        choices=["auto", "bfloat16", "fp8", "fp8_e4m3", "mxfp8"],
        help='Data type for kv cache storage. "auto" and "bfloat16" store BF16 '
        'rows (fp16 activations convert on write). "fp8" is an alias for '
        '"fp8_e4m3" (unit scale, or the checkpoint\'s per-tensor KV-cache '
        'scales where the model loads them). "mxfp8" stores '
        "block-scaled fp8-e4m3 (one UE8M0 scale per 32 head_dim elements) and "
        "requires an MHA attention backend.",
    )
    parser.add_argument(
        "--kv-cache-quant-method",
        type=str,
        default=ServerArgsOptions.kv_cache_quant_method,
        choices=["none", "per_token_head"],
        help="kv cache quant method",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=ServerArgsOptions.quantization,
        choices=[
            "fp8",
            "mxfp4",
            "nvfp4",
            "w8a8_fp8",
            "w8a16_fp8_block",
            "compressed-tensors",
        ],
        help="The quantization method.",
    )
    parser.add_argument(
        "--quantization-param-path",
        type=nullable_str,
        default=None,
        help="Path to the JSON file containing the KV cache "
        "scaling factors. FP8 KV cache runs unscaled, so under FP8 every "
        "factor in the file must be 1.0.",
    )
    parser.add_argument(
        "--max-model-len",
        metavar="MAX_MODEL_LEN",
        type=int,
        default=ServerArgsOptions.max_model_len,
        help="The model's maximum context length. Defaults to None (will use the value from the model's config.json instead).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "npu"],
        help="The device type.",
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        default=ServerArgsOptions.served_model_name,
        help="The model's name in metric labels. Under `ts serve`, requests may also name "
        "the model by it.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="The specific model version to use. It can be a branch "
        "name, a tag name, or a commit id. If unspecified, will use "
        "the default version.",
    )
    # Memory and scheduling
    parser.add_argument(
        "--gpu-memory-utilization",
        metavar="GPU_MEMORY_UTILIZATION",
        type=float,
        default=ServerArgsOptions.gpu_memory_utilization,
        help="The fraction of GPU memory to use for model weights and KV cache. Use a smaller value if you see out-of-memory errors.",
    )
    parser.add_argument(
        "--max-num-seqs",
        metavar="MAX_NUM_SEQS",
        type=int,
        default=ServerArgsOptions.max_num_seqs,
        help="Maximum number of sequences to process concurrently.",
    )
    parser.add_argument(
        "--disaggregation-decode-extra-slots",
        type=int,
        default=ServerArgsOptions.disaggregation_decode_extra_slots,
        help="Additional resident request slots per attention-DP rank on a "
        "decode server. These receive KV without increasing the execution "
        "concurrency set by --max-num-seqs. If omitted, defaults to "
        "--max-num-seqs divided by the attention DP size (integer division) "
        "on decode, and zero on other roles. Set to zero to disable the "
        "extra capacity. Persistent request state grows with resident "
        "capacity and can reduce the KV budget; all requests share one KV "
        "pool. The snapshot request limit defaults to resident capacity "
        "unless explicitly configured.",
    )
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=ServerArgsOptions.max_total_tokens,
        help="The maximum number of tokens in the memory pool. If not specified, it will be automatically calculated based on the memory usage fraction. "
        "This overrides the automatically calculated token pool size.",
    )
    parser.add_argument(
        "--chunked-prefill-size",
        metavar="CHUNKED_PREFILL_SIZE",
        type=int,
        default=ServerArgsOptions.chunked_prefill_size,
        help="Maximum number of tokens the scheduler may issue in a single iteration. Setting this to -1 disables chunked prefill.",
    )
    parser.add_argument(
        "--prefill-chunk-yield-tokens",
        type=int,
        default=ServerArgsOptions.prefill_chunk_yield_tokens,
        metavar="N",
        help="Prefill servers: the room, in tokens, a prefill chunk that "
        "cannot finish its prompt makes for bootstrapped prompts waiting "
        "behind it that fit whole (counted past their prefix hit), so they "
        "run complete in the same forward instead of waiting for the long "
        "prompt's chunk to pass. The chunk ends on an earlier page and gives "
        "up fewer than N plus one page, only while such prompts wait and "
        "only what they take. 0 (default) is off: no chunk gives way; at "
        "most half of --chunked-prefill-size. A decode server ignores it. "
        "Whatever N, a prefill server fills the budget a chunk leaves on "
        "its own with bootstrapped prompts that fit whole in it. See "
        "docs/design/scheduler.md section 1.1.",
    )
    parser.add_argument(
        "--enable-short-prefill-first",
        action="store_true",
        default=ServerArgsOptions.enable_short_prefill_first,
        help="Prefer new local prefill requests with less uncached work, with "
        "oldest-request starvation protection. Off by default; resident "
        "chunks retain priority. Applies to fused and prefill roles.",
    )
    parser.add_argument(
        "--enable-mixed-batch",
        action="store_true",
        dest="enable_mixed_batch",
        default=ServerArgsOptions.enable_mixed_batch,
        help="Allow the scheduler to issue prefill and decode requests in the "
        "same iteration.",
    )
    parser.add_argument(
        "--prefill-decode-interval",
        type=int,
        default=ServerArgsOptions.prefill_decode_interval,
        help="After a round that runs a prefill chunk, run this many decode "
        "rounds before the next prefill round, unless no decode can run. "
        "Every chunk starts a new interval, but no prompt is held back "
        "more than this many decode rounds in total. Trades TTFT for "
        "fewer decode stalls. Aggregated serving only. 0 (the default) is "
        "off.",
    )
    parser.add_argument(
        "--prefill-decode-min-interval",
        type=int,
        default=ServerArgsOptions.prefill_decode_min_interval,
        help="After every round that runs a prefill chunk, run at least "
        "this many decode rounds before the next prefill round, unless no "
        "decode can run, even while --prefill-decode-interval is suspended "
        "for a prompt it has held back in full. A long prompt's chunks "
        "then alternate with decode rounds instead of stalling the decodes "
        "back to back. Aggregated serving only. 0 (the default) is off.",
    )
    parser.add_argument(
        "--max-consecutive-prefill-rounds",
        type=int,
        default=ServerArgsOptions.max_consecutive_prefill_rounds,
        help="With --prefill-decode-interval: the most back-to-back prefill "
        "rounds that hold a decode back while an overdue prompt suspends "
        "the interval; the round that reaches it owes the decodes the whole "
        "interval again, and a prompt's held-back count restarts whenever "
        "one of its chunks runs. 0 (the default) keeps the unbounded "
        "suspension.",
    )
    parser.add_argument(
        "--prefill-interval-min-decoding",
        type=int,
        default=ServerArgsOptions.prefill_interval_min_decoding,
        help="With --max-consecutive-prefill-rounds (which needs "
        "--prefill-decode-interval): owe the "
        "--prefill-decode-interval and --prefill-decode-min-interval decode "
        "rounds only while at least this many requests are ready to decode. "
        "With fewer, waiting prompts are prefilled at once, but once "
        "--max-consecutive-prefill-rounds prefill rounds in a row have held "
        "a decode back, the next round decodes. Aggregated serving only. "
        "0 (the default) always owes them.",
    )
    parser.add_argument(
        "--enable-state-checkpoint-in-forward",
        action="store_true",
        dest="enable_state_checkpoint_in_forward",
        default=ServerArgsOptions.enable_state_checkpoint_in_forward,
        help="Hybrid (recurrent-state) models: the prefill forward that spans "
        "the last prefix-page boundary before a long prompt's replay tail "
        "(normally the final token) also writes the recurrent state there, so "
        "a later request that resends the prompt or diverges from it after "
        "that boundary resumes there. No extra prefill round; that forward "
        "holds one more state block. The prompt end in the page that "
        "boundary starts and every finish past the prompt end then publish "
        "no recurrent state and no KV tail to the prefix cache, since a next "
        "turn that re-renders the end of the previous prompt and its reply "
        "cannot resume there (a prompt end published at the request's "
        "finish, as on a prefill server, is kept if the state at the "
        "boundary has left the cache); a next turn that repeats the "
        "previous reply verbatim resumes at an earlier checkpoint instead "
        "and recomputes the reply. When a request publishes an in-forward "
        "checkpoint past the one it resumed from, which its own predecessor "
        "wrote past an earlier one, that earlier checkpoint -- two turns "
        "back on the conversation -- is evicted early from the device and "
        "host caches, and no new store or eviction write-back copies it to "
        "host, unless another request also resumed from it, or acquired or "
        "republished it after the conversation's previous turn resumed "
        "there. On a prefill server, a prompt too short for the in-forward "
        "checkpoint (by default, one no longer than the chunk budget) whose "
        "end lies inside a prefix page publishes only its full pages, no "
        "recurrent state and no KV tail at its end, since a next turn that "
        "re-renders the end of the prompt cannot resume there; a page-aligned "
        "prompt end keeps its state, and a fused server keeps short prompt "
        "ends. Requires prefix caching; a decode server ignores it. See "
        "docs/design/scheduler.md section 1.2.",
    )
    parser.add_argument(
        "--checkpoint-retention-requests",
        type=int,
        dest="checkpoint_retention_requests",
        default=ServerArgsOptions.checkpoint_retention_requests,
        help="Hybrid (recurrent-state) models: under cache pressure, a State "
        "checkpoint that a later request's checkpoint extends is evicted "
        "before cache entries used within this many new requests on its "
        "attention-DP rank, while entries unused for longer are evicted "
        "first. 0 evicts every entry by its own last use (plain LRU). Set "
        "it to about 16 times the expected concurrent requests per "
        "attention-DP rank. "
        "Default: 16 times --max-num-seqs per attention-DP rank. See "
        "docs/design/cache-concepts.md.",
    )
    parser.add_argument(
        "--checkpoint-retention-leaf-distance",
        type=int,
        dest="checkpoint_retention_leaf_distance",
        default=ServerArgsOptions.checkpoint_retention_leaf_distance,
        help="Hybrid (recurrent-state) models: State checkpoints within this "
        "many later requests of the newest on their branch rank as the newest "
        "does under --checkpoint-retention-requests. 0 protects only the "
        "newest request's checkpoints; 1 also the previous turn's. From 0 "
        "to 4. Default: 0.",
    )

    # KVStore
    parser.add_argument(
        "--kv-offload-config",
        type=str,
        default=ServerArgsOptions.kv_offload_config,
        help='Sparse KV offload JSON object with explicit "layers", "hot_tokens", '
        '"host_gb" (GiB), and "overlap". Model-specific options remain in '
        "--hf-overrides; unsupported cache recipes reject this configuration.",
    )
    parser.add_argument(
        "--disable-kvstore",
        action="store_true",
        help="Disable KVStore",
    )
    parser.add_argument(
        "--kvstore-ratio",
        type=float,
        default=ServerArgsOptions.kvstore_ratio,
        help="The ratio of Host L2 LCM block count to Device LCM block count. "
        "Host blocks use compact payload packing, so this is not a ratio of allocated bytes. "
        "Capped, with a warning, to what fits in this rank's share of the Host memory its node has available.",
    )
    parser.add_argument(
        "--kvstore-size",
        type=int,
        default=ServerArgsOptions.kvstore_size,
        help="The size of the KVStore host memory pool in gigabytes, which will override kvstore_ratio if set. "
        "Refused at startup if it does not fit in this rank's share of the Host memory its node has available, "
        "or if the ranks resolve it to different page counts, as pipeline stages may (use --kvstore-ratio there).",
    )
    parser.add_argument(
        "--kvstore-io-backend",
        type=str,
        choices=["direct", "kernel"],
        default=ServerArgsOptions.kvstore_io_backend,
        help="The IO backend for KVStore transfer between CPU and GPU.",
    )
    parser.add_argument(
        "--kvstore-release-checkpoint-cache",
        action="store_true",
        help="Advise Linux to evict local safetensors checkpoint pages after loading, before pinning the Host cache.",
    )
    parser.add_argument(
        "--kvstore-host-headroom-gib",
        type=int,
        default=ServerArgsOptions.kvstore_host_headroom_gib,
        help="GiB of available Host memory per local rank to reserve outside the pinned cache and retraction pools.",
    )
    parser.add_argument(
        "--kvstore-write-policy",
        type=str,
        choices=["write_through", "write_back"],
        default=ServerArgsOptions.kvstore_write_policy,
        help="What feeds Host L2 (the KVStore) from the GPU prefix cache. "
        "'write_through' (the default) copies every page the fused or "
        "prefill role caches. "
        "'write_back' copies only the pages the GPU evicts, in the "
        "scheduling step that evicts them (that step's forward waits for "
        "the copy), so the two tiers hold mostly different pages; on "
        "--disaggregation-mode decode a remote admission then loads its "
        "Host hit and the prefill node sends only the rest. "
        "write_back is refused with "
        "--disable-kvstore, --kvstore-storage-backend, "
        "--pipeline-parallel-size > 1 or --query-context-parallel-size > 1, "
        "and at startup for a model with more than 64 cache groups; see "
        "docs/configuration/server.md.",
    )
    parser.add_argument(
        "--shared-l2-config",
        type=str,
        default=ServerArgsOptions.shared_l2_config,
        help="JSON object joining a shared L2 pool, which takes the place of "
        "this rank's Host L2 (requires --disable-kvstore; prefill and fused "
        "instances of attention TP 1; not with L3 storage, KV offloading, "
        "retraction, --enable-memory-saver or --kvstore-write-policy "
        "write_back). Fields: owner_address, device_names, "
        "request_timeout_ms, join_timeout_s, read_workers, write_workers, "
        "scheduler {max_writes_in_flight, max_keys_per_write}; see "
        "docs/configuration/server.md.",
    )
    parser.add_argument(
        "--kvstore-storage-backend",
        type=str,
        choices=["mooncake", "memory"],
        default=ServerArgsOptions.kvstore_storage_backend,
        help="L3 store under compact Host (flat) KV. "
        "'mooncake' is Mooncake Store (SGLang/vLLM HiCache equivalent). "
        "'memory' is an in-process dict for tests. Requires Host L2 "
        "(do not pass --disable-kvstore).",
    )
    parser.add_argument(
        "--kvstore-storage-backend-extra-config",
        type=str,
        default=ServerArgsOptions.kvstore_storage_backend_extra_config,
        help="JSON object of extra L3 backend settings. For mooncake: "
        "master_server_address, local_hostname, metadata_server, "
        "global_segment_size, protocol, device_name, tenant_id.",
    )
    parser.add_argument(
        "--peer-prefix-port",
        type=int,
        default=ServerArgsOptions.peer_prefix_port,
        help="Share prefix caches with other replicas: a request the router "
        "sends with a peer holder reads the prefix that replica caches, in "
        "its Device or Host cache, over RDMA, instead of computing it, and "
        "this replica serves its own cache the same way. Its endpoint binds "
        "one port shared by the node's attention-DP replicas. Needs the Host "
        "tier (--enable-kvstore) and prefix caching; refused with "
        "--kvstore-storage-backend, "
        "--pipeline-parallel-size > 1 and --query-context-parallel-size > 1. "
        "A --disaggregation-mode decode replica serves the history it "
        "decoded but never reads: its prefill node sends what it lacks. "
        "Replicas share only with replicas of "
        "the same model, cache layout, weights and parallel layout. RDMA "
        "devices come from --disaggregation-ib-device. See "
        "docs/rfcs/0004-peer-prefix-sharing.md.",
    )
    parser.add_argument(
        "--peer-prefix-host",
        type=str,
        default=ServerArgsOptions.peer_prefix_host,
        help="The host other replicas and the router reach this replica's peer "
        "endpoint at; nonzero DP ranks append /rank to host:port. A holder hint "
        "must name it exactly so. Defaults to this host's detected address.",
    )
    parser.add_argument(
        "--peer-prefix-answer-timeout-ms",
        type=int,
        default=ServerArgsOptions.peer_prefix_answer_timeout_ms,
        help="How long a request waits for the peer holder's answer to each "
        "step (lookup, pin, release) before it computes the prefix itself.",
    )
    parser.add_argument(
        "--peer-prefix-min-fetch-tokens",
        type=int,
        default=ServerArgsOptions.peer_prefix_min_fetch_tokens,
        help="The fewest tokens past the local prefix hit worth reading from "
        "a peer holder; a shorter peer hit is computed instead.",
    )
    parser.add_argument(
        "--peer-prefix-landing-tier",
        type=str,
        choices=["device", "host"],
        default=ServerArgsOptions.peer_prefix_landing_tier,
        help="Where a peer prefix lands: 'device' reads it into the Device "
        "pages the request's admission allocates, which wait out the "
        "transfer; 'host' reads it into Host blocks before the admission, "
        "which then takes it as a Host hit at the cost of Host capacity and "
        "one host-to-device copy. Required with --peer-prefix-port.",
    )
    parser.add_argument(
        "--disk-tier-paths",
        type=str,
        nargs="+",
        default=ServerArgsOptions.disk_tier_paths,
        help="Cache prefixes on the node's local SSD below Host L2: one "
        "directory per drive, on a file system that supports O_DIRECT "
        "(XFS or ext4, not tmpfs); each rank keeps one file in one of them. "
        "Host L2 writes its entries back to the file in the background and "
        "a request whose prefix only the file holds reads it into Host "
        "before its admission. The file's contents never outlive the "
        "process. Needs --kvstore-write-policy write_back and prefix "
        "caching; refused with --kvstore-storage-backend and "
        "--pipeline-parallel-size > 1. See docs/rfcs/0006-disk-tier.md.",
    )
    parser.add_argument(
        "--disk-tier-size-gb",
        type=float,
        default=ServerArgsOptions.disk_tier_size_gb,
        help="The Disk tier's size per rank, in gigabytes.",
    )
    parser.add_argument(
        "--disk-tier-io",
        type=str,
        choices=["io_uring", "psync"],
        default=ServerArgsOptions.disk_tier_io,
        help="How the Disk tier reaches its file: 'io_uring' (needs a "
        "kernel and seccomp policy that allow it) or 'psync', a pool of "
        "threads calling pread/pwrite. Required with --disk-tier-paths.",
    )
    parser.add_argument(
        "--disk-tier-write-mbps",
        type=int,
        default=ServerArgsOptions.disk_tier_write_mbps,
        help="The bytes per second, in MB, each rank may write to its Disk "
        "tier file: the SSD's endurance budget. Required with "
        "--disk-tier-paths.",
    )
    parser.add_argument(
        "--disk-tier-config",
        type=str,
        default=ServerArgsOptions.disk_tier_config,
        help="JSON object of Disk tier tuning settings; see "
        "docs/configuration/server.md for the keys and their defaults.",
    )
    # Retraction snapshot pool
    parser.add_argument(
        "--retraction-snapshot-host-gb",
        type=float,
        default=ServerArgsOptions.retraction_snapshot_host_gb,
        help="Per-rank pinned Host pool, in gigabytes, that images a retracted "
        "request's unaligned tail pages, its groups Host L2 never holds, and "
        "its slot state (the hash-complete blocks go to Host L2 and stay "
        "pinned there until the request is restored; with --disable-kvstore "
        "the pool must hold whole images). A retracted request is suspended "
        "with its image and resumes exactly where it stopped once the image "
        "is copied back; nothing is recomputed. A victim whose image does not "
        "fit the Host (L2 pins, the pool or its rows) is aborted instead of "
        "imaged and the client told why, so the pool bounds how much can be "
        "suspended at once, never how long an admission waits. The explicit "
        "size, overriding --retraction-snapshot-ratio when set; 0 (the "
        "default) means not set. The prefill and encode roles never retract "
        "and ignore it.",
    )
    parser.add_argument(
        "--retraction-snapshot-ratio",
        type=float,
        default=ServerArgsOptions.retraction_snapshot_ratio,
        help="Size of the retraction snapshot pool as a fraction of this rank's "
        "Device KV capacity (the base --kvstore-ratio uses). An explicit 0 "
        "disables the pool: no image can be taken and every capacity block "
        "aborts its victim -- the only way to get abort-only behaviour. "
        "Unset (the default) derives the size: 0.1 of the Device KV without "
        "the KVStore (whole images; an on-by-default pool must stay cheap to "
        "pin; 1 uses Device KV's usable LCM-block count, with image rows "
        "limited separately), else the tails "
        "of --retraction-snapshot-max-requests images (one page per group "
        "that publishes to Host L2, a request's worst case for a group that "
        "never does), the slot-state arena coming on top.",
    )
    parser.add_argument(
        "--retraction-snapshot-max-requests",
        type=int,
        default=ServerArgsOptions.retraction_snapshot_max_requests,
        help="Slot-state image rows of the retraction snapshot pool, i.e. the "
        "most requests suspended at once. 0 (the default) derives to this "
        "rank's resident capacity C: --max-num-seqs / attention-DP size "
        "plus --disaggregation-decode-extra-slots. An explicit positive "
        "limit is preserved. Image rows and "
        "Host KV space are separate limits; when no eligible image fits, "
        "capacity recovery aborts a resident rather than imaging it. "
        "Refused with --retraction-snapshot-ratio 0.",
    )
    parser.add_argument(
        "--debug-force-retraction-interval",
        type=int,
        default=ServerArgsOptions.debug_force_retraction_interval,
        help="TEST ONLY. Every |N| scheduler plans retract the oldest quiescent "
        "decoding request (N > 0) or the one prefilling request between its "
        "chunks (N < 0) without capacity pressure, so the suspend/restore "
        "path runs on every request of a test run. Refused with "
        "--retraction-snapshot-ratio 0. 0 (the default) is off; never set it "
        "in serving.",
    )
    parser.add_argument(
        "--debug-force-retraction-max-per-request",
        type=int,
        default=ServerArgsOptions.debug_force_retraction_max_per_request,
        help="TEST ONLY. Bound successfully imaged forced retractions per "
        "request admission. Omission preserves unlimited forcing. An explicit "
        "value must be positive and requires a nonzero "
        "--debug-force-retraction-interval. Failed attempts and waits do not "
        "consume it; ordinary capacity retractions are unaffected.",
    )
    # Mamba Cache
    parser.add_argument(
        "--mamba-ssm-dtype",
        type=str,
        default=ServerArgsOptions.mamba_ssm_dtype,
        choices=["float32", "bfloat16"],
        help="Dtype of the stored linear-attention recurrent state (GDN "
        "and KDA SSM state). Scans accumulate in float32 either way; "
        "bfloat16 halves the state memory.",
    )
    parser.add_argument(
        "--max-prefill-tokens",
        metavar="MAX_PREFILL_TOKENS",
        type=int,
        default=ServerArgsOptions.max_prefill_tokens,
        help=(
            "Maximum prefill-token budget used when chunked prefill is "
            "disabled. Per-iteration scheduling is controlled by "
            "--chunked-prefill-size."
        ),
    )
    parser.add_argument(
        "--chunk-prefill-kv-size",
        type=int,
        default=ServerArgsOptions.chunk_prefill_kv_size,
        help=(
            "Prefill batch KV capacity in tokens, including cached prefixes. "
            "Converted to a page budget; each request is rounded up separately "
            "before summing. Its page-aligned capacity must exceed the maximum "
            "prompt length; an undersized budget fails startup. Omit to keep "
            "the existing unbounded batch policy."
        ),
    )
    parser.add_argument(
        "--prefill-attempt-interval",
        metavar="ROUNDS",
        type=int,
        default=ServerArgsOptions.prefill_attempt_interval,
        help="Prefill delay: rounds between ALL attempt points. On an "
        "ALL point the attention-DP ranks that would decode and also "
        "have prefill work open the group's prefill gate together, so "
        "the group batches prefills instead of each rank interrupting "
        "the others' decode rounds alone. Unset plans prefill every "
        "round. Requires --prefill-any-attempt-interval.",
    )
    parser.add_argument(
        "--prefill-any-attempt-interval",
        metavar="ROUNDS",
        type=int,
        default=ServerArgsOptions.prefill_any_attempt_interval,
        help="Prefill delay: rounds between ANY attempt points, where "
        "one rank's backlog opens the gate, bounding how long a prefill "
        "waits. Must be >= --prefill-attempt-interval.",
    )
    parser.add_argument(
        "--prefill-single-round",
        action="store_true",
        default=ServerArgsOptions.prefill_single_round,
        help="Prefill delay: one prefill round per window. The round "
        "after a prefill round is not an ALL point while any rank would "
        "decode, so a backlog does not drain in consecutive prefill "
        "rounds; the next ALL point is --prefill-attempt-interval rounds "
        "later. An idle engine still admits at once. Requires "
        "--prefill-attempt-interval.",
    )
    # Other runtime options
    parser.add_argument(
        "--stream-interval",
        type=int,
        default=ServerArgsOptions.stream_interval,
        help="Streaming buffer size in generated tokens: send the first step's "
        "tokens at once, then whenever at least this many tokens are unsent. A "
        "smaller value streams more smoothly; a larger value sends fewer, "
        "larger chunks.",
    )
    parser.add_argument(
        "--seed",
        metavar="SEED",
        type=int,
        default=ServerArgsOptions.seed,
        help="The random seed.",
    )
    parser.add_argument(
        "--distributed-timeout-seconds",
        metavar="DISTRIBUTED_TIMEOUT_SECONDS",
        type=int,
        default=ServerArgsOptions.distributed_timeout_seconds,
        help="Set the timeout of the torch.distributed process groups "
        "(initialization and collectives); the fused KVP history "
        "all-gather's in-kernel waits end before it (see "
        "docs/serving/parallelism.md).",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default=ServerArgsOptions.download_dir,
        help="Model download directory for huggingface.",
    )
    parser.add_argument(
        "--base-gpu-id",
        type=int,
        default=ServerArgsOptions.base_gpu_id,
        help="The base GPU ID to start allocating GPUs from. Useful when running multiple instances on the same machine.",
    )
    parser.add_argument(
        "--gpu-id-step",
        type=int,
        default=ServerArgsOptions.gpu_id_step,
        help="The delta between consecutive GPU IDs that are used. For example, setting it to 2 will use GPU 0,2,4,...",
    )

    # Logging
    parser.add_argument(
        "--enable-async-logging",
        action=argparse.BooleanOptionalAction,
        default=ServerArgsOptions.enable_async_logging,
        help=(
            "Write Rust host and default Python root logs on background "
            "threads. Bounded queues drop new records when full. Disabled "
            "by default; incompatible with TOKENSPEED_LOGGING_CONFIG_PATH."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=ServerArgsOptions.log_level,
        help="The logging level of all loggers.",
    )
    parser.add_argument(
        "--ttft-trace-sampling-percentage",
        type=int,
        default=ServerArgsOptions.ttft_trace_sampling_percentage,
        help="Log engine TTFT phases for this percentage of PD bootstrap rooms (0 disables).",
    )
    parser.add_argument(
        "--enable-log-request-stats",
        "--enable-request-time-stats-logging",
        action=argparse.BooleanOptionalAction,
        default=ServerArgsOptions.enable_log_request_stats,
        help=(
            "Log a one-line per-request performance summary when each request "
            "finishes or aborts: timings (queue/prefill/ttft/total/preemption), "
            "token counts (prompt/cache/output), cache-hit rate, decode "
            "throughput, and spec-decode acceptance. Measured entirely on the "
            "host (no GPU sync), so it adds no engine slowdown."
        ),
    )
    parser.add_argument(
        "--enable-metrics",
        action="store_true",
        help="Enable the selected metrics reporters.",
    )
    parser.add_argument(
        "--metrics-reporters",
        action="append",
        default=None,
        help="Select prometheus or a registered metrics reporter (repeat for multiple; defaults to prometheus).",
    )
    parser.add_argument(
        "--metrics-reporter-config",
        type=str,
        default=ServerArgsOptions.metrics_reporter_config,
        help="JSON file mapping external metrics reporter names to their configurations.",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=ServerArgsOptions.metrics_port,
        help="Port for the engine's Prometheus /metrics HTTP endpoint, served "
        "by the engine's main process when --enable-metrics is on. Defaults "
        "to an automatically allocated free port (logged at startup); 0 "
        "also auto-allocates.",
    )
    parser.add_argument(
        "--metrics-host",
        type=str,
        default=ServerArgsOptions.metrics_host,
        help="Bind host for the /metrics endpoint. Defaults to --host.",
    )

    parser.add_argument(
        "--app-key",
        type=str,
        default=ServerArgsOptions.app_key,
        help="Set app key of the server",
    )

    parser.add_argument(
        "--decode-log-interval",
        type=int,
        default=ServerArgsOptions.decode_log_interval,
        help="The log interval of decode batch.",
    )
    parser.add_argument(
        "--kv-events-config",
        type=str,
        default=ServerArgsOptions.kv_events_config,
        help=(
            "JSON KV cache event publisher config. Set "
            "'enable_kv_cache_events': true and publisher 'zmq' to "
            "publish device prefix-cache mutations."
        ),
    )
    parser.add_argument(
        "--otlp-traces-endpoint",
        type=str,
        default=ServerArgsOptions.otlp_traces_endpoint,
        help=(
            "OTLP/HTTP traces endpoint (a collector's /v1/traces, "
            "Jaeger, or Langfuse). Each finished request exports one "
            "span built from its lifecycle timestamps; the serving "
            "loop never blocks on it."
        ),
    )
    parser.add_argument(
        "--trace-sample-ratio",
        type=float,
        default=ServerArgsOptions.trace_sample_ratio,
        help="The share of requests the OTLP endpoint traces (0..1).",
    )

    parser.add_argument(
        "--trace-provider",
        type=str,
        default=ServerArgsOptions.trace_provider,
        help="Registered plugin provider of custom trace context; omission retains native generation.",
    )

    # Data parallelism
    parser.add_argument(
        "--data-parallel-size",
        metavar="DATA_PARALLEL_SIZE",
        type=int,
        default=ServerArgsOptions.data_parallel_size,
        help="The data parallelism size. If not set, inferred from world_size and attn_tp_size.",
    )
    parser.add_argument(
        "--pipeline-parallel-size",
        metavar="PIPELINE_PARALLEL_SIZE",
        type=int,
        default=ServerArgsOptions.pipeline_parallel_size,
        help="Number of pipeline stages for prefill chunk pipelining. "
        "Only supported with --disaggregation-mode prefill.",
    )
    parser.add_argument(
        "--pp-layer-partition",
        metavar="PP_LAYER_PARTITION",
        type=lambda arg: tuple(int(v) for v in arg.split(",")),
        default=ServerArgsOptions.pp_layer_partition,
        help="Explicit per-stage layer counts for pipeline parallelism, "
        'front to back, e.g. "8,11,11,8". Must have one entry per stage '
        "and sum to the model's layer count. Default: even split with "
        "the remainder on the front stages.",
    )
    parser.add_argument(
        "--load-watch-interval",
        type=float,
        default=ServerArgsOptions.load_watch_interval,
        help="Heartbeat compatibility interval for load snapshots in seconds. "
        "Changed load values publish immediately without debounce.",
    )

    # Expert parallelism
    parser.add_argument(
        "--expert-parallel-size",
        "--ep-size",
        type=int,
        default=ServerArgsOptions.ep_size,
        help="The expert parallelism size.",
    )
    parser.add_argument(
        "--init-expert-location",
        type=str,
        default=ServerArgsOptions.init_expert_location,
        help="Expert placement: 'trivial'; inline JSON (starts with '{'); "
        "a directory of per-rank *.expert-load.pt records; a .pt/.json "
        "file; otherwise a glob over record files. A 'logical_count' "
        "[layers, experts] load record derives the placement with the EPLB "
        "algorithm, a 'physical_to_logical_map' [layers, slots] pins it "
        "exactly. The EXPERT_LOAD profile activity writes load records.",
    )
    parser.add_argument(
        "--ep-num-redundant-experts",
        type=int,
        default=ServerArgsOptions.ep_num_redundant_experts,
        help="Add this many physical expert slots per MoE layer for replicas "
        "of hot experts; the total must divide over the EP size.",
    )
    parser.add_argument(
        "--ep-dispatch-algorithm",
        type=str,
        default=ServerArgsOptions.ep_dispatch_algorithm,
        choices=list(EP_DISPATCH_ALGORITHMS),
        help="How routing picks among an expert's replicas; required with an "
        "expert placement. static_with_zero_expert for models with zero "
        "experts (LongCat), static otherwise; dynamic* draw at random.",
    )
    parser.add_argument(
        "--eplb-algorithm",
        type=str,
        default=ServerArgsOptions.eplb_algorithm,
        help="EPLB algorithm deriving the placement from a load record: "
        "auto, deepseek or deepseek_hierarchical.",
    )
    parser.add_argument(
        "--expert-distribution-recorder-mode",
        type=str,
        default=ServerArgsOptions.expert_distribution_recorder_mode,
        choices=["stat"],
        help="'stat' counts the routes to every physical expert so the "
        "EXPERT_LOAD profile activity can dump a load record and "
        "--enable-eplb can rebalance from it.",
    )
    parser.add_argument(
        "--enable-eplb",
        action="store_true",
        help="Online expert rebalancing: every --eplb-rebalance-num-iterations "
        "forwards the routing load since the previous snapshot is balanced "
        "with the EPLB algorithm and the expert weights move between slots, "
        "--eplb-rebalance-layers-per-chunk layers per scheduling round. "
        "Requires --expert-distribution-recorder-mode stat and a static "
        "--ep-dispatch-algorithm, both explicit; the RebalanceExpertsReqInput "
        "control operation triggers one rebalance manually.",
    )
    parser.add_argument(
        "--eplb-rebalance-num-iterations",
        type=int,
        default=ServerArgsOptions.eplb_rebalance_num_iterations,
        help="Forwards between two expert load snapshots under --enable-eplb "
        "(required with it).",
    )
    parser.add_argument(
        "--eplb-rebalance-layers-per-chunk",
        type=int,
        default=ServerArgsOptions.eplb_rebalance_layers_per_chunk,
        help="MoE layers whose experts move in one scheduling round under "
        "--enable-eplb (required with it; at most the MoE layer count). "
        "Fewer layers per chunk bound the per-round stall.",
    )
    parser.add_argument(
        "--enable-fp8-attn-proj",
        action="store_true",
        default=ServerArgsOptions.enable_fp8_attn_proj,
        help="Use online block-FP8 QKV/O and GDN QKVZ/output projections "
        "in the Qwen3.5/Qwen4-Exp target model. GDN BA uses floating precision "
        "and MTP draft projections follow the checkpoint.",
    )
    parser.add_argument(
        "--dense-gemm-backend",
        type=str,
        default=ServerArgsOptions.dense_gemm_backend,
        choices=["auto", "trtllm_cutedsl", "deepgemm"],
        help="Backend for standard 128x128 block-FP8 dense linears. "
        "auto prefers compatible CuTe-DSL on Blackwell at M > 16 GEMM rows. "
        "trtllm_cutedsl requires Blackwell and preserves checkpoint FP8 "
        "weights and FP32 scales without requantization. "
        "deepgemm preserves FP32 scales on Hopper; SM100/SM103 use "
        "UE8M0 weight requantization and packed activation scales. "
        "Other quantization formats and routed experts are unchanged.",
    )
    parser.add_argument(
        "--moe-backend",
        type=str,
        default=ServerArgsOptions.moe_backend,
        help="MoE runner backend: auto, triton, gluon, flashinfer_trtllm, "
        "flashinfer_cutlass, flashinfer_cutedsl, deep_gemm, mega_moe, "
        "gluon_petit, or a registered plugin solution. Under --numerics "
        "rl-bitwise the selected kernel must declare batch invariance.",
    )
    parser.add_argument(
        "--moe-cutlass-fused-finalize",
        action=argparse.BooleanOptionalAction,
        default=ServerArgsOptions.moe_cutlass_fused_finalize,
        help="Enable GEMM2 atomic expert combine for unquantized FP16/BF16 "
        "MoE on flashinfer_cutlass, for target and draft models. On by "
        "default; the fused reduction is not run-to-run reproducible. Forced "
        "off by --numerics rl-bitwise.",
    )
    parser.add_argument(
        "--moe-mxfp4-fp8-activation",
        action="store_true",
        help="Run MXFP4 routed experts with FP8 activations (on Hopper the "
        "FlashInfer cutlass W4A8 Humming MoE: about 1.8x faster than the "
        "default W4A16 path, a few percent of relative error on the expert "
        "outputs; validate the served model before relying on it). Applies to "
        "every MXFP4 expert layer, target and draft; the MoE plan fails at "
        "startup where the selected backend has no FP8-activation kernel for "
        "the layer.",
    )
    parser.add_argument(
        "--draft-moe-backend",
        type=str,
        default=ServerArgsOptions.draft_moe_backend,
        help="MoE runner backend for the draft model in speculative decoding. "
        "If not set, defaults to --moe-backend.",
    )
    parser.add_argument(
        "--all2all-backend",
        metavar="ALL2ALL_BACKEND",
        type=str,
        default=ServerArgsOptions.all2all_backend,
        choices=["none", "agrs", "deepep", "flashinfer", "gluon_petit"],
        help="MoE communication backend. agrs and flashinfer explicitly select "
        "the Kimi-K3 attention-DP transport; gluon_petit selects the fused "
        "Petit MegaMoE transport; none preserves existing behavior.",
    )
    parser.add_argument(
        "--deepep-mode",
        type=str,
        choices=["normal", "low_latency", "auto"],
        default=ServerArgsOptions.deepep_mode,
        help="Select the mode when enable DeepEP MoE, could be `normal`, `low_latency` or `auto`. Default is `auto`, which means `low_latency` for decode batch and `normal` for prefill batch.",
    )
    parser.add_argument(
        "--disable-flashinfer-cutlass-moe-fp4-allgather",
        action="store_true",
        help="Disable flashinfer cutlass MoE FP4 allgather.",
    )

    # Multi-node distributed serving
    parser.add_argument(
        "--dist-init-addr",
        type=str,
        help="The host address for initializing distributed backend (e.g., `192.168.0.2:25000`, "
        "or `[fd00::2]:25000` for IPv6). "
        "Derived from the launcher environment under a multi-node Slurm step.",
    )
    parser.add_argument(
        "--nnodes",
        type=int,
        default=ServerArgsOptions.nnodes,
        help="The number of nodes. Derived from SLURM_STEP_NUM_NODES when unset.",
    )
    parser.add_argument(
        "--node-rank",
        type=int,
        default=ServerArgsOptions.node_rank,
        help="The node rank. Derived from SLURM_NODEID when unset.",
    )

    # Model override args
    parser.add_argument(
        "--hf-overrides",
        metavar="HF_OVERRIDES",
        type=str,
        help="A dictionary in JSON string format used to override default model configurations.",
        default=ServerArgsOptions.hf_overrides,
    )

    # Kernel backend. Names are validated against the backend registry
    # after plugin discovery, so plugins can add their own.
    attention_backend_names = (
        "mha, mla, fa3, fa4, triton, gluon, flashinfer, trtllm, trtllm_mla, "
        "flashmla, tokenspeed_mla, hybrid_linear_attn"
    )
    parser.add_argument(
        "--attention-backend",
        type=str,
        default=ServerArgsOptions.attention_backend,
        help="Choose the kernels for attention layers: "
        f"{attention_backend_names}, or a name a plugin registers. 'mha' "
        "selects compatible registered MHA kernels; 'gluon' forces "
        "registered Gluon MLA kernels.",
    )
    parser.add_argument(
        "--min-kv-split",
        type=int,
        default=None,
        help="Minimum decode KV splits for tokenspeed_mla target and draft attention "
        "(1–256). Omit to retain automatic selection. Larger values can improve "
        "long-context decoding at the cost of workspace and short-context latency.",
    )
    parser.add_argument(
        "--kda-backend",
        type=str,
        choices=["auto", "fla", "flashkda", "cutedsl_kda", "cudnn"],
        default=ServerArgsOptions.kda_backend,
        help="KDA (Kimi Delta Attention) prefill kernel policy. On AMD, "
        "this setting is ignored and compatible kernels are selected using "
        "registry priority. On NVIDIA, 'auto' selects the fastest available "
        "backend "
        "(cutedsl_kda > flashkda > fla). Named backends are NVIDIA-specific: "
        "'fla' uses the portable FLA scan, 'flashkda' uses the optional "
        "FlashKDA library (source build, SM90+), and 'cutedsl_kda' uses the "
        "FlashInfer CuTe DSL KDA prefill kernel on SM100/SM103. 'cudnn' uses FlashInfer's "
        "cuDNN KDA prefill engine on SM100/SM103. Decode is unaffected.",
    )
    parser.add_argument(
        "--kda-history-capacity",
        type=int,
        default=ServerArgsOptions.kda_history_capacity,
        help="Accepted KDA records each request keeps per layer between "
        "dense-state folds, counting the verify window (the speculative draft "
        "tokens). Default: max(16, draft tokens); it must be at least the "
        "draft tokens. A smaller capacity shrinks the per-request verify "
        "workspace but folds the records into the dense state more often: at "
        "the draft tokens themselves, every round that accepts any.",
    )
    parser.add_argument(
        "--drafter-attention-backend",
        type=str,
        help="Attention backend for drafter model in speculative decoding "
        f"({attention_backend_names}, or a plugin's). If not specified, uses "
        "the same backend as the main model (attention_backend).",
    )
    parser.add_argument(
        "--skip-softmax-threshold",
        type=float,
        default=ServerArgsOptions.skip_softmax_threshold,
        help="BLASST skip-softmax sparsity threshold for the gluon MHA "
        "prefill kernel (gfx950 only). A K/V block is skipped only when "
        "every row in the query tile has exp(block_max_score - "
        "running_max) below this threshold. 0.0 (default) is exact "
        "dense attention; the skip rate for a given threshold must be "
        "calibrated per model and sequence length. Only takes effect "
        "when every request in the batch has a zero-length cached "
        "prefix and the planner routes the batch to prefill; if any "
        "request in the batch has a cache hit, or the backend's "
        "registered prefill kernel does not clear the planner's "
        "performance cutoff (as with FP8/MXFP8 KV cache and the triton "
        "backend), the whole batch falls through to the KV-cache-extend "
        "path, which never reaches this kernel and silently ignores the "
        "threshold. Backends that do reach kernel selection raise an "
        "error there if they lack gluon skip-softmax support, rather "
        "than silently ignoring it.",
    )
    parser.add_argument(
        "--enable-qsa-prefill-bf16-scratch",
        action="store_true",
        dest="enable_qsa_prefill_bf16_scratch",
        default=ServerArgsOptions.enable_qsa_prefill_bf16_scratch,
        help="QSA sparse-attention models (Qwen4-Exp) with --kv-cache-dtype "
        "fp8_e4m3 on SM100 to SM107: a target prefill call of at least "
        "2048 rows copies the FP8 KV pages it can read into a BF16 scratch "
        "and attends there with the CuTe kernel's BF16 path instead of the "
        "FP8 prefill kernel. Smaller and mixed calls, calls whose context "
        "exceeds the scratch or 128 cache tokens per row, decode, verify "
        "and drafts keep the FP8 path; the cache stays FP8. The scratch "
        "holds one maximum-length context plus one --chunked-prefill-size "
        "chunk, with that chunk's remapped slots and attention output, and "
        "is reserved from the KV cache budget before the pool is sized. "
        "Startup fails for a model without QSA layers, another KV cache "
        "dtype (mxfp8 included), chunked prefill below 2048 tokens or "
        "disabled, or a GPU without the CuTe QSA kernel. The decode role, "
        "which runs no prefill, reserves no scratch.",
    )
    parser.add_argument(
        "--enable-ple-hot-cache",
        action="store_true",
        default=ServerArgsOptions.enable_ple_hot_cache,
        help="Cache frequently read rows of eligible offloaded Qwen4-Exp PLE "
        "tables in GPU memory. Disabled by default. "
        "Requires PLE offloading and a supported table format and CUDA kernel. "
        "Small lookups and CUDA graphs use regular host gather. "
        "TOKENSPEED_PLE_HOST_CACHE_SLOTS must be a power of two greater than one.",
    )
    parser.add_argument(
        "--qwen4-exp-flat-kv",
        action=argparse.BooleanOptionalAction,
        dest="qwen4_exp_flat_kv",
        default=ServerArgsOptions.qwen4_exp_flat_kv,
        help="Qwen4-Exp layout (on by default): target/draft KV, recurrent, PLE "
        "and QSA caches share one plane, preserving logical groups. "
        "Full-history block IDs translate to contiguous atom pages; state "
        "and recent windows use explicit strides. Requires PP=1 and non-MXFP8 KV. "
        "Use --no-qwen4-exp-flat-kv for the plane-based layout.",
    )
    parser.add_argument(
        "--sampling-backend",
        type=str,
        choices=[
            "greedy",
            "flashinfer",
            "flashinfer_full",
            "triton",
            "triton_full",
            "ascend_full",
            "sonic",
        ],
        default=ServerArgsOptions.sampling_backend,
        help="Sampling backend. "
        "'greedy': argmax + verify_chain_greedy, zero sampling-param plumbing. "
        "'flashinfer': temperature/top_k/top_p via fused softmax + top_k_top_p_sampling_from_probs; "
        "min_p and penalties silently ignored. "
        "'triton': temperature/top_k/top_p via MRV2-style logits-to-Gumbel-Max; "
        "min_p and penalties silently ignored. "
        "'flashinfer_full': adds min_p plus frequency/presence/repetition penalties and logit_bias "
        "via the softmax+renorm+min_p kernel sequence. "
        "'triton_full': adds min_p plus frequency/presence/repetition penalties and logit_bias "
        "with Triton Gumbel-Max for single-step sampling. "
        "'ascend_full': graph-safe Ascend top-k/top-p/min-p sampling and chain verify. "
        "Allocates a counts[max_req_pool_size, vocab_size] int32 buffer (substantial memory). "
        "ascend_full caps finite top_k at 4096; larger values use 4096. "
        "Other backends require finite top_k < 128. -1 disables top-k. "
        "'sonic': one fused Triton pipeline per step (vendored sonic-sampler; two kernels "
        "for sampling, three for chain verification) covering greedy, "
        "temperature/top_k/top_p/min_p, penalties, logit_bias and grammar for both "
        "single-step sampling and chain verification; a request with no top_k, top_p "
        "or min_p samples its whole vocabulary, the others select among its top-128 "
        "candidates. bf16 logits are ranked at bf16 resolution; fp32 logits "
        "(soft-capped heads) are sampled exactly in fp32. "
        "Its kernel config per batch size comes from a table measured on "
        "Blackwell, so nothing is tuned at startup. Allocates slot-indexed "
        "[max_req_pool_size, vocab_size] token counts and logit bias: about 10 bytes "
        "per element on device (substantial memory).",
    )
    parser.add_argument(
        "--sampling-stream",
        type=str,
        choices=list(SAMPLING_STREAMS),
        default=ServerArgsOptions.sampling_stream,
        help="Random stream of the non-greedy rows of the flashinfer and "
        "flashinfer_full sampling backends. 'batch': flashinfer's "
        "top_k_top_p / min_p sampling kernels, whose Philox stream is keyed "
        "by the batch row, so a request's draw depends on its co-batch. "
        "'per-request': the Gumbel-max pool kernels keyed by the request's "
        "seed and position (finite top_k is capped at 128). Chain verify "
        "coins are keyed by the request's seed and the round's first "
        "target position. Folded to per-request by --numerics rl-bitwise.",
    )
    parser.add_argument(
        "--dp-sampling",
        action="store_true",
        default=ServerArgsOptions.dp_sampling,
        help=(
            "Enable Batch-DP spec-verify sampling. Backend selection defaults "
            "to auto; override with TOKENSPEED_DP_SAMPLING_BACKEND."
        ),
    )
    parser.add_argument(
        "--dp-sampling-min-bs",
        type=int,
        default=ServerArgsOptions.dp_sampling_min_bs,
        help="Minimum effective decode batch for Batch-DP spec-verify. "
        "Defaults to 2 * TP size.",
    )
    parser.add_argument(
        "--attention-use-fp4-indexer-cache",
        "--attention-config.use-fp4-indexer-cache",
        "--attention_config.use_fp4_indexer_cache",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=ServerArgsOptions.attention_use_fp4_indexer_cache,
        help="Use the MXFP4 sparse attention indexer cache layout.",
    )
    parser.add_argument(
        "--attention-config.use-trtllm-ragged-deepseek-prefill",
        "--attention-config.use_trtllm_ragged_deepseek_prefill",
        "--attention_config.use_trtllm_ragged_deepseek_prefill",
        dest="use_trtllm_ragged_deepseek_prefill",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=ServerArgsOptions.use_trtllm_ragged_deepseek_prefill,
        help="Use ragged prefill for DeepSeek MLA attention.",
    )
    parser.add_argument(
        "--deepseek-v4-mega-moe-max-num-tokens",
        type=int,
        default=ServerArgsOptions.deepseek_v4_mega_moe_max_num_tokens,
        help=(
            "DeepSeek V4.1 MegaMoE staging-buffer cap on tokens per forward "
            "(0 = derive from chunked-prefill / cuda-graph budgets)."
        ),
    )
    parser.add_argument(
        "--deepseek-v4-indexer-prefill-max-logits-mb",
        type=int,
        default=ServerArgsOptions.deepseek_v4_indexer_prefill_max_logits_mb,
        help=(
            "Sparse indexer prefill workspace cap (MiB) for the "
            "softplus_sqrt logits buffer."
        ),
    )
    parser.add_argument(
        "--engram-host-table",
        action=argparse.BooleanOptionalAction,
        default=ServerArgsOptions.engram_host_table,
        help=(
            "DeepSeek V4.1 Engram: store the two FP8 n-gram tables in host "
            "memory and gather rows through UVA. Frees HBM for KV cache. "
            "See --engram-host-table-layout. Requires enough host RAM."
        ),
    )
    parser.add_argument(
        "--engram-host-table-dir",
        type=str,
        default=ServerArgsOptions.engram_host_table_dir,
        help=(
            "Directory for shared Engram mmap files. Default: /dev/shm "
            "when it has enough free space, otherwise /scratch or /tmp. "
            "Used only when several ranks share tables in mmap memory (see "
            "--engram-host-table-alloc). Docker often caps /dev/shm at "
            "32-64 GiB, too small for a full V4.1 table."
        ),
    )
    parser.add_argument(
        "--engram-host-table-layout",
        type=str,
        choices=["auto", "shared", "sharded"],
        default=ServerArgsOptions.engram_host_table_layout,
        help=(
            "Host Engram layout when --engram-host-table is set. shared: one "
            "full copy per node, skip the lookup all-reduce. sharded: each "
            "attention-TP rank holds a host shard and keeps the all-reduce "
            "(anonymous mapping, huge-page friendly). auto: sharded when "
            "attention TP > 1, else shared."
        ),
    )
    parser.add_argument(
        "--engram-host-table-alloc",
        type=str,
        choices=["auto", "cumem", "mmap"],
        default=ServerArgsOptions.engram_host_table_alloc,
        help=(
            "Host memory of shared Engram tables. cumem: pinned host-NUMA "
            "memory from CUDA virtual memory management, which GPUs map "
            "with 2 MiB pages, so random-row lookups miss the GPU TLB far "
            "less than with the 4-64 KiB base pages of a registered shared "
            "file; the ranks of a host share one copy through FABRIC handles; "
            "fails if unavailable. mmap: a registered shared file (see "
            "--engram-host-table-dir), or registered anonymous memory for "
            "a single rank. auto: cumem when every rank can use it, "
            "otherwise mmap on every rank (with a warning on NVIDIA)."
        ),
    )
    parser.add_argument(
        "--engram-host-table-numa",
        type=str,
        choices=["interleave", "writer"],
        default=ServerArgsOptions.engram_host_table_numa,
        help=(
            "Placement of cumem Engram tables. interleave: alternate 2 GiB "
            "chunks over the host NUMA nodes of the GPUs the job's ranks "
            "use on that host. writer: keep them on the NUMA node of the "
            "GPU of the rank that loads them. A single rank's table stays "
            "on its own GPU's node either way."
        ),
    )
    parser.add_argument(
        "--decode-monokernel",
        action=argparse.BooleanOptionalAction,
        default=ServerArgsOptions.decode_monokernel,
        help=(
            "Run each decoder layer's decode step as one persistent "
            "monokernel launch, for small decode and speculative verify "
            "steps (each model's seam names the steps and parallel "
            "layouts its kernels serve). Models: "
            f"{', '.join(DECODE_MONOKERNEL_ARCHITECTURES)} on gfx950. "
            "Startup fails when a layer cannot be packed for its "
            "monokernel. A speculative draft with monokernels of its "
            "own takes the switch too when its target is one they serve ("
            + "; ".join(
                f"{draft}: {', '.join(targets)}"
                for draft, targets in DRAFT_MONOKERNEL_TARGETS.items()
            )
            + "); every other draft keeps the regular path. Other steps "
            "keep the regular path."
        ),
    )
    parser.add_argument(
        "--monokernel-stuck-record-dir",
        type=str,
        default=ServerArgsOptions.monokernel_stuck_record_dir,
        help=(
            "With --decode-monokernel: a directory on tmpfs (for example "
            "/dev/shm) where each rank keeps "
            "its stuck-poll record file (stuck-rank<N>-pid<P>.bin and a "
            ".json sidecar). A launch poll that gives up writes what it "
            "waited for there before the launch traps, so the record "
            "outlives the process; a run without a stuck poll removes the "
            "files at exit. Without it the records stay in pinned memory "
            "and are logged when the failure reaches Python."
        ),
    )
    parser.add_argument(
        "--inkling2-loop-num-passes",
        type=int,
        default=ServerArgsOptions.inkling2_loop_num_passes,
        help=(
            "Loop passes per token for an inkling2_loop checkpoint with "
            "loop_config.is_varloop=true (at least 1); each pass runs "
            "the loop core's layers once more. Default: the "
            "checkpoint's num_passes. A draft model needs the "
            "checkpoint's count: it reads fixed layer visits."
        ),
    )
    parser.add_argument(
        "--inkling2-loop-share-cache",
        action="store_true",
        default=ServerArgsOptions.inkling2_loop_share_cache,
        help=(
            "For an inkling2_loop checkpoint with loop_config.is_varloop="
            "true: every loop pass of a core layer reads and writes that "
            "layer's one KV cache, so later tokens attend to the last "
            "pass's K/V, instead of one cache per pass. Cuts the loop "
            "core's KV memory; convolution state stays one per pass."
        ),
    )
    parser.add_argument(
        "--grammar-backend",
        type=str,
        choices=["xgrammar", "none"],
        default=ServerArgsOptions.grammar_backend,
        help=(
            "Grammar backend. 'none' disables grammar-guided decoding "
            "entirely; requests with json_schema, regex, ebnf or "
            "structural_tag are then rejected as invalid requests. The default "
            "is checkpoint-specific."
        ),
    )
    parser.add_argument(
        "--reasoning-parser",
        type=str,
        default=ServerArgsOptions.reasoning_parser,
        help=(
            "Reasoning parser name (e.g. 'minimax', 'kimi_k25'). "
            "Used to defer json_schema grammars past the model's "
            "reasoning channel."
        ),
    )
    parser.add_argument(
        "--grammar-compile-timeout-secs",
        type=float,
        default=ServerArgsOptions.grammar_compile_timeout_secs,
        help="Per-compile wallclock budget before the request is aborted.",
    )
    parser.add_argument(
        "--grammar-compile-max-retries",
        type=int,
        default=ServerArgsOptions.grammar_compile_max_retries,
        help="Compile timeouts allowed before a grammar key is permanently rejected.",
    )
    parser.add_argument(
        "--disable-any-whitespace",
        action="store_true",
        default=ServerArgsOptions.disable_any_whitespace,
        help="Compile xgrammar JSON grammars in tight mode (no arbitrary "
        "whitespace between tokens). Mitigates models that wedge into "
        "endless whitespace until length cutoff. xgrammar only.",
    )
    parser.add_argument(
        "--disable-capturable-grammar",
        action="store_true",
        default=ServerArgsOptions.disable_capturable_grammar,
        help="Force the synchronous eager grammar fallback even on CUDA. "
        "For parity-testing the captured-grammar path: output should "
        "match; throughput will be lower (sync stall every step).",
    )
    parser.add_argument(
        "--mla-disable-ragged",
        action="store_true",
        help="Disable the ragged prefill wrapper on MLA kernel backends during EXTEND.",
    )

    # Speculative decoding
    parser.add_argument(
        "--enable-dp-local-cuda-graph",
        action="store_true",
        default=ServerArgsOptions.enable_dp_local_cuda_graph,
        help="Select decode CUDA graph capacities independently on attention-DP "
        "ranks. Currently requires a supported DSPARK drafter, target and draft "
        "MegaMoE, TP1 for attention/dense/MoE/LM head, and greedy or adaptive "
        "DSpark sampling.",
    )
    parser.add_argument(
        "--enable-dsv41-adaptive-dspark",
        action="store_true",
        default=ServerArgsOptions.enable_dsv41_adaptive_dspark,
        help="Adapt DeepSeek V4.1 DSpark verification capacity using confidence "
        "and calibrated step costs. Requires CUDA graphs and greedy sampling "
        "or flashinfer with --enable-speculative-sampling.",
    )
    parser.add_argument(
        "--dsv41-dspark-min-batch-size",
        type=int,
        default=ServerArgsOptions.dsv41_dspark_min_batch_size,
        help="Enable adaptive DSpark when the maximum actual local decode "
        "batch size across attention ranks reaches this threshold. Uneven "
        "or idle DP ranks share the selected physical query capacity.",
    )
    parser.add_argument(
        "--draft-model-path-use-base",
        action="store_true",
        help="The path of the draft model weights use the path of the base model",
    )
    parser.add_argument(
        "--speculative-config",
        "--speculative_config",
        type=str,
        default=ServerArgsOptions.speculative_config,
        help="JSON speculative decoding configuration. Supported keys are method, model, num_speculative_tokens, synthetic_acceptance_length, rejection_sample_method, and draft_sample_method (vLLM names).",
    )
    parser.add_argument(
        "--speculative-algorithm",
        type=str,
        help="Speculative algorithm. In-tree: EAGLE3, EAGLE4, MTP, DFLASH, "
        "DSPARK, DUMMY; plugins may register more (validated after plugin "
        "discovery).",
    )
    parser.add_argument(
        "--speculative-draft-model-path",
        type=str,
        help="The path of the draft model weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--speculative-draft-model-revision",
        type=str,
        default=None,
        help="The draft model version to use when the draft has its own "
        "repository: a branch name, a tag name, or a commit id. If "
        "unspecified, the draft uses its repository's default version. A "
        "draft inside the target checkpoint uses --revision.",
    )
    parser.add_argument(
        "--speculative-draft-model-quantization",
        type=str,
        default=ServerArgsOptions.speculative_draft_model_quantization,
        help="Quantization method for the draft model. Defaults to 'unquant'.",
    )
    parser.add_argument(
        "--speculative-num-steps",
        type=int,
        help="The number of steps sampled from draft model in Speculative Decoding.",
        default=ServerArgsOptions.speculative_num_steps,
    )
    parser.add_argument(
        "--speculative-eagle-topk",
        type=int,
        help="Children each draft node expands to per step; above 1 the draft is a tree "
        "(EAGLE3, or EAGLE-style MTP), and --speculative-num-draft-tokens is its node budget.",
        default=ServerArgsOptions.speculative_eagle_topk,
    )
    parser.add_argument(
        "--speculative-num-draft-tokens",
        type=int,
        help="The number of tokens sampled from the draft model in Speculative Decoding.",
        default=ServerArgsOptions.speculative_num_draft_tokens,
    )
    parser.add_argument(
        "--speculative-exec-mode",
        type=int,
        choices=[-1, 0, 1, 2],
        help="EAGLE4 drafting: -1 a static chain or tree; 0 EAGLE-Harness dynamic "
        "trees sized from --speculative-data-path (serving); 1 the same for "
        "evaluation; 2 measures that cost data.",
        default=ServerArgsOptions.speculative_exec_mode,
    )
    parser.add_argument(
        "--speculative-data-path",
        type=str,
        help="EAGLE-Harness cost data directory (the cost/bs_<N> tree).",
        default=ServerArgsOptions.speculative_data_path,
    )
    parser.add_argument(
        "--speculative-compute-intensity",
        type=str,
        choices=["high_intensity", "low_intensity", "chain_intensity"],
        help="EAGLE-Harness policy tier (high_intensity: the verify buckets "
        "and thresholds of compute-bound targets; chain_intensity: a one-child "
        "chain of dynamic depth, with --speculative-eagle-topk 1).",
        default=ServerArgsOptions.speculative_compute_intensity,
    )
    parser.add_argument(
        "--speculative-chain-width-table",
        type=str,
        help="EAGLE-Harness chains (chain_intensity, exec modes 0 and 1): "
        "'MIN_BS:WIDTH,...'. A round of B >= MIN_BS live requests (the entry "
        "with the largest such MIN_BS) drafts WIDTH - 1 tokens per request (a "
        "verify tree of WIDTH nodes) without the confidence rule; smaller "
        "batches keep the rule, which decides at most 256 requests. Widths "
        "are capped at --speculative-num-draft-tokens. 'off' (the default): "
        "the rule decides every round, of at most 256 requests. The verify "
        "still runs every node of the window, so calibrate a table per "
        "workload (e.g. '64:4,384:3'). Ignored by trees and by the cost "
        "measurement (exec mode 2).",
        default=ServerArgsOptions.speculative_chain_width_table,
    )
    parser.add_argument(
        "--synthetic-acceptance-length",
        type=float,
        default=ServerArgsOptions.synthetic_acceptance_length,
        help="Benchmark-only mean accepted length including the guaranteed "
        "target token, in [1, speculative_num_draft_tokens]. Forces a "
        "floor/ceil acceptance distribution; generated text is not valid "
        "for correctness evaluation. Selects the synthetic rejection method. "
        "Requires --sampling-backend greedy or sonic (or flashinfer with "
        "--enable-speculative-sampling for sampled drafts) without "
        "--dp-sampling or --enable-output-logprobs. Unset disables it.",
    )
    parser.add_argument(
        "--enable-speculative-sampling",
        action="store_true",
        default=ServerArgsOptions.enable_speculative_sampling,
        help="Standard rejection sampling for chain speculative decoding: the "
        "drafter samples each step from its own distribution q (per-request "
        "temperature; greedy rows stay argmax) and verify accepts with "
        "coin * q(x) < p(x) instead of the target-only rule (vLLM's "
        "draft_sample_method=probabilistic). Needs EAGLE3, MTP, DSPARK or "
        "DFLASH with a base draft, with --speculative-eagle-topk 1 and the "
        "flashinfer or flashinfer_full sampling backend, EAGLE4 on "
        "EAGLE-Harness (--speculative-exec-mode 0 or 1, trees or chains) "
        "and the flashinfer backend, or DFLASH with a DFlash2 (candidate "
        "selector) draft and the triton backend; refused on the prefill "
        "role of a disaggregated deployment. Every drafter but a DFlash2 "
        "draft costs a per-request fp32 draft distribution buffer; see "
        "docs/configuration/server.md.",
    )
    parser.add_argument(
        "--speculative-rejection-sample-method",
        type=str,
        choices=REJECTION_SAMPLE_METHODS,
        default=ServerArgsOptions.speculative_rejection_sample_method,
        help="Verify rule, as vLLM's rejection_sample_method. 'standard': the "
        "existing token-wise rule. 'block': block verification of a sampled "
        "DSPARK or EAGLE3 chain (--speculative-eagle-topk 1); requires "
        "--enable-speculative-sampling and --sampling-backend flashinfer, "
        "aggregate TP (including DCP) without DP, PP or PD disaggregation. "
        "'synthetic': benchmark-only forced acceptance; set by "
        "--synthetic-acceptance-length.",
    )
    parser.add_argument(
        "--spec-reject-draft-prob-threshold",
        type=float,
        default=ServerArgsOptions.spec_reject_draft_prob_threshold,
        help="With --enable-speculative-sampling, recorded draft probabilities "
        "above this value mark a request with no proposal yet (fresh "
        "admission, PD landing) and always reject. Must lie within "
        "[1.0, 2**20].",
    )
    parser.add_argument(
        "--enable-deferred-gdn-state",
        action="store_true",
        default=ServerArgsOptions.enable_deferred_gdn_state,
        help="Defer accepted GDN recurrent replay to the next layer's WY (BF16 T4 chains).",
    )
    parser.add_argument(
        "--disable-replay-ssm",
        dest="enable_replay_ssm",
        action="store_false",
        default=ServerArgsOptions.enable_replay_ssm,
        help="Stage every verify position's recurrent state instead of "
        "replaying the accepted tokens (ReplaySSM, on by default for "
        "supported Qwen GDN and Nemotron-H Mamba2 targets).",
    )
    parser.add_argument(
        "--enable-replay-ssm",
        dest="enable_replay_ssm",
        action="store_true",
        help="Deprecated: ReplaySSM is on by default.",
    )
    parser.add_argument(
        "--enable-output-logprobs",
        action="store_true",
        default=ServerArgsOptions.enable_output_logprobs,
        help="Enable per-token sampled-token logprobs. OFF by default; enabling extends the captured CUDA-graph footprint. Requests asking for logprobs on a server without this flag receive empty logprobs.",
    )
    parser.add_argument(
        "--enable-output-cumulative-probs",
        action="store_true",
        default=ServerArgsOptions.enable_output_cumulative_probs,
        help="Return probability mass preceding each sampled token, ordered by "
        "descending probability and then ascending token ID, for sampling "
        "calibration. Requires output logprobs. Every sampling step keeps a "
        "copy of the score logits plus bounded temporaries; prefill and mixed "
        "steps sample outside any CUDA graph, so this memory comes from "
        "runtime headroom (leave room with --gpu-memory-utilization).",
    )
    parser.add_argument(
        "--input-logprob-chunk-tokens",
        type=int,
        default=ServerArgsOptions.input_logprob_chunk_tokens,
        help="Prompt rows pushed through the LM head per chunk when a request asks for prompt (input) logprobs (SGLang logprob_start_len). A sizing knob only: the transient per-chunk cost is about chunk x vocab x 8 bytes (bf16 logits shard, bf16 TP-gathered logits, fp32 log-softmax) and the value never changes a result.",
    )
    parser.add_argument(
        "--eagle3-layers-to-capture",
        type=str,
        help="The layers of Eagle3 to capture.",
        default=ServerArgsOptions.eagle3_layers_to_capture,
    )

    # Runtime options
    parser.add_argument(
        "--disable-pdl",
        action="store_true",
        help="Disable PDL launch.",
    )
    prefix_cache_group = parser.add_mutually_exclusive_group()
    prefix_cache_group.add_argument(
        "--enable-prefix-caching",
        action="store_true",
        default=ServerArgsOptions.enable_prefix_caching,
        help="Enable prefix caching.",
    )
    prefix_cache_group.add_argument(
        "--disable-prefix-caching",
        dest="enable_prefix_caching",
        action="store_false",
        help="Disable prefix caching.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graph.",
    )
    parser.add_argument(
        "--disable-cuda-graph-padding",
        action="store_true",
        help="Disable cuda graph when padding is needed. Still uses cuda graph when padding is not needed.",
    )
    parser.add_argument(
        "--disable-autotune",
        "--disable-flashinfer-autotune",
        action="store_true",
        help="Skip profiling missing kernel tactics during startup; kernel "
        "compilation warmup still runs. A matching "
        "persistent FlashInfer cache is still loaded; uncovered shapes use "
        "the library's heuristic fallback.",
    )
    parser.add_argument(
        "--enable-startup-requests",
        action="store_true",
        default=ServerArgsOptions.enable_startup_requests,
        help="Before registering with the frontend, serve a few requests of "
        "the engine's own (one short, one of a full prefill chunk, then a "
        "full batch) through the ordinary serving path, so the first client "
        "requests do not compile what that path reaches. Not supported with "
        "--disaggregation-mode, --kvstore-storage-backend or "
        "--shared-l2-config.",
    )
    parser.add_argument(
        "--enable-cudagraph-gc",
        action="store_true",
        help="Enable garbage collection during CUDA graph capture. If disabled (default), GC is frozen during capture to speed up the process.",
    )
    parser.add_argument(
        "--disable-nccl-nvls",
        action="store_true",
        help="Disable NCCL NVLS even when an MNNVL-capable fabric is detected.",
    )
    parser.add_argument(
        "--disable-symm-mem",
        action="store_true",
        help="Disable NCCL cuMem support even when an MNNVL-capable fabric is detected.",
    )
    parser.add_argument(
        "--disable-overlap-schedule",
        action="store_true",
        help="Disable the overlap scheduler, which overlaps the CPU scheduler with GPU model worker. "
        "On a PD prefill server at PP=1 this also returns the event loop to in-flight depth 0; "
        "under PP the chunk pipeline keeps one forward per stage in flight.",
    )
    parser.add_argument(
        "--disable-tf32",
        action="store_true",
        help="Disable forcing TF32 on for cuBLAS/cuDNN. By default the server sets "
        "NVIDIA_TF32_OVERRIDE=1 and TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1.",
    )
    parser.add_argument(
        "--max-cudagraph-capture-size",
        metavar="MAX_CUDAGRAPH_CAPTURE_SIZE",
        type=int,
        default=ServerArgsOptions.max_cudagraph_capture_size,
        help="Set the maximum batch size for CUDA graph capture.",
    )
    parser.add_argument(
        "--cudagraph-capture-sizes",
        metavar="CUDAGRAPH_CAPTURE_SIZE",
        type=int,
        nargs="+",
        help="Set the list of batch sizes for CUDA graph capture.",
    )
    parser.add_argument(
        "--disable-prefill-graph",
        action="store_true",
        help="Disable cuda graph for prefill.",
    )
    parser.add_argument(
        "--disable-kda-prefill-graph",
        action="store_true",
        help="Disable KDA prefill CUDA graphs while retaining ordinary "
        "prefill and decode graph settings. Supported cutedsl_kda prefill "
        "attention is included in prefill graphs by default.",
    )
    parser.add_argument(
        "--prefill-graph-max-tokens",
        type=int,
        default=ServerArgsOptions.prefill_graph_max_tokens,
        help="Largest token bucket captured by the breakable prefill CUDA "
        "graph. Default (unset) = min(8192, chunked-prefill size); "
        "0 disables.",
    )
    prefill_token_sizes = parser.add_mutually_exclusive_group()
    prefill_token_sizes.add_argument(
        "--prefill-graph-capture-token-sizes",
        dest="prefill_graph_capture_sizes",
        metavar="TOKENS",
        type=int,
        nargs="+",
        help="Total input-token capacities per forward, summed across the "
        "batch; not per-request sequence lengths. Shorter inputs are padded. "
        "For pure prefill, count newly computed tokens, excluding cached "
        "prefixes. Unset: a relative-stride ladder with ~12.5%% spacing, "
        "subject to a 16-token minimum step and a 512-token maximum step.",
    )
    prefill_token_sizes.add_argument(
        "--prefill-graph-capture-sizes",
        dest="prefill_graph_capture_sizes",
        metavar="TOKENS",
        type=int,
        nargs="+",
        help="Compatibility alias for --prefill-graph-capture-token-sizes. "
        "Specify only one spelling.",
    )
    parser.add_argument(
        "--prefill-graph-capture-batch-sizes",
        metavar="BS",
        type=int,
        nargs="+",
        help="Request capacities for inline prefill attention capture; "
        "replay rounds up to the smallest fitting captured batch size. "
        "Unset: 4, capped by --max-num-seqs per data-parallel rank and the "
        "token bucket, and at least the request count that fits each token "
        "bucket within the model context. KDA captures a fixed request count, "
        "so each token bucket needs one inline variant per request count. "
        "Adding request counts increases capture time and memory. "
        "This does not replace the scheduler's --max-num-seqs limit. "
        "Batches without a fitting capacity retain the ordinary attention breaks.",
    )
    parser.add_argument(
        "--enable-nan-detection",
        action="store_true",
        help="Enable the NaN guard: sanitize non-finite logits before "
        "sampling, detect requests whose logits contained NaN (or whose "
        "sampled token id escaped the vocab range), and terminate only "
        "those requests with a numerical error so corruption cannot "
        "spread to the rest of the batch.",
    )
    parser.add_argument(
        "--enable-nvtx",
        action="store_true",
        help="Emit NVTX ranges around input_prep / target_forward / "
        "sampling / drafter stages for nsys profiling. Off by default "
        "(true no-op — no NVTX calls are made). Also enabled by "
        "TOKENSPEED_NVTX=1.",
    )
    parser.add_argument(
        "--disable-weight-loader-prefetch-checkpoints",
        dest="weight_loader_prefetch_checkpoints",
        action="store_false",
        default=ServerArgsOptions.weight_loader_prefetch_checkpoints,
        help=(
            "Disable prefetching safetensors checkpoint shards into the OS "
            "page cache. Prefetch is enabled by default: shards are read "
            "in parallel ranges a bounded window ahead of weight loading "
            "(min(40 GiB, 25%% of available host memory)), so weight copies "
            "hit the cache at streaming bandwidth instead of demand-faulting "
            "cold pages from shared filesystems."
        ),
    )
    parser.add_argument(
        "--weight-loader-prefetch-num-threads",
        type=int,
        default=ServerArgsOptions.weight_loader_prefetch_num_threads,
        help="Maximum concurrent checkpoint range readers per rank.",
    )
    parser.add_argument(
        "--enable-memory-saver",
        action="store_true",
        help="Allow saving memory using release_memory_occupation and resume_memory_occupation",
    )
    parser.add_argument(
        "--allow-reload-forward",
        action="store_true",
        help="Development: allow the reload_forward control call (ts reload-forward), "
        "which drains the engine, re-executes edited model, layer and kernel modules "
        "in place, warms their kernels and recaptures the CUDA graphs. Nothing "
        "reloads until the call arrives.",
    )
    parser.add_argument(
        "--weight-memory-policy",
        choices=["cpu_backup", "discard"],
        default=None,
        help=(
            "Required with memory saver: back up weights to CPU, or discard "
            "them until a complete SDK update. Frozen draft weights retain CPU backup."
        ),
    )
    parser.add_argument(
        "--disable-cudagraph-memory-reserve",
        action="store_true",
        help="Do not reserve the projected CUDA-graph pool memory, nor what startup "
        "keeps resident on CUDA, in the KV cache budget.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "--tp",
        type=int,
        default=None,
        help="Sets tensor parallelism size uniformly (equivalent to --attn-tp-size). "
        "Cannot be used together with --attn-tp-size.",
    )
    parser.add_argument(
        "--enable-expert-parallel",
        action="store_true",
        help="Enable expert parallelism by automatically setting ep_size to world_size.",
    )

    # Specify different parallel strategies, different combinations correspond to different communication groups and weight partitioning, as well as different communication methods
    parser.add_argument(
        "--attn-tp-size",
        type=int,
        default=ServerArgsOptions.attn_tp_size,
        help="Specify tp size for attn part",
    )
    parser.add_argument(
        "--kv-parallel-size",
        type=int,
        default=ServerArgsOptions.kv_parallel_size,
        help="KV parallelism (KVP): shard full-history KV pages (MLA/DSA "
        "latent, DeepSeek V4 compressed KV) cyclically over a consecutive "
        "subgroup of attention TP, each rank storing one shard of every "
        "request's pages. Must divide --attn-tp-size. Aggregated and PD "
        "decode engines require equal --decode-context-parallel-size; "
        "the PD prefill role writes the shards and "
        "may split its query rows over them with "
        "--query-context-parallel-size. Works with the Host KVStore and the "
        "retraction snapshot pool; not with L3 storage "
        "(--kvstore-storage-backend).",
    )
    parser.add_argument(
        "--decode-context-parallel-size",
        type=int,
        default=ServerArgsOptions.decode_context_parallel_size,
        help="Decode context parallelism (DCP) in aggregated or PD decode engines: every "
        "rank attends the KV pages it owns under --kv-parallel-size and the "
        "partials are merged across the KVP group by their log-sum-exp. "
        "Requires --disaggregation-mode null or decode and must equal "
        "--kv-parallel-size.",
    )
    parser.add_argument(
        "--attn-head-tp-size",
        type=int,
        default=ServerArgsOptions.attn_head_tp_size,
        help="Shard the MLA head projections (q_b_proj, kv_b_proj, o_proj) "
        "by heads over this many contiguous ranks that hold different rows; "
        "the attention exchanges heads for tokens around core attention. "
        "Either attention-DP ranks of a decode engine (requires attention "
        "TP 1, attention DP and --disaggregation-mode decode; every rank "
        "keeps its own KV cache), or the query shards of a prefill engine "
        "(must equal --query-context-parallel-size; the extend rows run "
        "the absorbed sparse prefill through the exchange). Defaults to the "
        "ranks holding the same rows: the attention TP size, or 1 "
        "(head-replicated) under --query-context-parallel-size.",
    )
    parser.add_argument(
        "--head-tp-exchange-max-rows",
        type=int,
        default=ServerArgsOptions.head_tp_exchange_max_rows,
        help="Rows per rank the head-TP Lamport exchanges are prepared for "
        "(the absorbed query, the latent attention output, and the TP-BI "
        "o_proj rows). The symmetric-memory scratch is allocated before "
        "graph capture; larger forwards fall back to the generic all-to-all. "
        "Only used with --attn-head-tp-size > 1.",
    )
    parser.add_argument(
        "--ple-table-layout",
        choices=["global_sharded", "attn_tp_sharded", "shared", "peer_sharded"],
        default=ServerArgsOptions.ple_table_layout,
        help="PLE vocabulary layout: global_sharded splits across stage ranks; "
        "attn_tp_sharded splits within attention TP; shared maps one full host table "
        "per node; peer_sharded directly reads mapped host shards on one node.",
    )
    parser.add_argument(
        "--lm-head-tp-size",
        type=int,
        default=ServerArgsOptions.lm_head_tp_size,
        help="Vocab-shard the LM head over this many contiguous ranks. "
        "Under attention DP the default 1 "
        "keeps the head replicated; a wider group gathers distinct DP rows "
        "before the logits GEMM and transposes the vocab shards back. "
        "Without attention DP it must equal the attention TP size.",
    )
    parser.add_argument(
        "--tp-batch-invariant",
        type=str,
        choices=["none", "attn", "attn+dense"],
        default=ServerArgsOptions.tp_batch_invariant,
        help="Replace the reduce-scatter after a head-sharded o_proj "
        "(attn) and after a TP dense down_proj (attn+dense) with "
        "column-parallel GEMMs on hidden fed by an all-gather of the "
        "reduction dimension and followed by an all-to-all back to each "
        "rank's own rows. Every collective is then a permutation, so the "
        "bits match a TP1 full-K GEMM. attn needs --attn-head-tp-size > 1; "
        "attn+dense also needs --dense-tp-size > 1; both need unquantized "
        "o_proj / down_proj weights.",
    )
    parser.add_argument(
        "--disable-deepseek-v41-token-blacklist",
        action="store_true",
        default=ServerArgsOptions.disable_deepseek_v41_token_blacklist,
        help="Allow DeepSeek V4.1 to generate BOS, PAD, and image "
        "control/placeholder token IDs 0, 2, and 129262..129264. "
        "They are masked by default.",
    )
    parser.add_argument(
        "--deepseek-v41-attention-query-swa-min-tokens",
        type=int,
        default=ServerArgsOptions.deepseek_v41_attention_query_swa_min_tokens,
        help="Additional minimum query rows per request for SWA-only V4.1 "
        "attention QCP head exchange (default: 4096). The larger of this and "
        "--deepseek-v41-attention-query-min-tokens applies. Tune for the "
        "deployment; this is not a hardware auto-tuner.",
    )
    parser.add_argument(
        "--deepseek-v41-attention-query-min-tokens",
        type=int,
        default=ServerArgsOptions.deepseek_v41_attention_query_min_tokens,
        help="Minimum query rows per request in the current prefill window "
        "for V4.1 attention QCP head exchange (default: 1024). Also requires "
        "at least one query per rank. This threshold does not enable "
        "exchange or auto-tune for the GPU/interconnect.",
    )
    parser.add_argument(
        "--deepseek-v41-attention-query-parallel-size",
        type=int,
        default=ServerArgsOptions.deepseek_v41_attention_query_parallel_size,
        help="Enable DeepSeek V4.1 prefill attention QCP via query/head exchange "
        "over attention TP. Set 1 to disable (default), or attention TP "
        "size to enable for eligible long requests with 64 total heads. "
        "KV storage and model projections remain unchanged.",
    )
    parser.add_argument(
        "--deepseek-v41-indexer-query-parallel-size",
        type=int,
        default=ServerArgsOptions.deepseek_v41_indexer_query_parallel_size,
        help="Enable DeepSeek V4.1 long-prefill encoder indexer QCP for query "
        "projections and scoring over attention TP. Set 1 to disable "
        "(default), or the attention TP size to enable. KV remains "
        "replicated; this does not enable the generic QCP execution path.",
    )
    parser.add_argument(
        "--query-context-parallel-size",
        type=int,
        default=ServerArgsOptions.query_context_parallel_size,
        help="Query context parallelism (QCP) on the PD prefill role: shard "
        "every extend forward's query rows over the attention TP group, rank "
        "r computing a contiguous slice of the chunk's rows against the "
        "gathered KV history of its requests. Requires --disaggregation-mode "
        "prefill, must equal --attn-tp-size and keeps the prefill eager. "
        "--kv-parallel-size must be 1 or equal to it. GPU DSA attention "
        "needs it equal to --kv-parallel-size and a bf16 KV cache; its "
        "attention weights are head-replicated unless --attn-head-tp-size "
        "equals it, which shards them over the shard group. DeepSeek V4.1 "
        "keeps its attention weights head-replicated; this model-wide layout "
        "is separate from its operator-level "
        "--deepseek-v41-*-query-parallel-size options.",
    )
    parser.add_argument(
        "--dense-tp-size",
        type=int,
        default=ServerArgsOptions.dense_tp_size,
        help="Specify tp size for dense part. Defaults to the attention "
        "TP width: the full world without DP attention, one replica with "
        "it.",
    )
    parser.add_argument(
        "--moe-weight-sharing-size",
        type=int,
        default=ServerArgsOptions.moe_weight_sharing_size,
        help="Share immutable expert weights across this many workers while executing "
        "each batch locally (DWDP). Omission keeps resident weights. Requires "
        "prefill, TP1/EP1 compute, and disabled model CUDA graphs.",
    )
    parser.add_argument(
        "--moe-tp-size",
        type=int,
        default=ServerArgsOptions.moe_tp_size,
        help="Specify tp size for MoE part, default equals nprocs-per-node, if non dp_attn && combine_dense mode, this parameter will be overridden by attn_tp_size",
    )
    parser.add_argument(
        "--nprocs-per-node",
        type=int,
        default=ServerArgsOptions.nprocs_per_node,
        help="Number of processes to start per node",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=ServerArgsOptions.world_size,
        help="Total number of processes across all nodes.",
    )
    parser.add_argument(
        "--emulate-rank-zero",
        action="store_true",
        help="Run only global rank 0 of the configured parallel layout, "
        "on one GPU. Collectives become local stand-ins that keep the "
        "real shapes but not the values, so kernels, weight shards and "
        "cache sizing match rank 0 of the full deployment while outputs "
        "are meaningless. AMD GPUs only; requires one node and no "
        "pipeline, context or attention data parallelism.",
    )
    parser.add_argument(
        "--force-deterministic-rsag",
        action="store_true",
        help="NCCL and the rank-ordered fold only: no symmetric-memory "
        "path -- neither the Triton multicast all-gather/reduce-scatter, "
        "nor the trtllm and Triton all-reduce tiers, nor the distributed "
        "argmax. With --batch-invariant-collectives every reduction takes "
        "the fold; without it, NCCL. Not folded in by --numerics "
        "rl-bitwise, whose gathers keep the multicast kernels.",
    )
    parser.add_argument(
        "--batch-invariant-collectives",
        action="store_true",
        help="One association order per reduction, independent of the "
        "batch: every reduction is a fixed-rank-order fp32 fold, on the "
        "trtllm workspace kernels where one is armed, otherwise as data "
        "movement plus the fold, which for an all-reduce costs "
        "world_size times the traffic. NCCL sums are run-stable but chunk "
        "by message size, and the NVLS in-switch reduction's order moves "
        "with a row's offset in the buffer, so neither is "
        "batch-invariant. Gathers keep the multicast kernels. Turns off "
        "the fused all-reduce+norm. Folded in by --numerics rl-bitwise.",
    )
    parser.add_argument(
        "--owned-attention-output",
        type=str,
        default=ServerArgsOptions.owned_attention_output,
        choices=list(OWNED_ATTENTION_OUTPUTS),
        help="Muse Spark: how owned-row prefill (extend) forwards return "
        "attention outputs to the token owners. all-to-all sends each owner "
        "its rows of every rank's heads and projects them through the whole "
        "output projection, gathered on every rank after loading: 128 MiB "
        "(8192 x 8192 BF16) a layer, about 7.75 GiB a rank over 62 layers, "
        "taken from the KV pool. A numerics change. Refused where Iris owned "
        "rows do not run.",
    )
    parser.add_argument(
        "--prefill-collective-payload",
        type=str,
        default=ServerArgsOptions.prefill_collective_payload,
        choices=list(PREFILL_COLLECTIVE_PAYLOADS),
        help="Muse Spark: the payload encoding of the owned-row prefill "
        "collectives in extend forwards. int8-block16 sends INT8 codes with "
        "an FP32 scale per 16 elements (a numerics change); decode stays "
        "BF16, and so do the heads of --owned-attention-output all-to-all.",
    )
    parser.add_argument(
        "--dense-fp8-small-m",
        type=str,
        default=ServerArgsOptions.dense_fp8_small_m,
        choices=["none", "qkv"],
        help="Muse Spark on gfx950: run decode and verify forwards' QKV "
        "projection at 33-128 rows as an FP8 GEMM (qkv); prompt K / V "
        "stays BF16. Startup refuses what it cannot serve. A numerics "
        "change, refused under --numerics rl-bitwise and "
        "--batch-invariant-collectives. none keeps BF16.",
    )
    parser.add_argument(
        "--numerics",
        type=str,
        choices=list(numerics_envelopes) if numerics_envelopes is not None else None,
        default=ServerArgsOptions.numerics,
        help="Numerics envelope. rl-bitwise folds the determinism "
        "switches (batch-invariant collectives, no autotune/TF32/PDL, no "
        "fused all-reduce, the batch-invariant MoE leaves, per-request "
        "sampling) so outputs and logprobs are bitwise identical across "
        "runs and batch compositions within one deployment, and the "
        "trainer-operation-order switches (docs/design/numerics.md, "
        "alignment.trainer) so a teacher-forced pass reproduces the RL "
        "trainer's logprobs; a model serves it only once its profile "
        "declares it verified. Plugins may register additional envelopes.",
    )
    parser.add_argument(
        "--yarn-ramp-mask-device",
        type=str,
        choices=list(YARN_RAMP_MASK_DEVICES),
        default=ServerArgsOptions.yarn_ramp_mask_device,
        help="Device that computes the deepseek_yarn linear ramp mask. "
        "Defaults to cuda, independently of --yarn-frequency-device. "
        "Folded to cpu by --numerics rl-bitwise.",
    )
    parser.add_argument(
        "--yarn-frequency-device",
        type=str,
        choices=list(YARN_FREQUENCY_DEVICES),
        default=ServerArgsOptions.yarn_frequency_device,
        help="Device that computes deepseek_yarn powers, divisions and "
        "interpolation. Defaults to cuda, independently of --yarn-ramp-mask-device. "
        "Folded to cuda by --numerics rl-bitwise, overriding explicit cpu. "
        "Use cuda with --yarn-ramp-mask-device cpu for CUDA frequencies "
        "with a CPU-built ramp. CPU and CUDA arithmetic can round differently.",
    )
    parser.add_argument(
        "--mla-lora-scale",
        type=str,
        choices=list(MLA_LORA_SCALES),
        default=ServerArgsOptions.mla_lora_scale,
        help="Where LongCat-style MLA applies its sqrt(hidden / lora_rank) "
        "norm scales. 'folded': into the q_a_layernorm / kv_a_layernorm "
        "weights after loading. 'runtime': as separate bf16 multiplies "
        "after q_b_proj and after kv_a_layernorm, as the trainer does; the "
        "norm weights are never rewritten and the DSA indexer reads the "
        "unscaled q_lora. Folded to runtime by --numerics rl-bitwise.",
    )
    parser.add_argument(
        "--layer-boundary-norm",
        type=str,
        choices=list(LAYER_BOUNDARY_NORMS),
        default=ServerArgsOptions.layer_boundary_norm,
        help="The norm at each physical layer boundary (a layer's first "
        "norm and the final norm). 'fused': the fused add+norm kernel, "
        "whose residual sum stays fp32 into the norm. 'unfused': "
        "hidden + residual is materialized in bf16 first, then a "
        "standalone RMSNorm, as the trainer does; all-reduce+norm fusion "
        "is vetoed with it. Folded to unfused by --numerics rl-bitwise.",
    )
    parser.add_argument(
        "--router-topk",
        type=str,
        choices=list(ROUTER_TOPKS),
        default=ServerArgsOptions.router_topk,
        help="Correction-bias MoE routing (LongCat). 'fused': the fused "
        "CUDA softmax+bias+top-k kernel. 'torch': fp32 torch.softmax, "
        "torch.topk(probs + bias, sorted=True) in PyTorch tie order, "
        "weights = unbiased probs x routed_scaling_factor, zero experts "
        "become id -1 and keep their weight, as the trainer does. Folded "
        "to torch by --numerics rl-bitwise.",
    )
    parser.add_argument(
        "--logprob-order",
        type=str,
        choices=list(LOGPROB_ORDERS),
        default=ServerArgsOptions.logprob_order,
        help="Order of the selected-token log-softmax behind every "
        "returned logprob. 'torch': torch.log_softmax. 'blockwise': the "
        "fixed-width blockwise reduction (row max, shift, "
        "sum(exp) over fixed 32768-wide vocab blocks folded in block "
        "order), returning -(log(sum_exp) - target). 'blockwise-direct': "
        "the same reduction, returning target - log(sum_exp). "
        "Suffix either with -16384 to select 16384-wide blocks, "
        "independently of serving TP. "
        "Requests asking for "
        "temperature- or top-p-normalised logprobs are refused. Changes "
        "logprobs only, never the sampled tokens. Folded to blockwise by "
        "--numerics rl-bitwise, overriding an explicit torch; an explicit "
        "blockwise form and block width are preserved.",
    )
    parser.add_argument(
        "--moe-combine-order",
        type=str,
        choices=list(MOE_COMBINE_ORDERS),
        default=ServerArgsOptions.moe_combine_order,
        help="How a token's routed-expert contributions meet across the "
        "MoE TP-EP group. 'rank': the MoE kernel returns this rank's "
        "partial and the host sums the partials (all-reduce or "
        "reduce-scatter), adding the identity zero-expert residual once "
        "around it. 'slot': the MoE kernel folds the token's top-k slots "
        "in fp32 slot order across the EP group itself, zero-expert "
        "residual included, as the trainer's grouped MLP does, and the "
        "host reduces nothing; needs MoE TP 1 and a kernel declaring the "
        "combine_order trait with slot (a registered batch-invariant leaf), "
        "and vetoes all-reduce+norm fusion. Folded to slot by --numerics "
        "rl-bitwise.",
    )
    parser.add_argument(
        "--dsa-slot-order",
        type=str,
        choices=list(DSA_SLOT_ORDERS),
        default=ServerArgsOptions.dsa_slot_order,
        help="The order the sparse (DSA) attention cores reduce a token's "
        "selected KV slots in. 'selection': as the top-k leaf emitted "
        "them (every core). 'sorted': ascending token position, so the "
        "reduction is batch-invariant whenever the selected set is; "
        "served only by cores declaring the slot_order trait (the "
        "registered batch-invariant leaves). Folded to selection by --numerics "
        "rl-bitwise.",
    )
    parser.add_argument(
        "--dsa-index-k-format",
        type=str,
        choices=["fp8_scaled", "mxfp4"],
        default=ServerArgsOptions.dsa_index_k_format,
        help="Storage of Kimi-K3.1's DSA indexer key cache; other models "
        "refuse the flag. 'fp8_scaled' (the default): FP8 keys with one "
        "fp32 scale per 128 elements, 132 bytes per token per layer. "
        "'mxfp4': E2M1 keys with one UE8M0 scale per 32 elements, 68 "
        "bytes, scored against MXFP4 queries on AMD CDNA4; refused under "
        "bitwise numerics.",
    )
    parser.add_argument(
        "--disable-sampling-tp-sync",
        action="store_true",
        help="Skip broadcasting sampler outputs across the attention TP "
        "group. Only safe when the sampling kernels are deterministic.",
    )
    parser.add_argument(
        "--low-latency-max-num-tokens-per-gpu",
        type=int,
        default=ServerArgsOptions.low_latency_max_num_tokens_per_gpu,
        help="DeepEP low-latency send capacity per rank. Defaults to 256; "
        "set explicitly to cover the largest batch sent through low latency.",
    )

    parser.add_argument(
        "--mla-chunk-multiplier",
        type=int,
        default=ServerArgsOptions.mla_chunk_multiplier,
        help=(
            "Per-iter MLA chunked-prefill chunk capacity multiplier; "
            "the actual capacity is chunked_prefill_size * mla_chunk_multiplier."
        ),
    )

    # Multimodal
    mm_attention_backend_choices = [
        "fa3",
        "fa4",
        "triton_attn",
        "flashinfer_cudnn",
    ]
    parser.add_argument(
        "--mm-attention-backend",
        type=str,
        choices=mm_attention_backend_choices,
        default=ServerArgsOptions.mm_attention_backend,
        help="Set multimodal attention backend.",
    )
    parser.add_argument(
        "--mm-encoder-tp-mode",
        type=str,
        choices=["weights", "data"],
        default=ServerArgsOptions.mm_encoder_tp_mode,
        help=(
            "Multimodal encoder parallelism within each attention TP "
            "group. 'weights' shards encoder weights with TP (default); "
            "'data' replicates encoder weights and distributes whole "
            "multimodal items across the TP ranks."
        ),
    )
    # Disaggregation
    parser.add_argument(
        "--enable-external-encoder",
        action="store_true",
        help="Receive external encoder embeddings in aggregated or prefill serving.",
    )
    parser.add_argument(
        "--disaggregation-mode",
        type=str,
        default="null",
        choices=["null", "prefill", "decode", "encode"],
        help='Used for PD/EPD disaggregation. "prefill" for prefill-only server, "decode" for decode-only server, and "encode" for a vision-tower-only server that ships image embeddings to a prefill server. If not specified, it is not disaggregated',
    )
    parser.add_argument(
        "--disaggregation-decode-allocation-policy",
        type=str,
        choices=["early", "deferred"],
        default=ServerArgsOptions.disaggregation_decode_allocation_policy,
        help="Prefill servers only: what a request's Decode allocation "
        "gates. 'early' (the default) starts the prefill once every Decode "
        "rank has allocated the destination; it is deprecated and will be "
        "removed once 'deferred' supports the layouts that still need it. "
        "'deferred' (experimental) starts the prefill without waiting for "
        "the allocation, which the Decode server makes once the prefill "
        "server holds every page of the prompt, and holds only its KV "
        "transfer until then. deferred requires attention TP 1, "
        "--kv-parallel-size 1, --pipeline-parallel-size 1, no "
        "--enable-external-encoder, --disaggregation-layerwise-interval 0 "
        "and a model without multimodal input (or --language-model-only). "
        "Refused with any other --disaggregation-mode; see "
        "docs/configuration/server.md.",
    )
    parser.add_argument(
        "--comm-fusion-max-num-tokens",
        type=int,
        default=ServerArgsOptions.comm_fusion_max_num_tokens,
        help="Max num tokens for communication fusion workspace",
    )
    parser.add_argument(
        "--enable-allreduce-fusion",
        action="store_true",
        help="Enable allreduce fusion for improved decode performance. Auto-enabled on supported single-node TP configurations, and for Beam on any number of nodes (its boundaries fuse where an MNNVL workspace serves them).",
    )
    parser.add_argument(
        "--disaggregation-bootstrap-port",
        type=int,
        default=ServerArgsOptions.disaggregation_bootstrap_port,
        help="Bootstrap server port on the prefill server. Default is 8998.",
    )
    parser.add_argument(
        "--disaggregation-transfer-backend",
        type=str,
        default=ServerArgsOptions.disaggregation_transfer_backend,
        choices=["mooncake"],
        help="The backend for disaggregation transfer. Default is mooncake.",
    )
    parser.add_argument(
        "--disaggregation-ib-device",
        type=str,
        default=ServerArgsOptions.disaggregation_ib_device,
        help="The InfiniBand devices for disaggregation transfer, accepts single device (e.g., --disaggregation-ib-device mlx5_0) "
        "or multiple comma-separated devices (e.g., --disaggregation-ib-device mlx5_0,mlx5_1). "
        "Default is None, which triggers automatic device detection when mooncake backend is enabled.",
    )
    parser.add_argument(
        "--disaggregation-layerwise-interval",
        type=int,
        default=ServerArgsOptions.disaggregation_layerwise_interval,
        help="The interval of layerwise transfer for disaggregation. Default is 1.",
    )
    parser.add_argument(
        "--pdlb-url",
        type=str,
        default=None,
        help="The URL of the PD disaggregation load balancer. If set, the prefill/decode server will register with the load balancer.",
    )

    parser.add_argument(
        "--weight-version",
        type=str,
        default=ServerArgsOptions.weight_version,
        help="Initial model-weight version: the L3 cache's namespace, which a "
        "flushed weight update naming a new version replaces.",
    )
    parser.add_argument(
        "--model-update-config",
        type=str,
        default=ServerArgsOptions.model_update_config,
        help="JSON object handed to the Model Updater SDK for the "
        "UpdateWeightsFromMooncake control operation. Requires "
        "--model-update-sdk-module, "
        "--model-update-engine-type and --model-update-draft-weights.",
    )
    parser.add_argument(
        "--model-update-sdk-module",
        type=str,
        default=ServerArgsOptions.model_update_sdk_module,
        help="Import path of the Model Updater SDK module (imported lazily "
        "on the first Mooncake weight update).",
    )
    parser.add_argument(
        "--model-update-engine-type",
        type=str,
        default=ServerArgsOptions.model_update_engine_type,
        help="Model Updater SDK EngineType member name for this engine "
        "(resolved as EngineType[value.upper()]).",
    )
    parser.add_argument(
        "--model-update-draft-weights",
        type=str,
        choices=["retain", "refresh"],
        default=ServerArgsOptions.model_update_draft_weights,
        help="Whether a Mooncake weight update also streams the "
        "speculative draft model's weights: 'retain' updates the target "
        "only, 'refresh' updates target and draft.",
    )
