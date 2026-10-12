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

"""The arguments of the server."""

import argparse
import dataclasses
import json
import os
import random
import socket
from collections.abc import Sequence
from typing import Literal

from tokenspeed_kernel.ops.attention.gdn.triton import CHUNK_SIZE as FLA_CHUNK_SIZE
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.configs.numerics import (
    DSA_SLOT_ORDERS,
    LAYER_BOUNDARY_NORMS,
    LOGPROB_ORDERS,
    MLA_LORA_SCALES,
    MOE_COMBINE_ORDERS,
    NUMERICS_ENVELOPES,
    RL_BITWISE_SAMPLING_BACKENDS,
    ROUTER_TOPKS,
    SAMPLING_STREAMS,
    YARN_RAMP_MASK_DEVICES,
)
from tokenspeed.runtime.distributed.mapping import Mapping, _resolve_parallelism_sizes
from tokenspeed.runtime.moe.dispatch_algorithm import (
    EP_DISPATCH_ALGORITHMS,
    STATIC_EP_DISPATCH_ALGORITHMS,
)
from tokenspeed.runtime.utils import (
    get_amdgpu_memory_capacity,
    get_colorful_logger,
    get_nvgpu_memory_capacity,
    is_valid_ipv6_address,
    maybe_model_redirect,
    nullable_str,
)
from tokenspeed.runtime.utils.launcher import check_dist_init_port, detect_topology
from tokenspeed.runtime.utils.network import is_port_available
from tokenspeed.runtime.utils.spec_block_geometry import (
    BLOCK_SPEC_ALGORITHMS,
    BLOCK_SPEC_RULES,
)

logger = get_colorful_logger(__name__)

# Sampling backends whose verify runs the draft-prob chain kernel
# (--enable-speculative-sampling). greedy verifies by exact match and the
# Triton backends by target-sampled exact match; the drafter's recorded
# distribution never enters either.
SPECULATIVE_SAMPLING_BACKENDS = frozenset({"flashinfer", "flashinfer_full"})

# Usable range of --spec-reject-draft-prob-threshold. The sentinel rows are
# written as threshold + 1.0 in fp32 and detected by ``draft_prob > threshold``:
# below 1.0 a real probability would read as the sentinel, and from 2**24 on
# fp32 (24 significand bits, ulp 2.0 there) can no longer resolve the + 1.0;
# 2**20 leaves a wide margin, and nothing is gained from a larger sentinel.
SPEC_REJECT_DRAFT_PROB_THRESHOLD_MIN = 1.0
SPEC_REJECT_DRAFT_PROB_THRESHOLD_MAX = float(1 << 20)

# Spec-decode overshoot spans the physical KV extent must absorb past the
# logical context_len. The overlap scheduler steps a finished request at most
# ONE extra iteration (the depth-1 event loop commits the previous step every
# round, so the Finish event reaches the C++ scheduler before the next plan),
# and termination is checked CPU-side against max_new_tokens, so verify can
# commit past context_len for exactly that window:
#   1. the finishing step's own accept (up to spec_num_tokens past the limit),
#   2. the single lingering step's accept,
#   3. the lingering step laying out its next draft block after that accept.
# Hence 3 spans. Everything written past context_len belongs to a request
# whose output is already truncated by max_new_tokens; the garbage KV is
# evicted with the request one step later.
_SPEC_OVERSHOOT_SPANS = 3

# Speculative algorithms a prefill server runs on the chunk pipeline
# (--pipeline-parallel-size > 1). The drafter executes on the last stage, the
# only stage that samples: an MTP (NextN) draft needs only that stage's final
# hidden states, and DSPARK produces its draft context across stages. EAGLE3
# is excluded because its aux taps come from several stages and nothing
# carries them through the stage boundary. See
# ServerArgs.resolve_disaggregation.
PIPELINE_SPEC_ALGORITHMS = ("DSPARK", "MTP")


def expert_placement_requested(server_args) -> bool:
    """Whether serving needs an expert placement beyond the trivial identity.

    Redundant experts, a non-trivial initial location, load recording and
    online rebalancing all need the placement tables
    (``moe/expert_location.py``); plain EP serving does not and keeps its
    routing untouched.
    """
    return (
        server_args.ep_num_redundant_experts > 0
        or server_args.init_expert_location != "trivial"
        or server_args.expert_distribution_recorder_mode is not None
        or server_args.enable_eplb
    )


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


def validate_kvp_disaggregation_role(
    *, has_kvp: bool, disaggregation_mode: str
) -> None:
    """Reject KVP on PD roles whose transfer path cannot shard pages yet.

    An aggregated engine and the prefill role may shard: the prefill sender
    copies only the pages each rank owns and every rank of the KVP subgroup
    serves every decode rank. The decode role receives into an unsharded
    cache only -- no receive path lands a block on its owner alone -- and the
    encode role has no KV cache to shard.
    """
    if has_kvp and disaggregation_mode not in ("null", "prefill"):
        raise ValueError(
            "--kv-parallel-size > 1 requires --disaggregation-mode "
            f"null or prefill (got {disaggregation_mode!r}): only the prefill "
            "side of a PD transfer can be KVP-sharded"
        )


def _require_choice(flag: str, value: str, choices: tuple[str, ...]) -> None:
    """Refuse a launch value outside ``flag``'s closed set of ``choices``.

    ServerArgs is the one place a closed-set flag is validated; consumers
    read the resolved value and trust it.
    """
    if value not in choices:
        raise ValueError(f"{flag} must be one of {list(choices)}, got {value!r}")


# Attention backends whose sparse prefill can attend a query shard against the
# gathered history of its requests (the query-context-parallel extend arm).
QCP_ATTENTION_BACKENDS = frozenset({"dsa"})


def validate_qcp(
    *,
    qcp_size: int,
    attn_tp_size: int,
    attn_dp_size: int,
    dense_tp_size: int,
    moe_tp_ep_size: int,
    kvp_size: int,
    disaggregation_mode: str,
    disable_prefill_graph: bool,
    enable_mixed_batch: bool,
    attention_backend: str | None,
    kv_cache_dtype: str,
    kv_cache_quant_method: str,
) -> None:
    """Reject query-context-parallel layouts the first landing does not serve.

    QCP shards an extend forward's rows over the attention TP group. It is a
    prefill-role layout: the decode arm serves only the drafter's steps, the
    eager extend break gathers the request history, and the sparse DSA
    kernels attend the gathered buffer. ``qcp_size == 1`` is off and passes.
    """
    if qcp_size == 1:
        return
    if kv_cache_dtype not in ("auto", "bfloat16") or kv_cache_quant_method != "none":
        raise ValueError(
            "--prefill-context-parallel-size > 1 requires a bf16 KV cache (got "
            f"--kv-cache-dtype {kv_cache_dtype!r}, --kv-cache-quant-method "
            f"{kv_cache_quant_method!r}): the sharded KV write gathers the rotated "
            "latent and stores it with latent_store, which writes native rows only"
        )
    if qcp_size != attn_tp_size:
        raise ValueError(
            "--prefill-context-parallel-size must equal the attention TP size "
            f"(got {qcp_size} with attn_tp_size={attn_tp_size}): the query shard "
            "spans the whole attention TP group"
        )
    if attn_dp_size != 1:
        raise ValueError(
            "--prefill-context-parallel-size > 1 requires attention DP 1 (got "
            f"attn_dp_size={attn_dp_size}): the sampled-row table of a shard is "
            "per DP group and the DP metadata gather does not carry it"
        )
    if dense_tp_size not in (1, attn_tp_size) or moe_tp_ep_size not in (
        1,
        attn_tp_size,
    ):
        raise ValueError(
            "--prefill-context-parallel-size > 1 requires --dense-tp-size and the "
            f"MoE TP x EP group to be 1 or the attention TP width {attn_tp_size} "
            f"(got dense {dense_tp_size}, MoE {moe_tp_ep_size}): attention returns "
            "complete rows (head-replicated weights, or the head-TP tail), so the "
            "drafter's replicated decode rows are never scattered and a narrower "
            "dense or MoE group has no rows to gather"
        )
    if disaggregation_mode != "prefill":
        raise ValueError(
            "--prefill-context-parallel-size > 1 requires --disaggregation-mode "
            f"prefill (got {disaggregation_mode!r}): a sharded extend and "
            "replicated decode rows cannot share one forward"
        )
    if not disable_prefill_graph:
        raise ValueError(
            "--prefill-context-parallel-size > 1 requires --disable-prefill-graph: "
            "the history gather runs in the eager attention break"
        )
    if enable_mixed_batch:
        raise ValueError(
            "--prefill-context-parallel-size > 1 does not support "
            "--enable-mixed-batch: a MIXED round would carry sharded extend rows "
            "and replicated decode rows in one forward"
        )
    if (
        attention_backend is not None
        and attention_backend not in QCP_ATTENTION_BACKENDS
    ):
        raise ValueError(
            "--prefill-context-parallel-size > 1 requires a DSA-family attention "
            f"backend ({sorted(QCP_ATTENTION_BACKENDS)}), got "
            f"--attention-backend {attention_backend!r}"
        )
    if kvp_size not in (1, qcp_size):
        raise ValueError(
            "--kv-parallel-size must be 1 or equal to "
            f"--prefill-context-parallel-size (got kvp={kvp_size}, qcp={qcp_size}): "
            "the history gather splits by the page owners of the whole shard group"
        )


@dataclasses.dataclass
class ServerArgs:
    # Model and tokenizer
    model: str
    tokenizer: str | None = None
    tokenizer_mode: str = "auto"
    skip_tokenizer_init: bool = False
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

    # Direct SMG msgpack ZMQ path. When enabled, the scheduler skips the pickle
    # PULL/PUSH IPC and instead connects to SMG (which binds the handshake/input/
    # output sockets) over the msgpack wire. Default OFF;
    # SMG tokenizes/detokenizes, so pair with --skip-tokenizer-init.
    zmq_msgpack: bool = False
    # The frontend handshake endpoint dialed by each scheduler DP rank.
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
    max_total_tokens: int | None = None
    chunked_prefill_size: int | None = None
    max_prefill_tokens: int = 8192
    enable_mixed_batch: bool = False
    # Scheduler cache-reuse identity granularity in tokens.
    prefix_granularity: int = 64
    # special kv cache
    mamba_ssm_dtype: str = "float32"

    # Other runtime options
    stream_interval: int = 1
    stream_output: bool = False
    # Inline detokenization is the only supported path and is intentionally
    # not configurable from the CLI.
    enable_inline_detokenizer: bool = True
    seed: int | None = None
    distributed_timeout_seconds: int | None = None
    download_dir: str | None = None
    # Used for customizing extensible models
    ext_yaml: str | None = None
    base_gpu_id: int = 0
    gpu_id_step: int = 1

    # Logging
    log_level: str = "info"
    enable_log_requests: bool = True
    log_requests_level: int = 0
    enable_log_request_stats: bool = False
    enable_metrics: bool = False
    decode_log_interval: int = 40
    metrics_reporters: list[str] | None = None
    app_key: str | None = None

    # Cache events
    kv_events_config: str | None = None

    # Port for the in-engine SGLang-compatible RL control app (weight sync,
    # pause/resume, and memory occupation). Set by the ``ts serve`` orchestrator;
    # None disables the in-engine app.
    rl_control_port: int | None = None
    # Bind host for the in-engine RL control app. None binds the engine's
    # --host. Bind a reachable address (and set --rl-control-api-key) when an
    # external gateway drives this engine.
    rl_control_host: str | None = None
    # Bearer token the in-engine RL control app requires on every route. None
    # leaves it open, which is what slime expects by default. Never exported in server info.
    rl_control_api_key: str | None = dataclasses.field(default=None, repr=False)
    # Version identifier for the model weights. Stamped into every generation
    # response's meta_info so RL trainers know which policy version produced each
    # sample. Updated atomically after a successful weight push when the trainer
    # supplies a new version string.
    weight_version: str = "default"
    # Model Updater SDK (``/update_weights_from_mooncake``). The config is an
    # opaque JSON object handed to the SDK; the other three are required with
    # it and must stay unset without it (validated in ``validate``).
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
    load_balance_method: str = "shortest_queue"
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

    # MoE backend
    moe_backend: str = "auto"
    draft_moe_backend: str | None = None
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
    kvstore_io_backend: str = "kernel"
    # Optional L3 storage beneath the compact Host cache.
    kvstore_storage_backend: str | None = None
    kvstore_storage_backend_extra_config: str | None = None
    # The L3 prefetch that fills a waiting request's Host pages before its
    # admission: the shortest L3 prefix worth one (prefix pages), the lane's
    # deadline (base + per_page * pages seconds) and its batch. Explicit with
    # an L3 store, unset without one (see validate_l3_prefetch_options).
    kvstore_prefetch_min_pages: int | None = None
    kvstore_prefetch_timeout_base_s: float | None = None
    kvstore_prefetch_timeout_per_page_s: float | None = None
    kvstore_prefetch_batch_pages: int | None = None

    # Retraction snapshot pool (pinned Host, per rank) and its slot-state
    # arena rows: a retracted request is suspended with its image and
    # resumed by restore; a victim whose image does not fit is aborted
    # instead. Sized like the KVStore: an explicit size in gigabytes wins
    # (0 = not set), else a ratio of this rank's Device KV capacity (an
    # explicit 0 disables the pool: every capacity block then aborts its
    # victim), else derived at device build -- 0.1 of the Device KV without
    # L2, the tails of ``retraction_snapshot_max_requests`` images with L2
    # (cache/l2/sizing.py). The request cap derives from --max-num-seqs per
    # attention-DP rank when 0 (resolve_retraction_snapshot_options).
    retraction_snapshot_host_gb: float = 0.0
    retraction_snapshot_ratio: float | None = None
    retraction_snapshot_max_requests: int = 0
    # Test knob: every |N| plans the scheduler retracts the oldest quiescent
    # Decoding (N > 0) or Prefilling (N < 0) request without capacity
    # pressure, to exercise the suspend/restore path. 0 (off) in serving.
    debug_force_retraction_interval: int = 0

    # Multi-node distributed serving. ``None`` means "not given by the user",
    # which is what lets the launcher environment fill them in.
    dist_init_addr: str | None = None
    nnodes: int | None = None
    node_rank: int | None = None

    # Hugging Face model config overrides in JSON
    hf_overrides: str = "{}"
    preferred_sampling_params: str | None = None

    # Kernel backend
    attention_backend: str | None = None
    kda_backend: str = "auto"
    drafter_attention_backend: str | None = None
    # BLASST skip-softmax sparsity, gluon MHA prefill only (gfx950). 0.0
    # (default) is exact dense attention; see --skip-softmax-threshold help.
    skip_softmax_threshold: float = 0.0
    sampling_backend: str | None = None
    # Random stream of the non-greedy rows of the FlashInfer sampling backends:
    # "batch" keys flashinfer's Philox stream by the batch row, so a request's
    # draw depends on its co-batch; "per-request" keys it by (request seed,
    # position) through the Gumbel-max pool kernels. See
    # docs/design/numerics.md, sampling.deterministic.
    sampling_stream: str = "batch"
    dp_sampling: bool = False
    dp_sampling_min_bs: int | None = None
    attention_use_fp4_indexer_cache: bool | None = None
    use_trtllm_ragged_deepseek_prefill: bool | None = None

    # KV parallelism: shard the full-history KV pages page-cyclically over a
    # consecutive subgroup of attention TP (1 = off).
    kv_parallel_size: int = 1
    # Query context parallelism on the PD prefill role: shard every extend
    # forward's rows over the attention TP group (1 = off).
    prefill_context_parallel_size: int = 1
    deepseek_v4_mega_moe_max_num_tokens: int = 0
    deepseek_v4_indexer_prefill_max_logits_mb: int = 512
    deepseek_v4_prefill_chunk_size: int = 4
    # DeepSeek V4.1 Engram host tables. Off by default (GPU-sharded).
    engram_host_table: bool = False
    engram_host_table_dir: str | None = None
    engram_host_table_layout: str = "auto"

    # Grammar backend
    grammar_backend: str = "none"
    # Used by ``input_processor`` to defer json_schema grammars past the
    # model's reasoning channel.
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
    speculative_draft_model_quantization: str | None = "unquant"
    speculative_num_steps: int = 3
    speculative_eagle_topk: int = 1
    speculative_num_draft_tokens: int | None = None
    # Standard (draft-prob) rejection sampling for the chain drafters: the
    # drafter samples each step from its own distribution q and records it,
    # verify accepts with coin * q(x) < p(x). Off: the target-only rule
    # (accept with probability p(x) whatever the proposal). Both serve the
    # target distribution; this one trades a sampled proposal for a higher
    # acceptance rate under sampling temperatures.
    enable_speculative_sampling: bool = False
    # Recorded draft probabilities above this value mark a slot with no
    # proposal (fresh admission, PD landing): always reject. Sentinel rows
    # are written as threshold + 1, so it must lie within
    # [SPEC_REJECT_DRAFT_PROB_THRESHOLD_MIN, SPEC_REJECT_DRAFT_PROB_THRESHOLD_MAX].
    spec_reject_draft_prob_threshold: float = 2.0
    enable_replay_ssm: bool = True
    eagle3_layers_to_capture: str | None = None
    # Logprob support flags — all OFF by default. Enabling extends the
    # captured CUDA-graph footprint; requests asking for logprobs on a
    # server started without the matching flag will receive empty logprobs.
    enable_output_logprobs: bool = False
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
    disable_cuda_graph_padding: bool = False
    disable_autotune: bool = False
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
    # Trainer-operation-order switches (docs/design/numerics.md,
    # alignment.trainer). Each keeps the engine's own form by default and is
    # folded to the trainer's form by --numerics rl-bitwise.
    # Device that computes the deepseek_yarn RoPE inverse frequencies (the
    # position frequencies, both divisions and the YaRN linear ramp mask).
    yarn_ramp_mask_device: str = "cuda"
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
    low_latency_max_num_tokens_per_gpu: int = 256
    max_cudagraph_capture_size: int | None = None
    disable_prefill_graph: bool | None = False
    disable_kda_prefill_graph: bool = False
    # Breakable prefill graph bucket cap: None = auto min(2048, chunk); 0 disables.
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
    lm_head_tp_size: int | None = None
    tp_batch_invariant: Literal["none", "attn", "attn+dense"] = "none"
    dense_tp_size: int | None = None
    moe_tp_size: int | None = None
    mapping: Mapping | None = None
    emulate_rank_zero: bool = False

    mla_chunk_multiplier: int = 4
    mm_attention_backend: str | None = None
    mm_encoder_tp_mode: Literal["weights", "data"] = "weights"

    # For PD/EPD disaggregation: "null", "prefill", "decode", or "encode" (vision-tower-only).
    disaggregation_mode: str = "null"
    disaggregation_bootstrap_port: int = 8998
    disaggregation_transfer_backend: str = "mooncake"
    disaggregation_ib_device: str | None = None
    disaggregation_layerwise_interval: int = 1
    pdlb_url: str | None = None

    # For communication + norm fusion
    comm_fusion_max_num_tokens: int = 2048
    enable_allreduce_fusion: bool = False

    enable_expert_parallel: bool = False

    @property
    def mamba_cache_chunk_size(self) -> int:
        return max(FLA_CHUNK_SIZE, self.prefix_granularity)

    @property
    def spec_context_pad(self) -> int:
        """Tokens the physical KV extent must hold past the logical context_len.

        Zero without speculative decoding; see _SPEC_OVERSHOOT_SPANS for the
        derivation. Sizing consumers (page tables, graph capture widths) add
        this pad; user-semantic consumers (input validation, max_new_tokens
        folding, stop checks) keep the logical context_len.
        """
        if self.speculative_algorithm is None:
            return 0
        return _SPEC_OVERSHOOT_SPANS * int(self.speculative_num_draft_tokens)

    def __post_init__(self):
        self.resolve_basic_defaults()
        self.resolve_launcher_topology()
        self.resolve_parallelism()
        self.resolve_memory_and_scheduling()
        self.resolve_kernel_backends()
        self.resolve_cache()
        self.resolve_speculative_decoding()
        self.resolve_communication()
        self.resolve_numerics()
        self.resolve_disaggregation()
        self.validate()

    def resolve_basic_defaults(self):
        self.model = maybe_model_redirect(self.model)

        if self.kv_cache_dtype == "fp8":
            self.kv_cache_dtype = "fp8_e4m3"

        self.resolve_config_aliases()

        # Set missing default values
        if self.tokenizer is None:
            self.tokenizer = self.model

        if self.served_model_name is None:
            self.served_model_name = self.model

        if self.seed is None:
            self.seed = random.randint(0, 1 << 30)

    def resolve_config_aliases(self):
        # Whether the block-drafter widths were given rather than defaulted.
        self._speculative_widths_explicit = (
            self.speculative_num_steps != ServerArgs.speculative_num_steps
            or self.speculative_num_draft_tokens is not None
        )

        if self.use_trtllm_ragged_deepseek_prefill is not None:
            self.mla_disable_ragged = not self.use_trtllm_ragged_deepseek_prefill

        # Classify explicit DSpark arguments before cache validation, matching
        # the speculative-config path below. Model aliases that resolve to the
        # target checkpoint must use the same fail-closed cache contract.
        if self.speculative_config is None and self.speculative_algorithm == "DSPARK":
            explicit_draft_model = self.speculative_draft_model_path
            if (
                self.draft_model_path_use_base
                or explicit_draft_model is None
                or maybe_model_redirect(explicit_draft_model) == self.model
            ):
                self.draft_model_path_use_base = True

        if self.speculative_config is not None:
            try:
                config = json.loads(self.speculative_config)
            except json.JSONDecodeError as exc:
                raise ValueError("--speculative-config must be valid JSON") from exc

            if not isinstance(config, dict):
                raise ValueError("--speculative-config must be a JSON object")

            method = config.get("method")
            if method is not None and self.speculative_algorithm is None:
                self.speculative_algorithm = str(method).upper()

            draft_model = config.get("model")
            if draft_model is not None and self.speculative_draft_model_path is None:
                self.speculative_draft_model_path = str(draft_model)

            dspark_draft_model = self.speculative_draft_model_path
            dspark_uses_base_model = self.speculative_algorithm == "DSPARK" and (
                self.draft_model_path_use_base
                or dspark_draft_model is None
                or maybe_model_redirect(dspark_draft_model) == self.model
            )
            if dspark_uses_base_model:
                self.draft_model_path_use_base = True

            num_speculative_tokens = config.get("num_speculative_tokens")
            if num_speculative_tokens is not None:
                num_speculative_tokens = int(num_speculative_tokens)
                self._speculative_widths_explicit = True
                if self.speculative_algorithm == "DFLASH" or (
                    self.speculative_algorithm == "DSPARK"
                    and not dspark_uses_base_model
                ):
                    if self.speculative_num_draft_tokens is None:
                        self.speculative_num_draft_tokens = num_speculative_tokens
                    self.speculative_num_steps = max(num_speculative_tokens - 1, 0)
                elif self.speculative_algorithm == "DSPARK":
                    self.speculative_num_steps = num_speculative_tokens
                    if self.speculative_num_draft_tokens is None:
                        self.speculative_num_draft_tokens = num_speculative_tokens + 1
                else:
                    self.speculative_num_steps = num_speculative_tokens

        if self.speculative_eagle_topk != 1:
            if self.speculative_algorithm is None:
                raise ValueError(
                    f"--speculative-eagle-topk {self.speculative_eagle_topk} needs "
                    "--speculative-algorithm"
                )
            if self.speculative_num_draft_tokens is None:
                raise ValueError(
                    "--speculative-eagle-topk > 1 drafts a tree; set its node budget "
                    "with --speculative-num-draft-tokens"
                )
        if self.speculative_num_draft_tokens is None:
            self.speculative_num_draft_tokens = self.speculative_num_steps + 1

    def resolve_memory_and_scheduling(self):
        if current_platform().is_amd:
            gpu_mem = get_amdgpu_memory_capacity()
        elif current_platform().is_nvidia:
            gpu_mem = get_nvgpu_memory_capacity()
        else:
            # GPU memory is not known yet or no GPU is available.
            gpu_mem = None

        # Set GPU memory utilization.
        if self.gpu_memory_utilization is None:
            self.gpu_memory_utilization = 0.95

        # Set the chunked prefill token budget.
        if self.chunked_prefill_size is None:
            self.chunked_prefill_size = 8192

        # Set CUDA graph max capture size.
        if self.max_cudagraph_capture_size is None:
            # Based on detailed statistics, when serving TP1/TP2 models on lower-end GPUs with HBM<25G, you can either disable CUDA graph or set max_cudagraph_capture_size to a very small value to reduce graph memory overhead, with almost no impact on performance. TP4/TP8 serving still needs CUDA graph for high performance, and 80 is enough for lower-end GPUs.
            if gpu_mem is not None and gpu_mem < 25_000:
                if self.mapping.world_size < 4:
                    self.max_cudagraph_capture_size = 8
                else:
                    self.max_cudagraph_capture_size = 80
            elif self.speculative_algorithm:
                self.max_cudagraph_capture_size = 80
            else:
                self.max_cudagraph_capture_size = 160

        # Set max number of sequences.
        if self.max_num_seqs is None:
            if self.speculative_algorithm:
                self.max_num_seqs = 80
            else:
                self.max_num_seqs = 160

    def resolve_kernel_backends(self):
        _require_choice(
            "--dense-gemm-backend", self.dense_gemm_backend, ("auto", "trtllm_cutedsl")
        )
        # The numerics switches (docs/design/numerics.md) are closed sets.
        for flag, value, choices in (
            ("--sampling-stream", self.sampling_stream, SAMPLING_STREAMS),
            (
                "--yarn-ramp-mask-device",
                self.yarn_ramp_mask_device,
                YARN_RAMP_MASK_DEVICES,
            ),
            ("--mla-lora-scale", self.mla_lora_scale, MLA_LORA_SCALES),
            ("--layer-boundary-norm", self.layer_boundary_norm, LAYER_BOUNDARY_NORMS),
            ("--router-topk", self.router_topk, ROUTER_TOPKS),
            ("--logprob-order", self.logprob_order, LOGPROB_ORDERS),
            ("--moe-combine-order", self.moe_combine_order, MOE_COMBINE_ORDERS),
            ("--dsa-slot-order", self.dsa_slot_order, DSA_SLOT_ORDERS),
        ):
            _require_choice(flag, value, choices)
        if self.sampling_backend is None:
            # ``flashinfer`` is the only built-in backend that respects per-request
            # ``temperature`` / ``top_p`` / ``top_k``. ``greedy`` is argmax-only
            # (see ``GreedySamplingBackend.sample``: *"sampling_info is ignored
            # for single-step (always argmax)"*) — fast for hand-tuned greedy
            # decoding but silently wrong for any serving deployment where
            # requests carry sampling params, since the model collapses into
            # repetition-mode loops within a few hundred steps. Default to the
            # sampling-respecting backend on NVIDIA where flashinfer is
            # available, fall back to greedy elsewhere; users can still opt
            # into greedy explicitly via ``--sampling-backend greedy``.
            if current_platform().is_nvidia:
                self.sampling_backend = "flashinfer"
            else:
                self.sampling_backend = "greedy"

    def resolve_launcher_topology(self):
        """Fill in unset multi-node arguments from the launcher environment."""
        topology = detect_topology()
        if topology is None:
            self.nnodes = 1 if self.nnodes is None else self.nnodes
            self.node_rank = 0 if self.node_rank is None else self.node_rank
            return

        for flag, given, found in (
            ("--nnodes", self.nnodes, topology.nnodes),
            ("--node-rank", self.node_rank, topology.node_rank),
        ):
            if given is not None and given != found:
                raise ValueError(
                    f"{flag}={given} contradicts the {topology.source} environment, "
                    f"which reports {found}. Drop the flag to use the launcher's "
                    f"value, or correct the launch."
                )

        derived = []
        if self.nnodes is None:
            self.nnodes = topology.nnodes
            derived.append(f"--nnodes {self.nnodes}")
        if self.node_rank is None:
            self.node_rank = topology.node_rank
            derived.append(f"--node-rank {self.node_rank}")
        if self.dist_init_addr is None:
            check_dist_init_port(topology.dist_init_port)
            self.dist_init_addr = topology.dist_init_addr
            derived.append(f"--dist-init-addr {self.dist_init_addr}")
        if derived:
            logger.info(
                f"node {self.node_rank}/{self.nnodes} on {socket.gethostname()}: "
                f"derived from {topology.source}: {' '.join(derived)}"
            )

    def resolve_parallelism(self):
        world_size = self.world_size
        nprocs_per_node = self.nprocs_per_node
        nnodes = 1 if self.nnodes is None else self.nnodes
        pp_size = self.pipeline_parallel_size

        attn_tp_size = self.attn_tp_size
        attn_dp_size = self.data_parallel_size

        if world_size is None:
            world_size = pp_size
            if attn_tp_size is not None:
                world_size *= attn_tp_size
            if attn_dp_size is not None:
                world_size *= attn_dp_size
            logger.info(
                f"Inferred world_size ({world_size!s}) from attn_tp_size ("
                f"{attn_tp_size!s}) x attn_dp_size ({attn_dp_size!s}) x pp_size "
                f"({pp_size!s})",
            )
        else:
            logger.info(f"Specified world_size ({world_size!s})")

        # Pipeline stages are the outermost split: every per-layer-type
        # parallelism resolves inside one stage's world.
        if world_size % pp_size != 0:
            raise ValueError(
                f"world_size ({world_size}) must be divisible by "
                f"--pipeline-parallel-size ({pp_size})"
            )
        stage_world_size = world_size // pp_size

        attn_tp_size, attn_dp_size = _resolve_parallelism_sizes(
            stage_world_size, attn_tp_size, attn_dp_size
        )

        # Dense layers default to the attention replica's TP width
        # (attn_tp_size == world_size // attn_dp_size). Without DP attention
        # this is the full world, unchanged from before; with DP attention it
        # keeps each dense all-reduce inside one replica (matching attn)
        # instead of spanning the whole world, which would otherwise cross
        # nodes and force attn_tp != dense_tp. Pass --dense-tp-size to override.
        dense_tp_size = self.dense_tp_size
        if self.dense_tp_size is None:
            dense_tp_size = attn_tp_size
        dense_dp_size = None

        # --enable-expert-parallel auto-sets ep_size = the stage world (the
        # whole world when PP is off).
        if self.enable_expert_parallel and self.ep_size == 1:
            self.ep_size = stage_world_size
            logger.info(
                f"--enable-expert-parallel: auto-setting ep_size={stage_world_size!s}",
            )

        # MoE parallel sizes default to consuming the full stage world unless
        # the user overrides them explicitly.
        moe_ep_size = 1 if self.ep_size is None else self.ep_size
        moe_tp_size = (
            stage_world_size // moe_ep_size
            if self.moe_tp_size is None
            else self.moe_tp_size
        )
        moe_dp_size = None

        # The colocated multimodal encoder lives inside each attention TP
        # group. ``weights`` preserves the legacy weight-TP layout; ``data``
        # replicates the encoder weights and assigns whole multimodal items to
        # the ranks in that group.
        if self.mm_encoder_tp_mode not in ("weights", "data"):
            raise ValueError(
                "mm_encoder_tp_mode must be one of {'weights', 'data'}, got "
                f"{self.mm_encoder_tp_mode!r}"
            )
        if self.mm_encoder_tp_mode == "data":
            vision_tp_size = 1
            vision_dp_size = attn_tp_size
        else:
            vision_tp_size = attn_tp_size
            vision_dp_size = 1

        self.mapping = Mapping(
            world_size=world_size,
            attn_tp_size=attn_tp_size,
            attn_dp_size=attn_dp_size,
            attn_kvp_size=self.kv_parallel_size,
            attn_head_tp_size=self.attn_head_tp_size,
            lm_head_tp_size=self.lm_head_tp_size,
            attn_qcp_size=self.prefill_context_parallel_size,
            dense_tp_size=dense_tp_size,
            dense_dp_size=dense_dp_size,
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size,
            moe_dp_size=moe_dp_size,
            vision_tp_size=vision_tp_size,
            vision_dp_size=vision_dp_size,
            pp_size=pp_size,
            pp_layer_partition=self.pp_layer_partition,
            nprocs_per_node=nprocs_per_node,
            nnodes=nnodes,
            base_gpu_id=self.base_gpu_id,
            gpu_id_step=self.gpu_id_step,
        )

        # Impl constraints:
        validate_kvp_disaggregation_role(
            has_kvp=self.mapping.attn.has_kvp,
            disaggregation_mode=self.disaggregation_mode,
        )
        validate_qcp(
            qcp_size=self.mapping.attn.qcp_size,
            attn_tp_size=self.mapping.attn.tp_size,
            attn_dp_size=self.mapping.attn.dp_size,
            dense_tp_size=self.mapping.dense.tp_size,
            moe_tp_ep_size=self.mapping.moe.tp_ep_size,
            kvp_size=self.mapping.attn.kvp_size,
            disaggregation_mode=self.disaggregation_mode,
            disable_prefill_graph=bool(self.disable_prefill_graph),
            enable_mixed_batch=self.enable_mixed_batch,
            attention_backend=self.attention_backend,
            kv_cache_dtype=self.kv_cache_dtype,
            kv_cache_quant_method=self.kv_cache_quant_method,
        )
        if self.mapping.moe.has_tp and self.mapping.moe.has_ep:
            raise ValueError("MoE TP and EP cannot be both > 1")
        self._validate_decode_tp_layouts()

        if self.mm_encoder_tp_mode == "data":
            if self.disaggregation_mode not in ("null", "prefill"):
                raise ValueError(
                    "--mm-encoder-tp-mode data currently requires "
                    "--disaggregation-mode null (aggregate serving) or prefill"
                )
            if self.mapping.nnodes != 1:
                logger.warning("--mm-encoder-tp-mode data on nnodes>1 is experimental")

        logger.info(f"Parallelism configuration:\n{self.mapping!s}")

    def _validate_decode_tp_layouts(self):
        """Constraints of the TP layouts over ranks holding different rows:
        the decode-side layouts under attention DP, and head TP over the
        query shards of a prefill engine.

        The structural rules (head TP needs attention TP 1 or a full-TP query
        shard, is the query-shard group under QCP, tiles the stage world; LM
        head TP under DP needs attention TP 1) live in ``Mapping``; the
        prefill role's rules in ``validate_qcp``. This checks what only the
        server knows: the engine role and the batch-invariance selection. The
        weights' quantization is checked once the checkpoint's is resolved
        (:meth:`validate_tp_batch_invariant_weights`).
        """
        attn = self.mapping.attn
        if attn.head_tp_serves_decode_only:
            # Head TP over attention-DP ranks serves absorbed decode rows
            # only: an expanded prefill would need every head's K/V for the
            # cached prefix, which the head-sharded kv_b_proj cannot produce.
            # Over the query shards (attn.has_qcp) the extend rows run the
            # absorbed sparse prefill through the exchange, and validate_qcp
            # pinned the prefill role and the eager prefill already.
            if self.disaggregation_mode != "decode":
                raise ValueError(
                    "--attn-head-tp-size > 1 serves decode rows only and "
                    "requires --disaggregation-mode decode (or, on the prefill "
                    "role, --prefill-context-parallel-size equal to it)"
                )
            # The prefill CUDA graph records extend forwards, and startup
            # would capture (and tune on) extend-shaped dummies this layout
            # cannot run; the decode engine's warmup is decode-shaped instead
            # (ModelExecutor.autotune).
            if not self.disable_prefill_graph:
                logger.info(
                    "--attn-head-tp-size > 1 serves decode rows only: disabling "
                    "the prefill CUDA graph (--disable-prefill-graph)"
                )
                self.disable_prefill_graph = True
        if attn.has_head_tp and self.mapping.nprocs_per_node % attn.head_tp_size:
            logger.warning(
                f"attention head TP group of {attn.head_tp_size} ranks spans "
                f"nodes ({self.mapping.nprocs_per_node} ranks per node); the "
                "per-layer head exchanges will cross the network"
            )
        if self.tp_batch_invariant not in ("none", "attn", "attn+dense"):
            raise ValueError(
                "--tp-batch-invariant must be one of none, attn, attn+dense; got "
                f"{self.tp_batch_invariant!r}"
            )
        if self.tp_batch_invariant != "none" and not attn.has_head_tp:
            raise ValueError(
                f"--tp-batch-invariant {self.tp_batch_invariant} makes o_proj "
                "column-parallel over the attention head TP group and needs "
                "--attn-head-tp-size > 1"
            )
        if (
            self.tp_batch_invariant == "attn+dense"
            and self.mapping.dense.tp_size <= attn.tp_size
        ):
            # The batch-invariant dense tail replaces the token reduce-scatter
            # of a dense group wider than attention TP (CommManager refuses
            # it otherwise). Under query sharding the dense group is 1 or the
            # attention TP width (validate_qcp), so the selection has no
            # layout to apply to there.
            raise ValueError(
                "--tp-batch-invariant attn+dense makes the dense down_proj "
                "column-parallel over the dense TP group and needs a dense TP "
                f"group wider than attention TP (got --dense-tp-size "
                f"{self.mapping.dense.tp_size} with attention TP {attn.tp_size})"
                + (
                    "; under --prefill-context-parallel-size the dense group is 1 "
                    "or the attention TP width, so only --tp-batch-invariant attn "
                    "applies"
                    if attn.has_qcp
                    else ""
                )
            )
        if attn.has_dp and self.mapping.lm_head.has_tp and self.dp_sampling:
            raise ValueError(
                "--lm-head-tp-size > 1 under attention DP transposes the logits "
                "back to each rank's own rows and cannot combine with --dp-sampling"
            )

    def validate_tp_batch_invariant_weights(
        self,
        resolved_quantization: str | None,
        disable_quant_module: Sequence[str],
    ) -> None:
        """``--tp-batch-invariant`` needs an unquantized ``o_proj`` (and
        ``down_proj`` for ``attn+dense``): the column-parallel GEMM's full-K
        result is the point, and the quantized layouts do not offer it.

        ``resolved_quantization`` is the checkpoint's method after
        ``ModelConfig`` has merged ``--quantization`` with the checkpoint's own
        declaration; a quantized checkpoint passes only when its
        ``disable_quant_module`` keeps those modules in the loading dtype
        (``self_attn`` for o_proj; ``dense_mlp`` or ``mlps`` for down_proj).
        The layers check their own ``quant_config`` once built; this is the
        early, whole-deployment form of that check.
        """
        if self.tp_batch_invariant == "none" or resolved_quantization is None:
            return
        excluded = set(disable_quant_module)
        if "self_attn" not in excluded:
            raise ValueError(
                f"--tp-batch-invariant {self.tp_batch_invariant} needs an "
                f"unquantized o_proj, but the {resolved_quantization} checkpoint "
                "quantizes attention (disable_quant_module lacks 'self_attn')"
            )
        if self.tp_batch_invariant == "attn+dense" and not (
            {"dense_mlp", "mlps"} & excluded
        ):
            raise ValueError(
                "--tp-batch-invariant attn+dense needs an unquantized dense "
                f"down_proj, but the {resolved_quantization} checkpoint quantizes "
                "the dense MLPs (disable_quant_module lacks 'dense_mlp' / 'mlps')"
            )

    def resolve_cache(self):
        # Handle KVStore settings.
        self._handle_kvstore()
        self.validate_l3_prefetch_options()
        self.validate_cache_options()
        self.resolve_retraction_snapshot_options()

    @property
    def retracting_role(self) -> bool:
        """Whether this engine can retract at all: the fused and decode roles."""
        return self.disaggregation_mode not in ("prefill", "encode")

    @property
    def retraction_snapshot_pool_disabled(self) -> bool:
        """Whether no image can ever be taken, so a capacity block aborts.

        The one way to ask for that on a retracting role is
        ``--retraction-snapshot-ratio 0`` without a size override; the roles
        that never retract are resolved to the same knobs.
        """
        return (
            self.retraction_snapshot_host_gb == 0
            and self.retraction_snapshot_ratio == 0
        )

    def resolve_retraction_snapshot_options(self):
        """Resolve the retraction snapshot pool's knobs the way the KVStore's are.

        ``--retraction-snapshot-host-gb`` is the explicit size and wins when
        set (0 = not set); ``--retraction-snapshot-ratio`` sizes the pool as
        a fraction of this rank's Device KV capacity, an explicit 0 disabling
        it; neither given means the device builder derives the size
        (``cache/l2/sizing.py``). ``--retraction-snapshot-max-requests`` 0
        derives to the rank's running requests (``--max-num-seqs`` over the
        attention-DP ranks), the most that can ever need an image at once.
        The prefill and encode roles never retract: their knobs are resolved
        to no pool here with a log, as the KVStore's are off on the encode
        role.
        """
        host_gb = self.retraction_snapshot_host_gb
        ratio = self.retraction_snapshot_ratio
        max_requests = self.retraction_snapshot_max_requests
        if host_gb < 0 or max_requests < 0 or (ratio is not None and ratio < 0):
            raise ValueError(
                "--retraction-snapshot-host-gb, --retraction-snapshot-ratio and "
                "--retraction-snapshot-max-requests must be non-negative"
            )
        if not self.retracting_role:
            forced = self.debug_force_retraction_interval != 0
            if host_gb > 0 or ratio is not None or max_requests > 0 or forced:
                logger.info(
                    f"{self.disaggregation_mode!s} instance never retracts; ignoring "
                    "the retraction snapshot pool arguments"
                    + (" and --debug-force-retraction-interval" if forced else "")
                )
            self.retraction_snapshot_host_gb = 0.0
            self.retraction_snapshot_ratio = 0.0
            self.retraction_snapshot_max_requests = 0
            # The scheduler refuses the knob on a pool-less engine.
            self.debug_force_retraction_interval = 0
            return
        if self.retraction_snapshot_pool_disabled:
            if max_requests > 0:
                raise ValueError(
                    "--retraction-snapshot-ratio 0 disables the retraction snapshot "
                    "pool; --retraction-snapshot-max-requests has no rows to size"
                )
            if self.debug_force_retraction_interval != 0:
                raise ValueError(
                    "--debug-force-retraction-interval forces retractions, which the "
                    "pool --retraction-snapshot-ratio 0 disabled"
                )
            return
        if max_requests == 0:
            # The cap is max_num_seqs per attention-DP rank: __post_init__
            # resolves both before the cache.
            if self.mapping is None or self.max_num_seqs is None:
                raise ValueError(
                    "resolve_parallelism() and resolve_memory_and_scheduling() must "
                    "run before resolve_cache(): the derived "
                    "--retraction-snapshot-max-requests is max_num_seqs per "
                    "attention-DP rank"
                )
            dp_size = max(self.mapping.attn.dp_size, 1)
            self.retraction_snapshot_max_requests = self.max_num_seqs // dp_size
            if self.retraction_snapshot_max_requests == 0:
                raise ValueError(
                    "the retraction snapshot pool derives its request cap from "
                    f"--max-num-seqs ({self.max_num_seqs}) per attention-DP rank "
                    f"({dp_size}), which is 0 here; raise --max-num-seqs to at "
                    "least the attention-DP size or pass "
                    "--retraction-snapshot-max-requests"
                )

    def resolve_speculative_decoding(self):
        # Keep drafter backend consistent with the main model unless explicitly set.
        if (
            self.speculative_algorithm is not None
            and self.drafter_attention_backend is None
        ):
            self.drafter_attention_backend = self.attention_backend

        if (
            self.speculative_algorithm in ("MTP", "DSPARK")
            and self.speculative_draft_model_path is None
        ):
            self.draft_model_path_use_base = True

        if self.draft_model_path_use_base:
            self.speculative_draft_model_path = self.model

        if self.speculative_draft_model_path == self.model:
            self.draft_model_path_use_base = True

        if self.speculative_draft_model_quantization == "unquant":
            self.speculative_draft_model_quantization = None

        if self.speculative_algorithm in BLOCK_SPEC_ALGORITHMS:
            expected_steps = max(int(self.speculative_num_draft_tokens) - 1, 0)
            if self.speculative_num_steps == ServerArgs.speculative_num_steps:
                self.speculative_num_steps = expected_steps
            elif self.speculative_num_steps != expected_steps:
                raise ValueError(
                    f"{self.speculative_algorithm} requires "
                    "speculative_num_steps to equal "
                    "speculative_num_draft_tokens - 1. "
                    f"Got {self.speculative_num_steps=} and "
                    f"{self.speculative_num_draft_tokens=}. "
                    f"{BLOCK_SPEC_RULES}"
                )

        if self.eagle3_layers_to_capture is not None:
            self.eagle3_layers_to_capture = [
                int(x) for x in self.eagle3_layers_to_capture.split(",")
            ]

        if self.speculative_algorithm is not None and self.speculative_eagle_topk != 1:
            self._validate_tree_speculation()
        elif (
            self.speculative_algorithm in ("EAGLE3", "MTP")
            and self.speculative_num_draft_tokens != self.speculative_num_steps + 1
        ):
            raise ValueError(
                f"a draft chain verifies speculative_num_steps + 1 = "
                f"{self.speculative_num_steps + 1} tokens, got "
                f"speculative_num_draft_tokens={self.speculative_num_draft_tokens}"
            )

        if self.enable_speculative_sampling:
            self._validate_speculative_sampling()

    def _validate_tree_speculation(self) -> None:
        """Draft trees: EAGLE3/MTP with a node budget the draft can fill and a mask word can hold."""
        topk = self.speculative_eagle_topk
        steps = self.speculative_num_steps
        nodes = self.speculative_num_draft_tokens
        if self.speculative_algorithm not in ("EAGLE3", "MTP"):
            raise ValueError(
                f"speculative_eagle_topk={topk} (tree drafting) needs "
                f"--speculative-algorithm EAGLE3 or MTP, got {self.speculative_algorithm}"
            )
        if not 1 <= topk <= 16 or not 1 <= steps <= 10:
            raise ValueError(
                f"tree drafting needs 1..16 children per node and 1..10 steps: {topk=}, {steps=}"
            )
        if (steps - 1) * topk > nodes:
            raise ValueError(
                f"tree drafting writes (steps - 1) * topk = {(steps - 1) * topk} lane slots per "
                f"request into its {nodes}-slot draft window "
                "(--speculative-num-draft-tokens); lower topk or steps"
            )
        candidates = topk + (steps - 1) * topk * topk
        if not 2 <= nodes <= min(64, candidates + 1):
            raise ValueError(
                f"speculative_num_draft_tokens={nodes} must be in [2, {min(64, candidates + 1)}] "
                f"for topk={topk} over {steps} steps (root + drafted nodes, at most 64)"
            )
        if self.grammar_backend != "none" or self.enable_mixed_batch:
            raise ValueError(
                "tree drafting does not support structured output or mixed batches yet: "
                f"{self.grammar_backend=}, {self.enable_mixed_batch=}"
            )
        if self.disaggregation_mode != "null" or self.pipeline_parallel_size > 1:
            raise ValueError(
                "tree drafting does not carry the draft tree across prefill/decode "
                f"disaggregation or pipeline stages yet: {self.disaggregation_mode=}, "
                f"{self.pipeline_parallel_size=}"
            )
        if self.mapping.has_attn_dp:
            raise ValueError(
                "tree drafting does not support attention data parallelism yet: "
                f"attention DP size {self.mapping.attn.dp_size}"
            )

    def _validate_speculative_sampling(self):
        """Refuse ``--enable-speculative-sampling`` launches it cannot serve.

        The accept test needs a proposal drawn from the recorded draft
        distribution q, so the drafter must propose one token per step from
        its own logits (the Eagle family and the multi-depth MTP drafter;
        block drafters propose a whole block greedily), and the verifier must
        be a backend that runs the draft-prob chain kernel: ``greedy`` verifies
        by exact match and ``triton`` by a target-sampled exact match, so q
        never enters either. The prefill role of a disaggregated deployment
        never verifies a chain of its own and its drafted candidates ship to
        the decode node without q, so there the flag would only allocate the
        per-slot distribution buffer; it is refused. The sentinel threshold
        is validated here once for every layer below: at least 1.0 so no
        real probability reads as the sentinel, and at most 2**20 so the
        fp32 sentinel ``threshold + 1.0`` stays distinguishable from it.
        """
        if self.disaggregation_mode == "prefill":
            raise ValueError(
                "--enable-speculative-sampling has no effect on the prefill role "
                "of a disaggregated deployment: it never verifies a chain and its "
                "candidates reach the decode node without their draft "
                "distribution, so the flag would only cost the draft_probs "
                "buffer. Pass it to the decode role only."
            )
        if self.speculative_algorithm is None:
            raise ValueError(
                "--enable-speculative-sampling needs speculative decoding: pass "
                "--speculative-algorithm EAGLE3 or MTP"
            )
        if self.speculative_algorithm in BLOCK_SPEC_ALGORITHMS:
            raise ValueError(
                "--enable-speculative-sampling needs a chain drafter that samples "
                "one token per step from its own distribution; "
                f"{self.speculative_algorithm} proposes a whole block greedily"
            )
        if self.speculative_eagle_topk != 1:
            raise ValueError(
                "--enable-speculative-sampling supports only the topk=1 chain: "
                f"{self.speculative_eagle_topk=}"
            )
        if self.sampling_backend not in SPECULATIVE_SAMPLING_BACKENDS:
            if self.sampling_backend == "greedy":
                why = "verifies by exact match, so the draft distribution never enters"
            elif self.sampling_backend in ("triton", "triton_full"):
                why = (
                    "verifies by target-sampled exact match and has no draft-prob "
                    "rejection kernel"
                )
            else:
                why = "has no draft-prob rejection kernel"
            raise ValueError(
                "--enable-speculative-sampling needs a verifier with the draft-prob "
                f"chain kernel ({sorted(SPECULATIVE_SAMPLING_BACKENDS)}); "
                f"--sampling-backend {self.sampling_backend} {why}"
            )
        threshold = self.spec_reject_draft_prob_threshold
        if not (
            SPEC_REJECT_DRAFT_PROB_THRESHOLD_MIN
            <= threshold
            <= SPEC_REJECT_DRAFT_PROB_THRESHOLD_MAX
        ):
            raise ValueError(
                "--spec-reject-draft-prob-threshold must be within "
                f"[{SPEC_REJECT_DRAFT_PROB_THRESHOLD_MIN}, "
                f"{SPEC_REJECT_DRAFT_PROB_THRESHOLD_MAX}]: at least 1.0 so no real "
                "probability reads as the no-proposal sentinel, and small enough "
                f"that the fp32 sentinel threshold + 1.0 stays above it; got {threshold}"
            )

    def resolve_communication(self):
        # Auto-enable allreduce fusion on supported single-node TP configurations.
        platform = current_platform()
        if (
            not self.enable_allreduce_fusion
            and not self.emulate_rank_zero
            and (current_platform().is_hopper_plus or platform.is_amd)
            and self.mapping.nnodes == 1
            and self.mapping.has_attn_tp
            and not self.mapping.has_attn_dp
        ):
            self.enable_allreduce_fusion = True
            logger.info("Auto-enabled allreduce fusion")

        if self.mapping.attn.tp_size != self.mapping.dense.tp_size:
            self.comm_fusion_max_num_tokens = -1
            self.enable_allreduce_fusion = False
            logger.info(
                "allreduce is forbidden due to different attn_tp_size: "
                f"{self.mapping.attn.tp_size!s} and dense_tp_size: "
                f"{self.mapping.dense.tp_size!s}!",
            )

    def resolve_numerics(self):
        """Fold the ``--numerics`` envelope into the individual switches.

        ``rl-bitwise`` is the RL rollout contract: within one deployment the
        same request produces bitwise-identical tokens and logprobs across
        runs and regardless of batch composition, and the forward follows the
        training framework's operation order wherever the two engines differ
        (``_resolve_rl_bitwise``). The umbrella only ever tightens: it sets
        every switch it governs to its tight value and refuses explicit
        choices it cannot tighten (a named MoE or sampling backend without the
        guarantee). Each derived switch remains individually available for
        auto mode. Whether the served model is verified under the envelope is
        checked once its profile is known (``require_verified_numerics``).
        Runs after ``resolve_communication`` so it can veto the fused
        all-reduce that resolver auto-enables.
        """
        _require_choice("--numerics", self.numerics, NUMERICS_ENVELOPES)
        if self.numerics != "auto":
            self._resolve_rl_bitwise()
        # Individual switches that veto a fusion resolve_communication may
        # have auto-enabled, whatever the envelope. CommManager.should_fuse
        # re-derives the veto from the switches, so the fused kernels stay off
        # even where this flag is read before the fold.
        if self.layer_boundary_norm == "unfused":
            # The fused all-reduce+norm kernels add the residual inside the
            # fusion; the unfused boundary norm needs the bf16 sum first.
            self.enable_allreduce_fusion = False
        if self.moe_combine_order == "slot":
            # The MoE leaf returns complete rows; a fused all-reduce+norm at
            # the next layer boundary would sum them tp_size times.
            self.enable_allreduce_fusion = False
            # The slot-order fold runs over the EP group inside the MoE leaf:
            # a K-split (MoE TP) down projection would need a second,
            # rank-ordered fold after it, and DeepEP's all-to-all owns that
            # exchange itself (and hands the leaf a NCCL group, not the EP
            # device group the fold runs on).
            if self.mapping.moe.tp_size != 1:
                raise ValueError(
                    "--moe-combine-order slot needs MoE TP 1: a K-split down "
                    "projection would need a second, rank-ordered fold after "
                    f"the slot-order one (got --moe-tp-size {self.mapping.moe.tp_size})"
                )
            if self.all2all_backend == "deepep":
                raise ValueError(
                    "--moe-combine-order slot folds the routed outputs over the "
                    "EP group inside the MoE leaf; --all2all-backend deepep "
                    "performs that exchange itself and cannot be combined with it"
                )

    def _resolve_rl_bitwise(self):
        """The rl-bitwise block: every envelope beyond auto runs it."""
        # Collectives: one association order per reduction. NCCL's ring
        # chunks by message size, so a plain NCCL sum is run-stable but not
        # batch-size-invariant; batch_invariant_collectives routes the
        # all-reduce to the NVLS in-switch reduction with a fixed issuer where
        # multicast reaches (verified bitwise at startup) and every other
        # reduction to the rank-ordered fp32 fold (comm_backend/auto.py).
        # force_deterministic_rsag stays the user's "NCCL and the fold only"
        # knob; the envelope does not set it.
        self.batch_invariant_collectives = True
        self.enable_allreduce_fusion = False
        self.comm_fusion_max_num_tokens = -1
        # Kernels: heuristic tactics only (autotune picks shape-dependent
        # tactics), no TF32, and no programmatic dependent launches.
        self.disable_autotune = True
        self.disable_tf32 = True
        self.disable_pdl = True
        # MoE: the batch-invariant grouped leaves. An explicitly chosen
        # backend cannot honour the contract, so it is refused rather than
        # kept (a draft left unset inherits the target's).
        if self.moe_backend == "auto":
            self.moe_backend = "aok"
        elif self.moe_backend != "aok":
            raise ValueError(
                f"--numerics {self.numerics} needs the batch-invariant MoE "
                f"solution 'aok'; --moe-backend {self.moe_backend} makes no such "
                "claim"
            )
        if self.draft_moe_backend == "auto":
            self.draft_moe_backend = "aok"
        elif self.draft_moe_backend not in (None, "aok"):
            raise ValueError(
                f"--numerics {self.numerics} needs the batch-invariant MoE "
                f"solution 'aok'; --draft-moe-backend {self.draft_moe_backend} "
                "makes no such claim"
            )
        # Sampling: greedy rows must break exact logit ties canonically, and
        # sampled rows must draw from a stream the co-batch cannot move.
        if self.sampling_backend not in RL_BITWISE_SAMPLING_BACKENDS:
            raise ValueError(
                f"--numerics {self.numerics} needs a sampling backend with "
                "canonical greedy tie-breaking "
                f"({sorted(RL_BITWISE_SAMPLING_BACKENDS)}); --sampling-backend "
                f"{self.sampling_backend} resolves exact ties in reduction order"
            )
        self.sampling_stream = "per-request"
        # Sparse attention: the tuned top-k kernels' tie order moves with the
        # batch shape, so the cores reduce the selected slots sorted; the
        # batch-invariant cores the envelope pins declare the trait.
        self.dsa_slot_order = "sorted"
        # Trainer alignment: the training framework's operation order wherever
        # the two engines are known to differ. Each switch is documented in
        # ``docs/design/numerics.md`` under "alignment.trainer".
        # The trainer builds its RoPE inverse frequencies on the host; CPU and
        # CUDA division round each of them differently at ulp level.
        self.yarn_ramp_mask_device = "cpu"
        # The trainer multiplies the LoRA norm scales as separate bf16 ops.
        self.mla_lora_scale = "runtime"
        # The trainer materializes each layer's bf16 output before the next
        # layer's norm reads it.
        self.layer_boundary_norm = "unfused"
        # The trainer's router is softmax + topk(scores + bias) in torch.
        self.router_topk = "torch"
        # The trainer's logprobs come from its vocab-parallel cross-entropy.
        self.logprob_order = "megatron"
        # The trainer's grouped MLP applies the router weight inside the
        # activation and folds a token's slots in fp32 slot order.
        self.moe_combine_order = "slot"
        # Layouts stay explicit: the envelope does not fold them in. A
        # head-sharded o_proj that still sums its head partials across ranks
        # is batch-invariant under the envelope, but its bits are not the
        # full-K GEMM a replicated or column-parallel o_proj computes, and
        # they equal a TP-W engine's all-reduced o_proj only when both sides
        # sum in the same order: the exchanging forward's reduce-scatter
        # always takes the ordered fold, while an all-reduce (the TP-W
        # engine's, and the replicated-row decode steps of a query-sharding
        # engine) takes the in-switch reduction where multicast reaches, whose
        # order is a property of the GPU set (comm_backend/self_check.py).
        # --force-deterministic-rsag on the all-reducing side pins it to the
        # fold; docs/design/numerics.md, "Layout invariance of query context
        # parallelism".
        if self.mapping.attn.has_head_tp and self.tp_batch_invariant == "none":
            logger.warning(
                "--numerics rl-bitwise with --attn-head-tp-size > 1 but without "
                "--tp-batch-invariant attn: the o_proj head partials are summed "
                "across ranks (the ordered fold for the reduce-scatter, the "
                "in-switch all-reduce where it applies), which is batch-invariant "
                "but differs from the full-K o_proj of a TP1 or --tp-batch-invariant "
                "engine, and equals a TP-W engine's o_proj only when both sides "
                "sum in the same order (--force-deterministic-rsag on the "
                "all-reducing side)"
            )

    def resolve_disaggregation(self):
        # Pipeline parallelism is a prefill-node-only capability: the chunk
        # pipeline needs the P role's structural guarantees (no decode token
        # feedback, non-overlap loop).
        if self.pipeline_parallel_size > 1:
            # Debug escape hatch: run PP without PD to validate the stage
            # pipeline numerically (prefill + first token only — decode
            # autoregression is NOT correct with in-flight depth > 0).
            pp_debug = os.environ.get("TS_PP_DEBUG_ALLOW_NON_PREFILL") == "1"
            if self.disaggregation_mode != "prefill" and not pp_debug:
                raise ValueError(
                    "--pipeline-parallel-size > 1 requires "
                    "--disaggregation-mode prefill; PP is a prefill-node "
                    "chunk-pipeline feature"
                )
            if pp_debug and self.disaggregation_mode == "null":
                logger.warning(
                    "TS_PP_DEBUG_ALLOW_NON_PREFILL=1: running PP without PD "
                    "for pipeline validation; only prefill/first-token output "
                    "is meaningful"
                )
            # A pipeline stage threads its boundary state through an eager
            # stage forward (ModelExecutor._run_target_forward); no graph
            # subsystem captures that, so every stage runs eager.
            self.enforce_eager = True
            logger.info("CUDA graph is disabled under pipeline parallelism")
            if self.mapping.has_attn_dp:
                raise ValueError(
                    "--pipeline-parallel-size > 1 with attention DP is not "
                    "supported yet"
                )
            if self.speculative_algorithm is not None:
                # Pipeline speculation is a prefill-server feature: only the
                # last stage samples, so it alone runs the drafter and owns
                # the draft cache; the candidates ride the remote decode to
                # the peer. A decode role (or the PP debug mode) has no
                # token feedback on the chunk pipeline to draft against.
                if self.disaggregation_mode != "prefill":
                    raise ValueError(
                        "--pipeline-parallel-size > 1 supports speculation only "
                        "on a prefill server (--disaggregation-mode prefill)"
                    )
                # DSPARK produces its draft context across stages (each stage
                # projects the target taps it owns); an MTP (NextN) draft
                # needs only the last stage's captured hidden states.
                if self.speculative_algorithm not in PIPELINE_SPEC_ALGORITHMS:
                    raise ValueError(
                        f"--speculative-algorithm {self.speculative_algorithm} "
                        "is not supported with --pipeline-parallel-size > 1; "
                        f"pipeline speculation supports {PIPELINE_SPEC_ALGORITHMS}"
                    )
                # A draft layout limit rather than a PP limit: the DSPARK
                # draft reduces its attention-TP embedding partials over the
                # dense TP group. MTP drafts embed with an ordinary reduced
                # vocab-parallel lookup, so only DSPARK carries the rule.
                # Both TP groups are stride-1 over the stage, so equal widths
                # mean equal groups (the mapping has no rank yet here).
                if (
                    self.speculative_algorithm == "DSPARK"
                    and self.mapping.dense.tp_size != self.mapping.attn.tp_size
                ):
                    raise ValueError(
                        "Pipeline DSPARK requires matching dense/attention TP groups"
                    )
            if (
                self.pp_layer_partition is not None
                and len(self.pp_layer_partition) != self.pipeline_parallel_size
            ):
                raise ValueError(
                    f"--pp-layer-partition {self.pp_layer_partition} has "
                    f"{len(self.pp_layer_partition)} entries but "
                    f"--pipeline-parallel-size is {self.pipeline_parallel_size}"
                )
        elif self.pp_layer_partition is not None:
            raise ValueError(
                "--pp-layer-partition requires --pipeline-parallel-size > 1"
            )
        # PD disaggregation. The prefill role keeps the ordinary graph flags:
        # it never runs a decode step, so the decode graph has nothing to
        # capture (ModelExecutorConfig.prefill_only), while its extend
        # forwards replay the prefill graph like any server's.
        if self.disaggregation_mode == "decode":
            # Prefix caching stays configurable for decode servers.
            logger.info(
                f"enable_prefix_caching={self.enable_prefix_caching!r} for decode "
                "server",
            )
        elif self.disaggregation_mode == "encode":
            # Encode server: vision tower only, no LM / KV pool / prefix cache.
            # enforce_eager left as-is (the vision tower keeps its own CUDA graph).
            if self.mapping.has_attn_dp:
                raise ValueError(
                    "disaggregation_mode=encode currently supports "
                    "data_parallel_size == 1 inside one encode server; run "
                    "multiple independent encode servers for horizontal scale."
                )
            self.enable_prefix_caching = False

        if (
            self.disaggregation_mode == "prefill"
            and self.load_balance_method != "round_robin"
        ):
            if self.mapping.has_attn_dp:
                raise ValueError(
                    "Not supported when "
                    f"{self.disaggregation_mode=} {self.load_balance_method=} "
                    f"{self.mapping.attn.dp_size=}"
                )

    def _handle_kvstore(self):
        if self.disaggregation_mode == "encode":
            self.enable_kvstore = False
            logger.info(
                f"{self.disaggregation_mode!s} instance has set enable_kvstore to "
                "False!",
            )
        elif not self.disable_kvstore:
            self.enable_kvstore = True

        if self.kvstore_storage_backend is not None and not self.enable_kvstore:
            raise ValueError(
                "L3 storage (--kvstore-storage-backend) requires Host L2; "
                "unset --disable-kvstore"
            )

    def validate_l3_prefetch_options(self):
        """The L3 prefetch knobs: all four with an L3 store, none without.

        An L3 hit is fetched into Host pages before the request is admitted,
        on the Host cache executor's prefetch lane; these size that fetch.
        None of them has a silent default: the threshold trades a short hit's
        recompute against a round of admission latency, the deadline bounds
        how long a request may wait on the store, and the batch is the
        store's request granularity.
        """
        knobs = {
            "--kvstore-prefetch-min-pages": self.kvstore_prefetch_min_pages,
            "--kvstore-prefetch-timeout-base-s": self.kvstore_prefetch_timeout_base_s,
            "--kvstore-prefetch-timeout-per-page-s": (
                self.kvstore_prefetch_timeout_per_page_s
            ),
            "--kvstore-prefetch-batch-pages": self.kvstore_prefetch_batch_pages,
        }
        given = [name for name, value in knobs.items() if value is not None]
        if self.kvstore_storage_backend is None:
            if given:
                raise ValueError(
                    f"{', '.join(given)} belong to the L3 prefetch and need "
                    "--kvstore-storage-backend"
                )
            return
        missing = [name for name, value in knobs.items() if value is None]
        if missing:
            raise ValueError(
                f"--kvstore-storage-backend needs the L3 prefetch knobs: {', '.join(missing)}"
            )
        if self.kvstore_prefetch_min_pages < 1:
            raise ValueError("--kvstore-prefetch-min-pages must be at least 1 page")
        if self.kvstore_prefetch_timeout_base_s <= 0:
            raise ValueError("--kvstore-prefetch-timeout-base-s must be positive")
        if self.kvstore_prefetch_timeout_per_page_s < 0:
            raise ValueError("--kvstore-prefetch-timeout-per-page-s must be >= 0")
        if self.kvstore_prefetch_batch_pages < 1:
            raise ValueError("--kvstore-prefetch-batch-pages must be at least 1 page")

    def validate_cache_options(self):
        # Runs after _handle_kvstore() has applied the KVStore default, so the
        # check sees the effective setting rather than the pre-resolution flag.
        # Both Host tiers -- the L2 KVStore and the retraction snapshot pool --
        # are legal under KV-page sharding (--kv-parallel-size):
        # the scheduler allocates every Host block in its Device block's
        # residue class and the executor translates ownership on both ends of
        # every row (cache/transfer/ownership.py), so each rank copies the
        # blocks it owns. L3 is not: its keys name content and position, but
        # which rank owns a block is decided at allocation, so a rank cannot
        # answer an existence probe for blocks it did not own when the object
        # was written (docs/design/cache-concepts.md, the retraction image).
        if self.kv_parallel_size > 1 and (self.kvstore_storage_backend is not None):
            raise ValueError(
                "--kvstore-storage-backend (L3) is not supported under "
                "--kv-parallel-size > 1: a page-cyclic sharded group "
                "has no owner-stable L3 key. The Host KVStore itself and the "
                "retraction snapshot pool are supported."
            )
        # Same-checkpoint DSpark's KVStore support depends on where the draft
        # keeps its context; the engine decides once the draft config resolves
        # (resolve_dspark_prefix_replay_tokens).
        if (
            self.enable_kvstore
            and not self.enable_prefix_caching
            and self.disaggregation_mode != "decode"
        ):
            raise ValueError(
                "KVStore and disabled prefix caching are mutually exclusive "
                "and cannot be used at the same time. Please use only one of them."
            )

    def validate_model_update_options(self):
        """Require the Model Updater SDK flags together, or none of them.

        The config alone cannot select the SDK module, the engine type, or
        the draft policy, so those three are mandatory with it and
        meaningless without it.
        """
        companions = {
            "--model-update-sdk-module": self.model_update_sdk_module,
            "--model-update-engine-type": self.model_update_engine_type,
            "--model-update-draft-weights": self.model_update_draft_weights,
        }
        if self.model_update_config is None:
            given = [flag for flag, value in companions.items() if value is not None]
            if given:
                raise ValueError(f"{', '.join(given)} require --model-update-config")
            return
        missing = [flag for flag, value in companions.items() if value is None]
        if missing:
            raise ValueError(f"--model-update-config requires {', '.join(missing)}")
        try:
            parsed = json.loads(self.model_update_config)
        except json.JSONDecodeError as exc:
            raise ValueError("--model-update-config must be valid JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError("--model-update-config must be a JSON object")
        if self.model_update_draft_weights not in ("retain", "refresh"):
            raise ValueError(
                "--model-update-draft-weights must be 'retain' or 'refresh'"
            )

    def validate_petit_moe_options(self):
        """Validate shared backend, model, and scheduling options for Petit.

        MoELayer owns hardware, MoE topology, and expert compatibility checks.
        """
        active_moe_backends = [("target", self.moe_backend)]
        if self.speculative_algorithm is not None:
            active_moe_backends.append(
                ("draft", self.draft_moe_backend or self.moe_backend)
            )
        gluon_petit_roles = [
            role for role, backend in active_moe_backends if backend == "gluon_petit"
        ]
        if self.all2all_backend == "gluon_petit":
            mismatched_roles = [
                f"{role}={backend}"
                for role, backend in active_moe_backends
                if backend != "gluon_petit"
            ]
            if mismatched_roles:
                raise ValueError(
                    "Gluon Petit MegaMoE requires every active MoE backend to "
                    "match --all2all-backend gluon_petit; incompatible "
                    + ", ".join(mismatched_roles)
                )
        elif gluon_petit_roles:
            raise ValueError(
                "Gluon Petit MegaMoE requires --all2all-backend gluon_petit "
                f"for the active {', '.join(gluon_petit_roles)} MoE backend"
            )

        if gluon_petit_roles:
            if self.dtype != "bfloat16":
                raise ValueError(
                    "Gluon Petit MegaMoE requires --dtype bfloat16; "
                    f"configured dtype={self.dtype}"
                )
            if self.mapping.attn.tp_size != 1 or self.mapping.dense.tp_size != 1:
                raise ValueError(
                    "Gluon Petit MegaMoE requires attention TP1 and dense TP1"
                )
            decode_tokens_per_request = (
                self.speculative_num_draft_tokens
                if self.speculative_algorithm is not None
                else 1
            )
            decode_tokens_per_rank = (
                self.max_num_seqs // self.mapping.attn.dp_size
            ) * decode_tokens_per_request
            if decode_tokens_per_rank > 1024:
                raise ValueError(
                    "Gluon Petit MegaMoE supports at most 1024 decode tokens "
                    "per rank; reduce --max-num-seqs or the speculative draft "
                    f"token count (configured {decode_tokens_per_rank} tokens "
                    "per rank)"
                )
            if (
                self.chunked_prefill_size <= 0
                or self.chunked_prefill_size > 1024
                or self.max_prefill_tokens > 1024
            ):
                raise ValueError(
                    "Gluon Petit MegaMoE supports at most 1024 prefill tokens "
                    "per rank; set --chunked-prefill-size to a positive value "
                    "no greater than 1024 and --max-prefill-tokens no greater "
                    "than 1024"
                )

    def validate_rank_emulation(self):
        """Reject layouts ``--emulate-rank-zero`` cannot stand in for.

        The emulated rank replaces collectives through the comm backend and
        one-member process groups. Paths that exchange per-rank state outside
        them, or that keep their own peer communicators, need real peers.
        """
        if not self.emulate_rank_zero:
            return
        if not current_platform().is_amd:
            raise ValueError("--emulate-rank-zero is supported on AMD GPUs only")
        if self.mapping.world_size == 1:
            raise ValueError(
                "--emulate-rank-zero needs a parallel layout of more than one rank"
            )
        unsupported = []
        if self.mapping.nnodes != 1:
            unsupported.append(f"--nnodes {self.mapping.nnodes}")
        if self.mapping.has_pp:
            unsupported.append("pipeline parallelism")
        if self.mapping.attn.has_qcp:
            unsupported.append("query context parallelism")
        if self.mapping.has_attn_dp:
            unsupported.append("attention data parallelism")
        if self.mapping.moe.tp_ep_size != self.mapping.attn.tp_size:
            unsupported.append("an MoE TP x EP size other than the attention TP size")
        if self.mm_encoder_tp_mode == "data":
            unsupported.append("--mm-encoder-tp-mode data")
        if self.disaggregation_mode != "null":
            unsupported.append(f"--disaggregation-mode {self.disaggregation_mode}")
        if self.all2all_backend != "none":
            unsupported.append(f"--all2all-backend {self.all2all_backend}")
        if self.enable_allreduce_fusion:
            unsupported.append("--enable-allreduce-fusion")
        if self.enable_eplb:
            unsupported.append("--enable-eplb")
        if unsupported:
            raise ValueError(
                f"--emulate-rank-zero does not support {', '.join(unsupported)}"
            )

    def validate_expert_placement_options(self):
        """Check the expert placement flags (redundant experts, recorded load).

        A placement needs an explicit dispatch algorithm, and under rl-bitwise
        a deterministic one: the replicated-input EP path relies on every rank
        choosing the same replica for a route. Online rebalancing
        (``--enable-eplb``) spells out every choice it depends on -- the load
        counters, a static dispatch algorithm, the snapshot interval and the
        layers switched per round -- rather than auto-setting any of them.
        """
        if self.enable_eplb:
            if self.expert_distribution_recorder_mode != "stat":
                raise ValueError(
                    "--enable-eplb rebalances from the routing load counters; "
                    "pass --expert-distribution-recorder-mode stat explicitly."
                )
            if self.ep_dispatch_algorithm not in STATIC_EP_DISPATCH_ALGORITHMS:
                raise ValueError(
                    "--enable-eplb needs a static replica choice "
                    "(--ep-dispatch-algorithm static or static_with_zero_expert); "
                    f"got {self.ep_dispatch_algorithm!r}."
                )
            if (
                self.eplb_rebalance_num_iterations is None
                or self.eplb_rebalance_num_iterations <= 0
            ):
                raise ValueError(
                    "--enable-eplb requires --eplb-rebalance-num-iterations N > 0: "
                    "the forwards between two load snapshots."
                )
            if (
                self.eplb_rebalance_layers_per_chunk is None
                or self.eplb_rebalance_layers_per_chunk < 1
            ):
                raise ValueError(
                    "--enable-eplb requires --eplb-rebalance-layers-per-chunk L >= 1: "
                    "the MoE layers whose experts move in one scheduling round "
                    "(at most the model's MoE layer count)."
                )
            if self.mapping.moe.ep_size <= 1:
                raise ValueError(
                    "--enable-eplb balances expert load across expert-parallel "
                    f"ranks, but the MoE layers run with ep_size="
                    f"{self.mapping.moe.ep_size}; enable expert parallelism."
                )
            if self.numerics != "auto" and self.moe_combine_order != "slot":
                # Under a bitwise envelope the rank-order MoE combine makes the
                # output depend on the placement, which a rebalance changes;
                # the slot-order combine is placement-independent. The
                # envelope folds it in resolve_numerics, so this only guards
                # that fold.
                raise ValueError(
                    f"--enable-eplb under --numerics {self.numerics} requires "
                    "--moe-combine-order slot (a placement-independent MoE "
                    f"combine); got {self.moe_combine_order!r}."
                )
        elif (
            self.eplb_rebalance_num_iterations is not None
            or self.eplb_rebalance_layers_per_chunk is not None
        ):
            raise ValueError(
                "--eplb-rebalance-num-iterations and "
                "--eplb-rebalance-layers-per-chunk have no effect without "
                "--enable-eplb."
            )
        if self.expert_distribution_recorder_mode not in (None, "stat"):
            raise ValueError(
                "--expert-distribution-recorder-mode supports only 'stat' (per "
                "physical expert route counters dumped by the EXPERT_LOAD profile "
                f"activity), got {self.expert_distribution_recorder_mode!r}."
            )
        if self.ep_num_redundant_experts < 0:
            raise ValueError("--ep-num-redundant-experts must be non-negative")
        if self.ep_num_redundant_experts > 0 and self.mapping.moe.ep_size <= 1:
            raise ValueError(
                f"--ep-num-redundant-experts {self.ep_num_redundant_experts} "
                "replicates experts across expert-parallel ranks, but the MoE "
                f"layers run with ep_size={self.mapping.moe.ep_size}; enable "
                "expert parallelism (--ep-size > 1) or drop the redundant experts."
            )
        if expert_placement_requested(self):
            if self.ep_dispatch_algorithm is None:
                raise ValueError(
                    "--ep-dispatch-algorithm is required with "
                    "--ep-num-redundant-experts, a non-trivial "
                    "--init-expert-location or --expert-distribution-recorder-mode: "
                    "static_with_zero_expert for models with zero experts, "
                    "static otherwise."
                )
            if (
                self.numerics == "rl-bitwise"
                and self.ep_dispatch_algorithm not in STATIC_EP_DISPATCH_ALGORITHMS
            ):
                raise ValueError(
                    "--numerics rl-bitwise needs a deterministic expert placement; "
                    f"--ep-dispatch-algorithm {self.ep_dispatch_algorithm} picks "
                    "replicas at random. Use static or static_with_zero_expert."
                )
        elif self.ep_dispatch_algorithm is not None:
            raise ValueError(
                f"--ep-dispatch-algorithm {self.ep_dispatch_algorithm} has no effect "
                "without an expert placement (--ep-num-redundant-experts, "
                "--init-expert-location or --expert-distribution-recorder-mode)."
            )

    def validate(self):
        if self.low_latency_max_num_tokens_per_gpu <= 0:
            raise ValueError("--low-latency-max-num-tokens-per-gpu must be positive")
        if self.input_logprob_chunk_tokens <= 0:
            raise ValueError("--input-logprob-chunk-tokens must be positive")
        if self.device == "npu":
            if not self.disable_prefill_graph:
                raise ValueError("NPU execution requires --disable-prefill-graph")
            if not self.disable_pdl:
                raise ValueError("NPU execution requires --disable-pdl")

        self.validate_petit_moe_options()
        self.validate_rank_emulation()

        if (
            self.max_num_seqs is not None
            and self.max_num_seqs < self.mapping.attn.dp_size
        ):
            raise ValueError(
                f"max_num_seqs must be >= attn_dp_size: {self.max_num_seqs=} < {self.mapping.attn.dp_size=}"
            )

        self.validate_model_update_options()

        if self.mapping.has_attn_dp:
            if self.chunked_prefill_size > self.max_prefill_tokens:
                raise ValueError(
                    f"chunked_prefill_size must be <= max_prefill_tokens: {self.chunked_prefill_size=} > {self.max_prefill_tokens=}"
                )

        if self.deepseek_v4_prefill_chunk_size <= 0:
            raise ValueError("deepseek_v4_prefill_chunk_size must be positive")

        self.validate_expert_placement_options()

        from tokenspeed.runtime.utils.env import envs

        envs.TOKENSPEED_MAMBA_SSM_DTYPE.set(self.mamba_ssm_dtype)
        if not self.disable_pdl:
            os.environ.setdefault("TORCHINDUCTOR_ENABLE_PDL", "1")
            # Enable PDL for fused attention kernels.
            os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
        os.environ.setdefault("TLLM_LOG_LEVEL", "INFO")

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.allow_abbrev = False

        # Model and port args
        parser.add_argument(
            "model_path",
            nargs="?",
            metavar="model",
            default=None,
            help="The model name or path (positional argument). "
            "Equivalent to --model.",
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
            default=ServerArgs.tokenizer,
            help="The path of the tokenizer.",
        )
        parser.add_argument(
            "--host", type=str, default=ServerArgs.host, help="The host of the server."
        )
        parser.add_argument(
            "--port", type=int, default=ServerArgs.port, help="The port of the server."
        )
        parser.add_argument(
            "--tokenizer-mode",
            type=str,
            default=ServerArgs.tokenizer_mode,
            choices=["auto", "slow", "deepseek_v4"],
            help="Tokenizer mode. 'auto' will use the fast "
            "tokenizer and model-specific tokenizer hooks if available, "
            "'slow' will always use the slow tokenizer.",
        )
        parser.add_argument(
            "--skip-tokenizer-init",
            action=argparse.BooleanOptionalAction,
            default=ServerArgs.skip_tokenizer_init,
            help="If set, skip init tokenizer and pass input_ids in generate request",
        )
        parser.add_argument(
            "--language-model-only",
            action="store_true",
            default=ServerArgs.language_model_only,
            help="Skip vision/audio encoders on a multimodal checkpoint and "
            "run text-only. Multimodal requests are rejected.",
        )
        parser.add_argument(
            "--zmq-msgpack",
            action=argparse.BooleanOptionalAction,
            default=ServerArgs.zmq_msgpack,
            help="Drive the scheduler directly from SMG over the msgpack ZMQ "
            "wire instead of the Python tokenizer_manager (pickle IPC). SMG "
            "binds the handshake/input/output sockets; the scheduler connects "
            "in. Pair with --skip-tokenizer-init.",
        )
        parser.add_argument(
            "--data-parallel-address",
            type=_nonempty_str,
            default=ServerArgs.data_parallel_address,
            help="Host of the frontend-bound handshake ROUTER the scheduler "
            "connects to under --zmq-msgpack (default "
            f"{ServerArgs.data_parallel_address}).",
        )
        parser.add_argument(
            "--data-parallel-rpc-port",
            type=_uint16,
            default=ServerArgs.data_parallel_rpc_port,
            help="Port of the frontend-bound handshake ROUTER (default "
            f"{ServerArgs.data_parallel_rpc_port}, outside the smg frontend's "
            "derived-port band 20000..=29999). `smg serve --backend "
            "tokenspeed --connection-mode zmq` passes an explicit port "
            "derived from the worker's ipc:// URL instead.",
        )
        parser.add_argument(
            "--zmq-engine-index",
            type=_uint16,
            default=ServerArgs.zmq_engine_index,
            help="This engine's index under --zmq-msgpack; used as the two-byte "
            "little-endian ZMQ routing identity SMG addresses it by "
            "(0..65535).",
        )
        parser.add_argument("--ext-yaml", type=str, default=None)
        parser.add_argument(
            "--load-format",
            type=str,
            default=ServerArgs.load_format,
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
            default=ServerArgs.dtype,
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
            default=ServerArgs.kv_cache_dtype,
            choices=["auto", "bfloat16", "fp8", "fp8_e4m3", "mxfp8"],
            help='Data type for kv cache storage. "auto" and "bfloat16" store BF16 '
            'rows (fp16 activations convert on write). "fp8" is an alias for '
            '"fp8_e4m3" (unit scale). "mxfp8" stores '
            "block-scaled fp8-e4m3 (one UE8M0 scale per 32 head_dim elements) and "
            "requires --block-size 128 with an MHA attention backend.",
        )
        parser.add_argument(
            "--kv-cache-quant-method",
            type=str,
            default=ServerArgs.kv_cache_quant_method,
            choices=["none", "per_token_head"],
            help="kv cache quant method",
        )
        parser.add_argument(
            "--quantization",
            type=str,
            default=ServerArgs.quantization,
            choices=[
                "fp8",
                "mxfp4",
                "nvfp4",
                "w8a8_fp8",
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
            default=ServerArgs.max_model_len,
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
            default=ServerArgs.served_model_name,
            help="Override the model name returned by the v1/models endpoint in OpenAI API server.",
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
            default=ServerArgs.gpu_memory_utilization,
            help="The fraction of GPU memory to use for model weights and KV cache. Use a smaller value if you see out-of-memory errors.",
        )
        parser.add_argument(
            "--max-num-seqs",
            metavar="MAX_NUM_SEQS",
            type=int,
            default=ServerArgs.max_num_seqs,
            help="Maximum number of sequences to process concurrently.",
        )
        parser.add_argument(
            "--max-total-tokens",
            type=int,
            default=ServerArgs.max_total_tokens,
            help="The maximum number of tokens in the memory pool. If not specified, it will be automatically calculated based on the memory usage fraction. "
            "This overrides the automatically calculated token pool size.",
        )
        parser.add_argument(
            "--chunked-prefill-size",
            metavar="CHUNKED_PREFILL_SIZE",
            type=int,
            default=ServerArgs.chunked_prefill_size,
            help="Maximum number of tokens the scheduler may issue in a single iteration. Setting this to -1 disables chunked prefill.",
        )
        parser.add_argument(
            "--enable-mixed-batch",
            action="store_true",
            dest="enable_mixed_batch",
            default=ServerArgs.enable_mixed_batch,
            help="Allow the scheduler to issue prefill and decode requests in the same iteration.",
        )
        parser.add_argument(
            "--prefix-granularity",
            "--block-size",  # deprecated alias
            dest="prefix_granularity",
            metavar="PREFIX_GRANULARITY",
            type=int,
            default=ServerArgs.prefix_granularity,
            help="Scheduler prefix granularity in tokens: the identity "
            "boundary of cache reuse. (--block-size is a deprecated alias.)",
        )

        # KVStore
        parser.add_argument(
            "--disable-kvstore",
            action="store_true",
            help="Disable KVStore",
        )
        parser.add_argument(
            "--kvstore-ratio",
            type=float,
            default=ServerArgs.kvstore_ratio,
            help="The ratio of the size of the KVStore host memory pool to the size of the device pool.",
        )
        parser.add_argument(
            "--kvstore-size",
            type=int,
            default=ServerArgs.kvstore_size,
            help="The size of the KVStore host memory pool in gigabytes, which will override kvstore_ratio if set.",
        )
        parser.add_argument(
            "--kvstore-io-backend",
            type=str,
            choices=["direct", "kernel"],
            default=ServerArgs.kvstore_io_backend,
            help="The IO backend for KVStore transfer between CPU and GPU.",
        )
        parser.add_argument(
            "--kvstore-storage-backend",
            type=str,
            choices=["mooncake", "memory"],
            default=ServerArgs.kvstore_storage_backend,
            help="L3 store under compact Host (flat) KV. "
            "'mooncake' is Mooncake Store (SGLang/vLLM HiCache equivalent). "
            "'memory' is an in-process dict for tests. Requires Host L2 "
            "(do not pass --disable-kvstore).",
        )
        parser.add_argument(
            "--kvstore-storage-backend-extra-config",
            type=str,
            default=ServerArgs.kvstore_storage_backend_extra_config,
            help="JSON object of extra L3 backend settings. For mooncake: "
            "master_server_address, local_hostname, metadata_server, "
            "global_segment_size, protocol, device_name, tenant_id.",
        )
        parser.add_argument(
            "--kvstore-prefetch-min-pages",
            type=int,
            default=ServerArgs.kvstore_prefetch_min_pages,
            help="L3 prefetch threshold, in prefix pages: a waiting request whose "
            "L3-only prefix is at least this long has it fetched into Host pages "
            "before admission (the request waits, holding no Device page); a "
            "shorter one is simply computed. Required with "
            "--kvstore-storage-backend, refused without it.",
        )
        parser.add_argument(
            "--kvstore-prefetch-timeout-base-s",
            type=float,
            default=ServerArgs.kvstore_prefetch_timeout_base_s,
            help="L3 prefetch deadline, base term in seconds: a prefetch stops "
            "starting batches base + per-page * pages seconds after it began on "
            "the prefetch lane and lands the prefix it has. The deadline is "
            "checked between batches only -- a batch_get_into already issued "
            "runs to completion (the store API has no timeout), so one batch may "
            "overrun it and a hung store holds the lane and every later "
            "prefetch. Positive; required with --kvstore-storage-backend.",
        )
        parser.add_argument(
            "--kvstore-prefetch-timeout-per-page-s",
            type=float,
            default=ServerArgs.kvstore_prefetch_timeout_per_page_s,
            help="L3 prefetch deadline, per-page term in seconds (0 for a fixed "
            "deadline). Required with --kvstore-storage-backend.",
        )
        parser.add_argument(
            "--kvstore-prefetch-batch-pages",
            type=int,
            default=ServerArgs.kvstore_prefetch_batch_pages,
            help="Prefix pages per L3 get of a prefetch; the fetch stops at the "
            "first batch with a missing page. Required with "
            "--kvstore-storage-backend.",
        )
        # Retraction snapshot pool
        parser.add_argument(
            "--retraction-snapshot-host-gb",
            type=float,
            default=ServerArgs.retraction_snapshot_host_gb,
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
            default=ServerArgs.retraction_snapshot_ratio,
            help="Size of the retraction snapshot pool as a fraction of this rank's "
            "Device KV capacity (the base --kvstore-ratio uses). An explicit 0 "
            "disables the pool: no image can be taken and every capacity block "
            "aborts its victim -- the only way to get abort-only behaviour. "
            "Unset (the default) derives the size: 0.1 of the Device KV without "
            "the KVStore (whole images; an on-by-default pool must stay cheap to "
            "pin -- pass 1 to make every resident suspendable), else the tails "
            "of --retraction-snapshot-max-requests images (one page per group "
            "that publishes to Host L2, a request's worst case for a group that "
            "never does), the slot-state arena coming on top.",
        )
        parser.add_argument(
            "--retraction-snapshot-max-requests",
            type=int,
            default=ServerArgs.retraction_snapshot_max_requests,
            help="Slot-state image rows of the retraction snapshot pool, i.e. the "
            "most requests suspended at once. 0 (the default) derives to this "
            "rank's running requests, --max-num-seqs / attention-DP size, which "
            "covers every resident; a victim that finds no free row is aborted "
            "rather than imaged. Refused with --retraction-snapshot-ratio 0.",
        )
        parser.add_argument(
            "--debug-force-retraction-interval",
            type=int,
            default=ServerArgs.debug_force_retraction_interval,
            help="TEST ONLY. Every |N| scheduler plans retract the oldest quiescent "
            "decoding request (N > 0) or the one prefilling request between its "
            "chunks (N < 0) without capacity pressure, so the suspend/restore "
            "path runs on every request of a test run. Refused with "
            "--retraction-snapshot-ratio 0. 0 (the default) is off; never set it "
            "in serving.",
        )
        # Mamba Cache
        parser.add_argument(
            "--mamba-ssm-dtype",
            type=str,
            default=ServerArgs.mamba_ssm_dtype,
            choices=["float32", "bfloat16"],
            help="It is used to tune mamba ssm dtype",
        )
        parser.add_argument(
            "--max-prefill-tokens",
            metavar="MAX_PREFILL_TOKENS",
            type=int,
            default=ServerArgs.max_prefill_tokens,
            help=(
                "Maximum prefill-token budget used when chunked prefill is "
                "disabled. Per-iteration scheduling is controlled by "
                "--chunked-prefill-size."
            ),
        )
        # Other runtime options
        parser.add_argument(
            "--stream-interval",
            type=int,
            default=ServerArgs.stream_interval,
            help="The interval (or buffer size) for streaming in terms of the token length. A smaller value makes streaming smoother, while a larger value makes the throughput higher",
        )
        parser.add_argument(
            "--stream-output",
            action="store_true",
            help="Whether to output as a sequence of disjoint segments.",
        )
        parser.add_argument(
            "--seed",
            metavar="SEED",
            type=int,
            default=ServerArgs.seed,
            help="The random seed.",
        )
        parser.add_argument(
            "--distributed-timeout-seconds",
            metavar="DISTRIBUTED_TIMEOUT_SECONDS",
            type=int,
            default=ServerArgs.distributed_timeout_seconds,
            help="Set timeout for torch.distributed initialization.",
        )
        parser.add_argument(
            "--download-dir",
            type=str,
            default=ServerArgs.download_dir,
            help="Model download directory for huggingface.",
        )
        parser.add_argument(
            "--base-gpu-id",
            type=int,
            default=ServerArgs.base_gpu_id,
            help="The base GPU ID to start allocating GPUs from. Useful when running multiple instances on the same machine.",
        )
        parser.add_argument(
            "--gpu-id-step",
            type=int,
            default=ServerArgs.gpu_id_step,
            help="The delta between consecutive GPU IDs that are used. For example, setting it to 2 will use GPU 0,2,4,...",
        )

        # Logging
        parser.add_argument(
            "--log-level",
            type=str,
            default=ServerArgs.log_level,
            help="The logging level of all loggers.",
        )
        parser.add_argument(
            "--enable-log-requests",
            action=argparse.BooleanOptionalAction,
            default=ServerArgs.enable_log_requests,
            help="Log metadata, inputs, outputs of all requests (default on; --no-enable-log-requests to disable). The verbosity is decided by --log-requests-level",
        )
        parser.add_argument(
            "--log-requests-level",
            type=int,
            default=0,
            help="0: Log metadata. 1. Log metadata and partial input/output. 2. Log every input/output.",
            choices=[0, 1, 2],
        )
        parser.add_argument(
            "--enable-log-request-stats",
            action=argparse.BooleanOptionalAction,
            default=ServerArgs.enable_log_request_stats,
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
            help="Enable log metrics.",
        )
        parser.add_argument(
            "--metrics-reporters",
            action="append",
            choices=["prometheus"],
            default=["prometheus"],
            help="Select metrics reporter(can be specified multiple times)",
        )

        parser.add_argument(
            "--app-key",
            type=str,
            default=ServerArgs.app_key,
            help="Set app key of the server",
        )

        parser.add_argument(
            "--decode-log-interval",
            type=int,
            default=ServerArgs.decode_log_interval,
            help="The log interval of decode batch.",
        )
        parser.add_argument(
            "--kv-events-config",
            type=str,
            default=ServerArgs.kv_events_config,
            help=(
                "JSON KV cache event publisher config. Set "
                "'enable_kv_cache_events': true and publisher 'zmq' to "
                "publish device prefix-cache mutations."
            ),
        )

        # Data parallelism
        parser.add_argument(
            "--data-parallel-size",
            metavar="DATA_PARALLEL_SIZE",
            type=int,
            default=ServerArgs.data_parallel_size,
            help="The data parallelism size. If not set, inferred from world_size and attn_tp_size.",
        )
        parser.add_argument(
            "--pipeline-parallel-size",
            metavar="PIPELINE_PARALLEL_SIZE",
            type=int,
            default=ServerArgs.pipeline_parallel_size,
            help="Number of pipeline stages for prefill chunk pipelining. "
            "Only supported with --disaggregation-mode prefill.",
        )
        parser.add_argument(
            "--pp-layer-partition",
            metavar="PP_LAYER_PARTITION",
            type=lambda arg: tuple(int(v) for v in arg.split(",")),
            default=ServerArgs.pp_layer_partition,
            help="Explicit per-stage layer counts for pipeline parallelism, "
            'front to back, e.g. "8,11,11,8". Must have one entry per stage '
            "and sum to the model's layer count. Default: even split with "
            "the remainder on the front stages.",
        )
        parser.add_argument(
            "--load-balance-method",
            type=str,
            default=ServerArgs.load_balance_method,
            help="The load balancing strategy for data parallelism.",
            choices=[
                "round_robin",
                "shortest_queue",
                "minimum_cache_usage",
            ],
        )
        parser.add_argument(
            "--load-watch-interval",
            type=float,
            default=ServerArgs.load_watch_interval,
            help="Heartbeat compatibility interval for load snapshots in seconds. "
            "Changed load values publish immediately without debounce.",
        )

        # Expert parallelism
        parser.add_argument(
            "--expert-parallel-size",
            "--ep-size",
            type=int,
            default=ServerArgs.ep_size,
            help="The expert parallelism size.",
        )
        parser.add_argument(
            "--init-expert-location",
            type=str,
            default=ServerArgs.init_expert_location,
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
            default=ServerArgs.ep_num_redundant_experts,
            help="Add this many physical expert slots per MoE layer for replicas "
            "of hot experts; the total must divide over the EP size.",
        )
        parser.add_argument(
            "--ep-dispatch-algorithm",
            type=str,
            default=ServerArgs.ep_dispatch_algorithm,
            choices=list(EP_DISPATCH_ALGORITHMS),
            help="How routing picks among an expert's replicas; required with an "
            "expert placement. static_with_zero_expert for models with zero "
            "experts (LongCat), static otherwise; dynamic* draw at random.",
        )
        parser.add_argument(
            "--eplb-algorithm",
            type=str,
            default=ServerArgs.eplb_algorithm,
            help="EPLB algorithm deriving the placement from a load record: "
            "auto, deepseek or deepseek_hierarchical.",
        )
        parser.add_argument(
            "--expert-distribution-recorder-mode",
            type=str,
            default=ServerArgs.expert_distribution_recorder_mode,
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
            "--ep-dispatch-algorithm, both explicit; POST /rebalance_experts "
            "triggers one rebalance manually.",
        )
        parser.add_argument(
            "--eplb-rebalance-num-iterations",
            type=int,
            default=ServerArgs.eplb_rebalance_num_iterations,
            help="Forwards between two expert load snapshots under --enable-eplb "
            "(required with it).",
        )
        parser.add_argument(
            "--eplb-rebalance-layers-per-chunk",
            type=int,
            default=ServerArgs.eplb_rebalance_layers_per_chunk,
            help="MoE layers whose experts move in one scheduling round under "
            "--enable-eplb (required with it; at most the MoE layer count). "
            "Fewer layers per chunk bound the per-round stall.",
        )
        parser.add_argument(
            "--dense-gemm-backend",
            type=str,
            default=ServerArgs.dense_gemm_backend,
            choices=["auto", "trtllm_cutedsl"],
            help="Backend for standard 128x128 block-FP8 dense linears. "
            "trtllm_cutedsl requires Blackwell and preserves checkpoint FP8 "
            "weights and FP32 scales without requantization. "
            "Other quantization formats and routed experts are unchanged.",
        )
        parser.add_argument(
            "--moe-backend",
            type=str,
            default=ServerArgs.moe_backend,
            help="MoE runner backend: auto, triton, gluon, flashinfer_trtllm, "
            "flashinfer_cutlass, flashinfer_cutedsl, deep_gemm, mega_moe, "
            "gluon_petit, aok (the batch-invariant leaves; --numerics rl-bitwise "
            "folds auto to it)",
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
            default=ServerArgs.draft_moe_backend,
            help="MoE runner backend for the draft model in speculative decoding. "
            "If not set, defaults to --moe-backend.",
        )
        parser.add_argument(
            "--all2all-backend",
            metavar="ALL2ALL_BACKEND",
            type=str,
            default=ServerArgs.all2all_backend,
            choices=["none", "agrs", "deepep", "flashinfer", "gluon_petit"],
            help="MoE communication backend. agrs and flashinfer explicitly select "
            "the Kimi-K3 attention-DP transport; gluon_petit selects the fused "
            "Petit MegaMoE transport; none preserves existing behavior.",
        )
        parser.add_argument(
            "--deepep-mode",
            type=str,
            choices=["normal", "low_latency", "auto"],
            default=ServerArgs.deepep_mode,
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
            help="The host address for initializing distributed backend (e.g., `192.168.0.2:25000`). "
            "Derived from the launcher environment under a multi-node Slurm step.",
        )
        parser.add_argument(
            "--nnodes",
            type=int,
            default=ServerArgs.nnodes,
            help="The number of nodes. Derived from SLURM_STEP_NUM_NODES when unset.",
        )
        parser.add_argument(
            "--node-rank",
            type=int,
            default=ServerArgs.node_rank,
            help="The node rank. Derived from SLURM_NODEID when unset.",
        )

        # Model override args
        parser.add_argument(
            "--hf-overrides",
            metavar="HF_OVERRIDES",
            type=str,
            help="A dictionary in JSON string format used to override default model configurations.",
            default=ServerArgs.hf_overrides,
        )
        parser.add_argument(
            "--preferred-sampling-params",
            type=str,
            help="Default sampling settings as JSON for SMG's gRPC GetModelInfo response.",
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
            default=ServerArgs.attention_backend,
            help="Choose the kernels for attention layers: "
            f"{attention_backend_names}, or a name a plugin registers. 'gluon' "
            "forces registered Gluon kernels for supported attention "
            "architectures.",
        )
        parser.add_argument(
            "--kda-backend",
            type=str,
            choices=["auto", "fla", "flashkda", "cutedsl_kda"],
            default=ServerArgs.kda_backend,
            help="KDA (Kimi Delta Attention) prefill kernel policy. On AMD, "
            "this setting is ignored and compatible kernels are selected using "
            "registry priority. On NVIDIA, 'auto' selects the fastest available "
            "backend "
            "(cutedsl_kda > flashkda > fla). Named backends are NVIDIA-specific: "
            "'fla' uses the portable FLA scan, 'flashkda' uses the optional "
            "FlashKDA library (source build, SM90+), and 'cutedsl_kda' uses the "
            "CuteDSL KDA AOT kernel (prebuilt, sm_103a). Decode is unaffected.",
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
            default=ServerArgs.skip_softmax_threshold,
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
            "--sampling-backend",
            type=str,
            choices=[
                "greedy",
                "flashinfer",
                "flashinfer_full",
                "triton",
                "triton_full",
            ],
            default=ServerArgs.sampling_backend,
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
            "Allocates a counts[max_req_pool_size, vocab_size] int32 buffer (substantial memory). "
            "Finite top_k values must be < 128 or -1.",
        )
        parser.add_argument(
            "--sampling-stream",
            type=str,
            choices=list(SAMPLING_STREAMS),
            default=ServerArgs.sampling_stream,
            help="Random stream of the non-greedy rows of the flashinfer and "
            "flashinfer_full sampling backends. 'batch': flashinfer's "
            "top_k_top_p / min_p sampling kernels, whose Philox stream is keyed "
            "by the batch row, so a request's draw depends on its co-batch. "
            "'per-request': the Gumbel-max pool kernels keyed by the request's "
            "seed and position, so a request samples the same tokens alone and "
            "inside any batch (finite top_k is capped at 128). Folded to "
            "per-request by --numerics rl-bitwise.",
        )
        parser.add_argument(
            "--dp-sampling",
            action="store_true",
            default=ServerArgs.dp_sampling,
            help=(
                "Enable Batch-DP spec-verify sampling. Backend selection defaults "
                "to auto; override with TOKENSPEED_DP_SAMPLING_BACKEND."
            ),
        )
        parser.add_argument(
            "--dp-sampling-min-bs",
            type=int,
            default=ServerArgs.dp_sampling_min_bs,
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
            default=ServerArgs.attention_use_fp4_indexer_cache,
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
            default=ServerArgs.use_trtllm_ragged_deepseek_prefill,
            help="Use ragged prefill for DeepSeek MLA attention.",
        )
        parser.add_argument(
            "--deepseek-v4-mega-moe-max-num-tokens",
            type=int,
            default=ServerArgs.deepseek_v4_mega_moe_max_num_tokens,
            help=(
                "DeepSeek V4 MegaMoE staging-buffer cap on tokens per forward "
                "(0 = derive from chunked-prefill / cuda-graph budgets)."
            ),
        )
        parser.add_argument(
            "--deepseek-v4-indexer-prefill-max-logits-mb",
            type=int,
            default=ServerArgs.deepseek_v4_indexer_prefill_max_logits_mb,
            help=(
                "DeepSeek V4 sparse indexer prefill workspace cap (MiB) for the "
                "softplus_sqrt logits buffer."
            ),
        )
        parser.add_argument(
            "--deepseek-v4-prefill-chunk-size",
            type=int,
            default=ServerArgs.deepseek_v4_prefill_chunk_size,
            help=(
                "Maximum number of requests per DeepSeek V4 FlashMLA prefill " "chunk."
            ),
        )
        parser.add_argument(
            "--engram-host-table",
            action=argparse.BooleanOptionalAction,
            default=ServerArgs.engram_host_table,
            help=(
                "DeepSeek V4.1 Engram: store the two FP8 n-gram tables in host "
                "memory and gather rows through UVA. Frees HBM for KV cache. "
                "See --engram-host-table-layout. Requires enough host RAM."
            ),
        )
        parser.add_argument(
            "--engram-host-table-dir",
            type=str,
            default=ServerArgs.engram_host_table_dir,
            help=(
                "Directory for shared Engram mmap files. Default: /dev/shm "
                "when it has enough free space, otherwise /scratch or /tmp. "
                "Used only with --engram-host-table-layout shared. Docker often "
                "caps /dev/shm at 32-64 GiB, too small for a full V4.1 table."
            ),
        )
        parser.add_argument(
            "--engram-host-table-layout",
            type=str,
            choices=["auto", "shared", "sharded"],
            default=ServerArgs.engram_host_table_layout,
            help=(
                "Host Engram layout when --engram-host-table is set. shared: one "
                "full copy per node, skip the lookup all-reduce. sharded: each "
                "attention-TP rank holds a host shard and keeps the all-reduce "
                "(anonymous mapping, huge-page friendly). auto: sharded when "
                "attention TP > 1, else shared."
            ),
        )
        parser.add_argument(
            "--grammar-backend",
            type=str,
            choices=["xgrammar", "none"],
            default=ServerArgs.grammar_backend,
            help="Grammar backend. 'none' disables grammar-guided decoding entirely ",
        )
        parser.add_argument(
            "--reasoning-parser",
            type=str,
            default=ServerArgs.reasoning_parser,
            help=(
                "Reasoning parser name (e.g. 'minimax', 'kimi_k25'). "
                "Used to defer json_schema grammars past the model's "
                "reasoning channel."
            ),
        )
        parser.add_argument(
            "--grammar-compile-timeout-secs",
            type=float,
            default=ServerArgs.grammar_compile_timeout_secs,
            help="Per-compile wallclock budget before the request is aborted.",
        )
        parser.add_argument(
            "--grammar-compile-max-retries",
            type=int,
            default=ServerArgs.grammar_compile_max_retries,
            help="Compile timeouts allowed before a grammar key is permanently rejected.",
        )
        parser.add_argument(
            "--disable-any-whitespace",
            action="store_true",
            default=ServerArgs.disable_any_whitespace,
            help="Compile xgrammar JSON grammars in tight mode (no arbitrary "
            "whitespace between tokens). Mitigates models that wedge into "
            "endless whitespace until length cutoff. xgrammar only.",
        )
        parser.add_argument(
            "--disable-capturable-grammar",
            action="store_true",
            default=ServerArgs.disable_capturable_grammar,
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
            "--draft-model-path-use-base",
            action="store_true",
            help="The path of the draft model weights use the path of the base model",
        )
        parser.add_argument(
            "--speculative-config",
            "--speculative_config",
            type=str,
            default=ServerArgs.speculative_config,
            help="JSON speculative decoding configuration. Supported keys are method, model, and num_speculative_tokens.",
        )
        parser.add_argument(
            "--speculative-algorithm",
            type=str,
            help="Speculative algorithm. In-tree: EAGLE3, MTP, DFLASH, "
            "DSPARK; plugins may register more (validated after plugin "
            "discovery).",
        )
        parser.add_argument(
            "--speculative-draft-model-path",
            type=str,
            help="The path of the draft model weights. This can be a local folder or a Hugging Face repo ID.",
        )
        parser.add_argument(
            "--speculative-draft-model-quantization",
            type=str,
            default=ServerArgs.speculative_draft_model_quantization,
            help="Quantization method for the draft model. Defaults to 'unquant'.",
        )
        parser.add_argument(
            "--speculative-num-steps",
            type=int,
            help="The number of steps sampled from draft model in Speculative Decoding.",
            default=ServerArgs.speculative_num_steps,
        )
        parser.add_argument(
            "--speculative-eagle-topk",
            type=int,
            help="Children each draft node expands to per step; above 1 the draft is a tree "
            "(EAGLE3, or EAGLE-style MTP), and --speculative-num-draft-tokens is its node budget.",
            default=ServerArgs.speculative_eagle_topk,
        )
        parser.add_argument(
            "--speculative-num-draft-tokens",
            type=int,
            help="The number of tokens sampled from the draft model in Speculative Decoding.",
            default=ServerArgs.speculative_num_draft_tokens,
        )
        parser.add_argument(
            "--enable-speculative-sampling",
            action="store_true",
            default=ServerArgs.enable_speculative_sampling,
            help="Standard rejection sampling for chain speculative decoding: the "
            "drafter samples each step from its own distribution q (per-request "
            "temperature; greedy rows stay argmax) and verify accepts with "
            "coin * q(x) < p(x) instead of the target-only rule. Needs EAGLE3 "
            "or MTP with --speculative-eagle-topk 1 and the flashinfer or "
            "flashinfer_full sampling backend; refused on the prefill role of "
            "a disaggregated deployment. Costs a per-request fp32 draft "
            "distribution buffer; see docs/configuration/server.md.",
        )
        parser.add_argument(
            "--spec-reject-draft-prob-threshold",
            type=float,
            default=ServerArgs.spec_reject_draft_prob_threshold,
            help="With --enable-speculative-sampling, recorded draft probabilities "
            "above this value mark a request with no proposal yet (fresh "
            "admission, PD landing) and always reject. Must lie within "
            "[1.0, 2**20].",
        )
        parser.add_argument(
            "--disable-replay-ssm",
            dest="enable_replay_ssm",
            action="store_false",
            default=ServerArgs.enable_replay_ssm,
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
            default=ServerArgs.enable_output_logprobs,
            help="Enable per-token sampled-token logprobs. OFF by default; enabling extends the captured CUDA-graph footprint. Requests asking for logprobs on a server without this flag receive empty logprobs.",
        )
        parser.add_argument(
            "--input-logprob-chunk-tokens",
            type=int,
            default=ServerArgs.input_logprob_chunk_tokens,
            help="Prompt rows pushed through the LM head per chunk when a request asks for prompt (input) logprobs (SGLang logprob_start_len). A sizing knob only: the transient per-chunk cost is about chunk x vocab x 8 bytes (bf16 logits shard, bf16 TP-gathered logits, fp32 log-softmax) and the value never changes a result.",
        )
        parser.add_argument(
            "--eagle3-layers-to-capture",
            type=str,
            help="The layers of Eagle3 to capture.",
            default=ServerArgs.eagle3_layers_to_capture,
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
            default=ServerArgs.enable_prefix_caching,
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
            help="Skip profiling missing kernel tactics during startup. A matching "
            "persistent FlashInfer cache is still loaded; uncovered shapes use "
            "the library's heuristic fallback.",
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
            help="Disable the overlap scheduler, which overlaps the CPU scheduler with GPU model worker.",
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
            default=ServerArgs.max_cudagraph_capture_size,
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
            default=ServerArgs.prefill_graph_max_tokens,
            help="Largest token bucket captured by the breakable prefill CUDA "
            "graph. Default (unset) = min(2048, chunked-prefill size); "
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
            "Unset: the minimum request count that fits each token bucket within "
            "the model context. KDA uses fixed checkpoint slots, so each token "
            "bucket needs one inline variant per configured request count. "
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
            default=ServerArgs.weight_loader_prefetch_checkpoints,
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
            default=ServerArgs.weight_loader_prefetch_num_threads,
            help="Maximum concurrent checkpoint range readers per rank.",
        )
        parser.add_argument(
            "--enable-memory-saver",
            action="store_true",
            help="Allow saving memory using release_memory_occupation and resume_memory_occupation",
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
            default=ServerArgs.attn_tp_size,
            help="Specify tp size for attn part",
        )
        parser.add_argument(
            "--kv-parallel-size",
            type=int,
            default=ServerArgs.kv_parallel_size,
            help="KV parallelism (KVP): shard full-history KV pages (MLA/DSA "
            "latent, DeepSeek V4 compressed KV) cyclically over a consecutive "
            "subgroup of attention TP, each rank storing one shard of every "
            "request's pages; decode attends the owned pages and merges the "
            "partials (DCP). Allowed on aggregated engines and the PD prefill "
            "role, with the Host KVStore and the retraction snapshot pool; the "
            "decode role and L3 storage (--kvstore-storage-backend) are not "
            "supported yet.",
        )
        parser.add_argument(
            "--attn-head-tp-size",
            type=int,
            default=ServerArgs.attn_head_tp_size,
            help="Shard the MLA head projections (q_b_proj, kv_b_proj, o_proj) "
            "by heads over this many contiguous ranks that hold different rows; "
            "the attention exchanges heads for tokens around core attention. "
            "Either attention-DP ranks of a decode engine (requires attention "
            "TP 1, attention DP and --disaggregation-mode decode; every rank "
            "keeps its own KV cache), or the query shards of a prefill engine "
            "(must equal --prefill-context-parallel-size; the extend rows run "
            "the absorbed sparse prefill through the exchange). Defaults to the "
            "ranks holding the same rows: the attention TP size, or 1 "
            "(head-replicated) under --prefill-context-parallel-size.",
        )
        parser.add_argument(
            "--lm-head-tp-size",
            type=int,
            default=ServerArgs.lm_head_tp_size,
            help="Vocab-shard the LM head over this many contiguous ranks. "
            "Under attention DP (which needs attention TP 1) the default 1 "
            "keeps the head replicated; a wider group gathers the ranks' rows "
            "before the logits GEMM and transposes the vocab shards back. "
            "Without attention DP it must equal the attention TP size.",
        )
        parser.add_argument(
            "--tp-batch-invariant",
            type=str,
            choices=["none", "attn", "attn+dense"],
            default=ServerArgs.tp_batch_invariant,
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
            "--prefill-context-parallel-size",
            type=int,
            default=ServerArgs.prefill_context_parallel_size,
            help="Shard every extend forward's query rows over the attention TP "
            "group on the PD prefill role (query context parallelism): rank r "
            "computes a contiguous slice of the chunk's rows against the gathered "
            "KV history of its requests. Must equal --attn-tp-size and requires "
            "--disaggregation-mode prefill, --disable-prefill-graph, a DSA-family "
            "attention backend and --kv-parallel-size 1 or equal. "
            "The attention weights are head-replicated unless --attn-head-tp-size "
            "equals it, which shards them over the shard group.",
        )
        parser.add_argument(
            "--dense-tp-size",
            type=int,
            default=ServerArgs.dense_tp_size,
            help="Specify tp size for dense part. Defaults to the attention "
            "TP width: the full world without DP attention, one replica with "
            "it.",
        )
        parser.add_argument(
            "--moe-tp-size",
            type=int,
            default=ServerArgs.moe_tp_size,
            help="Specify tp size for MoE part, default equals nprocs-per-node, if non dp_attn && combine_dense mode, this parameter will be overridden by attn_tp_size",
        )
        parser.add_argument(
            "--nprocs-per-node",
            type=int,
            default=ServerArgs.nprocs_per_node,
            help="Number of processes to start per node",
        )
        parser.add_argument(
            "--world-size",
            type=int,
            default=ServerArgs.world_size,
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
            "path -- neither the Triton multicast all-gather/reduce-scatter "
            "and in-switch all-reduce, nor the trtllm and Triton all-reduce "
            "tiers, nor the distributed argmax. With --batch-invariant-collectives every "
            "reduction takes the fold; without it, NCCL. Not folded in by "
            "--numerics rl-bitwise, which keeps the multicast paths and "
            "verifies the in-switch reduction at startup.",
        )
        parser.add_argument(
            "--batch-invariant-collectives",
            action="store_true",
            help="One association order per reduction, independent of the "
            "batch. A 2-D bf16 all-reduce on a multicast-reachable group runs "
            "as the NVLS in-switch reduction issued by one fixed rank, "
            "verified bitwise at startup; every other reduction (other "
            "payloads, unreachable groups, the reduce-scatters) runs as NCCL "
            "data movement plus a fixed-rank-order fp32 fold, which for an "
            "all-reduce costs world_size times the traffic. NCCL sums are "
            "run-stable but chunk by message size, so they are not "
            "batch-size-invariant. Folded in by --numerics rl-bitwise.",
        )
        parser.add_argument(
            "--numerics",
            type=str,
            choices=list(NUMERICS_ENVELOPES),
            default=ServerArgs.numerics,
            help="Numerics envelope. rl-bitwise folds the determinism "
            "switches (batch-invariant collectives, no autotune/TF32/PDL, no "
            "fused all-reduce, the batch-invariant MoE leaves, per-request "
            "sampling) so outputs and logprobs are bitwise identical across "
            "runs and batch compositions within one deployment, and the "
            "trainer-operation-order switches (docs/design/numerics.md, "
            "alignment.trainer) so a teacher-forced pass reproduces the RL "
            "trainer's logprobs; a model serves it only once its profile "
            "declares it verified.",
        )
        parser.add_argument(
            "--yarn-ramp-mask-device",
            type=str,
            choices=list(YARN_RAMP_MASK_DEVICES),
            default=ServerArgs.yarn_ramp_mask_device,
            help="Device that computes the deepseek_yarn RoPE inverse "
            "frequencies (the position frequencies, both divisions and the "
            "YaRN linear ramp mask) before the table is moved to the model "
            "device once. The trainer builds it on the host, and CPU and CUDA "
            "division round differently at ulp level. Folded to cpu by "
            "--numerics rl-bitwise.",
        )
        parser.add_argument(
            "--mla-lora-scale",
            type=str,
            choices=list(MLA_LORA_SCALES),
            default=ServerArgs.mla_lora_scale,
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
            default=ServerArgs.layer_boundary_norm,
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
            default=ServerArgs.router_topk,
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
            default=ServerArgs.logprob_order,
            help="Order of the selected-token log-softmax behind every "
            "returned logprob. 'torch': torch.log_softmax. 'megatron': the "
            "trainer's vocab-parallel cross-entropy order (row max, shift, "
            "sum(exp) over fixed 32768-wide vocab blocks folded in block "
            "order, logp = -(log(sum_exp) - target)); requests asking for "
            "temperature- or top-p-normalised logprobs are refused. Changes "
            "logprobs only, never the sampled tokens. Folded to megatron by "
            "--numerics rl-bitwise.",
        )
        parser.add_argument(
            "--moe-combine-order",
            type=str,
            choices=list(MOE_COMBINE_ORDERS),
            default=ServerArgs.moe_combine_order,
            help="How a token's routed-expert contributions meet across the "
            "MoE TP-EP group. 'rank': the MoE kernel returns this rank's "
            "partial and the host sums the partials (all-reduce or "
            "reduce-scatter), adding the identity zero-expert residual once "
            "around it. 'slot': the MoE kernel folds the token's top-k slots "
            "in fp32 slot order across the EP group itself, zero-expert "
            "residual included, as the trainer's grouped MLP does, and the "
            "host reduces nothing; needs MoE TP 1 and a kernel declaring the "
            "combine_order trait with slot (the batch-invariant 'aok' leaf), "
            "and vetoes all-reduce+norm fusion. Folded to slot by --numerics "
            "rl-bitwise.",
        )
        parser.add_argument(
            "--dsa-slot-order",
            type=str,
            choices=list(DSA_SLOT_ORDERS),
            default=ServerArgs.dsa_slot_order,
            help="The order the sparse (DSA) attention cores reduce a token's "
            "selected KV slots in. 'selection': as the top-k leaf emitted "
            "them (every core). 'sorted': ascending slot order, so the "
            "reduction is batch-invariant whenever the selected set is; "
            "served only by cores declaring the slot_order trait (the "
            "batch-invariant 'aok' leaves). Folded to sorted by --numerics "
            "rl-bitwise.",
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
            default=ServerArgs.low_latency_max_num_tokens_per_gpu,
            help="DeepEP low-latency send capacity per rank. Defaults to 256; "
            "set explicitly to cover the largest batch sent through low latency.",
        )

        parser.add_argument(
            "--mla-chunk-multiplier",
            type=int,
            default=ServerArgs.mla_chunk_multiplier,
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
            default=ServerArgs.mm_attention_backend,
            help="Set multimodal attention backend.",
        )
        parser.add_argument(
            "--mm-encoder-tp-mode",
            type=str,
            choices=["weights", "data"],
            default=ServerArgs.mm_encoder_tp_mode,
            help=(
                "Multimodal encoder parallelism within each attention TP "
                "group. 'weights' shards encoder weights with TP (default); "
                "'data' replicates encoder weights and distributes whole "
                "multimodal items across the TP ranks."
            ),
        )
        # Disaggregation
        parser.add_argument(
            "--disaggregation-mode",
            type=str,
            default="null",
            choices=["null", "prefill", "decode", "encode"],
            help='Used for PD/EPD disaggregation. "prefill" for prefill-only server, "decode" for decode-only server, and "encode" for a vision-tower-only server that ships image embeddings to a prefill server. If not specified, it is not disaggregated',
        )
        parser.add_argument(
            "--comm-fusion-max-num-tokens",
            type=int,
            default=ServerArgs.comm_fusion_max_num_tokens,
            help="Max num tokens for communication fusion workspace",
        )
        parser.add_argument(
            "--enable-allreduce-fusion",
            action="store_true",
            help="Enable allreduce fusion for improved decode performance. Auto-enabled on supported single-node TP configurations.",
        )
        parser.add_argument(
            "--disaggregation-bootstrap-port",
            type=int,
            default=ServerArgs.disaggregation_bootstrap_port,
            help="Bootstrap server port on the prefill server. Default is 8998.",
        )
        parser.add_argument(
            "--disaggregation-transfer-backend",
            type=str,
            default=ServerArgs.disaggregation_transfer_backend,
            choices=["mooncake"],
            help="The backend for disaggregation transfer. Default is mooncake.",
        )
        parser.add_argument(
            "--disaggregation-ib-device",
            type=str,
            default=ServerArgs.disaggregation_ib_device,
            help="The InfiniBand devices for disaggregation transfer, accepts single device (e.g., --disaggregation-ib-device mlx5_0) "
            "or multiple comma-separated devices (e.g., --disaggregation-ib-device mlx5_0,mlx5_1). "
            "Default is None, which triggers automatic device detection when mooncake backend is enabled.",
        )
        parser.add_argument(
            "--disaggregation-layerwise-interval",
            type=int,
            default=ServerArgs.disaggregation_layerwise_interval,
            help="The interval of layerwise transfer for disaggregation. Default is 1.",
        )
        parser.add_argument(
            "--pdlb-url",
            type=str,
            default=None,
            help="The URL of the PD disaggregation load balancer. If set, the prefill/decode server will register with the load balancer.",
        )

        # SGLang-compatible RL control app.
        parser.add_argument(
            "--rl-control-port",
            type=int,
            default=ServerArgs.rl_control_port,
            help="Port for the in-engine RL control-plane HTTP app (weight sync, "
            "pause/resume, memory occupation). Normally allocated automatically "
            "by the `ts serve` orchestrator.",
        )
        parser.add_argument(
            "--rl-control-host",
            type=str,
            default=ServerArgs.rl_control_host,
            help="Bind host for the in-engine RL control-plane HTTP app. Defaults to "
            "--host. Bind a reachable address when an external gateway drives the "
            "engine, and set --rl-control-api-key.",
        )
        parser.add_argument(
            "--rl-control-api-key",
            type=str,
            default=ServerArgs.rl_control_api_key,
            help="Bearer token required on every RL control-plane route. Unset "
            "leaves the app open, which is what slime expects by default.",
        )
        parser.add_argument(
            "--weight-version",
            type=str,
            default=ServerArgs.weight_version,
            help="Initial model-weight version stamped into generation metadata.",
        )
        parser.add_argument(
            "--model-update-config",
            type=str,
            default=ServerArgs.model_update_config,
            help="JSON object handed to the Model Updater SDK for "
            "/update_weights_from_mooncake. Requires --model-update-sdk-module, "
            "--model-update-engine-type and --model-update-draft-weights.",
        )
        parser.add_argument(
            "--model-update-sdk-module",
            type=str,
            default=ServerArgs.model_update_sdk_module,
            help="Import path of the Model Updater SDK module (imported lazily "
            "on the first /update_weights_from_mooncake).",
        )
        parser.add_argument(
            "--model-update-engine-type",
            type=str,
            default=ServerArgs.model_update_engine_type,
            help="Model Updater SDK EngineType member name for this engine "
            "(resolved as EngineType[value.upper()]).",
        )
        parser.add_argument(
            "--model-update-draft-weights",
            type=str,
            choices=["retain", "refresh"],
            default=ServerArgs.model_update_draft_weights,
            help="Whether /update_weights_from_mooncake also streams the "
            "speculative draft model's weights: 'retain' updates the target "
            "only, 'refresh' updates target and draft.",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        args.ep_size = args.expert_parallel_size

        # Resolve model (positional model arg vs --model)
        positional_model = getattr(args, "model_path", None)
        if positional_model is not None and args.model is not None:
            raise ValueError(
                "Cannot specify model both as a positional argument and --model. "
                "Use one or the other."
            )
        if positional_model is not None:
            args.model = positional_model
        if args.model is None:
            raise ValueError(
                "Model is required. Provide it as a positional argument "
                "(e.g., `tokenspeed serve <model>`) or via --model/--model-path."
            )

        # --tensor-parallel-size → --attn-tp-size
        tensor_parallel_size = getattr(args, "tensor_parallel_size", None)
        if tensor_parallel_size is not None:
            if args.attn_tp_size is not None:
                raise ValueError(
                    "Cannot specify both --tensor-parallel-size and --attn-tp-size. "
                    "--tensor-parallel-size is an alias for --attn-tp-size."
                )
            args.attn_tp_size = tensor_parallel_size

        # Only pass fields that argparse actually produced. Falling back to
        # ``None`` for missing attrs would silently clobber dataclass defaults
        # for non-CLI-exposed fields (e.g. ``enable_inline_detokenizer``).
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(
            **{attr: getattr(args, attr) for attr in attrs if hasattr(args, attr)}
        )

    def url(self):
        if is_valid_ipv6_address(self.host):
            return f"http://[{self.host}]:{self.port}"
        return f"http://{self.host}:{self.port}"

    def zmq_handshake_endpoint(self) -> str:
        """The frontend handshake endpoint dialed under ``--zmq-msgpack``,
        composed from ``--data-parallel-address``/``--data-parallel-rpc-port``."""
        return f"tcp://{self.data_parallel_address}:{self.data_parallel_rpc_port}"


def prepare_server_args(argv: list[str]) -> ServerArgs:
    """
    Prepare the server arguments from the command line arguments.

    Args:
        args: The command line arguments. Typically, it should be `sys.argv[1:]`.

    Returns:
        The server arguments.
    """
    parser = argparse.ArgumentParser(allow_abbrev=False)
    ServerArgs.add_cli_args(parser)
    raw_args = parser.parse_args(argv)
    server_args = ServerArgs.from_cli_args(raw_args)
    return server_args


ZMQ_TCP_PORT_DELTA = 233


@dataclasses.dataclass
class PortArgs:
    # The ipc filename for AsyncLLM to receive BatchTokenIDOut directly
    # from the scheduler (zmq).
    tokenizer_ipc_name: str
    # The ipc filename for scheduler (rank 0) to receive inputs from tokenizer (zmq)
    scheduler_input_ipc_name: str

    # The port for nccl initialization (torch.dist)
    nccl_port: int

    # The resolved rendezvous address after moving past busy local ports.
    dist_init_addr: str

    # The ipc filename for rpc call between Engine and Scheduler
    rpc_ipc_name: str

    # The ipc filename for Scheduler to send metrics
    metrics_ipc_name: str

    # The ipc filename for Tokenizer and worker tokenizer
    tokenizer_worker_ipc_name: str | None

    @staticmethod
    def init_new(server_args: ServerArgs, dp_rank: int | None = None) -> "PortArgs":
        port = server_args.port + random.randint(100, 1000)
        while True:
            if is_port_available(port):
                break
            if port < 60000:
                port += 42
            else:
                port -= 43

        # DP attention. Use TCP + port to handle both single-node and multi-node.
        if server_args.mapping.nnodes == 1 and server_args.dist_init_addr is None:
            # Only use default port fallback when dp_size == 1
            # For dp_size > 1, we need explicit dist_init_addr to avoid port conflicts
            if server_args.mapping.has_attn_dp:
                raise ValueError(
                    f"When dp_size > 1 (dp_size={server_args.mapping.attn.dp_size}), you must provide --dist-init-addr. "
                    f"Example: --dist-init-addr 127.0.0.1:4000"
                )
            dist_init_addr = ("127.0.0.1", server_args.port + ZMQ_TCP_PORT_DELTA)
        elif server_args.dist_init_addr is None:
            raise ValueError(
                f"--dist-init-addr is required for nnodes={server_args.mapping.nnodes} "
                "and could not be derived from the launcher environment. "
                "Example: --dist-init-addr <head-node-ip>:20000"
            )
        else:
            dist_init_addr = server_args.dist_init_addr.split(":")
        if len(dist_init_addr) != 2:
            raise ValueError(
                "please provide --dist-init-addr as host:port of head node"
            )

        dist_init_host, dist_init_port = dist_init_addr
        dist_init_port = int(dist_init_port)

        # Scan forward until we find a port cluster where all derived ports are
        # free. This handles the case where a previous engine instance left
        # ports in TIME_WAIT or its child processes haven't fully terminated
        # yet. Note: the port at offset +1 (formerly detokenizer_port) is
        # intentionally skipped so the rest of the port layout stays stable for
        # any external tooling that indexed off the historical port cluster.
        #
        # The whole cluster is bound on node 0 alone, so scanning is only
        # meaningful there: is_port_available binds the local wildcard, and a
        # follower moving its own base would simply address ports the head
        # never bound. Multi-node therefore takes the cluster as derived and
        # reports a conflict instead of relocating it.
        while True:
            port_base = dist_init_port + 1
            rpc_port = port_base + 2
            metrics_ipc_port = port_base + 3
            if dp_rank is None:
                # TokenizerManager to DataParallelController
                scheduler_input_port = port_base + 4
            else:
                scheduler_input_port = port_base + 2 + 1 + dp_rank
            rpc_ipc_port = scheduler_input_port + 1
            cluster = [
                dist_init_port,
                port_base,
                rpc_port,
                metrics_ipc_port,
                scheduler_input_port,
                rpc_ipc_port,
            ]
            if server_args.mapping.nnodes > 1:
                if server_args.node_rank == 0:
                    busy = [p for p in cluster if not is_port_available(p)]
                    if busy:
                        raise ValueError(
                            f"control-plane ports {busy} are already in use on the "
                            "head node. Every node derives this cluster from "
                            "--dist-init-addr, so it cannot be moved on one node "
                            "alone; restart with a different --dist-init-addr port."
                        )
                break
            if all(is_port_available(p) for p in cluster):
                break
            dist_init_port += 10

        return PortArgs(
            tokenizer_ipc_name=f"tcp://{dist_init_host}:{port_base}",
            scheduler_input_ipc_name=f"tcp://{dist_init_host}:{scheduler_input_port}",
            nccl_port=port,
            dist_init_addr=f"{dist_init_host}:{dist_init_port}",
            rpc_ipc_name=f"tcp://{dist_init_host}:{rpc_port}",
            metrics_ipc_name=f"tcp://{dist_init_host}:{metrics_ipc_port}",
            tokenizer_worker_ipc_name=None,
        )
