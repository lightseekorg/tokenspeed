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
import math
import os
import random
import socket
from collections.abc import Sequence

from tokenspeed_cli import add_server_cli_args
from tokenspeed_cli.server_args import (
    DECODE_MONOKERNEL_ARCHITECTURES,
    DRAFT_MONOKERNEL_TARGETS,
    OWNED_ATTENTION_OUTPUTS,
    PREFILL_COLLECTIVE_PAYLOADS,
    REJECTION_SAMPLE_METHODS,
    ServerArgsOptions,
)
from tokenspeed_cli.server_args import nullable_str as nullable_str
from tokenspeed_cli.server_args import str_to_bool as str_to_bool
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.configs.numerics import (
    BITWISE_ENVELOPES,
    BLOCKWISE_LOGPROB_ORDERS,
    DSA_SLOT_ORDERS,
    LAYER_BOUNDARY_NORMS,
    LOGPROB_ORDERS,
    MLA_LORA_SCALES,
    MOE_COMBINE_ORDERS,
    NUMERICS_ENVELOPES,
    RL_BITWISE_SAMPLING_BACKENDS,
    ROUTER_TOPKS,
    SAMPLING_STREAMS,
    YARN_FREQUENCY_DEVICES,
    YARN_RAMP_MASK_DEVICES,
    resolve_numerics_envelope,
)
from tokenspeed.runtime.distributed.mapping import Mapping, _resolve_parallelism_sizes
from tokenspeed.runtime.execution.drafter.eagle_harness.config import (
    validate_server_args as validate_eagle_harness_args,
)
from tokenspeed.runtime.metrics.config import resolve_metrics_options
from tokenspeed.runtime.moe.dispatch_algorithm import (
    EP_DISPATCH_ALGORITHMS as EP_DISPATCH_ALGORITHMS,
)
from tokenspeed.runtime.moe.dispatch_algorithm import (
    STATIC_EP_DISPATCH_ALGORITHMS,
)
from tokenspeed.runtime.utils import (
    get_amdgpu_memory_capacity,
    get_colorful_logger,
    get_nvgpu_memory_capacity,
    is_valid_ipv6_address,
    maybe_model_redirect,
)
from tokenspeed.runtime.utils.launcher import check_dist_init_port, detect_topology
from tokenspeed.runtime.utils.network import (
    is_port_available,
    join_host_port,
    split_host_port,
)
from tokenspeed.runtime.utils.spec_block_geometry import (
    BLOCK_SPEC_ALGORITHMS,
    BLOCK_SPEC_RULES,
)

logger = get_colorful_logger(__name__)

# Sampling backends whose verify runs a draft-prob chain kernel
# (--enable-speculative-sampling) for the step drafters (EAGLE3, MTP): the
# FlashInfer backends over whole rows, sonic over its top-128 candidates.
# greedy verifies by exact match and the Triton backends by target-sampled
# exact match; the step drafters' dense recorded distribution never enters
# either.
SPECULATIVE_SAMPLING_BACKENDS = frozenset({"flashinfer", "flashinfer_full", "sonic"})
# The one backend whose verify reads the DFlash2 candidate selector's sparse
# draft q (RuntimeStates.future_draft_q): a DFlash2 draft needs it. A base DFLASH
# draft records the dense q and takes SPECULATIVE_SAMPLING_BACKENDS; DFlash.
# bind_sampling_backend checks the pairing once the draft is loaded.
DFLASH_SPECULATIVE_SAMPLING_BACKEND = "triton"

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
# round, so the Finish event reaches the C++ scheduler before the next plan).
# The device cuts a round at max_new_tokens and stops the request, but a
# finish only the control plane decides (a stop string) leaves the rounds
# uncut, so verify can still commit past context_len for exactly that window:
#   1. the finishing step's own accept (up to spec_num_tokens past the limit),
#   2. the single lingering step's accept,
#   3. the lingering step laying out its next draft block after that accept.
# Hence 3 spans. Everything written past context_len belongs to a request
# whose output is already truncated by max_new_tokens; the garbage KV is
# evicted with the request one step later.
_SPEC_OVERSHOOT_SPANS = 3


def spec_rope_positions(context: int, verify_width: int) -> int:
    """How many positions a RoPE table needs for a context.

    The RoPE kernels index the table by raw position, and speculative
    decoding writes up to ServerArgs.spec_context_pad positions past the
    context (_SPEC_OVERSHOOT_SPANS verify widths). The table covers the
    context, that pad and one more verify width of headroom; the verify width
    is 0 without speculative decoding.
    """
    return context + (_SPEC_OVERSHOOT_SPANS + 1) * verify_width


# Speculative algorithms a prefill server runs on the chunk pipeline
# (--pipeline-parallel-size > 1). The drafter executes on the last stage, the
# only stage that samples: an MTP (NextN) draft needs only that stage's final
# hidden states, and DSPARK produces its draft context across stages. EAGLE3
# is excluded because its aux taps come from several stages and nothing
# carries them through the stage boundary. See
# ServerArgs.resolve_disaggregation.
PIPELINE_SPEC_ALGORITHMS = ("DSPARK", "MTP")
#: Drafters fed by the target's aux hidden-state taps (``eagle3_layers_to_capture``).
AUX_CAPTURE_SPEC_ALGORITHMS = ("EAGLE3", "EAGLE4")
#: Drafters that propose one token per draft step (a chain of steps + 1 tokens, or a tree).
STEP_SPEC_ALGORITHMS = ("EAGLE3", "EAGLE4", "MTP")
#: Drafters whose prefill rounds replay graphs recorded at each request count.
EXTEND_GRAPH_SPEC_ALGORITHMS = (*STEP_SPEC_ALGORITHMS, "DFLASH", "DSPARK")
# Drafters whose topk=1 chain can record sampled proposals for block
# verification (flashinfer or triton) or synthetic rejection (flashinfer).
BLOCK_REJECTION_ALGORITHMS = ("DSPARK", "EAGLE3")
# Draft sampling, named as vLLM's draft_sample_method; "probabilistic" is
# --enable-speculative-sampling.
DRAFT_SAMPLE_METHODS = ("greedy", "probabilistic")


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


def _checkpoint_architectures(
    model: str, revision: str | None, download: bool = False
) -> list[str] | None:
    """The ``architectures`` of ``model``'s ``config.json``: from the model
    directory, else from the local Hugging Face cache, else, with ``download``,
    by downloading that file alone as ``get_config`` resolves a hub id
    (HF_HUB_OFFLINE keeps it to the cache; a ModelScope id is not looked up on
    the Hugging Face Hub). None when it cannot be read.
    """
    if not model:
        return []
    path = os.path.join(model, "config.json")
    if not os.path.isdir(model):
        import huggingface_hub.constants
        from huggingface_hub import hf_hub_download, try_to_load_from_cache

        from tokenspeed.runtime.utils.env import envs

        try:
            path = try_to_load_from_cache(model, "config.json", revision=revision)
            if (
                not isinstance(path, str)
                and download
                and not envs.TOKENSPEED_USE_MODELSCOPE.get()
            ):
                from tokenspeed.runtime.model_loader.weight_utils import get_lock

                with get_lock(model):
                    path = hf_hub_download(
                        model,
                        "config.json",
                        revision=revision,
                        local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                    )
        except Exception:  # not a repo id, offline and not cached, unreachable
            return None
        if not isinstance(path, str):
            return None
    try:
        with open(path) as file:
            config = json.load(file)
    except (OSError, ValueError):
        return None
    architectures = config.get("architectures") if isinstance(config, dict) else None
    return list(architectures) if isinstance(architectures, list) else []


# Checkpoint architectures that run --dense-fp8-small-m's FP8 projections.
DENSE_FP8_SMALL_M_ARCHITECTURES = ("MuseSparkForConditionalGeneration",)


def draft_takes_decode_monokernel(
    switch: bool, target_architectures: list[str], draft_architectures: list[str]
) -> bool:
    """Whether a draft under ``--decode-monokernel`` (``switch``) builds its
    layers from its monokernel class: only a draft in DRAFT_MONOKERNEL_TARGETS
    of one of the targets it lists."""
    return switch and any(
        target in DRAFT_MONOKERNEL_TARGETS.get(draft, ())
        for draft in draft_architectures
        for target in target_architectures
    )


def decode_monokernel_refusal(architectures: list[str]) -> str | None:
    """Why ``--decode-monokernel`` cannot serve a checkpoint of
    ``architectures``, or None when a monokernel exists for one of them."""
    if any(a in DECODE_MONOKERNEL_ARCHITECTURES for a in architectures):
        return None
    return (
        f"--decode-monokernel: no decode monokernel for {architectures or 'this model'} "
        f"(models with one: {', '.join(DECODE_MONOKERNEL_ARCHITECTURES)})"
    )


def dense_fp8_small_m_refusal(architectures: list[str]) -> str | None:
    """Why ``--dense-fp8-small-m`` cannot serve a checkpoint of
    ``architectures``, or None when one of them runs its FP8 projections."""
    if any(a in DENSE_FP8_SMALL_M_ARCHITECTURES for a in architectures):
        return None
    return (
        f"no FP8 projections for {architectures or 'this model'} (models with "
        f"them: {', '.join(DENSE_FP8_SMALL_M_ARCHITECTURES)})"
    )


def kernel_override_env_key(family: str, mode: str) -> str:
    """The variable ``tokenspeed_kernel.selection.select_kernel`` reads."""
    return f"TOKENSPEED_KERNEL_OVERRIDE_{family.upper()}_{mode.upper()}"


def validate_context_parallel_roles(
    *,
    kvp_size: int,
    dcp_size: int,
    qcp_size: int,
    disaggregation_mode: str,
) -> None:
    """Validate KVP/DCP roles and the PD-prefill-only QCP layout.

    Aggregated and PD decode engines require explicit DCP matching KVP.
    """
    if dcp_size < 1:
        raise ValueError(
            f"--decode-context-parallel-size must be a positive integer, got {dcp_size}"
        )
    if kvp_size > 1 and disaggregation_mode not in ("null", "prefill", "decode"):
        raise ValueError(
            "--kv-parallel-size > 1 requires --disaggregation-mode null, prefill "
            f"or decode (got {disaggregation_mode!r})"
        )
    if qcp_size > 1:
        if disaggregation_mode != "prefill":
            raise ValueError(
                "--query-context-parallel-size > 1 requires --disaggregation-mode "
                f"prefill (got {disaggregation_mode!r}): the query shard is a "
                "prefill-role layout whose decode arm runs every row on every "
                "rank (the drafter's steps only), which no engine serving "
                "steady-state decode should pay"
            )
        if kvp_size not in (1, qcp_size):
            raise ValueError(
                "--query-context-parallel-size must equal --kv-parallel-size "
                f"when KVP is enabled (got qcp={qcp_size}, kvp={kvp_size}): "
                "page owners must span the whole query shard group"
            )
    if dcp_size > 1:
        if disaggregation_mode not in ("null", "decode"):
            raise ValueError(
                "--decode-context-parallel-size > 1 requires --disaggregation-mode "
                f"null or decode (got {disaggregation_mode!r})"
            )
        if dcp_size != kvp_size:
            raise ValueError(
                "--decode-context-parallel-size must equal --kv-parallel-size "
                f"(got dcp={dcp_size}, kvp={kvp_size}): every rank of the KVP "
                "group owns pages of every request, so all of them merge the "
                "partials"
            )
    elif kvp_size > 1 and disaggregation_mode in ("null", "decode"):
        raise ValueError(
            f"--kv-parallel-size > 1 on --disaggregation-mode {disaggregation_mode} requires "
            f"--decode-context-parallel-size {kvp_size}: decode over sharded "
            "pages is DCP, and the launch names it explicitly"
        )


def _require_choice(flag: str, value: str, choices: tuple[str, ...]) -> None:
    """Refuse a launch value outside ``flag``'s closed set of ``choices``.

    ServerArgs is the one place a closed-set flag is validated; consumers
    read the resolved value and trust it.
    """
    if value not in choices:
        raise ValueError(f"{flag} must be one of {list(choices)}, got {value!r}")


def validate_qcp(
    *,
    qcp_size: int,
    attn_tp_size: int,
    attn_dp_size: int,
    dense_tp_size: int,
    moe_tp_ep_size: int,
    enable_mixed_batch: bool,
) -> None:
    """Reject query-context-parallel layouts the first landing does not serve.

    QCP shards an extend forward's rows over the attention TP group; the
    decode arm serves only the drafter's steps, every row on every rank.
    These are the layout gaps that rule; the role and size relations live in
    ``validate_context_parallel_roles``, the shard width in ``Mapping``, and
    the backend, device and KV-cache capabilities in ``AttnConfig``, which
    sees them resolved. ``qcp_size == 1`` is off and passes.
    """
    if qcp_size == 1:
        return
    if attn_dp_size != 1:
        raise ValueError(
            "--query-context-parallel-size > 1 requires attention DP 1 (got "
            f"attn_dp_size={attn_dp_size}): the sampled-row table of a shard is "
            "per DP group and the DP metadata gather does not carry it"
        )
    if dense_tp_size not in (1, attn_tp_size) or moe_tp_ep_size not in (
        1,
        attn_tp_size,
    ):
        raise ValueError(
            "--query-context-parallel-size > 1 requires --dense-tp-size and the "
            f"MoE TP x EP group to be 1 or the attention TP width {attn_tp_size} "
            f"(got dense {dense_tp_size}, MoE {moe_tp_ep_size}): attention returns "
            "complete rows (head-replicated weights, or the head-TP tail), so the "
            "drafter's replicated decode rows are never scattered and a narrower "
            "dense or MoE group has no rows to gather"
        )
    if enable_mixed_batch:
        raise ValueError(
            "--query-context-parallel-size > 1 does not support "
            "--enable-mixed-batch: a MIXED round would carry sharded extend rows "
            "and replicated decode rows in one forward"
        )


def _filesystem_type(path: str) -> str:
    """The type of the filesystem whose mount point is ``path``'s longest
    prefix in ``/proc/self/mounts``; ``path`` need not exist yet."""
    path = os.path.realpath(path)
    best, fs_type = "", "unknown"
    with open("/proc/self/mounts") as mounts:
        for line in mounts:
            fields = line.split()
            mount_point = fields[1].replace("\\040", " ")
            inside = path == mount_point or path.startswith(
                mount_point.rstrip("/") + "/"
            )
            if inside and len(mount_point) >= len(best):
                best, fs_type = mount_point, fields[2]
    return fs_type


@dataclasses.dataclass
class ServerArgs(ServerArgsOptions):
    """Resolve server options into a validated engine configuration."""

    mapping: Mapping | None = None

    # Resolved in __post_init__; not options.
    _grammar_backend_unset: bool = dataclasses.field(
        default=False, init=False, repr=False
    )
    _beam_checkpoint: bool | None = dataclasses.field(
        default=False, init=False, repr=False
    )
    _beam_grammar_none_warned: bool = dataclasses.field(
        default=False, init=False, repr=False
    )
    _speculative_widths_explicit: bool = dataclasses.field(
        default=False, init=False, repr=False
    )

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
        self.validate_decode_extra_slots()
        self.validate_weight_memory_policy()
        self.validate_forward_reload()
        self.resolve_basic_defaults()
        self.resolve_model_defaults()
        self.resolve_launcher_topology()
        self.resolve_parallelism()
        self.resolve_memory_and_scheduling()
        self.resolve_kernel_backends()
        self.resolve_cache()
        self.resolve_speculative_decoding()
        self.resolve_communication()
        self.resolve_numerics()
        self.resolve_disaggregation()
        self.validate_output_cumulative_probs()
        self.validate_dp_local_cuda_graph()
        self.validate_dsv41_adaptive_dspark()
        self.validate()

    def validate_dp_local_cuda_graph(self):
        if not self.enable_dp_local_cuda_graph:
            return
        if self.disaggregation_mode in ("prefill", "encode"):
            raise ValueError("DP-local CUDA graphs require a decode-capable worker")
        if self.enforce_eager:
            raise ValueError("DP-local CUDA graphs require CUDA graphs")
        if self.speculative_algorithm != "DSPARK":
            raise ValueError("DP-local CUDA graphs currently require DSPARK")
        mega_moe_backends = ("mega_moe", "deep_gemm_mega_moe")
        if (
            self.moe_backend not in mega_moe_backends
            or (self.draft_moe_backend or self.moe_backend) not in mega_moe_backends
        ):
            raise ValueError(
                "DP-local CUDA graphs require MegaMoE for both target and draft"
            )
        if (
            self.mapping.attn.tp_size != 1
            or self.mapping.dense.tp_size != 1
            or self.mapping.moe.tp_size != 1
            or self.mapping.lm_head.tp_size != 1
            or self.mapping.attn.has_head_tp
            or self.pipeline_parallel_size != 1
            or self.kv_parallel_size != 1
            or self.decode_context_parallel_size != 1
            or self.query_context_parallel_size != 1
            or self.dp_sampling
        ):
            raise ValueError(
                "DP-local CUDA graphs require attention/dense/MoE/LM-head TP1, "
                "no head TP, PP1/KVP1/DCP1/QCP1 and no DP sampling"
            )
        if self.sampling_backend != "greedy" and not self.enable_dsv41_adaptive_dspark:
            raise ValueError(
                "DP-local CUDA graphs require greedy sampling "
                "or adaptive DSpark sampling"
            )

    def validate_dsv41_adaptive_dspark(self):
        if (
            type(self.dsv41_dspark_min_batch_size) is not int
            or self.dsv41_dspark_min_batch_size < 1
        ):
            raise ValueError("DSpark minimum batch size must be a positive integer")
        if not self.enable_dsv41_adaptive_dspark:
            return
        if self.enforce_eager:
            raise ValueError("adaptive DSpark requires CUDA graphs")
        if self.synthetic_acceptance_length is not None:
            raise ValueError("adaptive DSpark requires real acceptance")
        from tokenspeed.runtime.utils.env import envs

        if self.speculative_algorithm != "DSPARK" or self.sampling_backend not in (
            "greedy",
            "flashinfer",
        ):
            raise ValueError(
                "V4.1 adaptive verification requires DSPARK and greedy "
                "or flashinfer sampling"
            )
        if self.sampling_backend == "flashinfer" and (
            not self.enable_speculative_sampling
            or self.sampling_stream != "per-request"
            or self.mapping.attn.tp_size != 1
            or self.speculative_rejection_sample_method != "standard"
        ):
            raise ValueError(
                "sampled adaptive DSpark requires --enable-speculative-sampling, "
                "--sampling-stream per-request, standard rejection sampling "
                "and attention TP1"
            )
        if not self.disable_prefill_graph:
            raise ValueError(
                "V4.1 adaptive verification requires --disable-prefill-graph"
            )
        if (
            self.kv_parallel_size != 1
            or self.decode_context_parallel_size != 1
            or self.query_context_parallel_size != 1
            or self.pipeline_parallel_size != 1
            or self.dp_sampling
        ):
            raise ValueError(
                "V4.1 adaptive verification requires QCP1/KVP1/DCP1 and PP1"
            )
        if (
            self.mapping.attn.has_head_tp
            or self.mapping.lm_head.tp_size != self.mapping.attn.tp_size
        ):
            raise ValueError(
                "adaptive DSpark requires head and vocabulary groups to follow attention TP"
            )
        if self.enable_output_cumulative_probs:
            raise ValueError(
                "V4.1 adaptive verification does not support "
                "--enable-output-cumulative-probs"
            )
        if (
            (self.enable_speculative_sampling and self.sampling_backend == "greedy")
            or self.speculative_draft_tree
            or envs.TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN.get()
            or self.disable_sampling_tp_sync
        ):
            raise ValueError(
                "V4.1 adaptive verification requires chain sampling "
                "and TP synchronization"
            )

    def validate_output_cumulative_probs(self):
        from tokenspeed.runtime.utils.env import envs

        if not self.enable_output_cumulative_probs:
            return
        if not self.enable_output_logprobs:
            raise ValueError(
                "--enable-output-cumulative-probs requires --enable-output-logprobs"
            )
        # Synthetic acceptance needs no check here: resolve_speculative_decoding
        # already refuses it with --enable-output-logprobs.
        if envs.TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN.get():
            raise ValueError(
                "--enable-output-cumulative-probs does not support "
                "TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN"
            )
        if self.speculative_draft_tree:
            raise ValueError(
                "--enable-output-cumulative-probs currently supports chain "
                "verification only, without draft trees or EAGLE-Harness"
            )
        if self.dp_sampling or self.mapping.has_attn_dp:
            raise ValueError(
                "--enable-output-cumulative-probs currently requires DP1 without DP sampling"
            )
        if self.pipeline_parallel_size != 1 or self.disaggregation_mode != "null":
            raise ValueError(
                "--enable-output-cumulative-probs currently requires one pipeline stage "
                "and aggregated serving"
            )

    def resolve_basic_defaults(self):
        resolve_metrics_options(self)
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

        self._grammar_backend_unset = self.grammar_backend is None
        if self.grammar_backend is None:
            self.grammar_backend = "none"

    def resolve_model_defaults(self, download: bool = False):
        """Model defaults that must be known before any process starts.

        ModelConfig applies a model's other defaults in the scheduler, but TF32
        follows the environment the launch exports before it starts the
        scheduler (``engine._set_envs_and_config``), so the checkpoint is read
        here (``_checkpoint_architectures``). Beam serves without TF32: its FP32
        router hands rows past the decode GEMV leaf's 16 to a Torch FP32
        matmul, which TF32 would run with a 10-bit mantissa. Without the
        network, ``_beam_checkpoint`` is None for a hub id the cache does not
        hold yet; engine setup calls this again with ``download``.
        """
        architectures = _checkpoint_architectures(self.model, self.revision, download)
        if (
            (
                self.owned_attention_output != "reduce-scatter"
                or self.prefill_collective_payload != "bf16"
            )
            and architectures is not None
            and "MuseSparkForConditionalGeneration" not in architectures
        ):
            raise ValueError(
                "--prefill-collective-payload and --owned-attention-output apply "
                "to Muse Spark's owned-row prefill only; this checkpoint is "
                f"{architectures}"
            )
        if self.decode_monokernel and architectures is not None:
            refusal = decode_monokernel_refusal(architectures)
            if refusal is not None:
                raise ValueError(refusal)
        if self.dense_fp8_small_m != "none" and architectures is not None:
            refusal = dense_fp8_small_m_refusal(architectures)
            if refusal is not None:
                raise ValueError(
                    f"--dense-fp8-small-m {self.dense_fp8_small_m}: {refusal}"
                )
        self._beam_checkpoint = (
            None if architectures is None else "BeamForCausalLM" in architectures
        )
        if self._beam_checkpoint and not self.disable_tf32:
            self.disable_tf32 = True
            logger.info(
                "Beam default disable_tf32=True: the FP32 router runs without TF32."
            )
        if (
            self._beam_checkpoint
            and self._grammar_backend_unset
            and self.grammar_backend == "none"
        ):
            self.grammar_backend = "xgrammar"
            logger.info("Beam default grammar_backend=xgrammar.")
        elif (
            self._beam_checkpoint
            and not self._grammar_backend_unset
            and self.grammar_backend == "none"
            and not self._beam_grammar_none_warned
        ):
            self._beam_grammar_none_warned = True
            logger.warning(
                "Beam was explicitly launched with --grammar-backend none; "
                "constrained decoding requests will be rejected. This override is "
                "intended only for benchmarking."
            )

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

            if "synthetic_acceptance_length" in config:
                length = config["synthetic_acceptance_length"]
                if (
                    isinstance(length, bool)
                    or not isinstance(length, (int, float))
                    or not math.isfinite(length)
                ):
                    raise ValueError(
                        "synthetic_acceptance_length must be a finite number"
                    )
                if (
                    self.synthetic_acceptance_length is not None
                    and self.synthetic_acceptance_length != length
                ):
                    raise ValueError(
                        "--synthetic-acceptance-length conflicts with "
                        "synthetic_acceptance_length in --speculative-config"
                    )
                if self.synthetic_acceptance_length is None:
                    self.synthetic_acceptance_length = length

            if "rejection_sample_method" in config:
                rejection = config["rejection_sample_method"]
                if rejection not in REJECTION_SAMPLE_METHODS:
                    raise ValueError(
                        f"rejection_sample_method must be one of "
                        f"{REJECTION_SAMPLE_METHODS}, got {rejection!r}"
                    )
                if (
                    self.speculative_rejection_sample_method != "standard"
                    and self.speculative_rejection_sample_method != rejection
                ):
                    raise ValueError(
                        "--speculative-rejection-sample-method conflicts with "
                        "rejection_sample_method in --speculative-config"
                    )
                self.speculative_rejection_sample_method = rejection

            if "draft_sample_method" in config:
                draft_sample = config["draft_sample_method"]
                if draft_sample not in DRAFT_SAMPLE_METHODS:
                    raise ValueError(
                        f"draft_sample_method must be one of "
                        f"{DRAFT_SAMPLE_METHODS}, got {draft_sample!r}"
                    )
                if self.enable_speculative_sampling and draft_sample == "greedy":
                    raise ValueError(
                        "--enable-speculative-sampling conflicts with "
                        "draft_sample_method='greedy' in --speculative-config"
                    )
                if draft_sample == "probabilistic":
                    self.enable_speculative_sampling = True

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

        if self.disaggregation_decode_extra_slots is None:
            self.disaggregation_decode_extra_slots = (
                self.max_num_seqs // self.mapping.attn.dp_size
                if self.disaggregation_mode == "decode"
                else 0
            )

    def resolve_kernel_backends(self):
        _require_choice(
            "--dense-gemm-backend",
            self.dense_gemm_backend,
            ("auto", "trtllm_cutedsl", "deepgemm"),
        )
        # The numerics switches (docs/design/numerics.md) are closed sets.
        for flag, value, choices in (
            ("--sampling-stream", self.sampling_stream, SAMPLING_STREAMS),
            (
                "--yarn-ramp-mask-device",
                self.yarn_ramp_mask_device,
                YARN_RAMP_MASK_DEVICES,
            ),
            (
                "--yarn-frequency-device",
                self.yarn_frequency_device,
                YARN_FREQUENCY_DEVICES,
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
            # Choose the platform sampler that respects request-level parameters.
            if current_platform().is_nvidia:
                self.sampling_backend = "flashinfer"
            elif current_platform().is_npu:
                self.sampling_backend = "ascend_full"
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
            attn_qcp_size=self.query_context_parallel_size,
            dense_tp_size=dense_tp_size,
            dense_dp_size=dense_dp_size,
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size,
            moe_dp_size=moe_dp_size,
            moe_weight_sharing_size=self.moe_weight_sharing_size,
            vision_tp_size=vision_tp_size,
            vision_dp_size=vision_dp_size,
            pp_size=pp_size,
            pp_layer_partition=self.pp_layer_partition,
            nprocs_per_node=nprocs_per_node,
            nnodes=nnodes,
            base_gpu_id=self.base_gpu_id,
            gpu_id_step=self.gpu_id_step,
        )

        if self.moe_weight_sharing_size is not None:
            if self.load_format not in ("auto", "safetensors"):
                raise ValueError(
                    "Peer expert weights require auto or safetensors checkpoint loading"
                )
            if self.enable_memory_saver:
                raise ValueError("Peer expert weights do not support memory saver")
            if self.disaggregation_mode != "prefill":
                raise ValueError(
                    "Peer expert weights currently require a prefill instance"
                )
            if self.mapping.requires_forward_coordination:
                raise ValueError(
                    "Peer expert weights require fully local execution groups"
                )
            if self.kvstore_storage_backend is not None:
                raise ValueError(
                    "Peer expert weights do not support shared L3 cache control"
                )
            if self.enable_eplb:
                raise ValueError(
                    "Peer expert weights are immutable and do not support expert rebalancing"
                )

        # Impl constraints:
        if self.deepseek_v41_attention_query_swa_min_tokens < 1:
            raise ValueError(
                "--deepseek-v41-attention-query-swa-min-tokens must be positive"
            )
        if self.deepseek_v41_attention_query_min_tokens < 1:
            raise ValueError(
                "--deepseek-v41-attention-query-min-tokens must be positive"
            )
        if self.deepseek_v41_attention_query_parallel_size not in (1, attn_tp_size):
            raise ValueError(
                "--deepseek-v41-attention-query-parallel-size must be 1 or the "
                f"attention TP size {attn_tp_size}"
            )
        if self.deepseek_v41_indexer_query_parallel_size not in (1, attn_tp_size):
            raise ValueError(
                "--deepseek-v41-indexer-query-parallel-size must be 1 or the "
                f"attention TP size {attn_tp_size}"
            )
        validate_context_parallel_roles(
            kvp_size=self.mapping.attn.kvp_size,
            dcp_size=self.decode_context_parallel_size,
            qcp_size=self.mapping.attn.qcp_size,
            disaggregation_mode=self.disaggregation_mode,
        )
        validate_qcp(
            qcp_size=self.mapping.attn.qcp_size,
            attn_tp_size=self.mapping.attn.tp_size,
            attn_dp_size=self.mapping.attn.dp_size,
            dense_tp_size=self.mapping.dense.tp_size,
            moe_tp_ep_size=self.mapping.moe.tp_ep_size,
            enable_mixed_batch=self.enable_mixed_batch,
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
        if self.head_tp_exchange_max_rows < 1:
            raise ValueError("--head-tp-exchange-max-rows must be positive")
        attn = self.mapping.attn
        if attn.head_tp_serves_decode_only:
            # Head TP over attention-DP ranks serves absorbed decode rows
            # only: an expanded prefill would need every head's K/V for the
            # cached prefix, which the head-sharded kv_b_proj cannot produce.
            # Over the query shards (attn.has_qcp) the extend rows run the
            # absorbed sparse prefill through the exchange;
            # validate_context_parallel_roles pinned the prefill role and the
            # GPU DSA backend declares its prefill eager under a shard.
            if self.disaggregation_mode != "decode":
                raise ValueError(
                    "--attn-head-tp-size > 1 serves decode rows only and "
                    "requires --disaggregation-mode decode (or, on the prefill "
                    "role, --query-context-parallel-size equal to it)"
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
                    "; under --query-context-parallel-size the dense group is 1 "
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
        self.resolve_kv_offload()
        # Handle KVStore settings.
        self._handle_kvstore()
        self.validate_cache_options()
        self.validate_startup_requests()
        self.resolve_retraction_snapshot_options()

    @property
    def retracting_role(self) -> bool:
        """Whether this engine can retract at all: the fused and decode roles."""
        return self.disaggregation_mode not in ("prefill", "encode")

    @property
    def has_host_tier(self) -> bool:
        """Whether this engine's scheduler has a Host tier: the local Host L2,
        or a shared L2 pool in its place."""
        return self.enable_kvstore or self.shared_l2_config is not None

    @property
    def writes_state_checkpoints_in_forward(self) -> bool:
        """Whether this engine's prefill forwards write in-forward State checkpoints.

        This is the flag, except on a decode server, whose prompts arrive by
        transfer. Its scheduler then also skips the dead prompt-end and finish
        checkpoints and demotes superseded in-forward checkpoints; a decode
        server publishes the former as ever.
        """
        return (
            self.enable_state_checkpoint_in_forward
            and self.disaggregation_mode != "decode"
        )

    @property
    def drops_short_prompt_end_checkpoints(self) -> bool:
        """Whether this engine's scheduler drops short prompts' end checkpoints.

        A prefill server does where it writes in-forward checkpoints; no other
        role does (a fused server keeps its short prompt ends).
        """
        return (
            self.writes_state_checkpoints_in_forward
            and self.disaggregation_mode == "prefill"
        )

    @property
    def scheduler_prefill_chunk_yield_tokens(self) -> int:
        """The tokens this engine's prefill chunks give up for whole prompts
        waiting behind them: the flag on a prefill server, 0 on a decode
        server, which computes no prompt (validate_cache_options refuses the
        flag on every other server)."""
        if self.disaggregation_mode != "prefill":
            return 0
        return self.prefill_chunk_yield_tokens

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
        derives to rank-local resident capacity C: max-num-seqs over the
        attention-DP ranks plus decode extra slots. Explicit positive caps
        are preserved; snapshot request rows and Host memory are separate limits.
        The prefill and encode roles never retract: their knobs are resolved
        to no pool here with a log, as the KVStore's are off on the encode
        role.
        """
        forced_cap = self.debug_force_retraction_max_per_request
        if forced_cap is not None:
            if type(forced_cap) is not int or not 1 <= forced_cap <= 2**31 - 1:
                raise ValueError(
                    "--debug-force-retraction-max-per-request must be an integer in [1, INT32_MAX]"
                )
            if self.debug_force_retraction_interval == 0:
                raise ValueError(
                    "--debug-force-retraction-max-per-request requires a nonzero "
                    "--debug-force-retraction-interval"
                )
            if not self.retracting_role:
                raise ValueError(
                    "--debug-force-retraction-max-per-request requires a role that can retract"
                )
        host_gb = self.retraction_snapshot_host_gb
        ratio = self.retraction_snapshot_ratio
        max_requests = self.retraction_snapshot_max_requests
        if host_gb < 0 or max_requests < 0 or (ratio is not None and ratio < 0):
            raise ValueError(
                "--retraction-snapshot-host-gb, --retraction-snapshot-ratio and "
                "--retraction-snapshot-max-requests must be non-negative"
            )
        if self.kv_offload_config is not None:
            if host_gb > 0 or ratio not in (None, 0) or max_requests > 0:
                raise ValueError(
                    "KV offloading aborts requests under decode memory pressure; "
                    "retraction snapshot pools are unsupported. Remove the "
                    "--retraction-snapshot-host-gb, --retraction-snapshot-ratio "
                    "and --retraction-snapshot-max-requests overrides "
                    "(a ratio of 0 is allowed)."
                )
            if self.debug_force_retraction_interval != 0:
                raise ValueError(
                    "KV offloading has no recovery images; "
                    "--debug-force-retraction-interval is unsupported."
                )
            # Reuse the scheduler's no-pool capacity abort path. Host history
            # and fixed hot partitions follow the ordinary request ownership.
            self.retraction_snapshot_host_gb = 0.0
            self.retraction_snapshot_ratio = 0.0
            self.retraction_snapshot_max_requests = 0
            return
        if self.shared_l2_config is not None and self.retracting_role:
            # A retraction image's Host part would go to the pool's proxy
            # tier, which holds no bytes.
            forced = self.debug_force_retraction_interval != 0
            if host_gb > 0 or ratio not in (None, 0) or max_requests > 0 or forced:
                raise ValueError(
                    "a rank with --shared-l2-config does not retract; remove the "
                    "--retraction-snapshot-host-gb, --retraction-snapshot-ratio, "
                    "--retraction-snapshot-max-requests and "
                    "--debug-force-retraction-interval overrides (a ratio of 0 is "
                    "allowed)."
                )
            self.retraction_snapshot_host_gb = 0.0
            self.retraction_snapshot_ratio = 0.0
            self.retraction_snapshot_max_requests = 0
            return
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
            # Extra slots are already rank-local; only the execution limit
            # is divided by attention DP. __post_init__ resolves both first.
            if (
                self.mapping is None
                or self.max_num_seqs is None
                or self.disaggregation_decode_extra_slots is None
            ):
                raise ValueError(
                    "resolve_parallelism() and resolve_memory_and_scheduling() must "
                    "run before resolve_cache(): the derived "
                    "--retraction-snapshot-max-requests uses rank-local "
                    "execution capacity plus resolved decode extra slots"
                )
            dp_size = max(self.mapping.attn.dp_size, 1)
            self.retraction_snapshot_max_requests = (
                self.max_num_seqs // dp_size + self.disaggregation_decode_extra_slots
            )
            if self.retraction_snapshot_max_requests == 0:
                raise ValueError(
                    "the retraction snapshot pool derives its request cap from "
                    f"--max-num-seqs ({self.max_num_seqs}) per attention-DP rank "
                    f"({dp_size}) plus --disaggregation-decode-extra-slots "
                    f"({self.disaggregation_decode_extra_slots}), which is 0 here; "
                    "raise --max-num-seqs to at "
                    "least the attention-DP size or pass "
                    "--retraction-snapshot-max-requests"
                )

    def resolve_kv_offload(self):
        """Validate deployment options without model-specific cache geometry."""
        options = self.kv_offload_config
        if options is None:
            return
        if isinstance(options, str):
            try:
                options = json.loads(options)
            except json.JSONDecodeError as exc:
                raise ValueError("--kv-offload-config must be valid JSON") from exc
        required = {"layers", "hot_tokens", "host_gb", "overlap"}
        if not isinstance(options, dict) or set(options) != required:
            raise ValueError(
                f"--kv-offload-config requires a JSON object with exactly {sorted(required)}"
            )
        layers = options["layers"]
        if (
            not isinstance(layers, list)
            or not layers
            or any(type(layer) is not int or layer < 0 for layer in layers)
            or len(set(layers)) != len(layers)
        ):
            raise ValueError("offload layers must be unique nonnegative integer IDs")
        hot = options["hot_tokens"]
        if type(hot) is not int or hot <= 0 or hot & (hot - 1):
            raise ValueError("offload hot_tokens must be a positive power of two")
        host = options["host_gb"]
        if (
            type(host) not in (int, float)
            or not math.isfinite(host)
            or host <= 0
            or int(host * (1 << 30)) <= 0
        ):
            raise ValueError("offload host_gb must be a finite positive GiB budget")
        if type(options["overlap"]) is not bool:
            raise ValueError("offload overlap must be an explicit bool")
        self.kv_offload_config = dict(options, layers=list(layers))

    def resolve_speculative_decoding(self):
        if self.speculative_algorithm == "DUMMY":
            if (
                self.speculative_draft_model_path is not None
                or self.draft_model_path_use_base
            ):
                raise ValueError("DUMMY drafting does not accept a draft model path")
            if int(self.speculative_num_draft_tokens) <= 1:
                raise ValueError(
                    "DUMMY drafting requires --speculative-num-draft-tokens > 1"
                )
        if self.speculative_rejection_sample_method not in REJECTION_SAMPLE_METHODS:
            raise ValueError(
                f"--speculative-rejection-sample-method must be one of "
                f"{REJECTION_SAMPLE_METHODS}, got "
                f"{self.speculative_rejection_sample_method!r}"
            )
        if self.synthetic_acceptance_length is not None:
            # A synthetic length selects synthetic rejection, as on main.
            if self.speculative_rejection_sample_method == "block":
                raise ValueError(
                    "--synthetic-acceptance-length selects the synthetic rejection "
                    "method; it cannot be combined with "
                    "--speculative-rejection-sample-method block"
                )
            self.speculative_rejection_sample_method = "synthetic"
        elif self.speculative_rejection_sample_method == "synthetic":
            raise ValueError(
                "--speculative-rejection-sample-method synthetic requires "
                "--synthetic-acceptance-length"
            )
        if self.synthetic_acceptance_length is not None:
            length = self.synthetic_acceptance_length
            if (
                isinstance(length, bool)
                or not isinstance(length, (int, float))
                or not math.isfinite(length)
            ):
                raise ValueError("synthetic_acceptance_length must be a finite number")
            if self.speculative_algorithm is None:
                raise ValueError(
                    "--synthetic-acceptance-length requires speculative decoding"
                )
            # Block checkpoints may replace the default width in ModelConfig.
            # SamplingBackend checks the final width before allocating buffers.
            checkpoint_sets_width = (
                self.speculative_algorithm in BLOCK_SPEC_ALGORITHMS
                and not self._speculative_widths_explicit
            )
            if length < 1 or (
                not checkpoint_sets_width and length > self.speculative_num_draft_tokens
            ):
                raise ValueError(
                    "synthetic_acceptance_length must be in "
                    f"[1, {self.speculative_num_draft_tokens}] "
                    "(the speculative verify width including the target token)"
                )
            # With sampled drafts (vLLM's draft_sample_method=probabilistic) the
            # drafter stays on the path and only the verify outcome is forced,
            # so it uses the flashinfer verifier; sonic forces either kind.
            if self.sampling_backend not in ("greedy", "sonic") and not (
                self.enable_speculative_sampling
                and self.sampling_backend == "flashinfer"
            ):
                raise ValueError(
                    "--synthetic-acceptance-length requires --sampling-backend greedy "
                    "or sonic, or --sampling-backend flashinfer with "
                    "--enable-speculative-sampling"
                )
            if self.speculative_eagle_topk != 1:
                raise ValueError(
                    "--synthetic-acceptance-length does not support draft trees"
                )
            if self.dp_sampling or self.enable_output_logprobs:
                raise ValueError(
                    "--synthetic-acceptance-length does not support "
                    "--dp-sampling or --enable-output-logprobs"
                )
            logger.warning(
                f"Synthetic acceptance length {length:.4f} enabled for benchmarking. "
                "Generated text is synthetic and must not be used for correctness "
                "or accuracy evaluation."
            )

        if self.min_kv_split is not None:
            if type(self.min_kv_split) is not int or not 1 <= self.min_kv_split <= 256:
                raise ValueError("min_kv_split must be an integer in [1, 256]")
            if self.attention_backend != "tokenspeed_mla" or (
                self.speculative_algorithm is not None
                and self.drafter_attention_backend not in (None, "tokenspeed_mla")
            ):
                raise ValueError(
                    "--min-kv-split requires tokenspeed_mla for target and draft attention"
                )

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

        draft_revision = self.speculative_draft_model_revision
        if draft_revision is not None and self.speculative_draft_model_path is None:
            raise ValueError(
                "--speculative-draft-model-revision applies only to a draft in its "
                "own repository, but no draft model is set. Pass "
                f"--speculative-draft-model-path (got {draft_revision!r})."
            )
        if draft_revision is not None and self.draft_model_path_use_base:
            raise ValueError(
                "--speculative-draft-model-revision applies only to a draft in its "
                "own repository, but this draft is the target checkpoint. Set "
                f"--revision instead (got {draft_revision!r})."
            )

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

        validate_eagle_harness_args(self)
        if self.speculative_draft_tree:
            self._validate_tree_speculation()
        elif (
            self.speculative_algorithm in STEP_SPEC_ALGORITHMS
            and self.speculative_num_draft_tokens != self.speculative_num_steps + 1
        ):
            raise ValueError(
                f"a draft chain verifies speculative_num_steps + 1 = "
                f"{self.speculative_num_steps + 1} tokens, got "
                f"speculative_num_draft_tokens={self.speculative_num_draft_tokens}"
            )

        # Block verification, and synthetic rejection over sampled drafts, run
        # the sampled-chain verify on the flashinfer backend (block also on
        # triton); greedy and sonic force the synthetic length over their own
        # verify.
        sampled_chain_verify = self.speculative_rejection_sample_method == "block" or (
            self.speculative_rejection_sample_method == "synthetic"
            and self.sampling_backend not in ("greedy", "sonic")
        )
        if sampled_chain_verify:
            if (
                not self.enable_speculative_sampling
                or self.speculative_algorithm not in BLOCK_REJECTION_ALGORITHMS
                or self.speculative_eagle_topk != 1
                or not (
                    self.sampling_backend == "flashinfer"
                    or self._block_verifies_on_triton
                )
                or self.dp_sampling
                or self.mapping.has_attn_dp
                or self.pipeline_parallel_size != 1
                or self.disaggregation_mode != "null"
            ):
                raise ValueError(
                    f"--speculative-rejection-sample-method "
                    f"{self.speculative_rejection_sample_method} with sampled drafts "
                    "requires a DSPARK or EAGLE3 chain "
                    "(--speculative-eagle-topk 1), --enable-speculative-sampling, "
                    "the flashinfer sampling backend (or triton for block), and "
                    "aggregate TP without attention DP, DP sampling, pipeline "
                    "parallelism, or PD disaggregation"
                )
            if self.speculative_rejection_sample_method == "synthetic":
                # vLLM's draft_sample_method=probabilistic with
                # rejection_sample_method=synthetic: drafts stay sampled, the
                # standard draft-probability verify runs for its cost, and the
                # synthetic length replaces its outcome.
                logger.warning(
                    "Synthetic rejection over sampled drafts: drafts are sampled "
                    "and recorded, but the accepted length is forced and the "
                    "verify outcome is discarded."
                )
        if self.enable_speculative_sampling:
            self._validate_speculative_sampling()

    @property
    def _block_verifies_on_triton(self) -> bool:
        """Block verification on the triton backend, which reads the dense q."""
        return (
            self.speculative_rejection_sample_method == "block"
            and self.sampling_backend == "triton"
        )

    @property
    def eagle_harness(self) -> bool:
        """EAGLE-4 drafting through the EAGLE-Harness dynamic trees."""
        return (
            self.speculative_algorithm == "EAGLE4" and self.speculative_exec_mode >= 0
        )

    @property
    def speculative_draft_tree(self) -> bool:
        """Whether drafts go through the draft-tree path: ``topk > 1``, or any
        EAGLE-Harness round (a ``chain_intensity`` chain is a one-child tree)."""
        return self.speculative_algorithm is not None and (
            self.speculative_eagle_topk != 1 or self.eagle_harness
        )

    def _validate_tree_speculation(self) -> None:
        """Draft trees: EAGLE3/EAGLE4/MTP with a node budget the draft can fill and a mask word can hold."""
        topk = self.speculative_eagle_topk
        steps = self.speculative_num_steps
        nodes = self.speculative_num_draft_tokens
        if self.speculative_algorithm not in STEP_SPEC_ALGORITHMS:
            raise ValueError(
                f"speculative_eagle_topk={topk} (tree drafting) needs "
                f"--speculative-algorithm EAGLE3, EAGLE4 or MTP, got {self.speculative_algorithm}"
            )
        # EAGLE-Harness trees have their own limits (eagle_harness/config.py).
        if not self.eagle_harness and (not 1 <= topk <= 16 or not 1 <= steps <= 10):
            raise ValueError(
                f"tree drafting needs 1..16 children per node and 1..10 steps: {topk=}, {steps=}"
            )
        if not self.eagle_harness and (steps - 1) * topk > nodes:
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
        its own logits (the Eagle family, the multi-depth MTP drafter, and
        DSpark's Markov steps) and the verifier must be a backend that runs
        the draft-prob chain kernel:
        ``greedy`` verifies by exact match, so a dense q never enters it, and
        ``triton`` by a target-sampled exact match unless it runs block
        verification, which reads the dense q. DFLASH is admitted with either
        ``triton`` (a DFlash2 draft's sparse q) or a draft-prob backend (a base
        draft's dense q); only the loaded checkpoint says which, so the drafter
        refuses the wrong pairing when it binds the verifier. The prefill role of a
        disaggregated deployment never verifies a chain of its own and its
        drafted candidates ship to the decode node without q, so there the
        flag would only allocate the per-slot distribution buffer; it is
        refused. The sentinel threshold is validated here once for every
        layer below: at least 1.0 so no real probability reads as the
        sentinel, and at most 2**20 so the fp32 sentinel ``threshold + 1.0``
        stays distinguishable from it.
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
                "--speculative-algorithm EAGLE3, MTP, DSPARK, or DFLASH"
            )
        # EAGLE-Harness samples its trees (validate_eagle_harness_args).
        if self.speculative_eagle_topk != 1 and not self.eagle_harness:
            raise ValueError(
                "--enable-speculative-sampling supports only the topk=1 chain: "
                f"{self.speculative_eagle_topk=}"
            )
        if self.speculative_algorithm == "DFLASH":
            if (
                self.sampling_backend != DFLASH_SPECULATIVE_SAMPLING_BACKEND
                and self.sampling_backend not in SPECULATIVE_SAMPLING_BACKENDS
            ):
                raise ValueError(
                    "--enable-speculative-sampling with DFLASH needs "
                    f"--sampling-backend {DFLASH_SPECULATIVE_SAMPLING_BACKEND} "
                    f"(DFlash2) or one of {sorted(SPECULATIVE_SAMPLING_BACKENDS)} "
                    f"(base draft); got --sampling-backend {self.sampling_backend}"
                )
        elif (
            self.sampling_backend not in SPECULATIVE_SAMPLING_BACKENDS
            and not self._block_verifies_on_triton
        ):
            if self.sampling_backend == "greedy":
                why = "verifies by exact match, so the draft distribution never enters"
            elif self.sampling_backend in ("triton", "triton_full"):
                why = (
                    "verifies by target-sampled exact match and has no dense "
                    "draft-prob rejection kernel (triton reads only the DFlash2 "
                    "selector's sparse q, under DFLASH, and the dense q under "
                    "--speculative-rejection-sample-method block)"
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
        # Auto-enable allreduce fusion on supported single-node TP configurations,
        # and for Beam on Hopper or newer at any node count: across nodes Beam's
        # boundaries fuse only where the attention TP group has an MNNVL
        # workspace and keep the unfused chain otherwise (beam_sandwich_fusion.py).
        platform = current_platform()
        if (
            not self.enable_allreduce_fusion
            and not self.emulate_rank_zero
            and (current_platform().is_hopper_plus or platform.is_amd)
            and (
                self.mapping.nnodes == 1
                or (self._beam_checkpoint and platform.is_hopper_plus)
            )
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

        if (
            self.all2all_backend == "deepep"
            and self.deepep_mode == "normal"
            and not self.enforce_eager
        ):
            # A decode graph captures the MoE's dispatch and combine. Only the
            # low-latency legs replay; the normal dispatch reads its
            # per-expert receive counts on the host, which a graph cannot
            # capture (the prefill graph is off under DeepEP for the same
            # reason).
            self.enforce_eager = True
            logger.info("CUDA graph is disabled under --deepep-mode normal")

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
        from tokenspeed.runtime.plugins import ensure_loaded

        ensure_loaded()
        _require_choice("--numerics", self.numerics, NUMERICS_ENVELOPES)
        _require_choice(
            "--owned-attention-output",
            self.owned_attention_output,
            OWNED_ATTENTION_OUTPUTS,
        )
        if (
            self.numerics == "rl-bitwise"
            and self.owned_attention_output != "reduce-scatter"
        ):
            # It moves the attention output's sum to another order; the
            # envelope promises the trainer's bits.
            raise ValueError(
                "--owned-attention-output all-to-all changes prefill numerics; "
                "--numerics rl-bitwise needs reduce-scatter"
            )
        _require_choice(
            "--prefill-collective-payload",
            self.prefill_collective_payload,
            PREFILL_COLLECTIVE_PAYLOADS,
        )
        if self.numerics == "rl-bitwise" and self.prefill_collective_payload != "bf16":
            # The envelope promises the trainer's bits; the INT8 payload rounds
            # every prefill collective's inputs.
            raise ValueError(
                "--prefill-collective-payload int8-block16 changes prefill "
                "numerics; --numerics rl-bitwise needs bf16"
            )
        resolve_numerics_envelope(self)
        _require_choice(
            "--dense-fp8-small-m",
            self.dense_fp8_small_m,
            ("none", "qkv"),
        )
        if self.dense_fp8_small_m != "none" and (
            self.numerics in BITWISE_ENVELOPES or self.batch_invariant_collectives
        ):
            # Only batches inside the FP8 row windows project in FP8; other
            # batches stay BF16, so a row's bits depend on the batch's row count.
            # Run-to-run determinism (--force-deterministic-rsag) still holds
            # and is not refused.
            raise ValueError(
                f"--dense-fp8-small-m {self.dense_fp8_small_m} is not "
                f"batch-invariant: it cannot run under --numerics {self.numerics} "
                f"or --batch-invariant-collectives"
            )
        if self.batch_invariant_collectives:
            # The fused all-reduce+norm sums off the fold; some layers fuse by token count alone.
            self.enable_allreduce_fusion = False
            self.comm_fusion_max_num_tokens = -1
        # Individual switches that veto a fusion resolve_communication may
        # have auto-enabled, whatever the envelope. CommManager.should_fuse
        # re-derives the veto from the switches, so the fused kernels stay off
        # even where this flag is read before the fold.
        if self.layer_boundary_norm == "unfused":
            # The fused all-reduce+norm kernels add the residual inside the
            # fusion; the unfused boundary norm needs the bf16 sum first.
            self.enable_allreduce_fusion = False
            self.comm_fusion_max_num_tokens = -1
        if self.moe_combine_order == "slot":
            # The MoE leaf returns complete rows; a fused all-reduce+norm at
            # the next layer boundary would sum them tp_size times.
            self.enable_allreduce_fusion = False
            # The slot-order fold runs over the EP group inside the MoE leaf:
            # a K-split (MoE TP) down projection would need a second,
            # rank-ordered fold after it. Under --all2all-backend deepep the
            # leaf must itself own the exchange and fold the routes in slot
            # order: the MoE plan selects only a kernel declaring both traits
            # and refuses the launch otherwise.
            if self.mapping.moe.tp_size != 1:
                raise ValueError(
                    "--moe-combine-order slot needs MoE TP 1: a K-split down "
                    "projection would need a second, rank-ordered fold after "
                    f"the slot-order one (got --moe-tp-size {self.mapping.moe.tp_size})"
                )

    def _resolve_rl_bitwise(self):
        """Resolve the built-in RL policy; plugins provide their own resolver."""
        # Collectives: one association order per reduction. NCCL's ring
        # chunks by message size, so a plain NCCL sum is run-stable but not
        # batch-size-invariant; batch_invariant_collectives routes every
        # reduction to the rank-ordered fp32 fold (comm_backend/auto.py).
        # Gathers move bytes and keep the multicast kernels;
        # force_deterministic_rsag, which also moves them to NCCL, stays the
        # user's knob and the envelope does not set it.
        self.batch_invariant_collectives = True
        self.enable_allreduce_fusion = False
        self.comm_fusion_max_num_tokens = -1
        # Kernels: heuristic tactics only (autotune picks shape-dependent
        # tactics), no TF32, and no programmatic dependent launches.
        self.disable_autotune = True
        self.disable_tf32 = True
        self.disable_pdl = True
        self.moe_cutlass_fused_finalize = False
        # MoE keeps auto or the user's provider restriction. The expert
        # plan checks the batch_invariant feature against actual layer geometry.
        # The NPU alignment contract compares model logits for the same history.
        # Serving keeps its configured sampler and random stream.
        if self.device != "npu":
            if self.sampling_backend not in RL_BITWISE_SAMPLING_BACKENDS:
                raise ValueError(
                    f"--numerics {self.numerics} needs a sampling backend with "
                    "canonical greedy tie-breaking "
                    f"({sorted(RL_BITWISE_SAMPLING_BACKENDS)}); --sampling-backend "
                    f"{self.sampling_backend} resolves exact ties in reduction order"
                )
            self.sampling_stream = "per-request"
        # Keep the trainer's selection order. The envelope still requires
        # batch-invariant top-k and attention leaves.
        self.dsa_slot_order = "selection"
        # Trainer alignment: the training framework's operation order wherever
        # the two engines are known to differ. Each switch is documented in
        # ``docs/design/numerics.md`` under "alignment.trainer".
        # The trainer constructs the ramp on CPU, then computes powers,
        # divisions and interpolation on CUDA. Both choices affect rounding.
        self.yarn_ramp_mask_device = "cpu"
        self.yarn_frequency_device = "cuda"
        # The trainer multiplies the LoRA norm scales as separate bf16 ops.
        self.mla_lora_scale = "runtime"
        # The trainer materializes each layer's bf16 output before the next
        # layer's norm reads it.
        self.layer_boundary_norm = "unfused"
        # The trainer's router is softmax + topk(scores + bias) in torch.
        self.router_topk = "torch"
        # The trainer's logprobs come from its vocab-parallel cross-entropy.
        if self.logprob_order not in BLOCKWISE_LOGPROB_ORDERS:
            self.logprob_order = "blockwise"
        # The trainer's grouped MLP applies the router weight inside the
        # activation and folds a token's slots in fp32 slot order.
        self.moe_combine_order = "slot"
        # Layouts stay explicit: the envelope does not fold them in. A
        # head-sharded o_proj that still sums its head partials across ranks
        # is batch-invariant under the envelope, but its bits are not the
        # full-K GEMM a replicated or column-parallel o_proj computes. Every
        # batch-invariant reduction (the exchanging forward's reduce-scatter,
        # a TP-W engine's all-reduce) is the same rank-ordered fold, so it does
        # equal a TP-W engine's all-reduced o_proj; docs/design/numerics.md,
        # "Layout invariance of query context parallelism".
        if self.mapping.attn.has_head_tp and self.tp_batch_invariant == "none":
            logger.warning(
                "--numerics rl-bitwise with --attn-head-tp-size > 1 but without "
                "--tp-batch-invariant attn: the o_proj head partials are summed "
                "across ranks by the rank-ordered fold, which is batch-invariant "
                "and equals a TP-W engine's o_proj, but differs from the full-K "
                "o_proj of a TP1 or --tp-batch-invariant engine"
            )

    def resolve_disaggregation(self):
        # Pipeline parallelism is a prefill-node-only capability: the chunk
        # pipeline needs the P role's structural guarantees (no decode token
        # feedback, no overlap schedule).
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
        if self.enable_external_encoder and self.disaggregation_mode not in (
            "null",
            "prefill",
        ):
            raise ValueError(
                "--enable-external-encoder requires aggregated or prefill serving"
            )
        # Only the prefill role has a decode allocation policy. Omitted, it is
        # early: a prompt waits for its Decode destination before it is
        # admitted, as every prefill server did before the flag existed.
        if self.disaggregation_mode == "prefill":
            if self.disaggregation_decode_allocation_policy is None:
                self.disaggregation_decode_allocation_policy = "early"
            _require_choice(
                "--disaggregation-decode-allocation-policy",
                self.disaggregation_decode_allocation_policy,
                ("early", "deferred"),
            )
        elif self.disaggregation_decode_allocation_policy is not None:
            raise ValueError(
                "--disaggregation-decode-allocation-policy requires "
                "--disaggregation-mode prefill"
            )
        if self.disaggregation_decode_allocation_policy == "deferred":
            # Experimental: one attention rank per prompt, no external
            # encoder, and the KV sent from the computed prompt rather than
            # streamed during its forward. Refused, never rewritten; a
            # multimodal model is refused by the rank, which reads the model
            # (HostBootstrap).
            unsupported = [
                setting
                for setting, refused in (
                    ("attention TP > 1", self.mapping.attn.tp_size > 1),
                    ("--kv-parallel-size > 1", self.mapping.attn.kvp_size > 1),
                    ("--pipeline-parallel-size > 1", self.mapping.has_pp),
                    ("--enable-external-encoder", self.enable_external_encoder),
                    (
                        "--disaggregation-layerwise-interval other than 0",
                        self.disaggregation_layerwise_interval != 0,
                    ),
                )
                if refused
            ]
            if unsupported:
                raise ValueError(
                    "--disaggregation-decode-allocation-policy deferred does "
                    f"not support {', '.join(unsupported)}"
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

    def validate_startup_requests(self):
        if not self.enable_startup_requests:
            return
        # A disaggregated node's requests need a peer node, dropping the
        # startup prefixes would delete an L3 store's shared namespace, and it
        # would leave them in a shared L2 pool.
        if self.disaggregation_mode != "null":
            raise ValueError(
                "--enable-startup-requests is not supported with --disaggregation-mode"
            )
        if self.kvstore_storage_backend is not None:
            raise ValueError(
                "--enable-startup-requests is not supported with "
                "--kvstore-storage-backend"
            )
        if self.shared_l2_config is not None:
            raise ValueError(
                "--enable-startup-requests is not supported with --shared-l2-config"
            )

    def validate_peer_prefix_options(self):
        """``--peer-prefix-port`` and its settings: what peer prefix sharing
        runs with (``docs/rfcs/0004-peer-prefix-sharing.md``)."""
        if self.peer_prefix_port is None:
            if self.peer_prefix_landing_tier is not None:
                raise ValueError(
                    "--peer-prefix-landing-tier requires --peer-prefix-port"
                )
            return
        _require_choice(
            "--peer-prefix-landing-tier",
            self.peer_prefix_landing_tier,
            ("device", "host"),
        )
        unsupported = [
            setting
            for setting, refused in (
                ("--disable-kvstore (no Host L2)", not self.enable_kvstore),
                ("--disable-prefix-caching", not self.enable_prefix_caching),
                (
                    "--kvstore-storage-backend",
                    self.kvstore_storage_backend is not None,
                ),
                (
                    "--pipeline-parallel-size > 1",
                    self.pipeline_parallel_size > 1,
                ),
                (
                    "--query-context-parallel-size > 1",
                    self.query_context_parallel_size > 1,
                ),
                (
                    "--enable-memory-saver (Peer Mooncake MRs do not yet re-register across TorchMemorySaver HBM unmap/remap)",
                    self.enable_memory_saver,
                ),
            )
            if refused
        ]
        if unsupported:
            raise ValueError(
                "--peer-prefix-port is not supported with " + ", ".join(unsupported)
            )
        if not 1 <= self.peer_prefix_port <= 65535:
            raise ValueError(
                f"--peer-prefix-port must be in 1..65535; got {self.peer_prefix_port}"
            )
        if self.peer_prefix_answer_timeout_ms <= 0:
            raise ValueError("--peer-prefix-answer-timeout-ms must be positive")
        if self.peer_prefix_min_fetch_tokens <= 0:
            raise ValueError("--peer-prefix-min-fetch-tokens must be positive")

    def validate_disk_tier_options(self):
        """``--disk-tier-paths`` and its settings: what the Disk tier runs
        with (``docs/rfcs/0006-disk-tier.md``)."""
        if self.disk_tier_paths is None:
            for flag, value in (
                ("--disk-tier-io", self.disk_tier_io),
                ("--disk-tier-write-mbps", self.disk_tier_write_mbps),
                ("--disk-tier-config", self.disk_tier_config),
            ):
                if value is not None:
                    raise ValueError(f"{flag} requires --disk-tier-paths")
            return
        unsupported = [
            setting
            for setting, refused in (
                ("--disable-kvstore (no Host L2)", not self.enable_kvstore),
                (
                    "--kvstore-write-policy other than write_back",
                    self.kvstore_write_policy != "write_back",
                ),
                ("--disable-prefix-caching", not self.enable_prefix_caching),
                (
                    "--kvstore-storage-backend",
                    self.kvstore_storage_backend is not None,
                ),
                (
                    "--pipeline-parallel-size > 1",
                    self.pipeline_parallel_size > 1,
                ),
            )
            if refused
        ]
        if unsupported:
            raise ValueError(
                "--disk-tier-paths is not supported with " + ", ".join(unsupported)
            )
        if not self.disk_tier_paths:
            raise ValueError("--disk-tier-paths needs at least one directory")
        if self.disk_tier_size_gb <= 0:
            raise ValueError("--disk-tier-size-gb must be positive")
        _require_choice("--disk-tier-io", self.disk_tier_io, ("io_uring", "psync"))
        if self.disk_tier_write_mbps is None or self.disk_tier_write_mbps <= 0:
            raise ValueError("--disk-tier-write-mbps must be a positive budget")
        from tokenspeed.runtime.cache.disk import DiskTierTuning

        DiskTierTuning.from_json(self.disk_tier_config)

    def validate_cache_options(self):
        self.validate_peer_prefix_options()
        self.validate_disk_tier_options()
        if self.kvstore_storage_backend is not None and self.pipeline_parallel_size > 1:
            raise ValueError(
                "Asynchronous L3 currently requires --pipeline-parallel-size 1"
            )
        # Host tiers and L3 use the same cyclic owner translation. L3 page
        # keys name a replica, so a new Host allocation can have another owner.
        # Device construction validates that sharded payloads are replicated
        # within the owner subgroup; head-partitioned payloads are rejected.
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
        if self.writes_state_checkpoints_in_forward:
            if not self.enable_prefix_caching:
                raise ValueError(
                    "--enable-state-checkpoint-in-forward writes a checkpoint only "
                    "the prefix cache can reuse; it requires prefix caching"
                )
        if self.shared_l2_config is not None:
            from tokenspeed.runtime.cache.shared_l2.client import SharedL2ClientConfig

            SharedL2ClientConfig.from_json(self.shared_l2_config)
            # The pool takes the Host L2's place on prefill and fused ranks
            # of attention TP 1 (docs/configuration/server.md, "Shared L2
            # pool"). Any other sharded cache group is refused when the cache
            # is built.
            unsupported = [
                setting
                for setting, refused in (
                    ("the local Host L2 (pass --disable-kvstore)", self.enable_kvstore),
                    (
                        "--kvstore-storage-backend",
                        self.kvstore_storage_backend is not None,
                    ),
                    ("--kv-offload-config", self.kv_offload_config is not None),
                    (
                        "--disaggregation-mode decode or encode",
                        self.disaggregation_mode in ("decode", "encode"),
                    ),
                    ("attention TP > 1", self.mapping.attn.tp_size != 1),
                    (
                        "--kv-parallel-size > 1 (the first version of the pool "
                        "holds replicated KV only)",
                        self.kv_parallel_size > 1,
                    ),
                    ("--pipeline-parallel-size > 1", self.mapping.has_pp),
                    ("disabled prefix caching", not self.enable_prefix_caching),
                    # A sleep unmaps the Device cache, which the pool's client
                    # registered for RDMA once, and a wake maps other memory.
                    ("--enable-memory-saver", self.enable_memory_saver),
                )
                if refused
            ]
            if unsupported:
                raise ValueError(
                    f"--shared-l2-config does not support {', '.join(unsupported)}"
                )
        _require_choice(
            "--kvstore-write-policy",
            self.kvstore_write_policy,
            ("write_through", "write_back"),
        )
        if self.kvstore_write_policy == "write_back":
            # L3 staging would split what is written back; pipeline stages and query context
            # parallelism have not run write-back. Attention TP and
            # --kv-parallel-size need no refusal: a written-back segment
            # publishes once every rank of the replica has acknowledged it
            # (docs/design/scheduler.md section 3.1). More than 64 cache
            # groups are refused by SchedulerConfig.Validate, once the model's
            # cache groups exist.
            unsupported = [
                setting
                for setting, refused in (
                    # A pool is written when a page is published.
                    ("--shared-l2-config", self.shared_l2_config is not None),
                    ("--disable-kvstore (no Host L2)", not self.enable_kvstore),
                    (
                        "--kvstore-storage-backend",
                        self.kvstore_storage_backend is not None,
                    ),
                    (
                        "--query-context-parallel-size > 1",
                        self.query_context_parallel_size > 1,
                    ),
                    ("--pipeline-parallel-size > 1", self.mapping.has_pp),
                )
                if refused
            ]
            if unsupported:
                raise ValueError(
                    "--kvstore-write-policy write_back does not support "
                    f"{', '.join(unsupported)}"
                )
        # The yield belongs to the prefill scheduler, which never retracts; a
        # decode server takes the flag as a no-op, so a PD launcher may pass
        # it to both roles.
        if self.prefill_chunk_yield_tokens < 0:
            raise ValueError(
                "--prefill-chunk-yield-tokens must be non-negative; got "
                f"{self.prefill_chunk_yield_tokens}"
            )
        if self.prefill_chunk_yield_tokens > 0:
            if self.disaggregation_mode not in ("prefill", "decode"):
                raise ValueError(
                    "--prefill-chunk-yield-tokens is for prefill servers "
                    "(--disaggregation-mode prefill; a decode server ignores it)"
                )
            if (
                self.disaggregation_mode == "prefill"
                and self.prefill_chunk_yield_tokens > self.chunked_prefill_size // 2
            ):
                raise ValueError(
                    "--prefill-chunk-yield-tokens must be at most half of "
                    f"--chunked-prefill-size ({self.chunked_prefill_size}); got "
                    f"{self.prefill_chunk_yield_tokens}"
                )

    def validate_forward_reload(self):
        """A forward reload recaptures graphs, which tagged pools cannot, and spans one engine, which PD roles are not."""
        if not self.allow_reload_forward:
            return
        if self.enable_memory_saver:
            raise ValueError(
                "--allow-reload-forward cannot be combined with --enable-memory-saver: "
                "a tagged CUDA-graph pool cannot be discarded and captured again"
            )
        if self.disaggregation_mode != "null":
            raise ValueError(
                "--allow-reload-forward does not support prefill/decode disaggregation: "
                "each role would reload and resume on its own, mixing code versions in "
                "one request"
            )

    def validate_weight_memory_policy(self):
        """Memory saving chooses its weight semantics explicitly."""
        if self.weight_memory_policy not in (None, "cpu_backup", "discard"):
            raise ValueError("--weight-memory-policy must be cpu_backup or discard")
        if self.enable_memory_saver and self.weight_memory_policy is None:
            raise ValueError("--enable-memory-saver requires --weight-memory-policy")
        if self.weight_memory_policy == "discard":
            if not self.enable_memory_saver or self.model_update_config is None:
                raise ValueError(
                    "Weight discard requires memory saver and Model Updater configuration"
                )
            if self.model_update_draft_weights not in ("retain", "refresh"):
                raise ValueError(
                    "Weight discard requires an explicit draft update policy"
                )
            # Quantization is checked against the resolved model config by
            # WeightMemoryState; CLI names alone cannot prove safe regeneration.
            if self.enable_eplb:
                raise ValueError(
                    "Weight discard currently requires running without EPLB"
                )
            if self.dense_fp8_small_m != "none":
                # The FP8 weight copies are made once, after loading, and a
                # weight update refuses to start with them; discarding them
                # would leave nothing that rebuilds them.
                raise ValueError(
                    f"--dense-fp8-small-m {self.dense_fp8_small_m} cannot run "
                    "with --weight-memory-policy discard: its FP8 weight copies "
                    "are made once after loading, and no weight update rebuilds "
                    "them after a discard"
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
        if self.dense_fp8_small_m != "none":
            # Every weight update refuses to start while the FP8 copies exist
            # (MuseSparkForConditionalGeneration.begin_weight_update).
            raise ValueError(
                f"--dense-fp8-small-m {self.dense_fp8_small_m} cannot run with "
                "--model-update-config: its FP8 weight copies are made once "
                "after loading, and no weight update rebuilds them"
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
        self.validate_decode_extra_slots()
        if (
            type(self.ttft_trace_sampling_percentage) is not int
            or not 0 <= self.ttft_trace_sampling_percentage <= 100
        ):
            raise ValueError(
                "--ttft-trace-sampling-percentage must be an integer in [0, 100]"
            )
        self.validate_weight_memory_policy()
        self.validate_forward_reload()
        if self.low_latency_max_num_tokens_per_gpu <= 0:
            raise ValueError("--low-latency-max-num-tokens-per-gpu must be positive")
        if self.input_logprob_chunk_tokens <= 0:
            raise ValueError("--input-logprob-chunk-tokens must be positive")
        if self.chunk_prefill_kv_size is not None and self.chunk_prefill_kv_size <= 0:
            raise ValueError("--chunk-prefill-kv-size must be positive")
        if self.prefill_decode_interval < 0:
            raise ValueError("--prefill-decode-interval must be non-negative")
        if self.prefill_decode_interval and self.disaggregation_mode != "null":
            raise ValueError(
                "--prefill-decode-interval needs --disaggregation-mode null: a "
                "prefill engine runs no decodes and a decode engine no local "
                "prefill rounds"
            )
        if self.prefill_decode_min_interval < 0:
            raise ValueError("--prefill-decode-min-interval must be non-negative")
        if self.prefill_decode_min_interval and self.disaggregation_mode != "null":
            raise ValueError(
                "--prefill-decode-min-interval needs --disaggregation-mode null: "
                "a prefill engine runs no decodes and a decode engine no local "
                "prefill rounds"
            )
        if self.max_consecutive_prefill_rounds < 0:
            raise ValueError("--max-consecutive-prefill-rounds must be non-negative")
        if self.max_consecutive_prefill_rounds and not self.prefill_decode_interval:
            raise ValueError(
                "--max-consecutive-prefill-rounds needs --prefill-decode-interval: "
                "it bounds how long an overdue prompt suspends the interval"
            )
        if self.prefill_interval_min_decoding < 0:
            raise ValueError("--prefill-interval-min-decoding must be non-negative")
        if (
            self.prefill_interval_min_decoding
            and not self.max_consecutive_prefill_rounds
        ):
            raise ValueError(
                "--prefill-interval-min-decoding needs "
                "--max-consecutive-prefill-rounds: below the threshold the round "
                "cap is what still bounds how long running decodes stall"
            )
        if self.monokernel_stuck_record_dir is not None:
            fs_type = _filesystem_type(self.monokernel_stuck_record_dir)
            if fs_type != "tmpfs":
                # The record file is mapped and registered as GPU-visible host
                # memory. On a disk filesystem the kernel's periodic dirty-page
                # writeback of those pages stalls decode for 1-2 s every few
                # seconds; tmpfs pages are never written back.
                raise ValueError(
                    "--monokernel-stuck-record-dir must be on tmpfs (for "
                    f"example /dev/shm); {self.monokernel_stuck_record_dir} is "
                    f"on {fs_type}"
                )
        if self.monokernel_stuck_record_dir is not None and not self.decode_monokernel:
            raise ValueError(
                "--monokernel-stuck-record-dir needs --decode-monokernel: only "
                "the monokernel launches write stuck-poll records"
            )
        if self.inkling2_loop_num_passes is not None and (
            self.inkling2_loop_num_passes < 1
        ):
            raise ValueError(
                "--inkling2-loop-num-passes must be at least 1, got "
                f"{self.inkling2_loop_num_passes}"
            )
        if self.device == "npu":
            if not self.disable_prefill_graph:
                raise ValueError("NPU execution requires --disable-prefill-graph")
            if not self.disable_pdl:
                raise ValueError("NPU execution requires --disable-pdl")

        self.validate_petit_moe_options()
        self.validate_rank_emulation()
        self.validate_prefill_delay()

        if (
            self.max_num_seqs is not None
            and self.max_num_seqs < self.mapping.attn.dp_size
        ):
            raise ValueError(
                f"max_num_seqs must be >= attn_dp_size: {self.max_num_seqs=} < {self.mapping.attn.dp_size=}"
            )

        self.validate_model_update_options()

        if (
            self.speculative_algorithm is not None
            and self.disaggregation_mode != "prefill"
            and self.max_num_seqs is not None
            # Only FlashInfer's distributed tactic selection (NVIDIA) needs
            # every attention TP peer in lockstep; elsewhere startup tuning
            # runs no cross-rank collectives, and the draft warmup stays
            # within the request cap (ModelExecutor.autotune).
            and current_platform().is_nvidia
        ):
            dp_size = self.mapping.attn.dp_size
            tp_size = self.mapping.attn.tp_size
            # Draft startup uses one request per TP peer, even when tactic
            # profiling is disabled. Fewer would leave empty tuning peers.
            if self.max_num_seqs // dp_size < tp_size:
                raise ValueError(
                    "Speculative decode warmup requires max_num_seqs // "
                    "data_parallel_size >= attn_tp_size, got "
                    f"{self.max_num_seqs} // {dp_size} < {tp_size}. "
                    f"Set --max-num-seqs to at least {dp_size * tp_size}."
                )

        if self.quantization == "fp8_weight_only":
            # A draft-weight format: its GEMV is decode-shaped, so a target's
            # prefill rows would re-read every weight per 32-row chunk.
            raise ValueError(
                "fp8_weight_only quantizes draft weights only; pass it as "
                "--speculative-draft-model-quantization"
            )

        if self.mapping.has_attn_dp:
            if self.chunked_prefill_size > self.max_prefill_tokens:
                raise ValueError(
                    f"chunked_prefill_size must be <= max_prefill_tokens: {self.chunked_prefill_size=} > {self.max_prefill_tokens=}"
                )

        self.validate_expert_placement_options()

        if not self.disable_pdl:
            os.environ.setdefault("TORCHINDUCTOR_ENABLE_PDL", "1")
            # Enable PDL for fused attention kernels.
            os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
        os.environ.setdefault("TLLM_LOG_LEVEL", "INFO")

    def validate_decode_extra_slots(self) -> None:
        extra = self.disaggregation_decode_extra_slots
        if extra is None:
            return
        if type(extra) is not int or extra < 0:
            raise ValueError(
                "--disaggregation-decode-extra-slots must be a non-negative integer"
            )
        if extra and self.disaggregation_mode != "decode":
            raise ValueError(
                "--disaggregation-decode-extra-slots requires --disaggregation-mode decode"
            )

    def validate_prefill_delay(self) -> None:
        intervals = self.prefill_delay
        if intervals is None:
            return
        attempt, any_attempt, _ = intervals
        if attempt < 1:
            raise ValueError(f"--prefill-attempt-interval must be >= 1, got {attempt}")
        if any_attempt < attempt:
            raise ValueError(
                "--prefill-any-attempt-interval must be >= --prefill-attempt-interval "
                f"(got {any_attempt} < {attempt})"
            )
        if self.disaggregation_mode != "null":
            raise ValueError(
                "the prefill delay gate applies to a fused engine "
                "(--disaggregation-mode null) only"
            )
        for flag, interval in (
            ("--prefill-decode-interval", self.prefill_decode_interval),
            ("--prefill-decode-min-interval", self.prefill_decode_min_interval),
        ):
            if interval:
                raise ValueError(
                    f"{flag} cannot be combined with "
                    "--prefill-attempt-interval and --prefill-any-attempt-interval: "
                    "local decode intervals and the attention-DP gate use "
                    "different prefill windows"
                )

    @property
    def prefill_delay(self) -> tuple[int, int, bool] | None:
        """The gate's (attempt, any, single_round) settings, or None; reject
        an incompletely configured gate."""
        attempt = self.prefill_attempt_interval
        any_attempt = self.prefill_any_attempt_interval
        if attempt is None and any_attempt is None:
            if self.prefill_single_round:
                raise ValueError(
                    "--prefill-single-round requires --prefill-attempt-interval "
                    "and --prefill-any-attempt-interval"
                )
            return None
        if attempt is None or any_attempt is None:
            raise ValueError(
                "--prefill-attempt-interval and --prefill-any-attempt-interval "
                "must be set together"
            )
        return (attempt, any_attempt, self.prefill_single_round)

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        """Register the shared options with the installed numerics profiles."""
        from tokenspeed.runtime.plugins import ensure_loaded

        ensure_loaded()
        add_server_cli_args(parser, numerics_envelopes=NUMERICS_ENVELOPES)

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
        # for non-CLI-exposed fields.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(
            **{attr: getattr(args, attr) for attr in attrs if hasattr(args, attr)}
        )

    def url(self):
        if is_valid_ipv6_address(self.host):
            return f"http://[{self.host}]:{self.port}"
        return f"http://{self.host}:{self.port}"

    def zmq_handshake_endpoint(self) -> str:
        """The frontend handshake endpoint the engine ranks dial, composed from
        ``--data-parallel-address``/``--data-parallel-rpc-port`` (an IPv6
        address bracketed)."""
        address = join_host_port(
            self.data_parallel_address, self.data_parallel_rpc_port
        )
        return f"tcp://{address}"


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
    # Reserved in the port cluster; no process binds it.
    tokenizer_ipc_name: str
    # Reserved in the port cluster; no process binds it.
    scheduler_input_ipc_name: str

    # The port for nccl initialization (torch.dist)
    nccl_port: int

    # The resolved rendezvous address after moving past busy local ports.
    dist_init_addr: str

    # Reserved in the port cluster; no process binds it.
    rpc_ipc_name: str

    # The ipc filename for Scheduler to send metrics
    metrics_ipc_name: str

    # The ipc filename for Tokenizer and worker tokenizer
    tokenizer_worker_ipc_name: str | None

    # Plain producer settings; live output resources remain in their owner process.
    metrics_producer_configs: dict[str, dict[str, object]] = dataclasses.field(
        default_factory=dict, repr=False
    )

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
            try:
                dist_init_addr = split_host_port(server_args.dist_init_addr)
            except ValueError:
                raise ValueError(
                    "please provide --dist-init-addr as host:port (or [ipv6]:port) "
                    "of head node"
                ) from None

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
            tokenizer_ipc_name=f"tcp://{join_host_port(dist_init_host, port_base)}",
            scheduler_input_ipc_name=f"tcp://{join_host_port(dist_init_host, scheduler_input_port)}",
            nccl_port=port,
            dist_init_addr=join_host_port(dist_init_host, dist_init_port),
            rpc_ipc_name=f"tcp://{join_host_port(dist_init_host, rpc_port)}",
            metrics_ipc_name=f"tcp://{join_host_port(dist_init_host, metrics_ipc_port)}",
            tokenizer_worker_ipc_name=None,
        )
