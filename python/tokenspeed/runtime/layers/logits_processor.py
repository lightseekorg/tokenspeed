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

"""Logits processing."""

import dataclasses

import torch
from tokenspeed_kernel.ops.communication.triton import all_gather_inner, create_state
from tokenspeed_kernel.ops.gemm.triton_gemv import decode_gemv, use_decode_gemv
from tokenspeed_kernel.ops.sampling import argmax as sampling_argmax
from tokenspeed_kernel.ops.sampling.cute_dsl import (
    DistArgmaxState,
    distributed_argmax,
)
from tokenspeed_kernel.ops.sampling.cute_dsl import (
    is_available as dist_argmax_available,
)
from tokenspeed_kernel.ops.sampling.cute_dsl import (
    supports_dist_argmax_shape,
    try_create_dist_argmax_state,
)
from tokenspeed_kernel.platform import current_platform
from torch import nn

from tokenspeed.runtime.configs.numerics import (
    BITWISE_ENVELOPES,
    MEGATRON_VOCAB_BLOCK,
)
from tokenspeed.runtime.distributed.comm_manager import (
    dp_group_row_counts,
    gather_sampled_rows,
)
from tokenspeed.runtime.distributed.comm_ops import (
    all_gather,
    all_gather_single,
    all_to_all_transpose,
    token_all_gather,
    token_all_gather_rows,
)
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.execution.context import ForwardContext, InputLogprobRows
from tokenspeed.runtime.execution.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from tokenspeed.runtime.execution.query_shard import QueryShardPlan
from tokenspeed.runtime.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from tokenspeed.runtime.sampling.dp_sampling_config import (
    DpSamplingRuntimeConfig,
)
from tokenspeed.runtime.sampling.logits_layout import (
    LogitsLayoutExecutor,
    LogitsLayoutPlan,
)
from tokenspeed.runtime.sampling.utils import gather_token_logprobs
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.triton import tl, triton

logger = get_colorful_logger(__name__)


_UNQUANTIZED_LM_HEAD_METHODS = frozenset(
    {"UnquantizedEmbeddingMethod", "UnquantizedLinearMethod"}
)


def _has_lm_head_runtime_attrs(lm_head, attr_names: tuple[str, ...]) -> bool:
    return all(hasattr(lm_head, attr_name) for attr_name in attr_names)


def should_apply_lm_head_quant_method(lm_head, quant_method) -> bool:
    """Whether ``lm_head``'s quant method should run the logits GEMM.

    Returns ``True`` only when the head is genuinely quantized and its runtime
    tensor layout matches the method (so a packed weight is never matmul'd, and
    a mismatched/stale method never runs). Otherwise the caller falls back to a
    dense ``weight`` matmul.
    """
    if (
        quant_method is None
        or not hasattr(lm_head, "weight")
        or not callable(getattr(quant_method, "apply", None))
    ):
        return False

    method_name = type(quant_method).__name__
    if method_name in _UNQUANTIZED_LM_HEAD_METHODS:
        return False

    if method_name == "Nvfp4W4A16LinearMethod":
        return lm_head.weight.dtype == torch.uint8 and _has_lm_head_runtime_attrs(
            lm_head,
            (
                "weight_scale",
                "alpha",
                "input_size_per_partition",
                "output_size_per_partition",
            ),
        )

    return True


def _force_deterministic_rsag() -> bool:
    """``--force-deterministic-rsag``: NCCL only, even for the pure-data-movement
    multicast gather of the logits."""
    from tokenspeed.runtime.utils.env import global_server_args_dict

    return bool(global_server_args_dict.get("force_deterministic_rsag", False))


def _dist_argmax_vetoed() -> bool:
    """Whether the distributed argmax (a cross-rank reduction over symmetric
    memory) stays off: under ``--force-deterministic-rsag`` like every
    symmetric-memory path, and under the bitwise envelope, which was verified
    with the gather plus the canonical local argmax and pins that form."""
    from tokenspeed.runtime.utils.env import global_server_args_dict

    return (
        _force_deterministic_rsag()
        or global_server_args_dict["numerics"] in BITWISE_ENVELOPES
    )


@dataclasses.dataclass
class LogitsProcessorOutput:
    ## Part 1: This part will be assigned in python/tokenspeed/runtime/layers/logits_processor.py::LogitsProcessor
    # The logits of the next tokens.       shape: [#seq, vocab_size]
    next_token_logits: torch.Tensor
    # Used when ``do_argmax=True``.   shape: [#seq]
    next_token_ids: torch.Tensor | None = None
    # Used by speculative decoding.
    # The last hidden layers
    hidden_states: torch.Tensor | None = None
    logits_layout_plan: LogitsLayoutPlan | None = None

    ## Part 2: Populated by the active SamplingBackend during sample()/verify().
    # The logprobs of the next tokens.                              shape: [#seq]
    next_token_logprobs: torch.Tensor | None = None
    # The logprobs and ids of the top-k tokens in output positions. shape: [#seq, k]
    next_token_top_logprobs_val: list | None = None
    next_token_top_logprobs_idx: list | None = None
    # The logprobs and ids of the requested token ids in output positions. shape: [#seq, n] (n is the number of requested token ids)
    next_token_token_ids_logprobs_val: list | None = None
    next_token_token_ids_logprobs_idx: list | None = None

    ## Part 3: Prefill-only. This part will be assigned in python/tokenspeed/runtime/layers/logits_processor.py::LogitsProcessor
    # The logprobs of the prompt rows the forward's ``InputLogprobPlan``
    # named, fp32 in plan order, whole on every rank.  shape: [#rows]
    input_token_logprobs: torch.Tensor | None = None


@dataclasses.dataclass
class LogitsMetadata:
    forward_mode: ForwardMode
    # The rows ``hidden_states`` holds are this rank's shard of the forward
    # (query context parallelism); None when they are the whole forward. It
    # selects the row-selection path (gathers over the TP group or local
    # indexing), so every constructor names it.
    query_shard: QueryShardPlan | None = dataclasses.field(kw_only=True)
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL
    gather_ids: torch.Tensor | None = None
    logits_rows_selected: bool = False
    # Prompt rows whose next-token logprob the forward returns (SGLang
    # ``logprob_start_len``); None when none is wanted.
    input_logprob_rows: InputLogprobRows | None = None
    # The forward's per-rank row tables (attention DP), for the LM-head TP
    # group's row counts; see LogitsProcessor._lm_head_tp_row_counts.
    all_decode_or_idle: bool = False
    global_num_tokens: list[int] | None = None
    collective_global_num_tokens: list[int] | None = None

    # DP attention metadata. Not needed when DP attention is not used.
    # Number of tokens in the request.
    global_num_tokens_gpu: torch.Tensor | None = None
    # The start position of local hidden states.
    dp_local_start_pos: torch.Tensor | None = None
    dp_local_num_tokens: torch.Tensor | None = None
    gathered_buffer: torch.Tensor | None = None
    # Buffer to gather logits from all ranks.
    forward_batch_gathered_buffer: torch.Tensor | None = None

    @classmethod
    def from_forward_context(cls, ctx: ForwardContext):
        return cls(
            forward_mode=ctx.forward_mode,
            capture_hidden_mode=ctx.capture_hidden_mode,
            gather_ids=ctx.gather_ids,
            logits_rows_selected=ctx.logits_rows_selected,
            input_logprob_rows=ctx.input_logprob_rows,
            query_shard=ctx.query_shard,
            all_decode_or_idle=ctx.all_decode_or_idle,
            global_num_tokens=ctx.global_num_tokens,
            collective_global_num_tokens=ctx.collective_global_num_tokens,
        )


_FUSED_LM_HEAD_GEMM = None


def _get_fused_lm_head_gemm():
    """Lazily import the fused lm_head GEMM kernel.

    The kernel is only present when tokenspeed-kernel was built with a
    compatible nvcc. Cache a sentinel when unavailable so we fall back
    to ``torch.matmul`` silently on subsequent calls.
    """
    global _FUSED_LM_HEAD_GEMM
    if _FUSED_LM_HEAD_GEMM is not None:
        return _FUSED_LM_HEAD_GEMM
    if not current_platform().is_nvidia:
        _FUSED_LM_HEAD_GEMM = (None, None)
        return _FUSED_LM_HEAD_GEMM
    try:
        from tokenspeed_kernel.thirdparty.cuda.lm_head_gemm import (
            lm_head_gemm,
            should_use_fused,
        )

        _FUSED_LM_HEAD_GEMM = (should_use_fused, lm_head_gemm)
    except Exception:
        _FUSED_LM_HEAD_GEMM = (None, None)
    return _FUSED_LM_HEAD_GEMM


def _lm_head_matmul(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute ``hidden_states @ weight.T``.

    Routes to the fused ``lm_head_gemm`` when the shape matches a compiled
    template and the bench-driven perf gate accepts (``should_use_fused``).
    Otherwise falls back to ``torch.matmul``.

    Only enabled for Kimi (``model_type == "kimi_k2"``) at the call site —
    on DSv3 the fused kernel's PDL launch surface caused a downstream EAGLE3
    spec decode AR regression that we have not characterised end-to-end; on
    Kimi the perf win is the largest and the regression has not been
    reproduced, so we gate the fused path to Kimi only.
    """
    cast_hidden = hidden_states.to(weight.dtype)
    should_use_fused, lm_head_gemm = _get_fused_lm_head_gemm()
    if should_use_fused is not None and should_use_fused(cast_hidden, weight):
        return lm_head_gemm(cast_hidden, weight)
    return torch.matmul(cast_hidden, weight.T)


class LogitsProcessor(nn.Module):

    _LOGITS_AG_MAX_TOKENS = 128
    _LOGITS_AG_STATE_UNINITIALIZED = object()
    _LOGITS_AG_STATES = {}

    _LOGITS_DIST_ARGMAX_MAX_TOKENS = 8192
    _LOGITS_DIST_ARGMAX_UNINITIALIZED = object()
    _LOGITS_DIST_ARGMAX_STATES = {}

    def __init__(
        self,
        config,
        skip_all_gather: bool = False,
        do_argmax: bool = False,
        logit_scale: float | None = None,
        tp_rank: int | None = None,
        tp_size: int | None = None,
        tp_group: tuple[int, ...] | None = None,
        *,
        dp_lm_head_tp: bool,
    ):
        """``tp_*`` describe the group ``lm_head`` is vocab-sharded over.

        ``dp_lm_head_tp`` selects the layout where that group's ranks are
        data-parallel for attention and hold different rows: the rows are
        all-gathered before the logits GEMM and the vocab shards transposed
        back to each rank's own rows afterwards (``skip_all_gather`` is then
        required, as the plain vocab all-gather does not apply). ``False`` is
        today's layout: every rank of the group holds the same rows.
        """
        super().__init__()
        self.config = config
        self.skip_all_gather = skip_all_gather
        self.do_argmax = do_argmax
        self.dp_sampling_enabled = False
        self.dp_num_tokens_per_req = 1
        self.dp_sampling_min_bs = 0
        self.logit_scale = logit_scale
        self._logits_layout_executor: LogitsLayoutExecutor | None = None
        from tokenspeed.runtime.utils.env import global_server_args_dict

        # --logprob-order: the log-softmax behind input (prompt) logprobs.
        self.logprob_order: str = global_server_args_dict["logprob_order"]
        if (
            self.logprob_order == "megatron"
            and config.vocab_size % MEGATRON_VOCAB_BLOCK != 0
        ):
            # The logits every logprob reads are sliced to config.vocab_size
            # (_get_logits), and the trainer's sum(exp) folds fixed-width
            # blocks of them; refuse at construction, not on the forward thread.
            raise ValueError(
                "--logprob-order megatron folds sum(exp) over fixed "
                f"{MEGATRON_VOCAB_BLOCK}-wide vocabulary blocks and needs a "
                f"vocab_size that is a multiple of it; got {config.vocab_size}"
            )

        if tp_rank is None:
            if tp_size is not None or tp_group is not None:
                raise ValueError("tp_size and tp_group require tp_rank.")
            tp_rank, tp_size = 0, 1
        elif tp_size is None:
            raise ValueError("tp_size is required when tp_rank is provided.")
        if not 0 <= tp_rank < tp_size:
            raise ValueError(f"Invalid tensor-parallel rank: {tp_rank}/{tp_size}.")
        if tp_size != 1 and tp_group is None:
            raise ValueError("tp_group is required when tp_size > 1.")
        self.tp_rank, self.tp_size, self.tp_group = tp_rank, tp_size, tp_group
        if dp_lm_head_tp and (tp_size == 1 or not skip_all_gather):
            raise ValueError(
                "dp_lm_head_tp needs a vocab-sharded head (tp_size > 1) and "
                "skip_all_gather=True"
            )
        self.dp_lm_head_tp = dp_lm_head_tp

        self._all_gather_state = self._LOGITS_AG_STATE_UNINITIALIZED
        self._dist_argmax_state = self._LOGITS_DIST_ARGMAX_UNINITIALIZED

        self.final_logit_softcapping = getattr(
            self.config, "final_logit_softcapping", None
        )
        if (
            self.final_logit_softcapping is not None
            and self.final_logit_softcapping < 0
        ):
            self.final_logit_softcapping = None

        # Gate the fused lm_head GEMM to Kimi only. See ``_lm_head_matmul``.
        self._use_fused_lm_head = getattr(self.config, "model_type", None) == "kimi_k2"

    def require_full_vocab_logits(self) -> None:
        """Turn the fused draft argmax off so every forward returns full-vocab logits.

        Draft models construct with ``do_argmax=True``: under tensor
        parallelism the fused path reduces the argmax across the vocab shards
        and hands back the local shard's logits, which only a greedy proposal
        can live with. A consumer that samples from the draft distribution
        (``--enable-speculative-sampling``) calls this once after construction
        and before the first forward; the ordinary vocab all-gather then runs
        on every draft step. Under attention DP the logits are full-vocab
        either way: the head is replicated (``skip_all_gather``), or
        ``dp_lm_head_tp`` transposes the vocab shards back to each rank's rows.
        Returns None.
        """
        self.do_argmax = False

    def configure_dp_logits_layout(self, runtime: DpSamplingRuntimeConfig) -> None:
        if (
            not runtime.enabled
            or runtime.topology is None
            or runtime.min_bs is None
            or runtime.max_bucket_bs is None
            or runtime.vocab_size is None
            or runtime.device is None
        ):
            raise RuntimeError("enabled DP sampling runtime is incomplete")
        topology = runtime.topology
        self.dp_sampling_enabled = True
        self.dp_num_tokens_per_req = runtime.num_tokens_per_req
        self.dp_sampling_min_bs = runtime.min_bs
        self._logits_layout_executor = LogitsLayoutExecutor(
            tp_rank=topology.tp_rank,
            tp_size=topology.tp_size,
            tp_group=topology.tp_group,
            max_bucket_bs=runtime.max_bucket_bs,
            num_tokens_per_req=runtime.num_tokens_per_req,
            vocab_size=runtime.vocab_size,
            device=runtime.device,
        )

    def _resolve_logits_layout_plan(
        self,
        hidden_states: torch.Tensor,
        logits_metadata: LogitsMetadata,
    ) -> LogitsLayoutPlan | None:
        if not self.dp_sampling_enabled:
            return None
        if not logits_metadata.forward_mode.is_decode():
            return None
        n = self.dp_num_tokens_per_req
        rows = hidden_states.shape[0]
        if rows % n != 0:
            raise ValueError(f"hidden_states have {rows} rows, not divisible by N={n}")
        effective_bs = rows // n
        bucket_bs = ((effective_bs + self.tp_size - 1) // self.tp_size) * self.tp_size
        if effective_bs < self.dp_sampling_min_bs:
            return None
        return LogitsLayoutPlan(
            effective_bs=effective_bs,
            bucket_bs=bucket_bs,
            tp_size=self.tp_size,
            num_tokens_per_req=n,
        )

    def _tp_group_multicast_reachable(self) -> bool:
        """Whether the gather's symmetric buffer can map multicast here.

        Topology now only admits: an NVLink domain can span hosts, so a
        host-spread group is asked of the fabric rather than refused outright,
        and without fabric the rendezvous hangs rather than failing over. The
        rank count cannot stand in for the topology test -- a strided group can
        be smaller than one host's device count while living on two.

        The world fabric map is gathered during distributed initialization, so
        the group verdict is a local lookup with no dispatch-time collective.
        """
        if self.tp_group is None:
            return False

        from tokenspeed_kernel.ops.communication.fabric import (
            group_has_fabric,
        )

        from tokenspeed.runtime.utils.env import global_server_args_dict

        mapping = global_server_args_dict.get("mapping")
        nprocs_per_node = getattr(mapping, "nprocs_per_node", None)
        spans_hosts = bool(nprocs_per_node) and (
            len({rank // nprocs_per_node for rank in self.tp_group}) > 1
        )
        if not spans_hosts:
            return True
        return group_has_fabric(self.tp_group)

    def _init_all_gather_state(self, lm_head: VocabParallelEmbedding):
        if not current_platform().is_nvidia or _force_deterministic_rsag():
            return None

        if (
            self.tp_size == 1
            or self.skip_all_gather
            or not self._tp_group_multicast_reachable()
        ):
            return None

        vocab_padded = lm_head.weight.size(0) * self.tp_size
        if vocab_padded % (self.tp_size * 8) != 0:
            return None

        key = (self.tp_group, vocab_padded)
        if key not in self._LOGITS_AG_STATES:
            self._LOGITS_AG_STATES[key] = create_state(
                enable_lamport=False,
                moe_tail_max_rows=0,
                group=pg_manager.get_process_group("nccl", self.tp_group),
                rank_in_group=self.tp_rank,
                attnres_max_numel=0,
                attnres_max_rows=0,
                max_tokens=self._LOGITS_AG_MAX_TOKENS,
                hidden_size=vocab_padded,
                device=None,
                max_numel=0,
                max_bytes=0,
            )
        return self._LOGITS_AG_STATES[key]

    def _agree_across_tp(self, ok: bool, group, device: torch.device) -> bool:
        """Reduce a per-rank verdict so the whole group takes the same path."""
        vote = torch.tensor([int(ok)], dtype=torch.int32, device=device)
        torch.distributed.all_reduce(
            vote, op=torch.distributed.ReduceOp.MIN, group=group
        )
        return bool(vote.item())

    def acquire_dist_argmax_state(
        self,
        lm_head: VocabParallelEmbedding,
        *,
        max_M: int,
        skip_ping_pong: bool,
        dtype: torch.dtype,
    ) -> DistArgmaxState | None:
        """Build this TP group's distributed-argmax state, or None to fall back.

        Shared by the sampler and the drafters. Construction rendezvouses and
        barriers, so a rank deciding alone would strand the others: platform
        eligibility and the build outcome are both reduced with MIN. Callers
        apply their own config-uniform gates first, which may return early
        because those cost no collective work.

        Args:
            lm_head: The vocab-parallel head whose shard the argmax reduces.
            max_M: Largest row count the caller will ever pass.
            skip_ping_pong: Pin the slot band instead of alternating; only
                when the caller synchronizes across ranks between calls.
            dtype: Value dtype of the logits selected by the caller.

        Returns:
            The state, or None when this group must use the gather path.
        """
        device = lm_head.weight.device
        key = (
            self.tp_group,
            lm_head.weight.size(0),
            max_M,
            skip_ping_pong,
            dtype,
            device,
        )
        if key in self._LOGITS_DIST_ARGMAX_STATES:
            return self._LOGITS_DIST_ARGMAX_STATES[key]
        if torch.cuda.is_current_stream_capturing():
            return None  # never rendezvous inside capture; warmup probes first

        group = pg_manager.get_process_group("nccl", self.tp_group)
        if self._agree_across_tp(
            current_platform().is_nvidia and dist_argmax_available(), group, device
        ):
            state = try_create_dist_argmax_state(
                group=group,
                rank_in_group=self.tp_rank,
                max_M=max_M,
                dtype=dtype,
                device=device,
                skip_ping_pong=skip_ping_pong,
            )
            if not self._agree_across_tp(state is not None, group, device):
                state = None
        else:
            state = None
        self._LOGITS_DIST_ARGMAX_STATES[key] = state
        return state

    def _init_dist_argmax_state(self, lm_head: VocabParallelEmbedding):
        if _dist_argmax_vetoed():
            return None
        if not 2 <= self.tp_size <= 32:
            return None  # the kernel's cross-rank reduce is a single warp shuffle
        if self.skip_all_gather or self.dp_sampling_enabled:
            return None

        vocab_per_rank = lm_head.weight.size(0)
        if vocab_per_rank * self.tp_size != self.config.vocab_size:
            return None  # padded vocab: sharded argmax could pick a pad column
        if not supports_dist_argmax_shape(
            vocab_per_rank, lm_head.weight.dtype, self.tp_size
        ):
            return None

        return self.acquire_dist_argmax_state(
            lm_head,
            max_M=self._LOGITS_DIST_ARGMAX_MAX_TOKENS,
            skip_ping_pong=True,
            dtype=lm_head.weight.dtype,
        )

    def forward(
        self,
        input_ids,
        hidden_states,
        lm_head: VocabParallelEmbedding,
        logits_metadata: LogitsMetadata,
        aux_hidden_states: torch.Tensor | None = None,
    ) -> LogitsProcessorOutput:
        """The model exit: prompt logprobs, the sampled rows, their logits.

        ``hidden_states`` are every input row of the forward, or this rank's
        shard of them under query context parallelism
        (``logits_metadata.query_shard``), or -- when the model selected its
        logits rows itself (``logits_rows_selected``) -- the sampled rows
        already. Prompt logprobs are scored first, where the activations
        live, then the sampled rows are selected: ``hidden_states[gather_ids]``
        on whole rows, :func:`gather_sampled_rows` over the group on a shard
        (every rank ends with the batch's ``[bs, hidden]`` rows in request
        order, so the LM head and the vocab all-gather run as without a
        shard). A FULL hidden capture is the rows as given -- the shard under
        a shard, which is what a drafter's extend step consumes; a LAST
        capture is the selected ``[bs, hidden]`` rows (the aux taps', when the
        model has them), whole on every rank.
        """
        shard = self._query_shard(logits_metadata)
        # A model may finish a cache-only chunk without any logits rows.
        # Return before LM-head/collective kernels, retaining the empty taps.
        if logits_metadata.logits_rows_selected and hidden_states.shape[0] == 0:
            if logits_metadata.input_logprob_rows is not None:
                raise ValueError("selected logits rows cannot provide input logprobs")
            capture = None
            if logits_metadata.capture_hidden_mode.need_capture():
                capture = (
                    torch.cat(aux_hidden_states, dim=-1)
                    if aux_hidden_states
                    else hidden_states
                )
            if self.dp_lm_head_tp:
                # The LM-head TP peers hold rows: join their exchange with none.
                logits = self._get_logits(
                    hidden_states, lm_head, logits_metadata, require_full_vocab=False
                )
            else:
                logits = hidden_states.new_empty(
                    (0, self.config.vocab_size), dtype=torch.float32
                )
            return LogitsProcessorOutput(
                next_token_logits=logits, hidden_states=capture
            )

        # Prompt logprobs read the [num_input_rows, hidden] activations this
        # rank holds, so they are scored before the sampled-row selection
        # below (on a shard: before the sampled rows leave it).
        input_token_logprobs = None
        if logits_metadata.input_logprob_rows is not None:
            input_token_logprobs = self.compute_input_token_logprobs(
                hidden_states, lm_head, logits_metadata, shard
            )

        # Get the last hidden states and last logits for the next token prediction
        gather_ids = logits_metadata.gather_ids
        # Only a LAST capture stores the sampled rows of the aux hidden states
        # (Eagle3's layer taps), so only then are they selected -- on a shard,
        # gathered: a collective per tap per forward otherwise spent for
        # nothing. Uniform across the group: the mode rides the context.
        aux_pruned_states: list[torch.Tensor] | None = None
        select_aux = (
            aux_hidden_states is not None
            and logits_metadata.capture_hidden_mode.is_last()
        )
        if gather_ids is not None:
            if logits_metadata.logits_rows_selected or (
                # Shapes align iff midlayer already pruned to one row per
                # request (draft first-step reduce). Other paths emit [N, H]
                # with N > bs. A shard's rows are never pre-selected.
                shard is None
                and gather_ids.shape[0] == hidden_states.shape[0]
            ):
                pruned_states = hidden_states
                if select_aux:
                    aux_pruned_states = list(aux_hidden_states)
            elif shard is not None:
                pruned_states = gather_sampled_rows(
                    hidden_states, shard, gather_ids, group=self.tp_group
                )
                if select_aux:
                    aux_pruned_states = [
                        gather_sampled_rows(h, shard, gather_ids, group=self.tp_group)
                        for h in aux_hidden_states
                    ]
            else:
                pruned_states = hidden_states[gather_ids]
                if select_aux:
                    aux_pruned_states = [h[gather_ids] for h in aux_hidden_states]
        else:
            if logits_metadata.forward_mode.is_extend_or_mixed():
                raise RuntimeError(
                    "EXTEND/MIXED forward must set gather_ids on ForwardContext"
                )
            pruned_states = hidden_states
            if select_aux:
                aux_pruned_states = list(aux_hidden_states)

        # Compute logits for the sampled tokens.
        logits_layout_plan = self._resolve_logits_layout_plan(
            pruned_states, logits_metadata
        )
        sampled_logits = self._get_logits(
            pruned_states,
            lm_head,
            logits_metadata,
            plan=logits_layout_plan,
            require_full_vocab=False,
        )

        hidden_states_to_store: torch.Tensor | None = None
        if logits_metadata.capture_hidden_mode.need_capture():
            if logits_metadata.capture_hidden_mode.is_full():
                if aux_hidden_states is not None:
                    aux_hidden_states = (
                        aux_hidden_states[0]
                        if len(aux_hidden_states) == 1
                        else torch.cat(aux_hidden_states, dim=-1)
                    )
                    hidden_states_to_store = aux_hidden_states
                else:
                    hidden_states_to_store = hidden_states
            elif logits_metadata.capture_hidden_mode.is_last():
                # Get the last token hidden states; pruned states only contain
                # the last tokens already (on a shard: the batch's gathered
                # [bs, hidden] rows, so the capture is whole on every rank).
                if aux_pruned_states is not None:
                    hidden_states_to_store = (
                        aux_pruned_states[0]
                        if len(aux_pruned_states) == 1
                        else torch.cat(aux_pruned_states, dim=-1)
                    )
                else:
                    hidden_states_to_store = pruned_states
            else:
                raise RuntimeError("Should never reach")

        # Greedy draft path: emit token ids here, fusing the cross-rank
        # vocab reduction into the argmax when gated on.
        next_token_ids = self._argmax(sampled_logits) if self.do_argmax else None
        return LogitsProcessorOutput(
            next_token_logits=sampled_logits,
            next_token_ids=next_token_ids,
            hidden_states=hidden_states_to_store,
            logits_layout_plan=logits_layout_plan,
            input_token_logprobs=input_token_logprobs,
        )

    def _query_shard(self, logits_metadata: LogitsMetadata) -> QueryShardPlan | None:
        """The forward's query shard when ``hidden_states`` are a shard.

        The shard's collectives (the sampled-row gather, the gather of the
        planned prompt rows) run over this processor's TP group: it is the LM
        head's vocab-shard group, every rank of which must hold the same rows
        for the vocab all-gather to be consistent, and query context
        parallelism shards queries over exactly that group
        (``validate_qcp``: ``qcp_size == attn_tp_size``). A plan of another
        width, or a replicated head (attention DP), names a layout this
        processor cannot serve.
        """
        plan = logits_metadata.query_shard
        if plan is None or plan.size == 1:
            return None
        if plan.size != self.tp_size or self.skip_all_gather:
            head = (
                "a replicated LM head"
                if self.skip_all_gather
                else f"an LM head sharded over {self.tp_size} ranks"
            )
            raise ValueError(
                f"query shard of {plan.size} ranks over {head}: the sampled rows "
                "and prompt logprobs are gathered over the head's vocab-shard "
                "group, so the query shard group must be it"
            )
        return plan

    def compute_input_token_logprobs(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        logits_metadata: LogitsMetadata,
        shard: QueryShardPlan | None,
    ) -> torch.Tensor:
        """Logprob of each named prompt row's next token, position-chunked.

        The rows (``logits_metadata.input_logprob_rows``) are pushed through the
        LM head ``chunk_tokens`` at a time so the transient ``[rows, vocab]``
        logits stay bounded; each chunk takes the same ``_get_logits`` route
        as the sampled rows (quantized head, rl-bitwise GEMM, TP gather,
        softcap) and the sampler's own ``gather_token_logprobs`` in the
        launch's ``--logprob-order``, so prompt and output logprobs of one
        token agree bitwise. Both orders are row-local, so the chunk size
        never changes a value. The chunks ask for
        a private full-vocab tensor (``require_full_vocab=True``): the
        multicast gather returns a view of the TP group's shared buffer that
        the next chunk's gather on a faster rank would overwrite while this
        rank still reads it, so the chunk loop never takes that path.

        Under a query shard (``InputLogprobRows.rows_per_rank`` is set) the
        rows this rank holds are the plan's rows inside its shard. The head
        is vocab-sharded over the same group, so a row's full-vocabulary
        logits need every rank's slice *of that row*: the planned rows'
        activations are all-gathered first (one collective with the
        per-rank counts, rank order being row order) and the chunk loop then
        runs over the whole plan on every rank, exactly as without a shard.
        Every rank ends with the same fp32 vector -- the unsharded forward's
        bit for bit, since each row meets the same operands -- so no result
        gather follows and the per-request NaN audit agrees across the
        group. A rank without any planned row contributes no activation and
        still joins the gather and every chunk's vocab all-gather.

        Args:
            hidden_states: The ``[num_input_rows, hidden]`` activations this
                rank holds, one row per input token it computed.
            lm_head: The vocab-parallel head.
            logits_metadata: Carries ``input_logprob_rows``.
            shard: The forward's query shard as ``forward`` resolved it
                (``_query_shard``): the plan whose rows ``hidden_states`` are,
                ``None`` when they are the whole forward's.

        Returns:
            fp32 ``[plan rows]`` logprobs in row order, the whole plan's.

        Raises:
            ValueError: The model narrowed its logits rows (``hidden_states``
                does not cover every input row it computed), so prompt rows
                have no activations to read.
            RuntimeError: The head is vocab-sharded over attention-DP ranks
                (``dp_lm_head_tp``); admission refuses such requests
                (``RequestHandler.supports_input_logprobs``), so reaching here
                is a routing error.
        """
        plan = logits_metadata.input_logprob_rows
        if self.dp_lm_head_tp:
            # Each chunk's LM-head TP exchange needs every peer, and the peers
            # hold different numbers of prompt rows (attention DP), so their
            # chunk loops would not line up; refuse rather than hang.
            raise RuntimeError(
                "prompt logprobs are not supported with --lm-head-tp-size > 1 "
                "under attention DP"
            )
        if logits_metadata.logits_rows_selected or (
            hidden_states.shape[0] != plan.num_input_rows
        ):
            raise ValueError(
                "input logprobs need one activation row per input token; this "
                f"model narrowed {plan.num_input_rows} input rows to "
                f"{hidden_states.shape[0]} logits rows"
            )
        if plan.chunk_tokens <= 0:
            raise ValueError("input_logprob_chunk_tokens must be positive")
        if (plan.rows_per_rank is None) != (shard is None):
            raise ValueError(
                "input logprob rows were staged for "
                f"{'a sharded' if plan.rows_per_rank is not None else 'an unsharded'} "
                f"forward but the forward is {'sharded' if shard else 'not'}"
            )
        if shard is None:
            # Chunks index the activations in place: no [rows, hidden] copy.
            source, index = hidden_states, plan.rows
            staged_rows = index.shape[0]
        else:
            if plan.rows.shape[0] != plan.rows_per_rank[shard.rank]:
                raise ValueError(
                    f"query shard rank {shard.rank} staged {plan.rows.shape[0]} "
                    f"prompt rows but its share of the plan is "
                    f"{plan.rows_per_rank[shard.rank]}"
                )
            source = token_all_gather_rows(
                hidden_states[plan.rows], self.tp_group, list(plan.rows_per_rank)
            )
            index = None
            staged_rows = source.shape[0]
        num_rows = plan.num_result_rows
        if staged_rows != num_rows:
            raise ValueError(
                f"{num_rows} prompt-logprob targets for {staged_rows} rows"
            )
        out = torch.empty(num_rows, dtype=torch.float32, device=hidden_states.device)
        for begin in range(0, num_rows, plan.chunk_tokens):
            end = min(begin + plan.chunk_tokens, num_rows)
            rows = source[begin:end] if index is None else source[index[begin:end]]
            logits = self._get_logits(
                rows,
                lm_head,
                logits_metadata,
                plan=None,
                require_full_vocab=True,
            )
            out[begin:end] = gather_token_logprobs(
                logits, plan.targets[begin:end], logprob_order=self.logprob_order
            )
            del logits
        return out

    def _lm_head_tp_row_counts(
        self, hidden_states: torch.Tensor, logits_metadata: LogitsMetadata
    ) -> list[int]:
        """Rows each LM-head TP rank brings to the logits GEMM.

        On the decode path every rank's logits rows follow the forward's
        host-side tables -- its decode tokens, or the live rows a narrowing
        drafter reported (``collective_global_num_tokens``) -- so the counts
        are read, not exchanged, and the step stays free of host syncs. The
        other shapes have no table: a prefill keeps one row per request or the
        logprob rows, a MIXED round mixes both, and a model selecting its own
        rows is on its own; those exchange the counts (a device sync, off the
        decode path). A graph capture records a uniform padded batch on every
        rank, so its counts are uniform and no sync is recorded.
        """
        rows = hidden_states.shape[0]
        if logits_metadata.all_decode_or_idle and not (
            logits_metadata.input_logprob_rows is not None
            or logits_metadata.logits_rows_selected
        ):
            table = (
                logits_metadata.collective_global_num_tokens
                if logits_metadata.collective_global_num_tokens is not None
                else logits_metadata.global_num_tokens
            )
            return dp_group_row_counts(
                table, self.tp_group, self.tp_group[self.tp_rank], rows
            )
        if hidden_states.is_cuda and torch.cuda.is_current_stream_capturing():
            return [rows] * self.tp_size
        counts = torch.tensor([rows], dtype=torch.int64, device=hidden_states.device)
        return all_gather(counts, self.tp_group, dim=0).tolist()

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        logits_metadata: LogitsMetadata,
        embedding_bias: torch.Tensor | None = None,
        plan: LogitsLayoutPlan | None = None,
        *,
        require_full_vocab: bool,
    ) -> torch.Tensor:
        """Get logits from hidden_states.

        Args:
            require_full_vocab: The caller reads the whole distribution of
                every row and keeps the tensor across further device work
                (prompt logprobs). Under TP this disables two shortcuts the
                sampled rows take: a ``do_argmax`` processor with the
                distributed argmax active keeps its logits TP-sharded for
                ``_argmax``, and the multicast all-gather returns a view of the
                group's shared comm buffer without an entry barrier (safe only
                because a whole forward separates consecutive sampled-row
                gathers). With it set the TP gather is the NCCL collective
                into a private tensor. ``False`` is the sampled-row route.
        """
        dp_sampling = plan is not None
        if dp_sampling and not self.dp_sampling_enabled:
            raise RuntimeError(
                "DP logits layout plan was provided but LogitsProcessor was not "
                "configured with dp_sampling"
            )

        if dp_sampling and self.skip_all_gather:
            if self._logits_layout_executor is None:
                raise RuntimeError(
                    "dp_sampling logits layout executor is not configured"
                )
            hidden_states = self._logits_layout_executor.slice_hidden_states(
                hidden_states, plan
            )

        lm_head_tp_row_counts: list[int] | None = None
        if self.dp_lm_head_tp:
            if dp_sampling:
                raise RuntimeError("dp_lm_head_tp cannot combine with DP sampling")
            lm_head_tp_row_counts = self._lm_head_tp_row_counts(
                hidden_states, logits_metadata
            )
            hidden_states = token_all_gather(
                hidden_states, self.tp_group, lm_head_tp_row_counts
            )

        quant_method = getattr(lm_head, "quant_method", None)
        if should_apply_lm_head_quant_method(lm_head, quant_method):
            logits = quant_method.apply(lm_head, hidden_states, embedding_bias)
        elif hasattr(lm_head, "weight"):
            from tokenspeed.runtime.utils.env import global_server_args_dict

            if global_server_args_dict["numerics"] in BITWISE_ENVELOPES:
                import tokenspeed_kernel

                logits = tokenspeed_kernel.mm(
                    hidden_states.to(lm_head.weight.dtype),
                    lm_head.weight,
                    override="aok",
                )
            else:
                cast_hidden = hidden_states.to(lm_head.weight.dtype)
                if current_platform().is_amd and use_decode_gemv(
                    cast_hidden, lm_head.weight
                ):
                    logits = decode_gemv(cast_hidden, lm_head.weight)
                elif self._use_fused_lm_head:
                    logits = _lm_head_matmul(cast_hidden, lm_head.weight)
                else:
                    logits = torch.matmul(cast_hidden, lm_head.weight.T)
        else:
            # GGUF models
            logits = quant_method.apply(lm_head, hidden_states, embedding_bias)

        if self.logit_scale is not None:
            logits.mul_(self.logit_scale)

        if lm_head_tp_row_counts is not None:
            # [T_full, V / W] -> [T_own, V]: this rank's rows with every
            # rank's vocab shard, in rank order like the plain all-gather.
            logits = all_to_all_transpose(
                logits, self.tp_group, input_split_sizes=lm_head_tp_row_counts
            )
        elif dp_sampling and not self.skip_all_gather:
            if self._logits_layout_executor is None:
                raise RuntimeError(
                    "dp_sampling logits layout executor is not configured"
                )
            logits = self._logits_layout_executor.swap_batch_vocab(logits, plan)

        elif not dp_sampling and self.tp_size > 1 and not self.skip_all_gather:
            if self.do_argmax and not require_full_vocab:
                if (
                    self._dist_argmax_state is self._LOGITS_DIST_ARGMAX_UNINITIALIZED
                    and not torch.cuda.is_current_stream_capturing()
                ):
                    self._dist_argmax_state = self._init_dist_argmax_state(lm_head)

                if (
                    self._dist_argmax_state
                    not in (self._LOGITS_DIST_ARGMAX_UNINITIALIZED, None)
                    and not self.final_logit_softcapping
                    and logits.size(0) <= self._LOGITS_DIST_ARGMAX_MAX_TOKENS
                ):
                    return logits

            # The multicast buffer/kernel is BF16-only; retain other logits dtypes
            # through the existing collective, including when a state is cached.
            # A private full-vocab result never uses the shared buffer either.
            state = (
                self._all_gather_state
                if logits.dtype == torch.bfloat16 and not require_full_vocab
                else None
            )
            if state is self._LOGITS_AG_STATE_UNINITIALIZED:
                # create_state rendezvouses; leave it for an eager call.
                if torch.cuda.is_current_stream_capturing():
                    state = None
                else:
                    state = self._all_gather_state = self._init_all_gather_state(
                        lm_head
                    )

            if state is not None and logits.size(0) <= self._LOGITS_AG_MAX_TOKENS:
                # skip_entry_sync=True assumes other sync points existing between two all_gather_inner calls.
                logits = all_gather_inner(
                    state,
                    logits,
                    tp_hidden_dim=logits.size(-1) * self.tp_size,
                    skip_entry_sync=True,
                    safe=False,
                )
            else:
                num_rows = logits.size(0)
                local_vocab_size = logits.size(1)
                gathered_logits = torch.empty(
                    self.tp_size * num_rows,
                    local_vocab_size,
                    dtype=logits.dtype,
                    device=logits.device,
                )
                all_gather_single(gathered_logits, logits, self.tp_group)
                logits = (
                    gathered_logits.view(self.tp_size, num_rows, local_vocab_size)
                    .transpose(0, 1)
                    .contiguous()
                    .view(num_rows, local_vocab_size * self.tp_size)
                )

        logits = logits[:, : self.config.vocab_size].contiguous()

        if self.final_logit_softcapping:
            fused_softcap_generic(logits, self.final_logit_softcapping)

        return logits

    def _argmax(self, logits: torch.Tensor) -> torch.Tensor:
        if (
            self._dist_argmax_state
            not in (self._LOGITS_DIST_ARGMAX_UNINITIALIZED, None)
            and not self.final_logit_softcapping
            and logits.size(0) <= self._LOGITS_DIST_ARGMAX_MAX_TOKENS
        ):
            _, idx = distributed_argmax(self._dist_argmax_state, logits)
            return idx
        else:
            return sampling_argmax(logits)


@triton.jit
def fused_softcap_kernel(
    full_logits_ptr,
    softcapping_value,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load values
    x = tl.load(full_logits_ptr + offsets, mask=mask).to(tl.float32)

    # Perform operations in-place
    x = x / softcapping_value

    # Stable tanh form; the exp ratio overflows to inf/inf for large logits.
    x = 2 * tl.sigmoid(2 * x) - 1

    x = x * softcapping_value

    # Store result
    tl.store(full_logits_ptr + offsets, x, mask=mask)


def fused_softcap(full_logits, final_logit_softcapping):
    n_elements = full_logits.numel()
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1)

    fused_softcap_kernel[grid](
        full_logits_ptr=full_logits,
        softcapping_value=final_logit_softcapping,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return full_logits


def fused_softcap_generic(full_logits, final_logit_softcapping):
    return fused_softcap(full_logits, final_logit_softcapping)
