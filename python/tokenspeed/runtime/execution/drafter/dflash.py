# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.kvcache.triton import (
    LatentPlanes,
    check_mla_latent_geometry,
    mla_latent_norm_rope_scatter,
)
from tokenspeed_kernel.ops.sampling.cute_dsl import distributed_argmax as _dist_argmax
from tokenspeed_kernel.ops.sampling.cute_dsl import (
    supports_dist_argmax_shape as _supports_dist_argmax_shape,
)
from typing_extensions import override

from tokenspeed.runtime.distributed.comm_ops import all_gather, all_gather_single
from tokenspeed.runtime.execution.cache_loc_kernel import (
    dflash_prepare_decode,
)
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.drafter._dflash_fused_kv import (
    _fused_norm_rope_stacked_scatter,
    _get_kv_buffer_ptrs,
    forget_kv_buffer_ptrs,
)
from tokenspeed.runtime.execution.drafter.base import BaseDrafter
from tokenspeed.runtime.execution.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
from tokenspeed.runtime.layers.logits_processor import (
    LogitsMetadata,
    LogitsProcessor,
    _dist_argmax_vetoed,
)
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.nvtx import nvtx_range
from tokenspeed.runtime.utils.spec_block_geometry import (
    read_checkpoint_block_size,
    validate_block_widths,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.input_buffer import InputBuffers
    from tokenspeed.runtime.execution.model_runner import ModelRunner
    from tokenspeed.runtime.execution.runtime_states import RuntimeStates
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
    from tokenspeed.runtime.sampling.backends.base import SamplingBackend

logger = get_colorful_logger(__name__)


def _resolve_block_geometry(
    cfg, spec_num_tokens: int, spec_algorithm: str
) -> tuple[int, int]:
    """Resolve (verify_width, draft_block_size) for a block drafter.

    Args:
        cfg: The draft model's config.
        spec_num_tokens: The verify width, one anchor row plus one row per
            drafted token, so ``draft_block_size = spec_num_tokens - 1``.
        spec_algorithm: ``"DFLASH"`` or ``"DSPARK"``; selects which
            ``block_size`` convention the checkpoint is held to.

    Returns:
        ``(verify_width, draft_block_size)``.

    Raises:
        ValueError: The verify width leaves no room for a draft, or the
            checkpoint was trained at a different block size.
    """
    verify_width = int(spec_num_tokens)
    if verify_width < 2:
        raise ValueError(
            "Block drafting requires speculative_num_draft_tokens >= 2 "
            f"(one anchor plus at least one draft); got {verify_width}."
        )
    draft_block_size = verify_width - 1

    ckpt_block_size = read_checkpoint_block_size(cfg)
    if ckpt_block_size is not None:
        validate_block_widths(
            spec_algorithm, ckpt_block_size, draft_block_size, verify_width
        )
    return verify_width, draft_block_size


def _resolve_draft_query_width(verify_width: int, sample_from_anchor: bool) -> int:
    """Resolve rows consumed by one native draft forward."""
    return verify_width - 1 if sample_from_anchor else verify_width


_UNSET = object()

# A short context projection that warmup_extend also compiles: the column
# all-gather picks its subgroup layout by the rank-local payload size, so a
# short chunk (here one 8-row draft block's verify rows) takes another
# layout than a long extend's.
_SHORT_CONTEXT_PROJECTION_ROWS = 8

# Dtypes a replicated draft head can carry over from the target's shards.
# A whitelist, not a blacklist: the draft matmuls the gathered rows directly,
# so a dtype belongs here only once its raw bytes ARE the vocabulary's values.
# That excludes packed and scaled weights (fp8, int4/int8 quantized), whose
# rows need their scales applied first. Adding a dtype means verifying that
# much, not just that the cast succeeds.
_REPLICABLE_HEAD_DTYPES = frozenset({torch.bfloat16, torch.float16, torch.float32})


class DFlash(BaseDrafter):
    """DFlash block drafter backed by a native TokenSpeed draft model."""

    supports_pd_layerwise_finalization = True
    sample_from_anchor = False
    spec_algorithm = "DFLASH"

    def __init__(
        self,
        spec_num_tokens: int,
        spec_num_steps: int,
        draft_model_runner: ModelRunner | None = None,
        attn_backend=None,
        token_to_kv_pool=None,
        runtime_states: RuntimeStates | None = None,
        input_buffers: InputBuffers | None = None,
        vocab_size: int | None = None,
    ) -> None:
        super().__init__(
            spec_num_tokens=spec_num_tokens,
            spec_num_steps=spec_num_steps,
            draft_model_runner=draft_model_runner,
            runtime_states=runtime_states,
            input_buffers=input_buffers,
            attn_backend=attn_backend,
            token_to_kv_pool=token_to_kv_pool,
            vocab_size=vocab_size,
        )
        if draft_model_runner is None:
            raise ValueError("Native DFLASH requires a draft model runner.")
        server_args = draft_model_runner.server_args
        if not server_args.speculative_draft_model_path:
            raise ValueError("DFLASH requires --speculative-draft-model-path.")

        self.device = torch.device(draft_model_runner.device)
        self.model = draft_model_runner.model
        # Bound by wire_target.
        self.target_model: torch.nn.Module | None = None
        self.target_language_model: torch.nn.Module | None = None
        self.embed_tokens: torch.nn.Module | None = None
        self.lm_head: torch.nn.Module | None = None
        self.logits_processor: torch.nn.Module | None = None
        self.attention_kind = getattr(self.model, "attention_kind", "qwen_mha")

        cfg = self.model.config
        dflash_cfg = getattr(cfg, "dflash_config", {}) or {}
        # The draft model resolved its checkpoint's taps once, for setup.
        self.target_layer_ids = list(self.model.target_layer_ids)
        mask_token_id = dflash_cfg.get("mask_token_id")
        if mask_token_id is None:
            mask_token_id = getattr(cfg, "mask_token_id", None)
        if mask_token_id is None:
            raise ValueError(
                "DFLASH draft config must define dflash_config.mask_token_id."
            )
        self.mask_token_id = int(mask_token_id)
        self.verify_width, self.draft_block_size = _resolve_block_geometry(
            cfg, int(spec_num_tokens), self.spec_algorithm
        )
        self.draft_query_width = _resolve_draft_query_width(
            self.verify_width, self.sample_from_anchor
        )
        # Legacy alias: callers that predate the verify/draft split.
        self.block_size = self.verify_width
        self.hidden_size = int(getattr(cfg, "hidden_size"))
        self._init_native_buffers()
        self._validate_draft_attention_window()
        self._validate_draft_window_sink()
        self._greedy_gathered_max: torch.Tensor | None = None
        # Set by wire_target, which binds the embedding.
        self._first_layer_input = "full"
        self._greedy_gathered_ids: torch.Tensor | None = None
        self._greedy_gather_cap = 0
        self._init_fused_kv_helper()
        self._init_context_taps()

    @override
    def bind_sampling_backend(self, sampling_backend: SamplingBackend) -> None:
        """Refuse --enable-speculative-sampling for greedy DFLASH, else bind.

        Base DFLASH proposes the argmax of every block row and records no
        draft distribution, so the base ``draft_probs`` gate never fires for
        it; the verifier's config says whether the flag is on. DFlash2 samples
        its walk (sparse q, see its override) and DSpark its Markov steps
        (dense q through the base sampler).

        Args:
            sampling_backend: The executor's verifier.

        Raises:
            ValueError: The flag is on and this drafter proposes greedily.
        """
        if (
            sampling_backend.config.enable_speculative_sampling
            and not self.supports_speculative_sampling
        ):
            raise ValueError(
                "--enable-speculative-sampling with DFLASH needs a DFlash2 draft "
                "(candidate selector weights) whose walk samples each step from "
                f"its own distribution; {type(self).__name__} proposes a whole "
                "block greedily"
            )
        # A no-op unless the executor allocated draft_probs (DSpark).
        super().bind_sampling_backend(sampling_backend)

    @override
    def idle_forward_global_num_tokens(
        self,
        global_num_tokens: list[int],
        global_bs: list[int],
        *,
        global_num_decodes: list[int],
    ) -> list[list[int]]:
        # Block drafter: one draft forward proposes the whole block.
        del global_bs
        return [global_num_tokens]

    def _validate_draft_attention_window(self) -> None:
        """Reject a drafter backend that would drop the draft's window.

        A backend that ignores a layer's ``sliding_window_size`` answers with
        full-history attention instead, which raises nothing and shows up only
        as wrong draft hidden states. Ask the built layers, not the config:
        only the layers know whether this draft family applies the window.

        Raises:
            ValueError: The draft declares a window the backend cannot apply.
        """
        if self.attn_backend.supports_layer_sliding_window:
            return
        windows = {
            int(module.sliding_window_size)
            for module in self.model.modules()
            if isinstance(module, PagedAttention) and module.sliding_window_size >= 0
        }
        if not windows:
            return
        backend_name = (
            self.draft_model_runner.server_args.drafter_attention_backend
            or type(self.attn_backend).__name__
        )
        raise ValueError(
            "The draft's attention layers declare a sliding window "
            f"(window_left={sorted(windows)}), but the {backend_name!r} drafter "
            "attention backend ignores per-layer sliding windows and would "
            "silently give those layers full-history attention. Launch with "
            "--drafter-attention-backend mla or tokenspeed_mla ('gluon' on "
            "AMD) for MLA drafts, or one of "
            "mha/fa3/fa4/triton/flashinfer/trtllm_mha for GQA drafts."
        )

    def _validate_draft_window_sink(self) -> None:
        """Reject a drafter backend that would drop the draft's window sink.

        A token sink keeps the first N tokens visible outside the sliding
        window. A backend without it answers with plain sliding-window
        attention, which raises nothing and costs acceptance quality only, so
        the mismatch has to be refused here. Ask the built layers, as
        :meth:`_validate_draft_attention_window` does.

        Raises:
            ValueError: The draft declares a sink the backend cannot apply.
        """
        sinks = {
            int(module.sliding_window_sink)
            for module in self.model.modules()
            if isinstance(module, PagedAttention) and module.sliding_window_sink > 0
        }
        if not sinks:
            return
        if self.attn_backend.supports_layer_window_sink:
            return
        backend_name = (
            self.draft_model_runner.server_args.drafter_attention_backend
            or type(self.attn_backend).__name__
        )
        raise ValueError(
            "The draft's attention layers declare a window sink "
            f"(sink={sorted(sinks)}), but the {backend_name!r} drafter "
            "attention backend cannot keep those leading tokens visible and "
            "would silently serve plain sliding-window attention. Launch with "
            "--drafter-attention-backend triton, or export the draft without "
            "sliding_window_sink."
        )

    def _init_native_buffers(self) -> None:
        if self.input_buffers is None:
            raise ValueError("Native DFLASH requires input buffers.")
        if self.attn_backend is None or self.token_to_kv_pool is None:
            raise ValueError("Native DFLASH requires draft attention components.")

        max_bs = self.input_buffers.max_bs
        self.draft_seq_lens_buf = torch.zeros(
            (max_bs,), dtype=torch.int32, device=self.device
        )
        # None means probed-and-unavailable; unset means not probed yet.
        self._dist_argmax_state: object = _UNSET
        self.draft_input_lengths_buf = torch.full(
            (max_bs,),
            self.draft_query_width,
            dtype=torch.int32,
            device=self.device,
        )
        self.draft_extend_seq_lens_cpu = torch.full(
            (max_bs,),
            self.draft_query_width,
            dtype=torch.int32,
            pin_memory=True,
        )
        self.block_offsets = torch.arange(
            self.draft_query_width, dtype=torch.int64, device=self.device
        )
        self.block_ids_buf = torch.full(
            (max_bs, self.draft_query_width),
            self.mask_token_id,
            dtype=torch.int32,
            device=self.device,
        )
        self.block_positions_buf = torch.empty(
            (max_bs, self.draft_query_width), dtype=torch.int64, device=self.device
        )
        self.next_tokens_buf = torch.empty(
            (max_bs, self.spec_num_tokens), dtype=torch.int32, device=self.device
        )
        self.current_tokens_buf = torch.empty(
            (max_bs,), dtype=torch.int32, device=self.device
        )
        self.decode_offsets_buf = (
            torch.arange(max_bs, dtype=torch.int64, device=self.device)
            * self.spec_num_tokens
            - 1
        )
        self.gather_indices_buf = torch.empty(
            (max_bs,), dtype=torch.int64, device=self.device
        )

    def wire_target(self, target_model) -> None:
        """Bind execution resources without changing model capture configuration."""
        language_model = getattr(target_model, "language_model", target_model)
        self.target_model = target_model
        self.target_language_model = language_model
        # Setup may provide a local draft embedding when the target's embedding
        # lives on another stage. Resource availability determines the binding.
        self.embed_tokens = target_model.get_input_embeddings()
        if self.embed_tokens is None:
            self.embed_tokens = self.model.embed_tokens
        self._first_layer_input = self._first_layer_input_mode()
        self.lm_head = target_model.lm_head
        self.logits_processor = language_model.logits_processor
        self._replicate_head_over_attention_dp()

    def _first_layer_embeds(self, flat_ids: torch.Tensor) -> torch.Tensor:
        """This rank's embedding input to the draft's first layer.

        The target's embedding may be vocab-sharded over the attention TP
        group (for example 8 shards of 36864 rows). ``_partial_lookup`` then
        answers with this rank's rows and ZERO rows for every id outside its
        shard, so the sum over the group is the embedding and one rank's view
        is not. Exactly one place may sum it: the lookup, or, when the draft's
        dense group spans several ranks, the first layer's reduce + norm (see
        :meth:`_first_layer_input_mode`). Adding zero rows is exact, so every
        mode hands the first layer the same embedding.
        """
        input_embeds = self.embed_tokens(
            flat_ids, reduce_results=self._first_layer_input == "full"
        )
        if (
            self._first_layer_input == "first_rank"
            and self.model.mapping.dense.tp_rank != 0
        ):
            input_embeds.zero_()
        return input_embeds

    def _first_layer_input_mode(self) -> str:
        """How the drafter hands the embedding to the draft's first layer.

        A draft whose dense group holds several ranks (attention DP 1) reduces
        its first layer's input over that group, so each rank must hand in a
        partial whose sum is the embedding: the lookup's own shard partial
        when the embedding is vocab-sharded over that group ("partial"), or
        the whole rows on the group's first rank and zeros on the others when
        the embedding is not sharded ("first_rank"). A replicated draft (dense
        TP 1) reduces nothing, so the lookup sums its shards itself ("full").

        Raises:
            ValueError: The embedding is sharded over another group than the
                draft's dense group.
        """
        dense = self.model.mapping.dense
        if dense.tp_size == 1:
            return "full"
        if self.embed_tokens.tp_size == 1:
            return "first_rank"
        if tuple(self.embed_tokens.tp_group) != tuple(dense.tp_group):
            raise ValueError(
                "the draft's dense group "
                f"{tuple(dense.tp_group)} is not the embedding's shard group "
                f"{tuple(self.embed_tokens.tp_group)}"
            )
        return "partial"

    def _replicate_head_over_attention_dp(self) -> None:
        """Replace the wired head and processor with replicated ones when the
        target's head is vocab-sharded over attention-DP ranks.

        Every route from a vocab-sharded head to a token is a collective over
        the head's TP group (this drafter's argmax all-gathers the per-shard
        maxima; ``LogitsProcessor._get_logits`` all-gathers the rows). Under
        ``dp_lm_head_tp`` that group spans attention-DP ranks holding DIFFERENT
        rows, and the target's forward keeps every one of them in the exchange.
        A drafter has no such guarantee: only the rank holding a request
        drafts, while its peers' idle forward skips the drafter
        (``ModelExecutor.execute_idle_forward``), so a collective one rank
        enters and the rest skip never completes.

        The shards are therefore gathered ONCE here, at setup, where every rank
        is running, into a rank-local full-vocabulary copy. Drafting then needs
        no collective, and the target keeps its ``--lm-head-tp-size``.
        """
        processor = self.logits_processor
        if not processor.dp_lm_head_tp:
            return
        head = self.lm_head
        if not isinstance(head, VocabParallelEmbedding):
            raise ValueError(
                "a vocab-sharded LM head under attention DP must be a "
                f"VocabParallelEmbedding to replicate; got {type(head).__name__}"
            )
        weight = head.weight
        shard = head.shard_indices
        if weight.dtype not in _REPLICABLE_HEAD_DTYPES:
            # The draft matmuls the gathered rows directly: a packed or scaled
            # weight's bytes are not the vocabulary's values.
            raise ValueError(
                "DFLASH replicates only an unquantized LM head for drafting; "
                f"the target's head holds {weight.dtype} rows"
            )
        if shard.num_added_elements:
            # An added-vocabulary block sits after this rank's padded org block,
            # so the gathered rows are not one contiguous vocabulary.
            raise ValueError(
                "DFLASH does not support an added-vocabulary LM head under "
                "--lm-head-tp-size > 1 with attention DP"
            )
        vocab_size = int(head.num_embeddings)
        gathered = all_gather(weight.detach(), processor.tp_group, dim=0)
        if gathered.shape[0] < vocab_size:
            raise ValueError(
                f"the gathered LM head holds {gathered.shape[0]} rows for a "
                f"vocabulary of {vocab_size}"
            )
        # Built with the SOURCE head's vocabulary and padding so both hold the
        # same row geometry: the gather returns the padded row count, not
        # vocab_size, and a head padded differently would place the vocabulary
        # at different indices than the shards it came from.
        replicated = ParallelLMHead(
            vocab_size,
            int(gathered.shape[1]),
            bias=False,
            params_dtype=gathered.dtype,
            org_num_embeddings=int(head.org_vocab_size),
            padding_size=int(head.padding_size),
            prefix="dflash_replicated_lm_head",
        ).to(gathered.device)
        rows = int(replicated.weight.shape[0])
        if rows != gathered.shape[0]:
            raise ValueError(
                f"the replicated LM head holds {rows} rows while the gathered "
                f"shards hold {gathered.shape[0]}"
            )
        with torch.no_grad():
            replicated.weight.copy_(gathered)
        self.lm_head = replicated
        # One rank, so no vocabulary gather or row exchange; do_argmax stays
        # off (the draft takes a local argmax over full-vocabulary logits). The
        # config is the replaced processor's: the head is the TARGET's, and the
        # processor slices logits to config.vocab_size.
        self.logits_processor = LogitsProcessor(
            processor.config,
            skip_all_gather=processor.skip_all_gather,
            logit_scale=processor.logit_scale,
            attn_tp_size=1,
            dp_lm_head_tp=False,
        )
        logger.info(
            "DFLASH replicated the LM head for drafting: the target's "
            f"{processor.tp_size}-way vocab shard spans attention-DP ranks, "
            "whose collectives a single drafting rank cannot complete"
        )

    def _probe_dist_argmax_state(self, dtype: torch.dtype, device: torch.device):
        """Ask for a drafting state, once the head's shard is a fit for one."""
        head = self.lm_head
        shard = int(head.shard_indices.num_org_elements)
        tp_size = int(self.logits_processor.tp_size)
        if (
            _dist_argmax_vetoed()
            or not 2 <= tp_size <= 32
            or int(head.num_embeddings) != int(head.org_vocab_size)
            or shard * tp_size != int(head.org_vocab_size)
            or not _supports_dist_argmax_shape(shard, dtype, tp_size)
        ):
            return None
        return self.logits_processor.acquire_dist_argmax_state(
            head,
            max_M=self.input_buffers.max_bs * max(self.spec_num_tokens - 1, 1),
            # Back-to-back walk rounds carry no cross-rank sync between them,
            # which skip_ping_pong would require.
            skip_ping_pong=False,
            dtype=dtype,
        )

    def _ensure_dist_argmax_state(self, dtype: torch.dtype, device: torch.device):
        """Probe once, before capture, and reuse the verdict for the process."""
        if self._dist_argmax_state is _UNSET:
            if torch.cuda.is_current_stream_capturing():
                return None  # rendezvous is collective; leave it to warmup
            self._dist_argmax_state = self._probe_dist_argmax_state(dtype, device)
        return self._dist_argmax_state

    def _greedy_gather_capacity(self) -> int:
        """Max element count for the greedy head's tensor-parallel all-gather
        scratch: a full ``max_bs`` decode block.

        The greedy head samples the last ``spec_num_tokens - 1`` block
        positions per request and all-gathers them across the TP group, so the
        worst case is ``tp_size * max_bs * (spec_num_tokens - 1)``.
        """
        tp_size = int(self.logits_processor.tp_size)
        return tp_size * self.input_buffers.max_bs * max(self.spec_num_tokens - 1, 1)

    def _ensure_greedy_gather_buffers(
        self,
        max_dtype: torch.dtype,
        ids_dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Lazily create the greedy all-gather scratch ONCE at its maximum
        capacity, then reuse it in place for every batch size.

        Sizing to the max ``max_bs`` block (rather than growing per batch size)
        is required for CUDA-graph correctness. Graphs are captured for
        increasing batch sizes (``[1, 2, ..., max_bs]``); a buffer grown lazily
        would be freed and reallocated when a larger bs needs more room, leaving
        every smaller-bs graph captured earlier with an
        ``all_gather_single`` recorded against freed memory. On replay
        those small-bs decode steps read garbage (out-of-vocab) draft token ids,
        which flow into the next verify forward's embedding lookup and trigger a
        CUDA illegal memory access. A fixed max-capacity buffer is allocated
        during warmup (before capture) and shared by every captured graph.

        Returns the (max, id) scratch tensors; callers slice ``[:needed]``.
        """
        cap = self._greedy_gather_capacity()
        if (
            self._greedy_gathered_max is None
            or self._greedy_gathered_ids is None
            or self._greedy_gather_cap < cap
            or self._greedy_gathered_max.dtype != max_dtype
            or self._greedy_gathered_max.device != device
            or self._greedy_gathered_ids.dtype != ids_dtype
        ):
            self._greedy_gathered_max = torch.empty(
                (cap,), dtype=max_dtype, device=device
            )
            self._greedy_gathered_ids = torch.empty(
                (cap,), dtype=ids_dtype, device=device
            )
            self._greedy_gather_cap = cap
        return self._greedy_gathered_max, self._greedy_gathered_ids

    def _greedy_sample_from_vocab_parallel_head(
        self,
        hidden_states: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._greedy_argmax_vocab_parallel(hidden_states, out=out)

    def _greedy_argmax_vocab_parallel(
        self,
        hidden_states: torch.Tensor,
        out: torch.Tensor | None = None,
        bias_fn: "Callable[[int, int], torch.Tensor] | None" = None,
        base_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Shared vocab-parallel greedy argmax primitive.

        ``base_logits`` lets a caller that walks several positions supply the
        org-vocab projection it already computed for all of them at once; only
        the bias and the argmax have to follow the previous token.
        """
        if not hasattr(self.lm_head, "weight") or not hasattr(
            self.lm_head, "shard_indices"
        ):
            # Replicated draft rows: no query shard.
            metadata = LogitsMetadata(forward_mode=ForwardMode.DECODE, query_shard=None)
            logits = self.logits_processor._get_logits(
                hidden_states, self.lm_head, metadata, require_full_vocab=False
            )
            if bias_fn is not None:
                logits = logits + bias_fn(0, int(logits.shape[-1])).to(logits.dtype)
            argmax = torch.argmax(logits, dim=-1)
            if out is not None:
                out.copy_(argmax.view_as(out))
                return out
            return argmax.to(torch.int32)

        shard = self.lm_head.shard_indices
        weight = self.lm_head.weight
        hidden_states = hidden_states.to(weight.dtype)

        num_org = int(shard.num_org_elements)
        num_org_padded = int(shard.num_org_elements_padded)
        num_added = int(shard.num_added_elements)
        org_vocab_start = int(shard.org_vocab_start_index)
        added_vocab_start = int(shard.added_vocab_start_index)

        chunk_len = int(hidden_states.shape[0])
        # The collective probe must sit outside every rank-local branch: a rank
        # with an empty org shard skipping it would strand the rest.
        dist_state = self._ensure_dist_argmax_state(weight.dtype, weight.device)
        if num_org > 0:
            if base_logits is None:
                base_logits = torch.matmul(hidden_states, weight[:num_org].T)
            if bias_fn is not None:
                base_logits = base_logits + bias_fn(org_vocab_start, num_org).to(
                    base_logits.dtype
                )
            if dist_state is not None:
                # Without padding the rank-major index IS the global id.
                _, idx = _dist_argmax(dist_state, base_logits)
                if out is not None:
                    out.copy_(idx.view_as(out))
                    return out
                return idx.to(torch.int32)
            local_max, local_arg = torch.max(base_logits, dim=-1)
        else:
            local_max = torch.full(
                (chunk_len,),
                torch.finfo(weight.dtype).min,
                dtype=weight.dtype,
                device=hidden_states.device,
            )
            local_arg = torch.zeros(
                (chunk_len,), dtype=torch.int64, device=hidden_states.device
            )

        if num_added > 0:
            added_start = num_org_padded
            added_end = num_org_padded + num_added
            added_weight = weight[added_start:added_end]
            added_logits = torch.matmul(hidden_states, added_weight.T)
            if bias_fn is not None:
                added_logits = added_logits + bias_fn(added_vocab_start, num_added).to(
                    added_logits.dtype
                )
            added_max, added_arg = torch.max(added_logits, dim=-1)
            use_added = added_max > local_max
            local_max = torch.where(use_added, added_max, local_max)
            local_arg = torch.where(
                use_added,
                added_arg.to(local_arg.dtype) + num_org_padded,
                local_arg,
            )

        if num_added == 0:
            global_ids = local_arg + org_vocab_start
        else:
            global_ids = torch.empty(
                (chunk_len,), dtype=torch.int64, device=hidden_states.device
            )
            is_base = local_arg < num_org
            global_ids[is_base] = org_vocab_start + local_arg[is_base]
            global_ids[~is_base] = added_vocab_start + (
                local_arg[~is_base] - num_org_padded
            )

        tp_size = int(self.logits_processor.tp_size)
        if tp_size == 1:
            if out is not None:
                out.copy_(global_ids.view_as(out))
                return out
            return global_ids.to(torch.int32)

        needed = tp_size * chunk_len
        gathered_max, gathered_ids = self._ensure_greedy_gather_buffers(
            local_max.dtype, global_ids.dtype, hidden_states.device
        )
        gathered_max = gathered_max[:needed]
        gathered_ids = gathered_ids[:needed]
        all_gather_single(
            gathered_max,
            local_max.contiguous(),
            self.logits_processor.tp_group,
        )
        all_gather_single(
            gathered_ids,
            global_ids.contiguous(),
            self.logits_processor.tp_group,
        )

        gathered_max = gathered_max.view(tp_size, chunk_len)
        gathered_ids = gathered_ids.view(tp_size, chunk_len)
        best_rank = torch.argmax(gathered_max, dim=0).unsqueeze(0)
        result = torch.gather(gathered_ids, 0, best_rank).view(-1)
        if out is not None:
            out.copy_(result.view_as(out))
            return out
        return result.to(torch.int32)

    @nvtx_range("dflash_update_native_cache", color="purple")
    def _update_native_cache_from_target(
        self,
        base_ctx: ForwardContext,
        logits_output: LogitsProcessorOutput,
        accept_lengths: torch.Tensor,
    ) -> None:
        """Advance accepted history and write context only when the drafter owns it."""
        self._update_draft_prefix_lengths(base_ctx, accept_lengths)
        if base_ctx.dspark_context_producer is not None:
            # The configured producer writes during target forward, before this
            # call on the same stream. Missing production is not a fallback mode.
            return

        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError("DFLASH requires target hidden states.")
        if hidden.shape[0] != base_ctx.input_num_tokens:
            # The width separates "taps collected but rows narrowed" from "no
            # taps, so the plain final hidden came through": a FULL capture
            # carries num_taps * hidden_size columns, and they need different fixes.
            raise RuntimeError(
                "DFLASH hidden-state/token mismatch: "
                f"hidden_tokens={hidden.shape[0]}, "
                f"input_tokens={base_ctx.input_num_tokens}, "
                f"hidden_width={hidden.shape[-1]}, "
                f"expected_width={self.model.num_context_features * self.hidden_size} "
                f"({self.model.num_context_features} taps x {self.hidden_size}), "
                f"bs={base_ctx.bs}, num_extends={base_ctx.num_extends}."
            )
        if base_ctx.input_num_tokens == 0:
            return

        positions = self.input_buffers.positions_buf[: base_ctx.input_num_tokens]
        # Context rows belong to the draft group's page-id space. This runs
        # before the draft publishes its next proposal window.
        backend = self.attn_backend
        if base_ctx.num_extends == base_ctx.bs:
            # A PD prefill worker may never publish a decode window.
            cache_locs = backend.extend_span_locations()
        elif base_ctx.num_extends == 0:
            cache_locs = backend.decode_window_locations()
        else:
            cache_locs = torch.cat(
                (
                    backend.extend_span_locations(),
                    backend.decode_window_locations(),
                )
            )
        self._write_native_cache(
            hidden,
            positions,
            cache_locs[: base_ctx.input_num_tokens],
            decode_only=base_ctx.num_extends == 0,
        )

    def _update_draft_prefix_lengths(
        self, base_ctx: ForwardContext, accept_lengths: torch.Tensor
    ) -> None:
        """Publish valid draft history independently of who produces cache bytes."""
        bs = base_ctx.bs
        req_pool_indices = self.input_buffers.req_pool_indices_buf[:bs]
        if (
            base_ctx.num_extends == 0
            and torch.cuda.is_available()
            and torch.cuda.is_current_stream_capturing()
        ):
            old_lens = self.runtime_states.valid_cache_lengths.index_select(
                0, req_pool_indices
            )
            self.draft_seq_lens_buf[:bs].copy_(
                old_lens.to(torch.int32) + accept_lengths[:bs].to(torch.int32)
            )
            return

        if base_ctx.input_num_tokens == 0:
            return

        # Target verification produces spec_num_tokens rows per decode request;
        # prefill rows retain their actual chunk lengths.
        lengths = self.input_buffers.input_lengths_buf[:bs].to(torch.int64).clone()
        lengths[base_ctx.num_extends :] = self.spec_num_tokens
        positions = self.input_buffers.positions_buf[: base_ctx.input_num_tokens]
        starts = torch.cumsum(lengths, 0) - lengths
        takes = lengths.clone()
        if bs > base_ctx.num_extends:
            takes[base_ctx.num_extends : bs] = (
                accept_lengths[base_ctx.num_extends : bs]
                .to(torch.int64)
                .clamp(min=0, max=self.spec_num_tokens)
            )
        last_row = (starts + takes - 1).clamp(min=0, max=base_ctx.input_num_tokens - 1)
        old_lens = self.runtime_states.valid_cache_lengths.index_select(
            0, req_pool_indices
        ).to(torch.int32)
        # Writers may materialize rejected rows too; only this accepted prefix
        # is visible to the next draft, and later rounds overwrite the rest.
        self.draft_seq_lens_buf[:bs].copy_(
            torch.where(takes > 0, (positions[last_row] + 1).to(torch.int32), old_lens)
        )

    def _write_native_cache(
        self,
        target_hidden: torch.Tensor,
        target_positions: torch.Tensor,
        target_cache_locs: torch.Tensor,
        decode_only: bool = False,
    ) -> None:
        model = self.draft_model_runner.model
        target_hidden = target_hidden.to(device=self.device, dtype=model.context_dtype)
        expected_width = model.context_in_features
        actual_width = int(target_hidden.shape[-1])
        if actual_width != expected_width:
            raise RuntimeError(
                "DFLASH captured hidden width mismatch: "
                f"expected {expected_width}, got {actual_width}. "
                "Check dflash_config.target_layer_ids against the target model."
            )
        with torch.inference_mode():
            ctx_hidden = model.project_target_hidden(target_hidden)
            if decode_only and self._fused_kv_enabled:
                self._write_native_cache_fused(
                    ctx_hidden, target_positions, target_cache_locs
                )
                return
            # The draft model owns its KV layout (GQA k/v vs MLA latent).
            model.write_context_kv(
                ctx_hidden,
                target_positions,
                target_cache_locs,
                self.token_to_kv_pool,
                attn_backend=self.attn_backend,
            )

    def warmup_extend(self, num_tokens: int) -> None:
        """Project ``num_tokens`` rows of zero taps, as an extend's context
        write does (``_write_native_cache``). The projection ends in a column
        all-gather over the attention TP group; a decode warmup whose context
        write takes the draft monokernels never reaches it. Warm both long
        and short chunks because the gather chooses its subgroup count by
        the rank-local payload size."""
        model = self.draft_model_runner.model
        taps = torch.zeros(
            (num_tokens, model.context_in_features),
            dtype=model.context_dtype,
            device=self.device,
        )
        with torch.inference_mode():
            model.project_target_hidden(taps)
            if num_tokens > _SHORT_CONTEXT_PROJECTION_ROWS:
                model.project_target_hidden(taps[:_SHORT_CONTEXT_PROJECTION_ROWS])

    def set_cache_pool(self, token_to_kv_pool: CachePool | None) -> None:
        """The stacked KV views and their raw pointers name the old arena."""
        super().set_cache_pool(token_to_kv_pool)
        # Keyed on layer 0, which the replacement arena may be handed again.
        forget_kv_buffer_ptrs()
        self._init_fused_kv_helper()
        # Same order as __init__.
        self._init_context_taps()

    def _init_fused_kv_helper(self) -> None:
        """Pre-stack KV weights, k_norm, eps, and cos_sin_cache for the bound pool."""
        self._fused_kv_enabled = False
        self._fused_kv_is_mla = False
        self._fused_kv_workspace_capacity = 0
        self._fused_kv_workspace_dtype = None
        self._fused_kv_proj_workspace = None
        self._fused_kv_latent_planes: LatentPlanes | None = None
        # torch.mm(out=...) is always used; workspace pre-allocated at warmup.
        self._fused_kv_k_buffers = []
        self._fused_kv_v_buffers = []
        if self.draft_model_runner.mapping.attn.kvp_size > 1:
            # The fused scatter takes physical slots and has no ownership
            # mask; the per-layer writer translates through the placement.
            logger.info(
                "DFLASH fused KV materialization disabled: the draft cache is "
                "page-sharded (--kv-parallel-size > 1)."
            )
            return
        try:
            layers = self.draft_model_runner.model.layers
            if not layers:
                return
            if getattr(self.draft_model_runner.model, "_uses_mla", False):
                self._init_fused_kv_helper_mla(layers)
                return
            first_attn = layers[0].self_attn
            is_neox = bool(getattr(first_attn.rotary_emb, "is_neox_style", True))
            if not is_neox:
                logger.info(
                    "DFLASH fused KV materialization disabled: this draft's RoPE "
                    "is interleaved (is_neox_style=False), which the stacked "
                    "GQA kernel does not implement."
                )
                return

            from tokenspeed.runtime.layers.dense.unquant import UnquantizedLinearMethod

            for layer in layers:
                attn = layer.self_attn
                if not isinstance(
                    getattr(attn.qkv_proj, "quant_method", None),
                    UnquantizedLinearMethod,
                ):
                    logger.info(
                        "DFLASH fused KV materialization disabled: the draft's "
                        "QKV projection is quantized."
                    )
                    return
                if not hasattr(attn.qkv_proj, "weight"):
                    logger.info(
                        "DFLASH fused KV materialization disabled: the draft's "
                        "QKV projection exposes no dense weight to stack."
                    )
                    return

            num_kv_heads = int(first_attn.num_kv_heads)
            head_dim = int(first_attn.head_dim)
            kv_size = int(first_attn.kv_size)
            rotary_dim = int(getattr(first_attn.rotary_emb, "rotary_dim", head_dim))
            n_layers = len(layers)

            self._fused_kv_num_kv_heads = num_kv_heads
            self._fused_kv_head_dim = head_dim
            self._fused_kv_kv_size = kv_size
            self._fused_kv_rotary_dim = rotary_dim
            self._fused_kv_n_layers = n_layers
            self._fused_kv_layer_out_dim = 2 * kv_size

            kv_weight_rows = []
            k_norm_rows = []
            eps_values = []
            for layer in layers:
                attn = layer.self_attn
                kv_weight_rows.append(
                    attn.qkv_proj.weight[attn.q_size : attn.q_size + 2 * attn.kv_size]
                )
                k_norm_rows.append(attn.k_norm.weight)
                eps_values.append(float(attn.k_norm.variance_epsilon))

            flat_kv_weight = torch.cat(kv_weight_rows, dim=0)
            self._fused_kv_flat_weight_t = flat_kv_weight.t().contiguous()
            self._fused_kv_k_norm_weight = torch.stack(k_norm_rows, dim=0).contiguous()
            self._fused_kv_eps = torch.tensor(
                eps_values, dtype=torch.float32, device=self.device
            )

            cos_sin_cache = first_attn.rotary_emb.cos_sin_cache
            if cos_sin_cache.device != self.device:
                cos_sin_cache = cos_sin_cache.to(self.device)
            self._fused_kv_cos_sin_cache = cos_sin_cache

            self._fused_kv_k_buffers = [
                self.token_to_kv_pool.get_key_buffer(layer.self_attn.attn.layer_id)
                for layer in layers
            ]
            self._fused_kv_v_buffers = [
                self.token_to_kv_pool.get_value_buffer(layer.self_attn.attn.layer_id)
                for layer in layers
            ]

            _get_kv_buffer_ptrs(self._fused_kv_k_buffers, self._fused_kv_v_buffers)

            self._fused_kv_enabled = True

            max_total_ctx = self.input_buffers.max_bs * self.spec_num_tokens
            ws_dtype = self.draft_model_runner.model.fc.weight.dtype
            self._fused_kv_proj_workspace = torch.empty(
                (max_total_ctx, n_layers * self._fused_kv_layer_out_dim),
                dtype=ws_dtype,
                device=self.device,
            )
            self._fused_kv_workspace_capacity = max_total_ctx
            self._fused_kv_workspace_dtype = ws_dtype

            logger.info(
                "DFLASH fused KV materialization enabled. "
                f"n_layers={n_layers:d}, num_kv_heads={num_kv_heads:d}, head_dim="
                f"{head_dim:d}",
            )
        except Exception as e:
            logger.warning(
                "DFLASH fused KV initialization failed, falling back to sequential: "
                f"{e!s}",
            )
            self._fused_kv_enabled = False
            self._fused_kv_is_mla = False

    def _init_fused_kv_helper_mla(self, layers) -> None:
        """Stack the MLA draft's latent KV projection and cache planes.

        Every layer's ``fused_qkv_a_proj_with_mqa`` weight contributes its KV
        half to one ``[hidden, n_layers * (kv_lora + rope)]`` buffer, so
        context injection becomes one GEMM plus one Triton launch instead of a
        per-layer chain of slices, norms, RoPE and scatters.
        """
        from tokenspeed.runtime.layers.attention.kv_cache.mla import MLATokenToKVPool
        from tokenspeed.runtime.layers.dense.unquant import UnquantizedLinearMethod

        def decline(reason: str) -> None:
            logger.info(f"DFLASH fused MLA KV write disabled: {reason!s}")

        pool = self.token_to_kv_pool
        if not isinstance(pool, MLATokenToKVPool):
            return decline("the draft KV pool is not an MLA latent pool")
        if getattr(pool, "quant_method", "none") == "per_token_head":
            return decline("the latent cache is per-token-head quantized")

        first_attn = layers[0].self_attn
        rotary = getattr(first_attn, "rotary_emb", None)
        if rotary is None:
            return decline("the draft's latent RoPE tables are missing")
        kv_lora_rank = int(first_attn.kv_lora_rank)
        rope_dim = int(first_attn.qk_rope_head_dim)
        kv_width = kv_lora_rank + rope_dim
        if int(rotary.cos_sin_cache.shape[-1]) != rope_dim:
            return decline("the RoPE table does not cover the whole latent tail")

        ws_dtype = self.draft_model_runner.model.fc.weight.dtype
        weight_rows, norm_rows, eps_values, latent_buffers = [], [], [], []
        for layer in layers:
            attn = layer.self_attn
            fused = getattr(attn, "fused_qkv_a_proj_with_mqa", None)
            weight = getattr(fused, "weight", None)
            if weight is None or weight.dtype != ws_dtype:
                return decline("the latent down-projection is quantized")
            if not isinstance(
                getattr(fused, "quant_method", None), UnquantizedLinearMethod
            ):
                return decline("the latent down-projection is quantized")
            if getattr(attn, "rotary_emb", None) is not rotary:
                return decline("the draft's layers do not share one RoPE table")
            if attn.kv_lora_scale is not None:
                # The fused write norms the latent but applies no runtime
                # scale (--mla-lora-scale runtime); the per-layer path does.
                return decline("the latent carries a runtime LoRA norm scale")
            start = int(attn.q_lora_rank)
            weight_rows.append(weight[start : start + kv_width])
            norm_rows.append(attn.kv_a_layernorm.weight)
            eps_values.append(float(attn.kv_a_layernorm.variance_epsilon))
            latent_buffers.append(pool.get_key_buffer(attn.attn_mqa.layer_id))

        if ws_dtype != torch.bfloat16:
            # The fused write rounds as the bf16 per-layer chain does.
            return decline(f"the latent projection is {ws_dtype}, not bf16")
        try:
            check_mla_latent_geometry(kv_lora_rank, rope_dim)
            planes = LatentPlanes.from_buffers(latent_buffers)
        except ValueError as error:
            return decline(str(error))
        if planes.width != kv_width:
            return decline(f"latent cache rows are {planes.width} wide, not {kv_width}")

        n_layers = len(layers)
        self._fused_kv_n_layers = n_layers
        self._fused_kv_layer_out_dim = kv_width
        self._fused_kv_flat_weight_t = torch.cat(weight_rows, dim=0).t().contiguous()
        self._fused_kv_k_norm_weight = torch.stack(norm_rows, dim=0).contiguous()
        self._fused_kv_eps = torch.tensor(
            eps_values, dtype=torch.float32, device=self.device
        )
        self._fused_kv_cos_sin_cache = rotary.cos_sin_cache.to(self.device)
        self._fused_kv_is_neox = bool(getattr(rotary, "is_neox_style", True))
        self._fused_kv_sanitize = bool(pool.latent_write_sanitizes)
        self._fused_kv_latent_planes = planes

        max_total_ctx = self.input_buffers.max_bs * self.spec_num_tokens
        self._fused_kv_proj_workspace = torch.empty(
            (max_total_ctx, n_layers * kv_width), dtype=ws_dtype, device=self.device
        )
        self._fused_kv_workspace_capacity = max_total_ctx
        self._fused_kv_workspace_dtype = ws_dtype
        self._fused_kv_is_mla = True
        self._fused_kv_enabled = True
        logger.info(
            "DFLASH fused MLA KV write enabled. "
            f"n_layers={n_layers:d}, kv_lora_rank={kv_lora_rank:d}, rope_dim="
            f"{rope_dim:d}, cache_dtype={planes.dtype!s}",
        )

    def _write_native_cache_fused_mla(
        self,
        ctx_hidden: torch.Tensor,
        target_positions: torch.Tensor,
        target_cache_locs: torch.Tensor,
    ) -> None:
        """One stacked GEMM plus one norm+RoPE+scatter launch for all layers."""
        total_ctx = int(ctx_hidden.shape[0])
        self._ensure_fused_workspace(total_ctx, ctx_hidden.dtype)
        proj_out = self._fused_kv_proj_workspace[:total_ctx]
        torch.mm(ctx_hidden, self._fused_kv_flat_weight_t, out=proj_out)
        mla_latent_norm_rope_scatter(
            proj_out.view(
                total_ctx, self._fused_kv_n_layers, self._fused_kv_layer_out_dim
            ),
            self._fused_kv_k_norm_weight,
            self._fused_kv_eps,
            self._fused_kv_cos_sin_cache,
            target_positions,
            target_cache_locs,
            self._fused_kv_latent_planes,
            write_mask=None,
            is_neox=self._fused_kv_is_neox,
            sanitize=self._fused_kv_sanitize,
        )

    def _init_context_taps(self) -> None:
        """Size the buffer the target's taps land in as the target produces them.

        A decode round's taps are copied (and rounded to the projection's
        dtype) into one ``[tokens, taps * hidden]`` row block as each capture
        layer runs; the last tap projects the block once and writes the
        draft's context KV, all in stream order with the target. Drafts that
        normalize each tap before projecting (``fc_norm``) keep the
        concatenated hidden states the target returns instead.
        """
        self._tap_rows: torch.Tensor | None = None
        self._tap_count = 0
        self._sink_tokens = 0
        self._sink_positions: torch.Tensor | None = None
        self._sink_cache_locs: torch.Tensor | None = None
        # Whether this round's last tap already wrote the draft's context KV.
        self._context_written = False
        model = self.draft_model_runner.model
        if getattr(model, "fc_norm", None) is not None:
            logger.info(
                "DFLASH capture sink disabled: this draft normalizes each "
                "target tap (fc_norm) before projecting."
            )
            return
        tap_count = len(self.target_layer_ids)
        if model.context_in_features != tap_count * self.hidden_size:
            logger.warning(
                "DFLASH capture sink disabled: the draft projects "
                f"{model.context_in_features:d} features, not "
                f"{tap_count:d} taps x {self.hidden_size:d}."
            )
            return
        max_tokens = self.input_buffers.max_bs * (self.spec_num_tokens + 1)
        self._tap_count = tap_count
        self._tap_rows = torch.empty(
            (max_tokens, tap_count * self.hidden_size),
            dtype=model.context_dtype,
            device=self.device,
        )

    def _sink_allowed(self, ctx: ForwardContext) -> bool:
        """Whether this round's taps go straight to the context projection.

        Decode-only rounds with rows (the target's verify window is the
        round's whole write vector) whose context the drafter writes itself.
        An idle forward may skip the final-norm tap, so it is never armed.
        """
        return (
            self._tap_rows is not None
            and self.target_model is not None
            # The target's decode monokernels store the taps themselves.
            and not self.target_model.stores_capture_taps(ctx)
            and ctx.dspark_context_producer is None
            and ctx.num_extends == 0
            and ctx.input_num_tokens > 0
            and not ctx.forward_mode.is_idle()
        )

    @override
    def prepare_target_forward(self, ctx: ForwardContext) -> None:
        """Attach the capture sink to a decode round's target forward.

        The target hands each captured tap to :meth:`on_target_capture` as
        it is produced; the last one writes the draft's context KV, so
        ``run`` finds it done. Everything the sink needs is fixed here, for
        this forward only.
        """
        self._context_written = False
        if not self._sink_allowed(ctx):
            return
        num_tokens = ctx.input_num_tokens
        self._sink_tokens = num_tokens
        self._sink_positions = self.input_buffers.positions_buf[:num_tokens]
        self._sink_cache_locs = self.attn_backend.decode_window_locations()[:num_tokens]
        ctx.target_capture_sink = self

    def on_target_capture(self, capture_idx: int, hidden: torch.Tensor) -> None:
        """Copy tap ``capture_idx`` into its columns of the round's tap block.

        The copy rounds a wider target residual (FP32) to the projection's
        dtype and owns the rows the target goes on to rewrite. The last tap
        projects the whole block once (``fc`` over the concatenated taps)
        and writes the draft's context KV, in stream order with the
        target's own work and collectives.
        """
        num_tokens = hidden.shape[0]
        if num_tokens != self._sink_tokens:
            raise RuntimeError(
                f"DFLASH target capture {capture_idx} carries {num_tokens} rows, "
                f"but this forward was armed for {self._sink_tokens}"
            )
        hidden_size = self.hidden_size
        rows = self._tap_rows[:num_tokens]
        rows[:, capture_idx * hidden_size : (capture_idx + 1) * hidden_size].copy_(
            hidden
        )
        if capture_idx != self._tap_count - 1:
            return
        self._write_native_cache(
            rows, self._sink_positions, self._sink_cache_locs, decode_only=True
        )
        self._context_written = True

    def _ensure_fused_workspace(self, total_ctx: int, dtype: torch.dtype) -> None:
        """Ensure the projection workspace is large enough.

        The workspace is pre-allocated at init to max_bs * spec_num_tokens,
        so this should always be a no-op.
        """
        if (
            self._fused_kv_workspace_capacity >= total_ctx
            and self._fused_kv_workspace_dtype == dtype
            and self._fused_kv_proj_workspace is not None
        ):
            return
        raise RuntimeError(
            f"DFLASH fused KV workspace too small: need {total_ctx}, "
            f"have {self._fused_kv_workspace_capacity}. "
            "This should not happen — workspace is pre-allocated at init."
        )

    def _write_native_cache_fused(
        self,
        ctx_hidden: torch.Tensor,
        target_positions: torch.Tensor,
        target_cache_locs: torch.Tensor,
    ) -> None:
        """Fused KV materialization for decode-only batches.

        One stacked GEMM for every layer's K|V projection, then one Triton
        kernel for fused RMSNorm + RoPE + direct scatter into KV pool.
        Total: 1 GEMM + 1 Triton launch, in either the GQA or MLA layout.
        """
        if self._fused_kv_is_mla:
            self._write_native_cache_fused_mla(
                ctx_hidden, target_positions, target_cache_locs
            )
            return

        layers = self.draft_model_runner.model.layers
        if not self._fused_kv_enabled:
            for layer in layers:
                attn = layer.self_attn
                k, v = attn.kv_proj_only(ctx_hidden)
                k = attn.apply_k_norm(k)
                k = attn.apply_k_rope(target_positions, k)
                k = k.view(-1, attn.num_kv_heads, attn.head_dim)
                v = v.view(-1, attn.num_kv_heads, attn.head_dim)
                self.token_to_kv_pool.set_kv_buffer(attn.attn, target_cache_locs, k, v)
            return

        total_ctx = int(ctx_hidden.shape[0])
        self._ensure_fused_workspace(total_ctx, ctx_hidden.dtype)

        proj_out_2d = self._fused_kv_proj_workspace[:total_ctx]
        torch.mm(ctx_hidden, self._fused_kv_flat_weight_t, out=proj_out_2d)

        proj_out = proj_out_2d.view(
            total_ctx, self._fused_kv_n_layers, self._fused_kv_layer_out_dim
        )

        _fused_norm_rope_stacked_scatter(
            proj_out,
            self._fused_kv_k_norm_weight,
            self._fused_kv_eps,
            self._fused_kv_cos_sin_cache,
            target_positions,
            target_cache_locs,
            self._fused_kv_k_buffers,
            self._fused_kv_v_buffers,
            self._fused_kv_num_kv_heads,
            self._fused_kv_head_dim,
            self._fused_kv_rotary_dim,
        )

    @staticmethod
    def _current_tokens_from_output(
        output_tokens: torch.Tensor,
        accept_lengths: torch.Tensor,
        num_extends: int,
        spec_num_tokens: int,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bs = accept_lengths.shape[0]
        current = (
            out
            if out is not None
            else torch.empty((bs,), dtype=torch.int32, device=output_tokens.device)
        )
        if num_extends > 0:
            current[:num_extends] = output_tokens[:num_extends]
        num_decodes = bs - num_extends
        if num_decodes > 0:
            offsets = (
                torch.arange(
                    num_decodes, dtype=torch.int64, device=output_tokens.device
                )
                * spec_num_tokens
                - 1
                + num_extends
            )
            current[num_extends:] = output_tokens[
                offsets + accept_lengths[num_extends:]
            ]
        return current

    def draft(self, current_tokens: torch.Tensor) -> torch.Tensor:
        return self._draft_native(current_tokens)

    @nvtx_range("dflash_native_draft", color="purple")
    def _draft_native(
        self,
        current_tokens: torch.Tensor,
        prepared: bool = False,
    ) -> torch.Tensor:
        bs = current_tokens.shape[0]
        req_pool_indices = self.input_buffers.req_pool_indices_buf[:bs]
        prefix_lens = self.draft_seq_lens_buf[:bs]
        seq_lens_after = prefix_lens + self.draft_query_width

        block_ids = self.block_ids_buf[:bs]
        # NOTE: callers (run) write current_tokens directly into
        # block_ids_buf[:bs, 0] before invoking _draft_native
        block_positions = self.block_positions_buf[:bs]
        if not prepared:
            torch.add(
                prefix_lens.unsqueeze(1),
                self.block_offsets,
                out=block_positions,
            )

        is_capturing = (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )
        # MLA block rows are all decode rows. Treating them as extends makes
        # MLA backends slice the entire block out of their decode metadata.
        metadata_num_extends = 0 if self.attention_kind == "kimi_mla" else bs
        if not is_capturing:
            # Same unified refresh the wrapper's decode path uses; a captured
            # graph instead re-derives the block-end seq_lens in-graph.
            self.attn_backend.refresh_decode_metadata(
                bs,
                bs,
                req_pool_indices,
                seq_lens_after,
                forward_mode=ForwardMode.DECODE,
                block_tables=self.round_block_tables,
                num_extends=metadata_num_extends,
            )
        else:
            self.attn_backend.fill_block_decode_seq_lens(bs, seq_lens_after)
        # The block's write window (draft_query_width slots per request at
        # prefix..prefix+width-1). Published AFTER the refresh: the refresh
        # republishes the verify-shaped window, and the draft model's
        # write_locations fetch must see the block window instead. In-graph:
        # one fused launch over the router's address-stable location stack.
        self.attn_backend.publish_draft_step_locations(
            cache_start=prefix_lens,
            num_tokens=self.draft_query_width,
        )

        ctx = ForwardContext(
            attn_backend=self.attn_backend,
            token_to_kv_pool=self.token_to_kv_pool,
            bs=bs,
            num_extends=metadata_num_extends,
            output_layout=ForwardOutputLayout(
                metadata_num_extends,
                metadata_num_extends,
                bs - metadata_num_extends,
                self.draft_query_width,
            ),
            input_num_tokens=bs * self.draft_query_width,
            forward_mode=ForwardMode.DECODE,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )

        flat_ids = block_ids.reshape(-1)
        input_embeds = self._first_layer_embeds(flat_ids)

        with torch.inference_mode():
            logits_output = self.draft_model_runner.forward(
                ctx=ctx,
                input_ids=flat_ids,
                positions=block_positions.reshape(-1),
                captured_hidden_states=None,
                input_embeds=input_embeds,
            )

        draft_hidden = logits_output.hidden_states
        if draft_hidden is None:
            raise RuntimeError(
                "Native DFLASH draft model did not return hidden states."
            )
        draft_hidden = draft_hidden.view(bs, self.draft_query_width, self.hidden_size)

        next_tokens = self.next_tokens_buf[:bs]
        return self._sample_block(draft_hidden, block_ids, next_tokens)

    def _sample_block(
        self,
        draft_hidden: torch.Tensor,
        block_ids: torch.Tensor,
        next_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Sample the block draft tokens from the draft hidden states."""
        next_tokens[:, 0] = block_ids[:, 0]
        self._greedy_sample_from_vocab_parallel_head(
            draft_hidden[:, 1:, :].reshape(-1, self.hidden_size),
            out=next_tokens[:, 1:],
        )
        next_tokens.clamp_(min=0)
        return next_tokens

    @nvtx_range("drafter:dflash", color="purple")
    def run(
        self,
        base_ctx: ForwardContext,
        logits_output: LogitsProcessorOutput,
        output_tokens: torch.Tensor,
        accept_lengths: torch.Tensor,
    ) -> torch.Tensor:
        if self.target_model is None:
            raise RuntimeError("DFLASH drafter is not bound to a target model.")

        if base_ctx.target_capture_sink is self and not self._context_written:
            raise RuntimeError(
                "DFLASH armed the capture sink, but the target never handed "
                "over its last tap."
            )
        if not self._context_written:
            self._update_native_cache_from_target(
                base_ctx, logits_output, accept_lengths
            )
        bs = base_ctx.bs
        current_tokens = self.block_ids_buf[:bs, 0]
        if base_ctx.num_extends == 0:
            history = self.attn_backend.draft_history_view()
            max_draft_prefix = history.max_tokens - self.draft_query_width
            dflash_prepare_decode(
                output_tokens=output_tokens,
                accept_lengths=accept_lengths[:bs],
                req_pool_indices=self.input_buffers.req_pool_indices_buf[:bs],
                valid_cache_lengths=self.runtime_states.valid_cache_lengths,
                draft_seq_lens=self.draft_seq_lens_buf[:bs],
                block_ids=self.block_ids_buf[:bs],
                block_positions=self.block_positions_buf[:bs],
                verify_width=self.spec_num_tokens,
                draft_query_width=self.draft_query_width,
                max_draft_prefix=max_draft_prefix,
            )
            return self._draft_native(current_tokens, prepared=True)

        self._current_tokens_from_output(
            output_tokens,
            accept_lengths,
            base_ctx.num_extends,
            self.spec_num_tokens,
            out=current_tokens,
        )
        return self.draft(current_tokens)
