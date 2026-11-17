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

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch
from tokenspeed_kernel.ops.sampling.triton import logprob_topk
from typing_extensions import override

from tokenspeed.runtime.execution.breakable_cuda_graph import (
    BreakableCapture,
    HandoffSlot,
    active_forward,
    run_eager,
)
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.drafter.base import (
    EXTEND_DECODE_MAX_REQUESTS,
    EXTEND_DECODE_SERIES,
    EXTEND_KV_SERIES,
    EXTEND_KV_TILE_ROWS,
    DsaTopKState,
    HiddenDraftInput,
    HiddenStateDrafter,
)
from tokenspeed.runtime.execution.drafter.tree import DraftTree
from tokenspeed.runtime.execution.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
from tokenspeed.runtime.layers.attention.backends.paged.tree_verify import (
    TreeDraftInputs,
)
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.common import maybe_inference_mode
from tokenspeed.runtime.utils.nvtx import nvtx_range

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.drafter.base import DrafterExtendCapture
    from tokenspeed.runtime.execution.input_buffer import InputBuffers
    from tokenspeed.runtime.execution.memory_delta import MemoryDeltaObserver
    from tokenspeed.runtime.execution.model_runner import ModelRunner
    from tokenspeed.runtime.execution.runtime_states import RuntimeStates
    from tokenspeed.runtime.execution.tree_spec import TreeSpec, TreeSpecConfig
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput

logger = get_colorful_logger(__name__)


def _advance_draft_forward_metadata_if_supported(attn_backend, seq_lens) -> None:
    advance = getattr(attn_backend, "advance_draft_forward_metadata", None)
    if advance is not None:
        advance(seq_lens)


@dataclass(frozen=True)
class AcceptedPrefixPublisher:
    """Eagle's :class:`DraftNarrowing` for the step-0 forward.

    ``frontier`` is the round's accepted frontier per request (the target's
    lengths for prompt rows, ``valid_cache_len + accept_len`` for decode
    rows). Publishing copies it into the backend's own seq-lens buffers
    through the drafter hook, so it is in-graph safe and idempotent. Under
    the drafter's breakable extend capture it is a break of its own
    (``run_eager``): the hook also refreshes the per-forward metadata the
    step's indexer last read, which a captured copy would pin to the
    capture's batch.
    """

    attn_backend: AttentionBackend
    frontier: torch.Tensor

    def publish_accepted_prefix(self) -> None:
        run_eager(self._publish)

    def _publish(self) -> None:
        self.attn_backend.advance_draft_forward_metadata(self.frontier)


EagleDraftInput = HiddenDraftInput


@dataclass
class _ExtendGraphBuffers:
    """Fixed-address inputs and outputs of the extend-round graphs.

    The graphs record these addresses; a replay stages the round's live
    values into them first. Sized once, for the plan's largest request count.
    """

    # [requests, width] the capture dummy's hidden rows, one per request.
    hidden_in: torch.Tensor
    # [requests] each request's last row, the rows step 0 narrows to.
    gather_ids: torch.Tensor
    # [requests] the accepted frontier step 0 publishes and the decode steps
    # start writing at.
    frontier: torch.Tensor
    # [requests, spec_num_tokens] the next verify window ``draft`` returns.
    next_tokens: torch.Tensor
    # The decode steps' inputs: step 0's outputs, or a split round's last rows.
    draft_ids: torch.Tensor | None = None
    hidden_out: torch.Tensor | None = None


@dataclass
class _ExtendGraph:
    """One captured part of the extend round and the context it replays under."""

    capture: BreakableCapture
    # The ambient context of the capture and of every replay: step 0's own
    # context, which its breaks read, or for the decode steps a marker whose
    # breaks hold the per-step contexts the capture built (fixed per
    # request count).
    ctx: ForwardContext


@runtime_checkable
class SplitExtendDraft(Protocol):
    """A one-layer draft whose rows meet only at attention: its extend rounds split."""

    def begin_split_round(
        self, ctx: ForwardContext, rows: int
    ) -> tuple[torch.Tensor, ...]:
        """Start a split round: each of ``rows`` rows' draft cache write locations."""

    def write_draft_kv(
        self,
        ctx: ForwardContext,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        captured_hidden_states: torch.Tensor,
        locations: tuple[torch.Tensor, ...],
    ) -> None:
        """Step 0 over every row up to its prologue: write each row's draft cache."""


@dataclass
class _KVTile:
    """The draft K/V writer's fixed-address inputs over one tile, and its graph."""

    ids: torch.Tensor
    positions: torch.Tensor
    # [rows, width] the target's hidden rows.
    hidden: torch.Tensor
    # Each row's write locations; a padded row's are zero.
    locations: tuple[torch.Tensor, ...]
    graph: _ExtendGraph | None = None

    def inputs(self) -> tuple[torch.Tensor, ...]:
        """The staged inputs, in ``write_draft_kv``'s order, the locations last."""
        return (self.ids, self.positions, self.hidden, *self.locations)


class Eagle(HiddenStateDrafter):
    """
    Draft model runner that implements the Eagle/Eagle3 algorithm.
    """

    supports_request_token_history = True

    def __init__(
        self,
        spec_num_tokens: int,
        spec_num_steps: int,
        draft_model_runner: ModelRunner,
        attn_backend: AttentionBackend | None = None,
        token_to_kv_pool: CachePool | None = None,
        runtime_states: RuntimeStates | None = None,
        input_buffers: InputBuffers | None = None,
        vocab_size: int | None = None,
    ) -> None:

        super().__init__(
            spec_num_tokens,
            spec_num_steps,
            draft_model_runner,
            runtime_states=runtime_states,
            input_buffers=input_buffers,
            attn_backend=attn_backend,
            token_to_kv_pool=token_to_kv_pool,
            vocab_size=vocab_size,
        )

        hot_token_ids = draft_model_runner.model.get_hot_token_id()

        if hot_token_ids is not None:
            self.hot_token_ids = hot_token_ids.to(self.device)
        else:
            self.hot_token_ids = None

        # For constructing fallback global_num_tokens during CUDA graph capture.
        self.dp_size = draft_model_runner.mapping.attn.dp_size
        self.world_size = draft_model_runner.mapping.world_size

        # Draft-side request-token history (e.g. a draft with its own n-gram
        # over-embedding). The drafter owns the per-slot write frontier: it
        # advances through the speculative chain, unlike valid_cache_lengths.
        self.draft_reads_token_history: bool = bool(
            draft_model_runner.model_config.requires_request_token_history
        )
        self.draft_history_lengths_buf: torch.Tensor | None = None
        self.draft_history_offsets_buf: torch.Tensor | None = None
        if self.draft_reads_token_history:
            target_history = self.runtime_states.request_token_history_ids
            if target_history is None:
                raise NotImplementedError(
                    "a draft model reading request-token history requires the "
                    "target model to keep one (the capacity comes from it)"
                )
            self.runtime_states.init_draft_request_token_history(
                target_history.shape[1]
            )
            pool_size = self.runtime_states.valid_cache_lengths.shape[0]
            self.draft_history_lengths_buf = torch.zeros(
                (pool_size,), dtype=torch.int32, device=self.device
            )
            self.draft_history_offsets_buf = torch.arange(
                self.input_buffers.max_bs + 1, dtype=torch.int32, device=self.device
            )

        # Draft-tree state (bind_tree); None for chains.
        self.tree_spec: TreeSpec | None = None
        self.draft_tree: DraftTree | None = None
        self.tree_lanes: TreeDraftInputs | None = None
        # Rows each request drafts per step: 1 for a chain, K for a draft tree.
        self.lanes_per_request = 1
        # Step 0's lane ancestor masks, lane r seeing only its own slot.
        self._tree_seed_lane_mask: torch.Tensor | None = None
        # Width of the full draft vocabulary the tree ranks (bind_tree).
        self.tree_vocab_size: int | None = None

        # Extend-round graphs and their one pool and handoff map; empty until captured.
        self._extend_buffers: _ExtendGraphBuffers | None = None
        self._decode_steps_graphs: dict[int, _ExtendGraph] = {}
        self._extend_graph_pool: Any | None = None
        self._extend_graph_handoffs: dict[HandoffSlot, torch.Tensor] = {}
        self._extend_graph_engaged = False
        self._kv_tile: _KVTile | None = None
        self._split_draft = isinstance(draft_model_runner.model, SplitExtendDraft)

    @override
    def slot_state_rows(self, slot: int) -> list[torch.Tensor]:
        # The draft history's write frontier is drafter-owned; the history
        # rows themselves are token-derived and reseeded (RuntimeStates).
        if self.draft_history_lengths_buf is None:
            return []
        return [self.draft_history_lengths_buf[slot]]

    def bind_tree(self, tree_spec: TreeSpec) -> None:
        """Draft trees: top-K lanes per step (tree.py) in the draft paged cache's lane window."""
        if self.draft_reads_token_history:
            raise NotImplementedError(
                "tree drafting does not support draft token history yet"
            )
        config = tree_spec.config
        max_bs = self.input_buffers.max_bs
        logits_processor = self.draft_model_runner.model.logits_processor
        # The fused distributed argmax leaves each TP rank only its vocab shard of the logits.
        logits_processor.do_argmax = False
        self.tree_vocab_size = (
            self.hot_token_ids.numel()
            if self.hot_token_ids is not None
            else logits_processor.config.vocab_size
        )
        self.tree_spec = tree_spec
        self.draft_tree = DraftTree(
            max_bs, config.topk, config.num_steps, config.num_nodes, self.device
        )
        self._bind_tree_lanes(config, max_bs)

    def _bind_tree_lanes(self, config: TreeSpecConfig, max_bs: int) -> None:
        """Bind the top-K draft lanes, as every tree drafter does."""
        self.tree_lanes = self._make_tree_lanes(config, max_bs)
        self.attn_backend.bind_tree_draft(self.tree_lanes)
        self.lanes_per_request = config.topk
        self._tree_seed_lane_mask = torch.ones(
            config.topk, dtype=torch.int64, device=self.device
        ) << torch.arange(config.topk, device=self.device)

    def _make_tree_lanes(self, config: TreeSpecConfig, max_bs: int) -> TreeDraftInputs:
        """The draft lanes ``_bind_tree_lanes`` binds."""
        return TreeDraftInputs(
            topk=config.topk,
            num_steps=config.num_steps,
            max_bs=max_bs,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _model_shares_mtp_topk(self) -> bool:
        return bool(
            getattr(
                self.draft_model_runner.model,
                "index_share_for_mtp_iteration",
                False,
            )
        )

    def _attach_dsa_topk(self, dsa_topk: DsaTopKState) -> None:
        """Hand the draft backend the top-k this step reuses -- or nothing.

        Always written: the draft backend's share is per forward, and the
        drafter, not a metadata refresh, is what separates its steps."""
        share = self.attn_backend.sparse_topk
        # QSA layout and logical candidates belong to one model invocation.
        # Draft steps reuse physical slots and rebuild their row geometry.
        share.qsa_metadata = None
        share.qsa_selection = None
        if self._model_shares_mtp_topk():
            share.prefill, share.decode = dsa_topk
        else:
            share.clear()

    def _extract_dsa_topk(self, dsa_topk: DsaTopKState) -> DsaTopKState:
        if not self._model_shares_mtp_topk():
            return dsa_topk
        share = self.attn_backend.sparse_topk
        return share.prefill, share.decode

    @override
    def _target_dsa_topk(self, base_ctx: ForwardContext) -> DsaTopKState:
        """The target's last indexer layer left its selection on the target
        backend; the MTP head that shares it starts from there."""
        if not self._model_shares_mtp_topk():
            return (None, None)
        share = base_ctx.attn_backend.sparse_topk
        return share.prefill, share.decode

    def _map_hot(self, ids: torch.Tensor) -> torch.Tensor:
        """Map token ids through hot_token_ids if available, otherwise return as-is."""
        return self.hot_token_ids[ids] if self.hot_token_ids is not None else ids

    @override
    def draft_vocab_map(self) -> torch.Tensor | None:
        return self.hot_token_ids

    def _get_first_step_input(
        self,
        draft_input: EagleDraftInput,
        bs: int,
        input_num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (input_ids, gather_ids) for the first draft step.

        The first-step input shape matches the base model's: ragged
        ``[prefill_part || decode_part]`` under MIXED, full prefill chunks
        under EXTEND, ``base_model_output`` directly under DECODE. Under a
        query shard the ids are the shard's rows of the shifted prefill
        ids (``QueryShardPlan.local_row_ids``) while ``gather_ids`` keep the batch's full layout, as on the
        target's forward: the draft model's exit selects the shard's rows
        with ``QueryShardPlan.local_sampled_index`` when it gathers the
        sampled rows across the group.
        """
        num_extends = draft_input.num_extends
        num_decodes = bs - num_extends
        if num_extends > 0:
            num_decode_tokens = num_decodes * self.spec_num_tokens
            num_prefill_tokens = input_num_tokens - num_decode_tokens

            input_ids = self.input_buffers.shifted_prefill_ids_buf[:input_num_tokens]
            unpadded_input_lengths = self.input_buffers.input_lengths_buf[:bs]
            if num_decodes > 0:
                input_ids[num_prefill_tokens:].copy_(
                    draft_input.base_model_output[num_extends:]
                )
                unpadded_input_lengths[num_extends:].copy_(
                    draft_input.accept_lengths[num_extends:]
                )

            last_indices = unpadded_input_lengths[:num_extends].cumsum(0) - 1
            last_input_ids = input_ids[last_indices]
            input_ids[last_indices] = torch.where(
                last_input_ids == -1,
                draft_input.base_model_output[:num_extends],
                last_input_ids,
            )

            gather_ids = last_indices
            if num_decodes > 0:
                gather_ids = torch.cat(
                    [
                        gather_ids,
                        self.padded_gather_ids_offsets_buf[:num_decodes]
                        + draft_input.accept_lengths[num_extends:]
                        + num_prefill_tokens,
                    ]
                )
            plan = draft_input.query_shard
            if plan is not None and plan.size > 1:
                if num_decodes > 0:
                    raise RuntimeError(
                        "a query-sharded draft step runs pure extend rounds"
                    )
                input_ids = input_ids[plan.local_row_ids]
        else:
            input_ids = draft_input.base_model_output
            gather_ids = (
                self.padded_gather_ids_offsets_buf[:bs] + draft_input.accept_lengths
            )

        return input_ids, gather_ids

    def _accepted_frontier(
        self,
        bs: int,
        draft_input: EagleDraftInput,
    ) -> torch.Tensor:
        """Per-request accepted frontier after this round's verify: the
        target's lengths for prompt rows, ``valid_cache_len + accept_len``
        for decode rows (the verify window minus the rejected tail). Step 0
        attends over it from the live rows; step ``i >= 1`` writes at
        ``frontier + i - 1`` and attends over ``frontier + i``."""
        num_extends = draft_input.num_extends
        frontier = self.input_buffers.seq_lens_buf[:bs].clone()
        if bs > num_extends:
            req_pool_indices = self.input_buffers.req_pool_indices_buf[num_extends:bs]
            frontier[num_extends:] = (
                self.runtime_states.valid_cache_lengths.index_select(
                    0, req_pool_indices
                )
                + draft_input.accept_lengths[num_extends:]
            )
        return frontier

    @nvtx_range("draft_first_step", color="purple")
    def _run_first_step(
        self,
        bs: int,
        draft_input: EagleDraftInput,
        narrowing: AcceptedPrefixPublisher,
    ) -> tuple[LogitsProcessorOutput, DsaTopKState]:

        buffers = self.input_buffers
        forward_mode = draft_input.forward_mode

        input_ids, gather_ids = self._get_first_step_input(
            draft_input, bs, draft_input.input_num_tokens
        )
        draft_model = self.draft_model_runner.model
        input_num_tokens = draft_input.input_num_tokens

        ctx = ForwardContext(
            attn_backend=self.attn_backend,
            token_to_kv_pool=self.token_to_kv_pool,
            bs=bs,
            num_extends=draft_input.num_extends,
            output_layout=ForwardOutputLayout(
                draft_input.num_extends,
                draft_input.num_extends,
                bs - draft_input.num_extends,
                1,
            ),
            input_num_tokens=input_num_tokens,
            forward_mode=forward_mode,
            capture_hidden_mode=CaptureHiddenMode.LAST,
            gather_ids=gather_ids,
            global_num_tokens=draft_input.global_num_tokens,
            global_bs=draft_input.global_bs,
            all_decode_or_idle=draft_input.all_decode_or_idle,
            draft_narrowing=narrowing,
            query_shard=draft_input.query_shard,
        )
        # The step-0 rows: the whole span, or the target's shard of it.
        rows = (
            slice(0, input_num_tokens)
            if draft_input.query_shard is None
            else draft_input.query_shard.local_row_ids
        )

        dsa_topk = draft_input.dsa_topk
        prepare_dsa_topk = getattr(draft_model, "prepare_dsa_topk_for_mtp_decode", None)
        compute_dsa_topk_first_step = bool(
            getattr(draft_model, "compute_dsa_topk_first_step", False)
        )
        if compute_dsa_topk_first_step:
            # The draft model has its own sparse indexer weights. Compute
            # first-step top-k, then select rows used by later MTP steps.
            dsa_topk = (None, None)
        elif draft_input.num_extends == 0 and prepare_dsa_topk is not None:
            dsa_topk = prepare_dsa_topk(dsa_topk, gather_ids)
        else:
            dsa_topk = (None, None)
        self._attach_dsa_topk(dsa_topk)

        history_kwargs = {}
        if self.draft_reads_token_history:
            # The step-0 rows mirror the target forward's packed layout, so
            # the offsets and mask the executor prepared for the target apply
            # verbatim; the write frontier is the pre-forward committed length.
            history_kwargs["request_token_history"] = (
                self.runtime_states.draft_request_token_history_view(
                    req_pool_indices=buffers.req_pool_indices_buf[:bs],
                    input_start_offsets=buffers.input_start_offsets_buf[: bs + 1],
                    active_request_mask=buffers.active_request_mask_buf[:bs],
                    committed_lengths=self.runtime_states.valid_cache_lengths,
                )
            )
        logits_output = self.draft_model_runner.forward(
            ctx=ctx,
            input_ids=input_ids,
            positions=buffers.positions_buf[rows],
            captured_hidden_states=draft_input.base_out_hidden_states,
            spec_step_idx=0,
            **history_kwargs,
        )
        dsa_topk = self._extract_dsa_topk(dsa_topk)
        if compute_dsa_topk_first_step and prepare_dsa_topk is not None:
            dsa_topk = prepare_dsa_topk(
                dsa_topk,
                gather_ids,
                num_prefill_rows=draft_input.num_extends,
            )
        return logits_output, dsa_topk

    @nvtx_range("draft_multi_step", color="purple")
    def _run_multi_step_decode(
        self,
        bs: int,
        draft_ids: torch.Tensor,
        hidden: torch.Tensor,
        next_tokens: torch.Tensor,
        dsa_topk: DsaTopKState,
        frontier: torch.Tensor,
        *,
        global_num_tokens: list[int] | None,
        global_bs: list[int] | None,
        all_decode_or_idle: bool,
        first_input_ids: torch.Tensor | None,
    ) -> None:
        """Draft steps ``1 .. S - 1``: ``lanes`` rows per request (1 for a
        chain, K for a draft tree) forward from the accepted frontier.

        ``first_input_ids`` (last-row tokens) start at step 0, one below the frontier.

        ``global_num_tokens`` / ``global_bs`` / ``all_decode_or_idle`` are the
        round's attention-DP sizing (the target forward's). The steps that
        publish backend state between the forwards go through ``run_eager``:
        a direct call here, and under the extend-round breakable capture a
        break that re-runs on every replay (the backend's write window and
        seq_lens publication carry host state the next breaks read).
        """
        lanes = self.lanes_per_request
        first_step = 1 if first_input_ids is None else 0
        # Step 1 writes at the accepted frontier (vc + accept_length after the
        # target's verify) so rotary/cache metadata stay on the accepted
        # prefix, not the rejected tail.
        cache_start = frontier if first_step == 1 else frontier - 1

        # +1 is the kernel's read-inclusive convention; advanced per iter.
        draft_seq_lens = self.draft_seq_lens_buf[:bs]
        torch.add(cache_start, 1, out=draft_seq_lens)

        positions = cache_start.repeat_interleave(lanes)
        # Step i's write window: a chain's one advancing slot, or K lane slots per request.
        slot_starts = positions if self.tree_lanes is None else cache_start.clone()

        history_pool_indices = None
        if self.draft_reads_token_history:
            # Step i appends its one input token at frontier + (i - 1), the
            # same advancing position the KV chain writes. Tensor-only writes
            # on persistent buffers, so graph capture records the update.
            history_pool_indices = self.input_buffers.req_pool_indices_buf[:bs]
            self.draft_history_lengths_buf.index_copy_(
                0, history_pool_indices, cache_start.to(torch.int32)
            )

        for i in range(first_step, self.spec_num_steps):
            # make a ctx every time model runner forward
            # Multi-step decode is pure DECODE mode: one token per request.
            # global_num_tokens must reflect each rank's batch size, not the
            # target model's total tokens (which may be bs * spec_num_tokens).
            step_global_num_tokens = global_num_tokens

            if self.dp_size > 1:
                if global_bs is not None:
                    step_global_num_tokens = global_bs
                else:
                    # CUDA graph capture path: uniform batch size across ranks.
                    step_global_num_tokens = [bs] * self.world_size

            ctx = ForwardContext(
                bs=bs,
                num_extends=0,
                output_layout=ForwardOutputLayout(0, 0, bs, lanes),
                attn_backend=self.attn_backend,
                token_to_kv_pool=self.token_to_kv_pool,
                input_num_tokens=bs * lanes,
                forward_mode=ForwardMode.DECODE,
                capture_hidden_mode=CaptureHiddenMode.LAST,
                global_num_tokens=step_global_num_tokens,
                global_bs=global_bs,
                all_decode_or_idle=all_decode_or_idle,
            )

            def publish_step(topk: DsaTopKState = dsa_topk) -> None:
                self._attach_dsa_topk(topk)
                if self.tree_lanes is None:
                    # Keep attention metadata on the accepted prefix; rejected
                    # verify tail slots may still contain stale draft KV.
                    _advance_draft_forward_metadata_if_supported(
                        self.attn_backend, draft_seq_lens
                    )
                else:
                    # Tree lanes attend over the accepted prefix and their
                    # window (TreeDraftInputs).
                    self.tree_lanes.active = True
                # Publish this step's one write slot per request (the chain's
                # advancing position, seq_len - 1): step i writes position
                # cache_start + (i - 1).
                self.attn_backend.publish_draft_step_locations(
                    cache_start=slot_starts,
                    num_tokens=lanes,
                )

            run_eager(publish_step)

            history_kwargs = {}
            if self.draft_reads_token_history:
                history_kwargs["request_token_history"] = (
                    self.runtime_states.draft_request_token_history_view(
                        req_pool_indices=history_pool_indices,
                        input_start_offsets=self.draft_history_offsets_buf[: bs + 1],
                        active_request_mask=(
                            self.input_buffers.active_request_mask_buf[:bs]
                        ),
                        committed_lengths=self.draft_history_lengths_buf,
                    )
                )
            # The lane flag ends with this forward, even one that raises.
            lane_forward = (
                nullcontext()
                if self.tree_lanes is None
                else self.tree_lanes.lane_forward()
            )
            with nvtx_range("draft_forward", color="red"), lane_forward:
                logits_output = self.draft_model_runner.forward(
                    ctx=ctx,
                    input_ids=(
                        first_input_ids
                        if i == 0
                        else self._map_hot(draft_ids.reshape(-1))
                    ),
                    positions=positions,
                    captured_hidden_states=hidden,
                    spec_step_idx=i,
                    **history_kwargs,
                )
                dsa_topk = self._extract_dsa_topk(dsa_topk)
            if self.draft_reads_token_history and i + 1 < self.spec_num_steps:
                self.draft_history_lengths_buf.index_copy_(
                    0,
                    history_pool_indices,
                    draft_seq_lens.to(torch.int32),
                )

            with nvtx_range("draft_sample", color="yellow"):
                if self.tree_lanes is None:
                    draft_ids = self.sample_draft_step(logits_output, step=i)
                    # Column 0 holds last_verified_ids; drafter writes step `i` into column `i + 1`.
                    next_tokens[:, i + 1] = self._map_hot(draft_ids)
                    hidden = logits_output.hidden_states
                else:
                    draft_ids, hidden = self._expand_tree_lanes(bs, i, logits_output)
                if i + 1 < self.spec_num_steps:
                    positions.add_(1)
                    draft_seq_lens.add_(1)
                    if self.tree_lanes is not None:
                        slot_starts.add_(lanes)

    def _seed_tree_lanes(
        self,
        bs: int,
        logits_output: LogitsProcessorOutput,
        frontier: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Step 0 of a draft tree: the best K candidates become the lanes.
        Returns their ``[bs, K]`` tokens and ``[bs * K, hidden]`` rows."""
        topk = self.lanes_per_request
        lane_tokens = self.draft_tree.seed(bs, *self._score_candidates(logits_output))
        self.tree_lanes.lane_mask[: bs * topk].view(bs, topk).copy_(
            self._tree_seed_lane_mask
        )
        self.tree_lanes.set_frontier(bs, frontier)
        return lane_tokens, logits_output.hidden_states.repeat_interleave(topk, dim=0)

    def _expand_tree_lanes(
        self, bs: int, step: int, logits_output: LogitsProcessorOutput
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Keep each request's best K children as the next lanes; returns
        their tokens and hidden rows (``None`` after the last step)."""
        topk = self.lanes_per_request
        last = step == self.spec_num_steps - 1
        next_hidden = None if last else torch.empty_like(logits_output.hidden_states)
        lane_mask = self.tree_lanes.lane_mask[: bs * topk].view(bs, topk)
        lane_tokens = self.draft_tree.expand(
            bs,
            step,
            *self._score_candidates(logits_output),
            None if last else (lane_mask, logits_output.hidden_states, next_hidden),
        )
        return lane_tokens, next_hidden

    def _draft_window(self, bs: int, next_tokens: torch.Tensor) -> torch.Tensor:
        """The next verify window: the chain's tokens, or the best tree's
        (whose parents become the next round's tree)."""
        if self.tree_spec is None:
            return next_tokens
        tokens, parent = self.draft_tree.finalize(bs, next_tokens[:, 0])
        tokens[:, 1:] = self._map_hot(tokens[:, 1:].long()).to(torch.int32)
        self.tree_spec.draft_parent_buf[:bs].copy_(parent)
        return tokens

    def _score_candidates(
        self, logits_output: LogitsProcessorOutput
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Each draft row's best K candidates: ``[rows, K]`` log-probabilities
        (best first) and tokens. The one place a drafter decides how the tree
        scores candidates; DraftTree only records and selects."""
        logits = logits_output.next_token_logits
        if logits.shape[-1] != self.tree_vocab_size:
            raise RuntimeError(
                f"tree drafting ranks all {self.tree_vocab_size} draft tokens, "
                f"got logits of width {logits.shape[-1]}"
            )
        return logprob_topk(logits, self.draft_tree.topk)

    # ------------------------------------------------------------------
    # Public entry point (type-based dispatch from ModelExecutor)
    # ------------------------------------------------------------------

    def _last_verified_window(
        self,
        bs: int,
        draft_input: EagleDraftInput,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The next verify window, every column the request's last verified id.

        Written into ``out`` (``[bs, spec_num_tokens]`` int32) when given --
        the extend-round graphs' fixed window -- else into a fresh tensor.
        """
        next_tokens = (
            torch.empty(
                (bs, self.spec_num_tokens),
                dtype=torch.int32,
                device=self.device,
            )
            if out is None
            else out
        )

        # Last verified id per request → next_tokens[:, 0].
        num_extends = draft_input.num_extends
        num_decodes = bs - num_extends
        if num_extends > 0:
            next_tokens[:num_extends, 0] = draft_input.base_model_output[:num_extends]
        if num_decodes > 0:
            indices = (
                self.padded_gather_ids_offsets_buf[:num_decodes]
                + draft_input.accept_lengths[num_extends:]
            )
            if num_extends > 0:
                indices.add_(num_extends)
            torch.index_select(
                draft_input.base_model_output,
                0,
                indices,
                out=next_tokens[num_extends:, 0],
            )
        if self.spec_num_tokens > 1:
            next_tokens[:, 1:] = next_tokens[:, :1]
        return next_tokens

    @override
    def draft(
        self,
        draft_input: EagleDraftInput,
    ) -> torch.Tensor:

        bs = draft_input.accept_lengths.shape[0]
        if self._splits_round(bs, draft_input):
            return self._draft_split_extend(bs, draft_input)
        # Step 0 eager, then the decode steps eager or replayed (the same steps).
        replay = self._replays_decode_steps(bs, draft_input)
        buffers = self._extend_buffers

        # Layout: column 0 holds the last verified id (the base model's accepted token);
        # columns 1..spec_num_steps hold the drafter's speculative tokens.
        next_tokens = self._last_verified_window(
            bs,
            draft_input,
            out=buffers.next_tokens[:bs] if replay else None,
        )

        # The runner refreshed the draft decode metadata over the target's
        # post-verify lengths (vc + N). Step 0 narrows to the live rows, whose
        # context is the accepted frontier; the model publishes it through
        # this handle at the moment it switches to those rows.
        frontier = self._accepted_frontier(bs, draft_input)
        if replay:
            # The graph reads the frontier where it recorded it.
            frontier = buffers.frontier[:bs].copy_(frontier)
        narrowing = AcceptedPrefixPublisher(self.attn_backend, frontier)

        # Step 0: the logits processor prunes its outputs to one row per request.
        logits_output, dsa_topk = self._run_first_step(bs, draft_input, narrowing)

        if self.tree_lanes is None:
            draft_ids = self.sample_draft_step(logits_output, step=0)
            next_tokens[:, 1] = self._map_hot(draft_ids)
            hidden = logits_output.hidden_states
        else:
            draft_ids, hidden = self._seed_tree_lanes(bs, logits_output, frontier)

        if self.spec_num_steps <= 1:
            return self._draft_window(bs, next_tokens)

        if self.input_buffers.all_extends_mid_chunk and self.dp_size == 1:
            # Skip multi-step when the whole batch is mid-chunk EXTEND:
            # no request completes a target-side speculative verification
            # after this forward, so any speculative tokens would be discarded.
            #
            # In DP we still run, because peer ranks may have completing
            # extends or decodes; diverging here would desync the drafter's
            # dense-TP / MoE-EP collectives (NCCL hang or RSAG mismatch).
            if self.tree_spec is not None:
                self.tree_spec.draft_parent_buf[:bs].copy_(self.tree_spec.chain_parent)
            return next_tokens

        # Draft step 2+ (multi-step decode).
        # Multi-step decode operates on full bs; drop the [num_extends:]
        # slice that step 0 may have set up for MIXED target. No-op on
        # backends that fill separate prefill/decode metadata at init
        # time.
        with self.attn_backend.override_num_extends(0):
            if replay:
                # Hand step 0's outputs to the graph's inputs.
                buffers.draft_ids[:bs].copy_(draft_ids)
                buffers.hidden_out[:bs].copy_(hidden)
                self._replay_decode_steps(bs)
            else:
                self._run_multi_step_decode(
                    bs,
                    draft_ids,
                    hidden,
                    next_tokens,
                    dsa_topk,
                    frontier,
                    global_num_tokens=draft_input.global_num_tokens,
                    global_bs=draft_input.global_bs,
                    all_decode_or_idle=draft_input.all_decode_or_idle,
                    first_input_ids=None,
                )
        return self._draft_window(bs, next_tokens)

    # ------------------------------------------------------------------
    # Extend-round CUDA graphs
    # ------------------------------------------------------------------
    #
    # The decode steps replay per request count; a split draft's cache, per tile.

    def _split_extend_supported(self) -> bool:
        """Whether extend rounds split: a split draft's chain, without attention DP."""
        return (
            self._split_draft
            and self.dp_size == 1
            and self.tree_spec is None
            and self.draft_sampler is None
            and not self.draft_reads_token_history
            and not self._model_shares_mtp_topk()
        )

    def _splits_round(self, bs: int, draft_input: EagleDraftInput) -> bool:
        """Whether this round splits: a pure extend, no DP sizing or query shard."""
        shard = draft_input.query_shard
        return (
            draft_input.forward_mode.is_extend()
            and draft_input.num_extends == bs
            and draft_input.global_num_tokens is None
            and (shard is None or shard.size == 1)
            and draft_input.base_out_hidden_states.shape[0]
            == draft_input.input_num_tokens
            and self._split_extend_supported()
        )

    def _draft_split_extend(
        self, bs: int, draft_input: EagleDraftInput
    ) -> torch.Tensor:
        """Every row's draft K/V, then steps 0 .. S - 1 from each request's last row."""
        graph = self._decode_steps_graphs.get(bs)
        buffers = self._extend_buffers
        next_tokens = self._last_verified_window(
            bs, draft_input, out=None if graph is None else buffers.next_tokens[:bs]
        )
        frontier = self._accepted_frontier(bs, draft_input)
        input_ids, gather_ids = self._get_first_step_input(
            draft_input, bs, draft_input.input_num_tokens
        )
        # The steps attend over the prefix: wait for the draft pool's Host load.
        tracker = self.token_to_kv_pool.layerwise_load_tracker
        if tracker is not None:
            tracker.wait_for_all_layers()
        self._write_draft_kv(bs, draft_input, input_ids)
        if self.input_buffers.all_extends_mid_chunk:
            # No request completes a verify after this forward (see draft).
            return next_tokens
        tokens = input_ids.index_select(0, gather_ids)
        hidden = draft_input.base_out_hidden_states.index_select(0, gather_ids)
        with self.attn_backend.override_num_extends(0):
            if graph is None:
                self._run_multi_step_decode(
                    bs,
                    tokens,
                    hidden,
                    next_tokens,
                    (None, None),
                    frontier,
                    global_num_tokens=None,
                    global_bs=None,
                    all_decode_or_idle=draft_input.all_decode_or_idle,
                    first_input_ids=tokens,
                )
                return next_tokens
            buffers.frontier[:bs].copy_(frontier)
            buffers.draft_ids[:bs].copy_(tokens)
            buffers.hidden_out[:bs].copy_(hidden)
            if not self._extend_graph_engaged:
                self._extend_graph_engaged = True
                logger.info(
                    "drafter prefill graph ENGAGED: split draft, the steps from "
                    f"step 0 at {bs} requests"
                )
            self._replay_decode_steps(bs)
        return next_tokens

    def _write_draft_kv(
        self, bs: int, draft_input: EagleDraftInput, input_ids: torch.Tensor
    ) -> None:
        """Write every row's draft K/V, tile by tile when the writer's graph exists."""
        num_tokens = draft_input.input_num_tokens
        model = self.draft_model_runner.model
        ctx = ForwardContext(
            attn_backend=self.attn_backend,
            token_to_kv_pool=self.token_to_kv_pool,
            bs=bs,
            num_extends=bs,
            output_layout=ForwardOutputLayout(bs, bs, 0, 1),
            input_num_tokens=num_tokens,
            forward_mode=draft_input.forward_mode,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )
        # The round's own layout: no top-k or row layout of the last step.
        self._attach_dsa_topk((None, None))
        locations = model.begin_split_round(ctx, num_tokens)
        positions = self.input_buffers.positions_buf[:num_tokens]
        hidden = draft_input.base_out_hidden_states
        tile = self._kv_tile
        if tile is None or tile.graph is None:
            model.write_draft_kv(ctx, input_ids, positions, hidden, locations)
            return
        if hidden.dtype != tile.hidden.dtype or hidden.shape[1] != tile.hidden.shape[1]:
            raise RuntimeError(
                f"the target's hidden rows ({hidden.dtype}, width {hidden.shape[1]}) "
                "are not the ones the draft K/V writer reads "
                f"({tile.hidden.dtype}, width {tile.hidden.shape[1]})"
            )
        rows = tile.ids.shape[0]
        for start in range(0, num_tokens, rows):
            n = min(rows, num_tokens - start)
            sources = (input_ids, positions, hidden, *locations)
            for buf, src in zip(tile.inputs(), sources, strict=True):
                buf[:n].copy_(src[start : start + n])
                # Padded rows: finite inputs at zero locations.
                buf[n:].zero_()
            with active_forward(tile.graph.ctx):
                tile.graph.capture.replay()

    def _capture_kv_tile(
        self,
        rows: int,
        capture: DrafterExtendCapture,
        observer: AbstractContextManager[None],
    ) -> None:
        """Warm up and capture the draft K/V writer over one tile of ``rows`` rows."""
        model = self.draft_model_runner.model
        # A dummy extend of one tile, for the write locations' shapes.
        capture.prepare_batch(rows, 1)
        ctx = ForwardContext(
            attn_backend=self.attn_backend,
            token_to_kv_pool=self.token_to_kv_pool,
            bs=1,
            num_extends=1,
            output_layout=ForwardOutputLayout(1, 1, 0, 1),
            input_num_tokens=rows,
            forward_mode=ForwardMode.EXTEND,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )
        self._attach_dsa_topk((None, None))
        locations = model.begin_split_round(ctx, rows)
        width, dtype = capture.target_hidden_width, capture.target_hidden_dtype
        tile = self._kv_tile
        if (
            tile is None
            or tile.hidden.shape != (rows, width)
            or tile.hidden.dtype != dtype
            or [(t.dtype, t.shape[1:]) for t in tile.locations]
            != [(t.dtype, t.shape[1:]) for t in locations]
        ):
            # Outside the measured capture: the utilization headroom funds them.
            ib = self.input_buffers
            tile = self._kv_tile = _KVTile(
                ids=torch.empty(
                    rows, dtype=ib.shifted_prefill_ids_buf.dtype, device=self.device
                ),
                positions=torch.empty(
                    rows, dtype=ib.positions_buf.dtype, device=self.device
                ),
                hidden=torch.empty((rows, width), dtype=dtype, device=self.device),
                locations=tuple(
                    torch.empty((rows, *t.shape[1:]), dtype=t.dtype, device=self.device)
                    for t in locations
                ),
            )
        # Every row a padded one: the warmups write only at zero locations.
        for buf in tile.inputs():
            buf.zero_()

        def body() -> None:
            model.write_draft_kv(
                ctx, tile.ids, tile.positions, tile.hidden, tile.locations
            )

        cap = BreakableCapture(
            pool=self._extend_graph_pool,
            stream=capture.stream,
            handoff_storage=self._extend_graph_handoffs,
        )
        cap.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(cap.stream):
            for _ in range(capture.num_warmup):
                body()
        torch.cuda.synchronize()
        with observer, active_forward(ctx), cap:
            body()
        if self._extend_graph_pool is None:
            self._extend_graph_pool = cap.pool
        # Capture records kernels without executing them: smoke-test a replay.
        with active_forward(ctx):
            cap.replay()
        tile.graph = _ExtendGraph(cap, ctx)

    def _stage_split_inputs(self, bs: int) -> None:
        """A capture's split-step inputs: a valid token and a zero hidden row each."""
        buffers = self._extend_buffers
        if buffers.draft_ids is None:
            rows = buffers.gather_ids.shape[0]
            buffers.draft_ids = torch.empty(
                rows,
                dtype=self.input_buffers.shifted_prefill_ids_buf.dtype,
                device=self.device,
            )
            buffers.hidden_out = torch.empty(
                (rows, buffers.hidden_in.shape[1]),
                dtype=buffers.hidden_in.dtype,
                device=self.device,
            )
        buffers.draft_ids[:bs].fill_(1)
        buffers.hidden_out[:bs].zero_()

    def _decode_steps_graphs_supported(self) -> bool:
        """Whether this configuration's extend-round decode steps can be recorded.

        The graphs bake a chain's one row per request, the argmax proposal (no
        draft sampler), no draft token history and no top-k carried across
        steps, and one replay decision shared by every rank that runs the
        round's collectives: attention DP decides per rank, and the pipeline's
        chunk rounds are not validated. Tensor- and expert-parallel ranks all
        run the same round.
        """
        return (
            self.spec_num_steps > 1
            and self.tree_spec is None
            and self.draft_sampler is None
            and not self.draft_reads_token_history
            and not self._model_shares_mtp_topk()
            and self.dp_size == 1
            and self.draft_model_runner.mapping.pp_size == 1
        )

    @override
    def extend_graph_plan(
        self, token_buckets: Sequence[int]
    ) -> dict[str, list[tuple[int, int]]]:
        plan = {}
        if self._split_extend_supported():
            # Its K/V writer goes with the steps, which every split round runs.
            if self._decode_steps_graphs_supported() and token_buckets:
                plan[EXTEND_KV_SERIES] = [
                    (min(EXTEND_KV_TILE_ROWS, max(token_buckets)), 1)
                ]
        if self._decode_steps_graphs_supported():
            # The steps bake their request count exactly.
            most = min(self.input_buffers.max_bs, EXTEND_DECODE_MAX_REQUESTS)
            plan[EXTEND_DECODE_SERIES] = [(bs, bs) for bs in range(most, 0, -1)]
        return plan

    @override
    def release_extend_graphs(self) -> None:
        """Drop the extend-round graphs and their pool (they name the cache
        pool's buffers); the static buffers stay for the next capture."""
        self._decode_steps_graphs.clear()
        if self._kv_tile is not None:
            self._kv_tile.graph = None
        self._extend_graph_pool = None
        self._extend_graph_handoffs = {}

    @override
    def capture_extend_graphs(
        self, capture: DrafterExtendCapture, *, observer: MemoryDeltaObserver
    ) -> None:
        """Capture the sampled decode steps, then the K/V writer, in ladder order.

        Each graph is warmed up eagerly on the capture stream first, so the
        lazily allocated kernel state lands outside the graph pool. The graphs
        share one private pool, apart from the target's prefill and decode
        pools: an eager op that memoized a buffer inside one of those pools
        would be rewritten by its replays (see ``PrefillGraph.capture``).
        """
        if capture.num_warmup < 1:
            raise ValueError("the drafter prefill graphs need at least one warmup")
        decode_plan = capture.plan.get(EXTEND_DECODE_SERIES, ())
        decode_counts = (
            [decode_plan[i][1] for i in capture.ladders[EXTEND_DECODE_SERIES].sampled]
            if decode_plan
            else []
        )
        self.release_extend_graphs()
        # Sized for the whole plan, so a full capture reuses what a probe allocated.
        most = max((bs for _, bs in decode_plan), default=1)
        self._allocate_extend_buffers(capture, requests=most)
        with maybe_inference_mode():
            for bs in decode_counts:
                capture.prepare_batch(bs, bs)
                self._stage_capture_inputs(bs)
                if self._split_extend_supported():
                    self._stage_split_inputs(bs)
                else:
                    self._run_first_step_eager(bs, capture.stream)
                self._decode_steps_graphs[bs] = self._capture_decode_steps(
                    bs, capture, observer.measure(EXTEND_DECODE_SERIES)
                )
            kv_plan = capture.plan.get(EXTEND_KV_SERIES, ())
            if kv_plan and capture.ladders[EXTEND_KV_SERIES].sampled:
                self._capture_kv_tile(
                    kv_plan[0][0], capture, observer.measure(EXTEND_KV_SERIES)
                )
        buffers = self._extend_buffers
        tile = self._kv_tile
        static_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                buffers.hidden_in,
                buffers.gather_ids,
                buffers.frontier,
                buffers.next_tokens,
                buffers.draft_ids,
                buffers.hidden_out,
                *(() if tile is None else tile.inputs()),
            )
            if tensor is not None
        )
        kv_writer = (
            ""
            if tile is None or tile.graph is None
            else f", the draft K/V writer at {tile.ids.shape[0]} rows"
        )
        logger.info(
            "drafter prefill graph: captured decode steps at requests "
            f"{sorted(self._decode_steps_graphs)!s}{kv_writer} (eager attention "
            f"breaks); static buffers {static_bytes / (1 << 20):.1f} MiB from the "
            "utilization headroom"
        )

    def _allocate_extend_buffers(
        self, capture: DrafterExtendCapture, *, requests: int
    ) -> None:
        """The static buffers for the plan's largest request count.

        Allocated once, outside every measured capture window: like the target
        prefill graph's static outputs, they come out of the utilization
        headroom, not the CUDA-graph reserve. A later capture (after a cache
        rebind) reuses them.
        """
        width = capture.target_hidden_width
        buffers = self._extend_buffers
        if (
            buffers is not None
            and buffers.hidden_in.shape[0] >= requests
            and buffers.hidden_in.shape[1] == width
            and buffers.hidden_in.dtype == capture.target_hidden_dtype
            and buffers.gather_ids.shape[0] >= requests
        ):
            return
        self._extend_buffers = _ExtendGraphBuffers(
            hidden_in=torch.zeros(
                (requests, width), dtype=capture.target_hidden_dtype, device=self.device
            ),
            gather_ids=torch.zeros(requests, dtype=torch.int64, device=self.device),
            frontier=torch.zeros(
                requests,
                dtype=self.input_buffers.seq_lens_buf.dtype,
                device=self.device,
            ),
            next_tokens=torch.zeros(
                (requests, self.spec_num_tokens), dtype=torch.int32, device=self.device
            ),
        )

    def _stage_capture_inputs(self, bs: int) -> None:
        """Stage the capture dummy's one-token requests: valid ids, zero hidden."""
        ib = self.input_buffers
        buffers = self._extend_buffers
        # The dummy batch leaves the shifted ids unset; a placeholder -1 would
        # index past the embedding.
        ib.shifted_prefill_ids_buf[:bs].fill_(1)
        buffers.hidden_in[:bs].zero_()
        lengths = ib.input_lengths_buf[:bs].to(torch.int64)
        buffers.gather_ids[:bs].copy_(lengths.cumsum(0) - 1)
        buffers.frontier[:bs].copy_(ib.seq_lens_buf[:bs])

    def _run_first_step_eager(self, bs: int, stream: torch.cuda.Stream) -> None:
        """Step 0 over the capture dummy, on its stream: the decode steps' inputs."""
        ib = self.input_buffers
        buffers = self._extend_buffers
        ctx = ForwardContext(
            attn_backend=self.attn_backend,
            token_to_kv_pool=self.token_to_kv_pool,
            bs=bs,
            num_extends=bs,
            output_layout=ForwardOutputLayout(bs, bs, 0, 1),
            input_num_tokens=bs,
            forward_mode=ForwardMode.EXTEND,
            capture_hidden_mode=CaptureHiddenMode.LAST,
            gather_ids=buffers.gather_ids[:bs],
            draft_narrowing=AcceptedPrefixPublisher(
                self.attn_backend, buffers.frontier[:bs]
            ),
        )
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self._attach_dsa_topk((None, None))
            with active_forward(ctx):
                logits_output = self.draft_model_runner.forward(
                    ctx=ctx,
                    input_ids=ib.shifted_prefill_ids_buf[:bs],
                    positions=ib.positions_buf[:bs],
                    captured_hidden_states=buffers.hidden_in[:bs],
                    spec_step_idx=0,
                )
                draft_ids = self.sample_draft_step(logits_output, step=0)
                hidden = logits_output.hidden_states
                if buffers.draft_ids is None:
                    # The first capture: allocate outside the graph pool.
                    buffers.draft_ids = torch.empty(
                        buffers.gather_ids.shape,
                        dtype=draft_ids.dtype,
                        device=self.device,
                    )
                    buffers.hidden_out = torch.empty(
                        (buffers.gather_ids.shape[0], hidden.shape[1]),
                        dtype=hidden.dtype,
                        device=self.device,
                    )
                buffers.draft_ids[:bs].copy_(draft_ids)
                buffers.hidden_out[:bs].copy_(hidden)
                buffers.next_tokens[:bs, 1] = self._map_hot(draft_ids)
        torch.cuda.synchronize()

    def _decode_steps_body(self, bs: int) -> None:
        """The decode steps over the static buffers (what the graph records)."""
        buffers = self._extend_buffers
        self._run_multi_step_decode(
            bs,
            buffers.draft_ids[:bs],
            buffers.hidden_out[:bs],
            buffers.next_tokens[:bs],
            (None, None),
            buffers.frontier[:bs],
            global_num_tokens=None,
            global_bs=None,
            all_decode_or_idle=False,
            first_input_ids=(
                buffers.draft_ids[:bs] if self._split_extend_supported() else None
            ),
        )

    def _capture_decode_steps(
        self,
        bs: int,
        capture: DrafterExtendCapture,
        observer: AbstractContextManager[None],
    ) -> _ExtendGraph:
        """Warm up and capture the decode steps for ``bs`` requests."""
        # The ambient context; no break receives it (each step's breaks hold
        # the step's own context, fixed per request count).
        ctx = ForwardContext(
            attn_backend=self.attn_backend,
            token_to_kv_pool=self.token_to_kv_pool,
            bs=bs,
            num_extends=0,
            output_layout=ForwardOutputLayout(0, 0, bs, 1),
            input_num_tokens=bs,
            forward_mode=ForwardMode.DECODE,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )
        # The steps' write window and the indexer's positions come from the
        # positions a segment computes: the capture must run the segments.
        cap = BreakableCapture(
            pool=self._extend_graph_pool,
            stream=capture.stream,
            handoff_storage=self._extend_graph_handoffs,
            run_segments_while_capturing=True,
        )
        cap.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(cap.stream):
            for _ in range(capture.num_warmup):
                with self.attn_backend.override_num_extends(0), active_forward(ctx):
                    self._decode_steps_body(bs)
        torch.cuda.synchronize()
        with (
            observer,
            self.attn_backend.override_num_extends(0),
            active_forward(ctx),
            cap,
        ):
            self._decode_steps_body(bs)
        if self._extend_graph_pool is None:
            self._extend_graph_pool = cap.pool
        with self.attn_backend.override_num_extends(0), active_forward(ctx):
            cap.replay()
        return _ExtendGraph(cap, ctx)

    def _replays_decode_steps(self, bs: int, draft_input: EagleDraftInput) -> bool:
        """Whether an unsplit pure extend replays its decode steps after step 0."""
        if bs not in self._decode_steps_graphs or self._split_extend_supported():
            return False
        if not draft_input.forward_mode.is_extend() or draft_input.num_extends != bs:
            return False
        if draft_input.global_num_tokens is not None:
            return False
        shard = draft_input.query_shard
        if shard is not None and shard.size > 1:
            return False
        if not self._extend_graph_engaged:
            self._extend_graph_engaged = True
            logger.info(
                "drafter prefill graph ENGAGED: decode steps at "
                f"{draft_input.input_num_tokens} tokens, {bs} requests"
            )
        return True

    def _replay_decode_steps(self, bs: int) -> None:
        """Replay the decode steps over the staged step-0 outputs."""
        graph = self._decode_steps_graphs[bs]
        with active_forward(graph.ctx):
            # Step 0 has waited for the draft layers' Host load.
            graph.capture.replay(noted_waits_done=True)
