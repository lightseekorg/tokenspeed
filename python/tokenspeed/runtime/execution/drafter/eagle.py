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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from tokenspeed_kernel.ops.sampling.triton import logprob_topk
from typing_extensions import override

from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.drafter.base import BaseDrafter
from tokenspeed.runtime.execution.drafter.tree import DraftTree
from tokenspeed.runtime.execution.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
from tokenspeed.runtime.execution.query_shard import QueryShardPlan
from tokenspeed.runtime.layers.attention.backends.paged.tree_verify import (
    TreeDraftInputs,
)
from tokenspeed.runtime.utils.nvtx import nvtx_range

DsaTopKState = tuple[Any | None, Any | None]

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.input_buffer import InputBuffers
    from tokenspeed.runtime.execution.model_runner import ModelRunner
    from tokenspeed.runtime.execution.runtime_states import RuntimeStates
    from tokenspeed.runtime.execution.tree_spec import TreeSpec
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput


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
    through the drafter hook, so it is in-graph safe and idempotent.
    """

    attn_backend: AttentionBackend
    frontier: torch.Tensor

    def publish_accepted_prefix(self) -> None:
        self.attn_backend.advance_draft_forward_metadata(self.frontier)


@dataclass
class EagleDraftInput:
    input_num_tokens: int
    num_extends: int
    forward_mode: ForwardMode
    base_model_output: torch.Tensor  # [bs]
    accept_lengths: torch.Tensor  # [bs]
    base_out_hidden_states: torch.Tensor
    global_num_tokens: list[int] | None = None
    global_bs: list[int] | None = None
    all_decode_or_idle: bool = False
    dsa_topk: DsaTopKState = (None, None)
    # The target forward's query shard: the draft's extend rows (step 0) are
    # the same shard of the same span, its decode steps run every row.
    query_shard: QueryShardPlan | None = None


class Eagle(BaseDrafter):
    """
    Draft model runner that implements the Eagle/Eagle3 algorithm.
    """

    shares_target_embed_head = True
    supports_pd_layerwise_finalization = True
    supports_request_token_history = True
    supports_speculative_sampling = True

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

        self.device = draft_model_runner.device
        hot_token_ids = draft_model_runner.model.get_hot_token_id()

        if hot_token_ids is not None:
            self.hot_token_ids = hot_token_ids.to(self.device)
        else:
            self.hot_token_ids = None

        # For constructing fallback global_num_tokens during CUDA graph capture.
        self.dp_size = draft_model_runner.mapping.attn.dp_size
        self.world_size = draft_model_runner.mapping.world_size

        # Drafter-owned alias source for the draft attn backend; advanced in
        # place during multi-step decode.
        self.draft_seq_lens_buf = torch.zeros_like(self.input_buffers.seq_lens_buf)

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

        # Precomputed `arange(max_bs) * spec_num_tokens - 1`
        # gather_ids = gather_ids_offsets + accept_lengths
        self.padded_gather_ids_offsets_buf = (
            torch.arange(
                self.input_buffers.max_bs, dtype=torch.int64, device=self.device
            )
            * spec_num_tokens
            - 1
        )

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
        self.tree_lanes = TreeDraftInputs(
            topk=config.topk,
            num_steps=config.num_steps,
            max_bs=max_bs,
            device=self.device,
        )
        self.attn_backend.bind_tree_draft(self.tree_lanes)
        self.lanes_per_request = config.topk
        self._tree_seed_lane_mask = torch.ones(
            config.topk, dtype=torch.int64, device=self.device
        ) << torch.arange(config.topk, device=self.device)

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
        # QSA row geometry belongs to one model invocation. Draft steps reuse
        # the selected slots below, but must rebuild their new row layout.
        share.qsa_metadata = None
        if self._model_shares_mtp_topk():
            share.prefill, share.decode = dsa_topk
        else:
            share.clear()

    def _extract_dsa_topk(self, dsa_topk: DsaTopKState) -> DsaTopKState:
        if not self._model_shares_mtp_topk():
            return dsa_topk
        share = self.attn_backend.sparse_topk
        return share.prefill, share.decode

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
        query shard the ids are the shard's slice of the shifted prefill
        ids while ``gather_ids`` keep the batch's full layout, as on the
        target's forward: the draft model's exit cuts them to its shard
        through ``QueryShardPlan.local_sampled_ids`` when it gathers the
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
                input_ids = input_ids[plan.local_slice]
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
            else draft_input.query_shard.local_slice
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
                    row_offset=rows.start,
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
        draft_input: EagleDraftInput,
        dsa_topk: DsaTopKState,
        frontier: torch.Tensor,
    ) -> None:
        """Draft steps ``1 .. S - 1``: ``lanes`` rows per request (1 for a
        chain, K for a draft tree) forward from the accepted frontier."""
        lanes = self.lanes_per_request
        # Step 1 writes at the accepted frontier (vc + accept_length after the
        # target's verify) so rotary/cache metadata stay on the accepted
        # prefix, not the rejected tail.
        cache_start = frontier

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

        for i in range(1, self.spec_num_steps):
            # make a ctx every time model runner forward
            # Multi-step decode is pure DECODE mode: one token per request.
            # global_num_tokens must reflect each rank's batch size, not the
            # target model's total tokens (which may be bs * spec_num_tokens).
            global_num_tokens = draft_input.global_num_tokens

            if self.dp_size > 1:
                if draft_input.global_bs is not None:
                    global_num_tokens = draft_input.global_bs
                else:
                    # CUDA graph capture path: uniform batch size across ranks.
                    global_num_tokens = [bs] * self.world_size

            ctx = ForwardContext(
                bs=bs,
                num_extends=0,
                output_layout=ForwardOutputLayout(0, 0, bs, lanes),
                attn_backend=self.attn_backend,
                token_to_kv_pool=self.token_to_kv_pool,
                input_num_tokens=bs * lanes,
                forward_mode=ForwardMode.DECODE,
                capture_hidden_mode=CaptureHiddenMode.LAST,
                global_num_tokens=global_num_tokens,
                global_bs=draft_input.global_bs,
                all_decode_or_idle=draft_input.all_decode_or_idle,
            )
            self._attach_dsa_topk(dsa_topk)

            if self.tree_lanes is None:
                # Keep attention metadata on the accepted prefix; rejected verify
                # tail slots may still contain stale draft KV.
                _advance_draft_forward_metadata_if_supported(
                    ctx.attn_backend,
                    draft_seq_lens,
                )
            else:
                # Tree lanes attend over the accepted prefix and their window (TreeDraftInputs).
                self.tree_lanes.active = True
            # Publish this step's one write slot per request (the chain's
            # advancing position, seq_len - 1): step i writes position
            # cache_start + (i - 1).
            self.attn_backend.publish_draft_step_locations(
                cache_start=slot_starts,
                num_tokens=lanes,
            )

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
                        row_offset=0,
                    )
                )
            with nvtx_range("draft_forward", color="red"):
                logits_output = self.draft_model_runner.forward(
                    ctx=ctx,
                    input_ids=self._map_hot(draft_ids.reshape(-1)),
                    positions=positions,
                    captured_hidden_states=hidden,
                    spec_step_idx=i,
                    **history_kwargs,
                )
                dsa_topk = self._extract_dsa_topk(dsa_topk)
            if self.tree_lanes is not None:
                self.tree_lanes.active = False
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

    @override
    def draft(
        self,
        draft_input: EagleDraftInput,
    ) -> torch.Tensor:

        bs = draft_input.accept_lengths.shape[0]

        # Layout: column 0 holds the last verified id (the base model's accepted token);
        # columns 1..spec_num_steps hold the drafter's speculative tokens.
        next_tokens = torch.empty(
            (bs, self.spec_num_tokens),
            dtype=torch.int32,
            device=self.device,
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

        # The runner refreshed the draft decode metadata over the target's
        # post-verify lengths (vc + N). Step 0 narrows to the live rows, whose
        # context is the accepted frontier; the model publishes it through
        # this handle at the moment it switches to those rows.
        frontier = self._accepted_frontier(bs, draft_input)
        narrowing = AcceptedPrefixPublisher(self.attn_backend, frontier)

        # First draft step. LogitsProcessor prunes `[num_prefill_tokens + num_decodes * spec_num_tokens, ...]`
        # down to `[bs, ...]`, so logits/hidden_states arrive here already aligned to one row per request.
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
            self._run_multi_step_decode(
                bs,
                draft_ids,
                hidden,
                next_tokens,
                draft_input,
                dsa_topk,
                frontier,
            )
        return self._draft_window(bs, next_tokens)

    @override
    @nvtx_range("drafter", color="purple")
    def run(
        self,
        base_ctx: ForwardContext,
        logits_output: LogitsProcessorOutput,
        output_tokens: torch.Tensor,
        accept_lengths: torch.Tensor,
    ) -> torch.Tensor:

        draft_input = EagleDraftInput(
            input_num_tokens=base_ctx.input_num_tokens,
            num_extends=base_ctx.num_extends,
            forward_mode=base_ctx.forward_mode,
            base_model_output=output_tokens,
            accept_lengths=accept_lengths,
            base_out_hidden_states=logits_output.hidden_states,
            global_num_tokens=base_ctx.global_num_tokens,
            global_bs=base_ctx.global_bs,
            all_decode_or_idle=base_ctx.all_decode_or_idle,
            dsa_topk=self._target_dsa_topk(base_ctx),
            query_shard=base_ctx.query_shard,
        )

        # next_tokens layout: column 0 = last verified id, columns 1.. = drafter tokens.
        return self.draft(draft_input)
