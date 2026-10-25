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
"""Runtime state tensors shared by the model executor."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence

import torch
from tokenspeed_kernel.ops.metadata import apply_stop_conditions

from tokenspeed.runtime.execution.request_token_history import RequestTokenHistoryView
from tokenspeed.runtime.execution.slot_state import (
    pack_slot_rows,
    slot_state_image_bytes,
    unpack_slot_rows,
)
from tokenspeed.runtime.execution.types import (
    MAX_STOP_TOKEN_IDS,
    RequestHistorySeeds,
    StopCriteria,
)
from tokenspeed.runtime.sampling.tree_verify import DraftTreeQ


class RuntimeStates:
    """Own runtime state tensors keyed by request-pool index.

    A retraction snapshot images each slot's rows (``slot_state_rows``); the
    token-derived rows below are reseeded from the control plane's token list
    instead and stay out of the image (``docs/design/cache-concepts.md``,
    "Retraction snapshot").
    """

    #: Per-slot rows a restore does not copy: the history rows are reseeded
    #: by ``seed_request_token_history`` and the n-gram tail by the per-round
    #: snapshot in ``InputBuffers.fill_ngram_inputs`` followed by the accepted
    #: history commit. An explicit restore token also reseeds an unchanged slot.
    token_derived_slot_state: tuple[str, ...] = (
        "request_token_history_ids",
        "draft_request_token_history_ids",
        "ngram_accepted_tokens",
        "ngram_needs_seed",
    )
    constant_slot_state: tuple[str, ...] = ()

    def __init__(
        self,
        req_pool_size: int,
        vocab_size: int,
        output_length: int,
        device: str = "cuda",
    ):
        self.device = device
        self.vocab_size = vocab_size
        self.ngram_accepted_tokens: torch.Tensor | None = None
        self.ngram_needs_seed: torch.Tensor | None = None
        self.ngram_request_ids: list[str | None] = []
        self.request_token_history_ids: torch.Tensor | None = None
        self.draft_request_token_history_ids: torch.Tensor | None = None

        self.valid_cache_lengths = torch.zeros(
            req_pool_size + 1, dtype=torch.int32, device=device
        )
        # Resolve input ids from here when overlap scheduling.
        self.future_input_map = torch.empty(
            (req_pool_size + 1, output_length), dtype=torch.int32, device=device
        )
        # True while ``future_input_map[slot, 1:]`` holds real draft candidates
        # for the slot's next step: set by the drafter's epilogue after every
        # forward and by a landed remote prefill's candidates, cleared when an
        # explicit decode id consumes the row (``write_decode_input_ids``) and
        # at admission (``reset_states``). The prologue verifies an explicit
        # id single-token unless this says the candidates beside it are real;
        # a retraction images it with the row, so a restored request's first
        # decode is exactly the unretracted step.
        self.spec_candidates_ready = torch.zeros(
            req_pool_size + 1, dtype=torch.bool, device=device
        )
        # The drafter's recorded proposal distributions
        # (--enable-speculative-sampling), see init_draft_probs.
        self.draft_probs: torch.Tensor | None = None
        self.draft_probs_sentinel: float = 0.0
        # Draft trees (init_draft_trees): next round's parents next to its tokens.
        self.chain_parents: torch.Tensor | None = None
        self.future_parent_map: torch.Tensor | None = None
        self.future_draft_q: DraftTreeQ | None = None
        # Each slot's StopCriteria (bind_stop_criteria), read by the sampling
        # epilogue that cuts verify rounds where the request ends
        # (apply_stop_conditions), and the request that last bound them.
        self.stop_token_ids = torch.full(
            (req_pool_size + 1, MAX_STOP_TOKEN_IDS),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.token_limits = torch.full(
            (req_pool_size + 1,),
            torch.iinfo(torch.int32).max,
            dtype=torch.int32,
            device=device,
        )
        self._stop_criteria_owners: list[str | None] = [None] * (req_pool_size + 1)
        # True once a slot's request ended on the device. A forward dispatched
        # behind the one that ended it (the overlap schedule) runs the slot as
        # a padding row: its state rows read nothing and write nothing
        # (masked_state_tables) and it keeps no tokens, so the finish
        # publishes the state and KV of the request's last kept token.
        self.stopped = torch.zeros(req_pool_size + 1, dtype=torch.bool, device=device)

    def init_draft_probs(
        self, *, spec_num_tokens: int, reject_threshold: float
    ) -> None:
        """Allocate the drafter's recorded proposal distributions.

        ``draft_probs[slot, s]`` (fp32 ``[pool + 1, spec_num_tokens, vocab]``)
        is the distribution the chain drafter sampled its step-``s`` token
        from, which the next round verifies as candidate column ``s + 1``.
        Rows start at the sentinel ``reject_threshold + 1``: a slot whose
        drafter has not recorded this round (fresh admission, PD landing)
        rejects every candidate and samples its first token from the full
        target distribution. The last step slot pairs with the bonus token,
        which has no proposal; it is never read and stays zero. The server
        args validate the threshold's range (at least 1.0, so no real
        probability reads as the sentinel, and small enough that ``+ 1.0``
        is representable in fp32). Returns None.
        """
        if spec_num_tokens < 1:
            raise ValueError(
                f"draft_probs needs at least one verify column, got {spec_num_tokens}"
            )
        pool_size = self.valid_cache_lengths.shape[0]
        self.draft_probs_sentinel = float(reject_threshold) + 1.0
        self.draft_probs = torch.full(
            (pool_size, spec_num_tokens, self.vocab_size),
            self.draft_probs_sentinel,
            dtype=torch.float32,
            device=self.device,
        )
        self.draft_probs[:, -1, :] = 0.0

    def reset_draft_probs(self, pool_indices: torch.Tensor) -> None:
        """Mark ``pool_indices`` as having no recorded proposal.

        One reset site, ``reset_states``, covers every way a slot (re)enters
        service: a local prefill resets its rows in the forward prologue, and
        a PD decode destination resets a landing request's row when it seeds
        the remote cache length (``_receive``), which the same data-plane FIFO
        orders ahead of the candidate landing and of the request's first
        local forward, the earliest point a drafter records into the row.
        Tensor-only (index_fill_), so it is stream-ordered without a host
        wait and legal inside a captured graph. Returns None.
        """
        if self.draft_probs is None:
            return
        self.draft_probs[:, :-1].index_fill_(0, pool_indices, self.draft_probs_sentinel)

    def init_draft_trees(self, num_nodes: int) -> None:
        """Allocate each pool slot's next-round tree parents, starting as the chain.

        The drafter's tree for a slot rides next to its candidate tokens in
        ``future_input_map``; rows reset to dummy tokens reset to the chain.
        """
        self.chain_parents = torch.arange(
            -1, num_nodes - 1, dtype=torch.int32, device=self.device
        )
        self.future_parent_map = self.chain_parents.repeat(
            self.future_input_map.shape[0], 1
        )

    def init_draft_tree_q(
        self, num_nodes: int, num_slots: int, num_candidates: int
    ) -> None:
        """Allocate each pool slot's next-round draft distributions (sampled
        trees); every node starts deterministic."""
        self.future_draft_q = DraftTreeQ.zeros(
            self.future_input_map.shape[0],
            num_slots,
            num_candidates,
            num_nodes,
            self.device,
        )

    def init_ngram_state(self, context_len: int) -> None:
        """Allocate a bounded, newest-first accepted input tail per pool slot.

        ``context_len`` is zero without Engram, three for V4.1. This executor
        state follows valid_cache_lengths, not sampled output or model KV. It
        is reseeded from the control plane's token list on admission and on a
        slot handoff (it is token-derived, so a retraction image leaves it
        out), never transferred or prefix-matched as a backend cache. Returns
        None.
        """
        if context_len == 0:
            return
        pool_size = self.valid_cache_lengths.shape[0]
        self.ngram_accepted_tokens = torch.full(
            (pool_size, context_len), -1, dtype=torch.int64, device=self.device
        )
        self.ngram_needs_seed = torch.ones(
            pool_size, dtype=torch.bool, device=self.device
        )
        self.ngram_request_ids = [None] * pool_size

    def init_request_token_history(self, capacity: int) -> None:
        """Allocate each slot's committed-token history row.

        ``capacity`` is zero for a model that reads no history, else the
        physical context length. Row ``s`` holds ``[0, valid_cache_lengths[s])``
        committed tokens; the model's kernels append every forward's inputs
        past that frontier, and resuming extends are reseeded from host
        snapshots (:meth:`seed_request_token_history`). Like the n-gram tail,
        this is executor state that follows ``valid_cache_lengths``; it is
        never transferred or prefix-matched as a cache. Returns None.
        """
        if capacity < 0:
            raise ValueError(
                f"request token history capacity must be non-negative, got {capacity}"
            )
        if capacity == 0:
            return
        pool_size = self.valid_cache_lengths.shape[0]
        self.request_token_history_ids = torch.zeros(
            (pool_size, capacity), dtype=torch.int32, device=self.device
        )

    @property
    def has_request_token_history(self) -> bool:
        """Whether this executor keeps request-token history."""
        return self.request_token_history_ids is not None

    def init_draft_request_token_history(self, capacity: int) -> None:
        """Allocate the draft model's own committed-token history rows.

        A draft stream is the request stream shifted one token left, so it can
        never append into the target's rows: the shifted ids would overwrite
        the verify window right before the target commits it. Row ``s`` holds
        the shifted stream at the target's positions; the drafter supplies the
        per-slot write frontier explicitly (it advances through the
        speculative chain, unlike ``valid_cache_lengths``). Idempotent per
        capacity; returns None.
        """
        if capacity <= 0:
            raise ValueError(
                f"draft request token history capacity must be positive, got "
                f"{capacity}"
            )
        if self.draft_request_token_history_ids is not None:
            return
        pool_size = self.valid_cache_lengths.shape[0]
        self.draft_request_token_history_ids = torch.zeros(
            (pool_size, capacity), dtype=torch.int32, device=self.device
        )

    def draft_request_token_history_view(
        self,
        *,
        req_pool_indices: torch.Tensor,
        input_start_offsets: torch.Tensor,
        active_request_mask: torch.Tensor,
        committed_lengths: torch.Tensor,
        row_offset: int,
    ) -> RequestTokenHistoryView:
        """Combine the draft history with one packed-batch layout.

        Args:
            req_pool_indices: ``[bs]`` slot of every request row.
            input_start_offsets: ``[bs + 1]`` packed offsets of the draft
                forward's inputs.
            active_request_mask: ``[bs]`` False rows are graph padding.
            committed_lengths: ``[pool + 1]`` per-slot write frontier for this
                draft step; the drafter owns and advances it.
            row_offset: Batch-global row of the forward's first local input
                row (a query shard's start; 0 otherwise).
        """
        if self.draft_request_token_history_ids is None:
            raise RuntimeError("draft request token history is not enabled")
        return RequestTokenHistoryView(
            history_token_ids=self.draft_request_token_history_ids,
            committed_lengths=committed_lengths,
            req_pool_indices=req_pool_indices,
            input_start_offsets=input_start_offsets,
            active_request_mask=active_request_mask,
            row_offset=row_offset,
        )

    def request_token_history_view(
        self,
        *,
        req_pool_indices: torch.Tensor,
        input_start_offsets: torch.Tensor,
        active_request_mask: torch.Tensor,
        row_offset: int,
    ) -> RequestTokenHistoryView:
        """Combine the persistent history with one packed-batch layout.

        ``row_offset`` is the batch-global row of the forward's first local
        input row (a query shard's start; 0 otherwise).
        """
        if self.request_token_history_ids is None:
            raise RuntimeError("request token history is not enabled")
        return RequestTokenHistoryView(
            history_token_ids=self.request_token_history_ids,
            committed_lengths=self.valid_cache_lengths,
            req_pool_indices=req_pool_indices,
            input_start_offsets=input_start_offsets,
            active_request_mask=active_request_mask,
            row_offset=row_offset,
        )

    def seed_request_token_history(self, seeds: RequestHistorySeeds) -> None:
        """Restore the committed prefix of each seeded slot.

        One pinned upload carries every seed; each row is then written from
        its device slice on the current stream. Returns None.
        """
        history = self.request_token_history_ids
        if history is None:
            raise RuntimeError("request token history is not enabled")
        pool_size, capacity = history.shape
        for slot, prefix_length in zip(seeds.slots, seeds.prefix_lengths):
            # The last row is graph padding and never belongs to a request.
            if not 0 <= slot < pool_size - 1:
                raise ValueError(f"request token history slot {slot} is out of range")
            if prefix_length > capacity:
                raise ValueError(
                    f"request token history prefix {prefix_length} exceeds "
                    f"capacity {capacity}"
                )
        if sum(seeds.prefix_lengths) == 0:
            return
        device_tokens = torch.tensor(
            list(itertools.chain.from_iterable(seeds.tokens)),
            dtype=torch.int32,
            pin_memory=torch.device(self.device).type != "cpu",
        ).to(self.device, non_blocking=True)
        offset = 0
        draft = self.draft_request_token_history_ids
        for slot, prefix_length in zip(seeds.slots, seeds.prefix_lengths):
            history[slot, :prefix_length].copy_(
                device_tokens[offset : offset + prefix_length]
            )
            if draft is not None and prefix_length > 1:
                # The draft stream is the request stream shifted one token
                # left, at the same positions.
                draft[slot, : prefix_length - 1].copy_(
                    device_tokens[offset + 1 : offset + prefix_length]
                )
            offset += prefix_length

    # ------------------------------------------------------------------
    # Slot-state image (SlotStateExporter)
    # ------------------------------------------------------------------

    def slot_state_rows(self, slot: int) -> list[torch.Tensor]:
        """The per-slot rows a retraction snapshot images, in a fixed order.

        The next step's inputs (``future_input_map``, the tree parents),
        whether their candidate columns are real, the committed frontier and
        the verifier's recorded proposal distributions: re-deriving any of
        them needs a forward, so a restore copies them byte for byte.
        """
        rows = [
            self.valid_cache_lengths[slot],
            self.future_input_map[slot],
            self.spec_candidates_ready[slot],
            self.stop_token_ids[slot],
            self.token_limits[slot],
            # Always False for a victim (only a quiescent, unfinished request
            # is retracted); imaging it clears a previous occupant's flag.
            self.stopped[slot],
        ]
        if self.draft_probs is not None:
            rows.append(self.draft_probs[slot])
        if self.future_parent_map is not None:
            rows.append(self.future_parent_map[slot])
        if self.future_draft_q is not None:
            q = self.future_draft_q
            rows += [q.q[slot], q.q_ids[slot], q.slot[slot], q.pos[slot]]
        return rows

    def slot_state_bytes(self) -> int:
        return slot_state_image_bytes(self.slot_state_rows(0))

    def export_slot_state(
        self, slot: int, out: torch.Tensor, stream, *, request_id: str
    ) -> None:
        del request_id
        pack_slot_rows(self.slot_state_rows(slot), out, stream)

    def import_slot_state(
        self, slot: int, src: torch.Tensor, stream, *, request_id: str
    ) -> None:
        # The restored row is the request's next-step input in full, and the
        # imaged ``spec_candidates_ready`` says whether its candidate columns
        # are real: True after a drafter forward or a landing that shipped
        # candidates, False for a victim landed without them. The scheduler
        # hands a restored request its first decode with an explicit
        # ``decode_input_id`` (no forward of its own is in flight), and the
        # prologue verifies that id single-token exactly when the bit is
        # False -- the unretracted step would have done the same.
        del request_id
        unpack_slot_rows(self.slot_state_rows(slot), src, stream)

    def reset_states(
        self,
        extend_request_pool_indices: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
    ) -> None:
        self.valid_cache_lengths[extend_request_pool_indices] = extend_prefix_lens
        # Scalar indexed assignment stages a CPU tensor and synchronizes CUDA.
        # Keep the reset ordered on the execution stream without a host wait.
        self.spec_candidates_ready.index_fill_(0, extend_request_pool_indices, False)
        self.stopped.index_fill_(0, extend_request_pool_indices, False)
        # Runs in the forward's prologue, before this round's drafter records
        # the rows, so a (re)admitted request verifies its first chain against
        # the sentinel and never a previous occupant's distributions.
        self.reset_draft_probs(extend_request_pool_indices)
        if self.ngram_accepted_tokens is not None:
            assert self.ngram_needs_seed is not None
            self.ngram_accepted_tokens.index_fill_(0, extend_request_pool_indices, -1)
            self.ngram_needs_seed.index_fill_(0, extend_request_pool_indices, True)

    def bind_stop_criteria(
        self,
        request_ids: Sequence[str],
        request_pool_indices: Sequence[int],
        stop_criteria: Sequence[StopCriteria],
    ) -> None:
        """Write the stop criteria of every slot whose request changed.

        Runs on the stream the forward runs on, ahead of it; a slot keeps its
        rows while the same request holds it (a restore images them).
        """
        if len(stop_criteria) != len(request_ids):
            raise ValueError(
                f"{len(stop_criteria)} stop criteria for {len(request_ids)} requests"
            )
        changed = [
            i
            for i, (request_id, slot) in enumerate(
                zip(request_ids, request_pool_indices, strict=True)
            )
            if self._stop_criteria_owners[slot] != request_id
        ]
        if not changed:
            return
        pin = torch.device(self.device).type == "cuda"
        slots = torch.tensor(
            [request_pool_indices[i] for i in changed],
            dtype=torch.int64,
            pin_memory=pin,
        )
        token_ids = torch.full(
            (len(changed), MAX_STOP_TOKEN_IDS), -1, dtype=torch.int32, pin_memory=pin
        )
        limits = torch.empty(len(changed), dtype=torch.int32, pin_memory=pin)
        for row, i in enumerate(changed):
            criteria = stop_criteria[i]
            ids = criteria.stop_token_ids
            token_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.int32)
            limits[row] = criteria.token_limit
            self._stop_criteria_owners[request_pool_indices[i]] = request_ids[i]
        slots = slots.to(self.device, non_blocking=True)
        self.stop_token_ids.index_copy_(
            0, slots, token_ids.to(self.device, non_blocking=True)
        )
        self.token_limits.index_copy_(
            0, slots, limits.to(self.device, non_blocking=True)
        )

    def apply_stop_conditions(
        self,
        req_pool_indices: torch.Tensor,
        output_tokens: torch.Tensor,
        accept_lengths: torch.Tensor,
        input_lengths: torch.Tensor,
        prompt_lengths: torch.Tensor,
        *,
        num_extends: int,
        num_prefill_outputs: int,
        decode_width: int,
        padding_index: int,
    ) -> None:
        """Cut this forward's accepted widths at each request's stop
        conditions and record which requests ended (``stopped``).

        Must run before anything reads ``accept_lengths``; see
        ``tokenspeed_kernel.ops.metadata.apply_stop_conditions``.
        """
        apply_stop_conditions(
            req_pool_indices,
            output_tokens,
            accept_lengths,
            input_lengths,
            prompt_lengths,
            self.valid_cache_lengths,
            self.stop_token_ids,
            self.token_limits,
            self.stopped,
            num_extends=num_extends,
            num_prefill_outputs=num_prefill_outputs,
            decode_width=decode_width,
            padding_index=padding_index,
        )

    def masked_state_tables(
        self,
        block_tables: Mapping[str, torch.Tensor],
        req_pool_indices: torch.Tensor,
        state_group_ids: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        """The batch's block tables with stopped requests' state rows padded.

        A padding row (both slots ``-1``) reads the zero state and writes
        nothing, the contract graph padding rows already follow, so a
        forward dispatched behind the one that ended a request leaves the
        request's live state block as that forward found it.

        The tables are this step's own upload, so the rows are padded in
        place: they stay views of the one packed upload, which the attention
        routers unpack in a single launch.
        """
        stopped_rows = self.stopped.index_select(0, req_pool_indices)[:, None]
        tables = dict(block_tables)
        for group_id in state_group_ids:
            tables[group_id].masked_fill_(stopped_rows, -1)
        return tables

    def mark_spec_candidates_drafted(self, req_pool_indices: torch.Tensor) -> None:
        """The drafter wrote these slots' next-step rows: their candidates are real.

        Tensor-only (``index_fill_``), so it records into a decode graph with
        the row writes it follows.
        """
        self.spec_candidates_ready.index_fill_(0, req_pool_indices, True)

    def write_remote_spec_candidate_ids(
        self, req_pool_idx: int, candidate_ids: list[int]
    ) -> None:
        width = self.future_input_map.shape[1]
        if len(candidate_ids) != width:
            raise RuntimeError(
                f"remote spec candidate width mismatch: got {len(candidate_ids)}, expected {width}"
            )
        ids = torch.tensor(
            candidate_ids,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        ).to(self.device, non_blocking=True)
        self.future_input_map[req_pool_idx, :width] = ids
        self.spec_candidates_ready[req_pool_idx] = True
        # The candidates come without their draft distribution; the row's
        # draft_probs still hold the sentinel reset_states wrote when the
        # remote cache length was seeded, so the first local verify rejects
        # them at column 1 and samples token 0 from the full target.
