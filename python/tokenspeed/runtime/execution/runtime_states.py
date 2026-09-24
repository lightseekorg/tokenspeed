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

import torch

from tokenspeed.runtime.execution.request_token_history import RequestTokenHistoryView
from tokenspeed.runtime.execution.types import RequestHistorySeeds


class RuntimeStates:
    """Own runtime state tensors keyed by request-pool index."""

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

        self.valid_cache_lengths = torch.zeros(
            req_pool_size + 1, dtype=torch.int32, device=device
        )
        # Resolve input ids from here when overlap scheduling.
        self.future_input_map = torch.empty(
            (req_pool_size + 1, output_length), dtype=torch.int32, device=device
        )
        self.remote_spec_candidate_ready = torch.zeros(
            req_pool_size + 1, dtype=torch.bool, device=device
        )

    def init_ngram_state(self, context_len: int) -> None:
        """Allocate a bounded, newest-first accepted input tail per pool slot.

        ``context_len`` is zero without Engram, three for V4.1. This executor
        state follows valid_cache_lengths, not sampled output or model KV. It
        is reseeded from host snapshots on admission/recovery, never transferred
        or prefix-matched as a backend cache. Returns None.
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

    def request_token_history_view(
        self,
        *,
        req_pool_indices: torch.Tensor,
        input_start_offsets: torch.Tensor,
        active_request_mask: torch.Tensor,
    ) -> RequestTokenHistoryView:
        """Combine the persistent history with one packed-batch layout."""
        if self.request_token_history_ids is None:
            raise RuntimeError("request token history is not enabled")
        return RequestTokenHistoryView(
            history_token_ids=self.request_token_history_ids,
            committed_lengths=self.valid_cache_lengths,
            req_pool_indices=req_pool_indices,
            input_start_offsets=input_start_offsets,
            active_request_mask=active_request_mask,
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
        for slot, prefix_length in zip(seeds.slots, seeds.prefix_lengths):
            history[slot, :prefix_length].copy_(
                device_tokens[offset : offset + prefix_length]
            )
            offset += prefix_length

    def reset_states(
        self,
        extend_request_pool_indices: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
    ) -> None:
        self.valid_cache_lengths[extend_request_pool_indices] = extend_prefix_lens
        # Scalar indexed assignment stages a CPU tensor and synchronizes CUDA.
        # Keep the reset ordered on the execution stream without a host wait.
        self.remote_spec_candidate_ready.index_fill_(
            0, extend_request_pool_indices, False
        )
        if self.ngram_accepted_tokens is not None:
            assert self.ngram_needs_seed is not None
            self.ngram_accepted_tokens.index_fill_(0, extend_request_pool_indices, -1)
            self.ngram_needs_seed.index_fill_(0, extend_request_pool_indices, True)

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
        self.remote_spec_candidate_ready[req_pool_idx] = True
