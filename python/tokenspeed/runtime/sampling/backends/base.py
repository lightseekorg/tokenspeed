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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
    from tokenspeed.runtime.sampling.dp_sampling_config import DpSamplingRuntimeConfig
    from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
    from tokenspeed.runtime.sampling.sampling_params import SamplingParams
    from tokenspeed.runtime.utils.server_args import ServerArgs


DEFAULT_RANDOM_SEED = 48
SPECULATIVE_ACCEPT_THRESHOLD_SINGLE = 1.0
SPECULATIVE_ACCEPT_THRESHOLD_ACC = 1.0


@dataclass
class SamplingBackendConfig:

    # Optional logprob features — OFF by default. These are checked at server
    # start / graph capture time so the fast path has zero extra compute.
    # Enabling any of these enlarges the captured graph footprint.
    enable_output_logprobs: bool = False

    # Sizing for pre-allocated per-backend buffers (e.g. coin buffers for
    # rejection sampling). Required to keep RNG out of the CUDA graph.
    max_bs: int = 1
    max_draft_tokens_per_req: int = 1

    # Sizing for backend-owned per-request state (e.g. token-count buffers
    # for penalties in FlashInferFullSamplingBackend). Indexed by req_pool_idx, not
    # batch row, so the data survives batch membership changes.
    max_req_pool_size: int = 0
    vocab_size: int = 0

    device: torch.device | None = None
    random_seed: int = DEFAULT_RANDOM_SEED

    # Attention TP group for sampler-output broadcast (rank 0 wins).
    tp_group: tuple[int, ...] | None = None
    enable_tp_sync: bool = True

    @classmethod
    def from_server_args(
        cls,
        server_args: ServerArgs,
        *,
        max_bs: int,
        max_draft_tokens_per_req: int,
        device: str,
        random_seed: int = DEFAULT_RANDOM_SEED,
        max_req_pool_size: int = 0,
        vocab_size: int = 0,
        tp_group: tuple[int, ...] | None = None,
    ) -> SamplingBackendConfig:

        return cls(
            enable_output_logprobs=server_args.enable_output_logprobs,
            max_bs=max_bs,
            max_draft_tokens_per_req=max(max_draft_tokens_per_req, 1),
            max_req_pool_size=max_req_pool_size,
            vocab_size=vocab_size,
            device=device,
            random_seed=random_seed,
            tp_group=tp_group,
            enable_tp_sync=not server_args.disable_sampling_tp_sync,
        )


class SamplingBackend(ABC):
    """Shared contract for single-step sampling and multi-step spec-decode verification.

    Both methods return (output_tokens, accept_lengths). For sample(),
    accept_lengths is all-ones so the downstream contract matches verify().

    Backends that need random state override prepare() to refill per-request
    buffers outside of any CUDA graph capture.

    Requests asking for params a backend doesn't implement are NOT rejected;
    the backend silently applies only what it supports, so all requests go
    through the same captured graph.
    """

    # Subclasses that hold per-pool-idx state (scalars like temperature /
    # top_k, plus large rows like _counts / _logit_bias) flip this to True
    # so prepare_step() performs flip detection + _reset_slot. Stateless
    # backends (greedy) leave it False and the whole prepare_step call is
    # a no-op.
    _HAS_POOL_STATE: bool = False
    _SUPPORTS_DP_VERIFY: bool = False

    def __init__(self, config: SamplingBackendConfig) -> None:

        self.config = config

        # Sentinel of "which rid currently owns each slot from this backend's
        # point of view". rid is just a comparison value here, not a lookup
        # key, so this is pool-keyed state (size O(pool_rows) strings), not
        # rid-keyed state. A mismatch against the incoming rid is a flip.
        if self._HAS_POOL_STATE:
            pool_rows = config.max_req_pool_size + 1
            self._last_rid_per_slot: list[str | None] = [None] * pool_rows

        # Resolved once; None means maybe_broadcast is a no-op.
        self._tp_pg = None
        self._tp_src_global_rank: int | None = None
        if (
            config.enable_tp_sync
            and config.tp_group is not None
            and len(config.tp_group) > 1
        ):
            from tokenspeed.runtime.distributed.process_group_manager import (
                process_group_manager as pg_manager,
            )

            self._tp_pg = pg_manager.get_process_group("nccl", config.tp_group)
            self._tp_src_global_rank = config.tp_group[0]

    def configure_dp_sampling(self, runtime: DpSamplingRuntimeConfig) -> None:
        """Configure optional DP sampling state.

        Stateless or unsupported backends ignore this; DP-capable backends
        override it to initialize backend-local communication buffers.
        """

    def maybe_broadcast(self, *tensors: torch.Tensor) -> None:
        """Broadcast each tensor from tp_group[0] so all attention-TP ranks
        agree. No-op when sync is off or tp_size <= 1. Graph-safe."""
        if self._tp_pg is None:
            return
        for t in tensors:
            dist.broadcast(t, src=self._tp_src_global_rank, group=self._tp_pg)

    def prepare_step(
        self,
        request_ids: list[str],
        request_pool_indices: list[int],
        sampling_params_list: list[SamplingParams],
        num_tokens_per_req: int = 1,
    ) -> None:
        """Called once per step, outside the CUDA graph. Two jobs:

        1. Flip detection: a slot's owning rid changed since last step
           (first-use and rid-recycling look the same). Delegates to
           _reset_slot which scatters all per-slot persistent state
           (scalars, counts, bias, generators).
        2. Per-step dynamic refill: coin buffers, etc. Delegated to the
           subclass via _prepare_step_hook.

        Stateless backends (greedy) short-circuit both phases.
        """

        if not self._HAS_POOL_STATE:
            return

        assert (
            len(request_ids) == len(request_pool_indices) == len(sampling_params_list)
        ), (
            f"prepare_step expects aligned per-request lists; got "
            f"rids={len(request_ids)}, pool_indices={len(request_pool_indices)}, "
            f"sp_list={len(sampling_params_list)}"
        )

        pool_rows = len(self._last_rid_per_slot)
        for rid, pool_idx, sp in zip(
            request_ids, request_pool_indices, sampling_params_list
        ):
            assert (
                0 <= pool_idx < pool_rows
            ), f"pool_idx {pool_idx} out of range [0, {pool_rows}) for rid={rid}"
            if self._last_rid_per_slot[pool_idx] != rid:
                self._reset_slot(pool_idx, sp)
                self._last_rid_per_slot[pool_idx] = rid

        self._prepare_step_hook(
            num_tokens_per_req=num_tokens_per_req,
            bs=len(request_pool_indices),
            request_pool_indices=request_pool_indices,
        )

    def prepare_capture(self, bs: int, num_tokens_per_req: int = 1) -> None:
        """Per-step refill for the capture/warm-up path. No flip detection;
        the backend uses its stub generator for any RNG-fed buffers so the
        captured graph sees a fully-written state.
        Default: no-op.
        """
        self._prepare_step_hook(
            num_tokens_per_req=num_tokens_per_req,
            bs=bs,
            request_pool_indices=None,
        )

    def _prepare_step_hook(
        self,
        num_tokens_per_req: int,
        bs: int,
        request_pool_indices: list[int] | None,
    ) -> None:
        """Subclass hook for per-step dynamic state (coin buffers, etc).
        request_pool_indices=None is the capture path; otherwise the CPU
        list from forward_op.request_pool_indices.
        Default: no-op."""

    def _reset_slot(self, pool_idx: int, sp: SamplingParams) -> None:
        """Scatter all per-slot persistent state for a newly-assigned slot.
        Called from prepare_step on flip. Stateful backends override."""
        raise NotImplementedError

    def reset_capture_state(self) -> None:
        """Clear any per-pool state that warm-up iterations may have dirtied
        before CUDA graph capture. Warm-up runs sample()/verify() against
        pool row 0 (see CudaGraphWrapper capture path); stateful backends
        override this to zero whatever row 0 accumulates. Default: no-op."""

    def get_packed_output_d2h(
        self,
        output_tokens: torch.Tensor,
        output_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """If the backend wrote both outputs into a single contiguous GPU
        buffer, return CPU views obtained from one D2H copy. Otherwise
        return None and let the caller fall back to two separate D2Hs."""
        return None

    @abstractmethod
    def sample(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def verify(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
