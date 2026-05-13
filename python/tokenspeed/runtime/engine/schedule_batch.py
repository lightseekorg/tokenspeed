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

"""
Store information about requests and batches.

The following is the flow of data structures for a batch:

ScheduleBatch -> executor inputs

- ScheduleBatch is managed by the runtime event loop and model executor.
  It contains high-level scheduling data. Most of the data is on the CPU.
- Executor inputs contain low-level tensor data. Most of the data consists of
  GPU tensors.
"""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from tokenspeed.runtime.cache.allocator import KVAllocator
from tokenspeed.runtime.cache.base_prefix_cache import BasePrefixCache
from tokenspeed.runtime.cache.req_to_token_pool import ReqToTokenPool
from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.engine.request import Req
from tokenspeed.runtime.execution.forward_batch_info import (
    ForwardMode,
)
from tokenspeed.runtime.layers.attention.kv_cache.base import BaseTokenToKVPool
from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
from tokenspeed.runtime.pd.disaggregation_decode_scheduler import (
    DisaggregationDecodeScheduler,
)
from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
from tokenspeed.runtime.utils import get_colorful_logger

if TYPE_CHECKING:
    from tokenspeed.runtime.spec_decode.algorithm import SpeculativeAlgorithm
    from tokenspeed.runtime.spec_decode.eagle import EagleDraftInput


logger = get_colorful_logger(__name__)

bid = 0


@dataclasses.dataclass
class ScheduleBatch(DisaggregationDecodeScheduler):
    """Store all information of a batch on the scheduler."""

    # Request, memory pool, and cache
    reqs: list[Req]
    req_to_token_pool: ReqToTokenPool = None
    kv_allocator: KVAllocator = None
    token_to_kv_pool: BaseTokenToKVPool = None
    tree_cache: BasePrefixCache = None

    # Batch configs
    model_config: ModelConfig = None
    forward_mode: ForwardMode = None
    enable_overlap: bool = False

    # Events
    launch_done: threading.Event | None = None

    # Sampling info
    sampling_info: SamplingBatchInfo = None
    next_batch_sampling_info: SamplingBatchInfo = None

    # Batched arguments to model runner
    input_ids: torch.Tensor = None  # shape: [b], int32
    input_multi_ids: torch.Tensor | None = None  # shape: [b, mm_heads], int32
    draft_input_ids: torch.Tensor = None  # shape: [b], int32
    input_embeds: torch.Tensor = None  # shape: [b, hidden_size], float32
    input_extra_infos: list[dict] | None = None
    req_pool_indices: torch.Tensor = None  # shape: [b], int32

    seq_lens: torch.Tensor = None  # shape: [b], int64
    output_ids: torch.Tensor = None  # shape: [b], int32
    output_multi_ids: torch.Tensor = None  # shape: [b], int32

    # The sum of all sequence lengths
    seq_lens_sum: int = None

    # For DP attention
    global_num_tokens: list[int] | None = (
        None  # e.g. dp = 4, attn-tp = 2, [A, A, B, B, C, C, D, D]
    )
    global_num_tokens_for_logprob: list[int] | None = None
    all_decode_or_idle: bool = False
    # For processing logprobs
    return_logprob: bool = False
    top_logprobs_nums: list[int] | None = None
    token_ids_logprobs: list[list[int]] | None = None

    # For extend and mixed chunekd prefill
    prefix_lens: list[int] = None
    extend_lens: list[int] = None
    extend_num_tokens: int = None
    decoding_reqs: list[Req] = None
    extend_logprob_start_lens: list[int] = None
    # It comes empty list if logprob is not required.
    extend_input_logprob_token_ids: torch.Tensor | None = None

    # Stream
    has_stream: bool = False

    # Has grammar
    has_grammar: bool = False

    # Device
    device: str = "cuda"

    # Speculative decoding
    spec_algorithm: SpeculativeAlgorithm = None
    spec_info: EagleDraftInput | None = None
    draft_token_num: int | None = 0
    spec_num_steps: int | None = 0
    # Reserve multiple positions for speculative decoding
    reserve_num_tokens_init: int = None

    # Enable custom logit processor
    enable_custom_logit_processor: bool = False

    # Whether to return hidden states
    return_hidden_states: bool = False

    # set aux data for Disaggregation
    disagg_set_aux_fn: Callable[[torch.Tensor, LogitsProcessorOutput], None] | None = (
        None
    )
    # kvstore pointer for synchronizing data loading from CPU to GPU
    kvstore_consumer_index: int = -1

    @classmethod
    def init_new(
        cls,
        reqs: list[Req],
        req_to_token_pool: ReqToTokenPool,
        kv_allocator: KVAllocator,
        token_to_kv_pool: BaseTokenToKVPool,
        tree_cache: BasePrefixCache,
        model_config: ModelConfig,
        enable_overlap: bool,
        spec_algorithm: SpeculativeAlgorithm,
        enable_custom_logit_processor: bool,
        reserve_num_tokens_init: int = 0,
        draft_token_num: int = 0,
        spec_num_steps: int = 0,
    ):
        return cls(
            reqs=reqs,
            req_to_token_pool=req_to_token_pool,
            kv_allocator=kv_allocator,
            token_to_kv_pool=token_to_kv_pool,
            tree_cache=tree_cache,
            model_config=model_config,
            enable_overlap=enable_overlap,
            return_logprob=any(req.return_logprob for req in reqs),
            has_stream=any(req.stream for req in reqs),
            has_grammar=any(req.grammar for req in reqs),
            device=req_to_token_pool.device,
            spec_algorithm=spec_algorithm,
            enable_custom_logit_processor=enable_custom_logit_processor,
            return_hidden_states=any(req.return_hidden_states for req in reqs),
            reserve_num_tokens_init=reserve_num_tokens_init,
            draft_token_num=draft_token_num,
            spec_num_steps=spec_num_steps,
        )

    def batch_size(self):
        return len(self.reqs)

    def alloc_token_slots(self, req_pool_index: int, num_tokens: int):
        out_cache_loc = self.kv_allocator.alloc(
            req_pool_index,
            num_tokens,
            self.req_to_token_pool.alloced_lens[req_pool_index].item(),
        )

        if out_cache_loc is None:
            if self.tree_cache is not None:
                logger.debug(
                    "[evict] before evict evict_tokens=%s evictable_size=%s",
                    num_tokens,
                    self.tree_cache.evictable_size(),
                )
                need_page_num = (
                    num_tokens + self.kv_allocator.page_size - 1
                ) // self.kv_allocator.page_size
                self.tree_cache.evict(need_page_num, self.kv_allocator.free)
                logger.debug(
                    "[evict] after evict evictable_size=%s",
                    self.tree_cache.evictable_size(),
                )
                out_cache_loc = self.kv_allocator.alloc(
                    req_pool_index,
                    num_tokens,
                    self.req_to_token_pool.alloced_lens[req_pool_index].item(),
                )
                logger.debug("[evict] out_cache_loc=%r after evict", out_cache_loc)

            if out_cache_loc is None:
                phase_str = "Prefill" if self.forward_mode.is_extend() else "Decode"
                logger.error(
                    "%s out of memory. Try to lower your batch size.\nTry to allocate %s tokens.\nAvailable tokens: %s\n",
                    phase_str,
                    num_tokens,
                    self.kv_allocator.available_size()
                    + self.tree_cache.evictable_size(),
                )
                if self.tree_cache is not None:
                    self.tree_cache.pretty_print()
                exit(1)

        return out_cache_loc

    def prealloc_for_draft_decode(self, is_disaggregation_decode: bool = False):
        """Pre-allocate a segment of slots for draft decode"""
        if self.enable_overlap:
            # Conceptually, each allocation during speculation + overlap is preparing for the next batch's launch.
            # Therefore, at the beginning, reserve enough space at the end of prefill for the next round's verify and draft decode.
            # Then, each time adjust the reserved space based on acceptance length to prevent allocation divergence causing insufficient space.
            # The reserved space for draft decode will always be overwritten by valid tokens in the next verify.
            # Initially allocate spec_num_steps, subsequent allocations are not needed.
            num_tokens_pre_alloc = self.draft_token_num + (self.spec_num_steps - 1)
        else:
            # Synchronously, each allocation is for the current batch's launch. Here we allocate spec_num_steps
            # extra slots to reserve enough space for draft decode.
            if self.spec_num_steps > 1:
                num_tokens_pre_alloc = self.spec_num_steps - 1
            else:
                return
        out_cache_loc_list = []
        req_indices = []
        for i, req in enumerate(self.reqs):
            # End of prefill or PD disaggregation mocked prefill
            if req.draft_fill_ids[-1] == -1 or is_disaggregation_decode:
                out_cache_loc_list.append(
                    self.alloc_token_slots(req.req_pool_idx, num_tokens_pre_alloc)
                )
                req_indices.append(req.req_pool_idx)
        bs = len(req_indices)
        if len(out_cache_loc_list) == 0:
            return
        out_cache_loc = torch.concat(out_cache_loc_list)
        out_cache_loc = out_cache_loc.to(self.device, non_blocking=True)
        req_indices = torch.tensor(req_indices, dtype=torch.int32).to(
            self.device, non_blocking=True
        )
        start_offsets = self.req_to_token_pool.alloced_lens[req_indices]
        end_offsets = start_offsets + num_tokens_pre_alloc
        assign_req_to_token_pool[(bs,)](
            req_indices,
            self.req_to_token_pool.req_to_token,
            start_offsets,
            end_offsets,
            out_cache_loc,
            self.req_to_token_pool.req_to_token.shape[1],
            triton.next_power_of_2(bs),
        )
        self.req_to_token_pool.alloced_lens[req_indices] += num_tokens_pre_alloc

    def __str__(self):
        return (
            f"ScheduleBatch(forward_mode={self.forward_mode.name}, "
            f"#req={(len(self.reqs))})"
        )


@triton.jit
def assign_req_to_token_pool(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    pool_len: tl.constexpr,
    bs_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    kv_start = tl.load(start_offset + pid)
    kv_end = tl.load(end_offset + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len

    # Get the offset for reading out_cache
    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < pid)
    end = tl.load(end_offset + length_offset, mask=length_offset < pid)
    out_offset = tl.sum(end - start, axis=0)

    out_cache_ptr = out_cache_loc + out_offset

    save_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    load_offset = tl.arange(0, BLOCK_SIZE)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = save_offset < kv_end
        data = tl.load(out_cache_ptr + load_offset, mask=mask)
        tl.store(token_pool + save_offset, data, mask=mask)
        save_offset += BLOCK_SIZE
        load_offset += BLOCK_SIZE
