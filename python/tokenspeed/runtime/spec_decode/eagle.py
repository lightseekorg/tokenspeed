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

import dataclasses

import torch
import triton
import triton.language as tl

from tokenspeed.runtime.execution.forward_batch_info import CaptureHiddenMode
from tokenspeed.runtime.layers.attention.utils import (
    create_flashinfer_kv_indices_triton,
)
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


@dataclasses.dataclass
class EagleDraftInput:
    # The inputs for decode
    # shape: (b, topk)
    topk_p: torch.Tensor = None
    topk_index: torch.Tensor = None
    # shape: (b, hidden_size)
    hidden_states: torch.Tensor = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL

    # Inputs for extend
    # shape: (b,)
    verified_id: torch.Tensor = None
    accept_length: torch.Tensor = None
    accept_length_cpu: list[int] = None
    accept_index: torch.Tensor = None

    # Inputs for the attention backends
    # shape: (b + 1,)
    kv_indptr: torch.Tensor = None
    kv_indices: torch.Tensor = None

    # For draft extend fast plan
    qo_indptr_cpu: torch.Tensor = None
    kv_indptr_cpu: torch.Tensor = None
    kv_indices_for_extend: torch.Tensor = None
    kv_len_arr_cpu: torch.Tensor = None

    draft_token_num: int = 0

    def set_input_ids(
        self,
        input_ids: torch.Tensor,
        draft_input_ids: torch.Tensor,
        extend_seq_lens: torch.Tensor,
    ) -> None:
        pt = 0
        for i, extend_seq_len in enumerate(extend_seq_lens):
            cur_input_ids = draft_input_ids[i]
            if cur_input_ids[-1] == -1:
                cur_input_ids[-1] = self.verified_id[i]
            input_ids[pt : pt + extend_seq_len] = cur_input_ids

            pt += extend_seq_len

    def prepare_extend_after_decode(self, batch_size: int) -> torch.Tensor:
        new_verified_id = torch.empty_like(self.accept_length, dtype=torch.long)
        create_extend_spec_info[(batch_size,)](
            self.verified_id,
            new_verified_id,
            self.accept_length,
            self.draft_token_num,
        )
        # Extract the last accepted token for each request
        self.verified_id = new_verified_id
        return self.verified_id

    def filter_batch(self, new_indices: torch.Tensor) -> None:
        if self.topk_p is not None:
            self.topk_p = self.topk_p[: len(new_indices)]
        self.topk_index = self.topk_index[: len(new_indices)]
        self.hidden_states = self.hidden_states[: len(new_indices)]
        self.verified_id = self.verified_id[: len(new_indices)]

    def merge_batch(self, spec_info: EagleDraftInput) -> None:
        if self.hidden_states is None:
            self.hidden_states = spec_info.hidden_states
            self.verified_id = spec_info.verified_id
            self.topk_p = spec_info.topk_p
            self.topk_index = spec_info.topk_index
            return
        if spec_info.hidden_states is None:
            return
        self.hidden_states = torch.cat(
            [self.hidden_states, spec_info.hidden_states], axis=0
        )
        self.verified_id = torch.cat([self.verified_id, spec_info.verified_id], axis=0)
        if self.topk_p is not None and spec_info.topk_p is not None:
            self.topk_p = torch.cat([self.topk_p, spec_info.topk_p])
        self.topk_index = torch.cat([self.topk_index, spec_info.topk_index])


@dataclasses.dataclass
class EagleDraftOutput:
    """
    Both prefill and decode batches end with draft. Used to store the previous draft's information,
    to construct verify's input at the next decode

    Args:
        last_verified_ids:
    """

    last_verified_ids: torch.Tensor
    token_list: torch.Tensor

    def filter_batch(self, keep_indices: torch.Tensor) -> None:
        # 1. chunked prefill
        # 2. retract
        # 3. Check finished when updating running and getting new
        self.last_verified_ids = self.last_verified_ids[keep_indices]
        self.token_list = self.token_list[keep_indices, :]

    def merge_batch(self, spec_info: EagleDraftOutput) -> None:
        if spec_info.last_verified_ids is None:
            return
        if self.last_verified_ids is None:
            # May reach here when all requests in running batch are finished
            self.last_verified_ids = spec_info.last_verified_ids
            self.token_list = spec_info.token_list
            return
        self.last_verified_ids = torch.cat(
            [self.last_verified_ids, spec_info.last_verified_ids]
        )
        self.token_list = torch.cat([self.token_list, spec_info.token_list], dim=0)


@triton.jit
def create_extend_spec_info(
    verified_id,  # padded verified id
    new_verified_id,
    accept_length_ptr,
    spec_num_tokens: int,
):
    pid = tl.program_id(axis=0)
    accept_len = tl.load(accept_length_ptr + pid)
    last_verified_id = tl.load(verified_id + pid * spec_num_tokens + accept_len)
    tl.store(accept_length_ptr + pid, accept_len + 1)
    tl.store(new_verified_id + pid, last_verified_id)


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


def generate_attn_arg_prefill(
    draft_token_num: int,
    req_pool_indices: torch.Tensor,
    paged_kernel_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    kv_indices_buf: torch.Tensor = None,
    draft_decode_step: int = None,
):
    batch_size = req_pool_indices.shape[0]
    if draft_decode_step is not None:
        qo_indptr = torch.arange(
            0,
            (1 + batch_size),
            step=1,
            dtype=torch.int32,
            device="cuda",
        )
    else:
        qo_indptr = torch.arange(
            0,
            (1 + batch_size) * draft_token_num,
            step=draft_token_num,
            dtype=torch.int32,
            device="cuda",
        )

    cum_kv_seq_len = torch.zeros((batch_size + 1,), dtype=torch.int32, device="cuda")

    if draft_decode_step is None:
        paged_kernel_lens = paged_kernel_lens + draft_token_num
    else:
        paged_kernel_lens = paged_kernel_lens + draft_decode_step + 1

    torch.cumsum(paged_kernel_lens, dim=0, out=cum_kv_seq_len[1:])
    if kv_indices_buf is not None:
        kv_indices = kv_indices_buf
    else:
        # Prevent kv_indices out of bounds in large steps
        kv_indices = torch.empty(
            cum_kv_seq_len[-1] + 256, dtype=torch.int32, device="cuda"
        )
    create_flashinfer_kv_indices_triton[(batch_size,)](
        req_to_token,
        req_pool_indices,
        paged_kernel_lens,
        cum_kv_seq_len,
        None,
        kv_indices,
        req_to_token.size(1),
    )
    return kv_indices, cum_kv_seq_len, qo_indptr, None
