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

from typing import TYPE_CHECKING

import torch

from tokenspeed.runtime.execution.cache_loc_kernel import compute_out_cache_loc
from tokenspeed.runtime.execution.forward_batch_info import compute_position_triton
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.nvtx import nvtx_range

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.runtime_states import RuntimeStates


logger = get_colorful_logger(__name__)


class InputBuffers:
    """
    ForwardContext tensor data source, read-only after fill. Holds only
    model-forward inputs; per-request sampling scalars (temperature, top_k,
    penalties, seed, etc.) live on the sampling backend as pool-indexed
    buffers populated on slot flips.
    """

    def __init__(
        self,
        max_bs: int,
        max_num_tokens: int,
        page_size: int,
        dummy_kv_slot: int,
        device: str = "cuda",
        has_mamba: bool = False,
    ):
        self.device = device
        self.page_size = page_size
        self.max_num_tokens = max_num_tokens
        self.dummy_kv_slot = dummy_kv_slot
        self.max_bs = max_bs
        self.all_extends_mid_chunk = False
        self.has_mamba = has_mamba

        with torch.device(device):
            self.input_ids_buf = torch.ones((max_num_tokens,), dtype=torch.int32)
            # Used in draft prefill
            self.shifted_prefill_ids_buf = torch.ones_like(self.input_ids_buf)
            self.input_lengths_buf = torch.ones((max_num_tokens,), dtype=torch.int32)
            self.positions_buf = torch.arange(0, max_num_tokens, dtype=torch.int64)
            self.req_pool_indices_buf = torch.zeros((max_bs,), dtype=torch.int32)
            self.seq_lens_buf = torch.ones((max_bs,), dtype=torch.int32)
            # Initialise to dummy_kv_slot so that padding positions (never
            # written by compute_out_cache_loc) always point to the reserved
            # dummy KV slot and never corrupt real KV cache entries.
            self.out_cache_loc_buf = torch.full(
                (max_num_tokens,), dummy_kv_slot, dtype=torch.int32
            )
            self.extend_prefix_lens_buf = torch.zeros(max_bs, dtype=torch.int32)
            self.extend_seq_lens_buf = torch.zeros(max_bs, dtype=torch.int32)
            if has_mamba:
                self.mamba_pool_indices_buf = torch.full(
                    (max_bs,), -1, dtype=torch.int32
                )
                self.mamba_cow_src_indices_buf = torch.full(
                    (max_bs,), -1, dtype=torch.int32
                )
                self.mamba_branching_seqlens_buf = torch.full(
                    (max_bs,), -1, dtype=torch.int32
                )
                self.mamba_track_pool_indices_buf = torch.full(
                    (max_bs,), -1, dtype=torch.int32
                )

        self.extend_prefix_lens_cpu = torch.zeros(
            max_bs, dtype=torch.int32, pin_memory=True
        )
        self.extend_seq_lens_cpu = torch.zeros(
            max_bs, dtype=torch.int32, pin_memory=True
        )
        if has_mamba:
            self._mamba_pool_indices_cpu = torch.full(
                (max_bs,), -1, dtype=torch.int32, pin_memory=True
            )
            self._mamba_cow_src_indices_cpu = torch.full(
                (max_bs,), -1, dtype=torch.int32, pin_memory=True
            )
            self._mamba_branching_seqlens_cpu = torch.full(
                (max_bs,), -1, dtype=torch.int32, pin_memory=True
            )
            self._mamba_track_pool_indices_cpu = torch.full(
                (max_bs,), -1, dtype=torch.int32, pin_memory=True
            )

    @nvtx_range("input_prep_fill", color="cyan")
    def fill_input_buffers(
        self,
        forward_op,
        runtime_states: RuntimeStates,
        req_to_page: torch.Tensor,
        total_tokens: int,
    ):
        batch_size = len(forward_op.request_ids)
        assert batch_size >= 0
        num_extends = forward_op.num_extends()
        req_pool_indices_cpu = torch.tensor(
            forward_op.request_pool_indices, device="cpu", pin_memory=True
        )
        self.req_pool_indices_buf[:batch_size].copy_(
            req_pool_indices_cpu,
            non_blocking=True,
        )
        input_lengths_cpu = torch.tensor(
            forward_op.input_lengths, device="cpu", pin_memory=True
        )
        self.input_lengths_buf[:batch_size].copy_(
            input_lengths_cpu,
            non_blocking=True,
        )

        self.all_extends_mid_chunk = (
            num_extends > 0
            and num_extends == batch_size
            and all(
                forward_op.extend_prefix_lens[i] + forward_op.input_lengths[i]
                < forward_op.prefill_lengths[i]
                for i in range(num_extends)
            )
        )

        if num_extends > 0:
            self.extend_prefix_lens_cpu[:num_extends] = torch.as_tensor(
                forward_op.extend_prefix_lens, dtype=torch.int32
            )
            self.extend_prefix_lens_buf[:num_extends].copy_(
                self.extend_prefix_lens_cpu[:num_extends], non_blocking=True
            )
            self.extend_seq_lens_cpu[:num_extends] = torch.as_tensor(
                forward_op.input_lengths[:num_extends], dtype=torch.int32
            )
            self.extend_seq_lens_buf[:num_extends].copy_(
                self.extend_seq_lens_cpu[:num_extends], non_blocking=True
            )

        # Get valid cache lengths for requests
        req_pool_indices_device = self.req_pool_indices_buf[:batch_size]
        input_lengths_device = self.input_lengths_buf[:batch_size]

        valid_cache_lengths = runtime_states.valid_cache_lengths[
            req_pool_indices_device
        ]

        # Compute out_cache_loc using Triton kernel
        compute_out_cache_loc(
            out_cache_loc_ptr=self.out_cache_loc_buf[:total_tokens],
            req_pool_indices=req_pool_indices_device,
            input_lengths=input_lengths_device,
            cache_start=valid_cache_lengths,
            req_to_pages=req_to_page,
            page_size=self.page_size,
        )

        # Compute positions. In mixed batches, prefill rows use their extend
        # prefix lengths while decode rows use the current valid cache lengths.
        prefill_prefix_lens = self.extend_prefix_lens_buf[:num_extends]
        if num_extends == 0:
            prefix_lens = valid_cache_lengths
        elif num_extends == batch_size:
            prefix_lens = prefill_prefix_lens
        else:
            prefix_lens = valid_cache_lengths.clone()
            prefix_lens[:num_extends].copy_(prefill_prefix_lens)
        positions, _ = compute_position_triton(
            extend_prefix_lens=prefix_lens,
            extend_seq_lens=input_lengths_device,
            extend_seq_lens_sum=total_tokens,
        )
        self.positions_buf[:total_tokens].copy_(positions)

        # Determine input_ids and forward_mode
        if num_extends > 0:
            prefill_token_count = sum(forward_op.input_lengths[:num_extends])
            input_ids_cpu = torch.tensor(
                forward_op.input_ids, device="cpu", pin_memory=True
            )
            self.input_ids_buf[:prefill_token_count].copy_(
                input_ids_cpu,
                non_blocking=True,
            )
            shifted_ids_cpu = torch.tensor(
                forward_op.shifted_input_ids, device="cpu", pin_memory=True
            )
            self.shifted_prefill_ids_buf[:prefill_token_count].copy_(
                shifted_ids_cpu,
                non_blocking=True,
            )
            if num_extends < batch_size:
                decode_req_pool_indices = req_pool_indices_device[
                    num_extends:batch_size
                ]
                if forward_op.decode_input_ids is not None:
                    decode_count = batch_size - num_extends
                    if len(forward_op.decode_input_ids) != decode_count:
                        raise RuntimeError(
                            "mixed forward decode_input_ids length mismatch: "
                            f"got {len(forward_op.decode_input_ids)}, "
                            f"expected {decode_count}"
                        )
                    decode_input_ids_tensor = torch.tensor(
                        forward_op.decode_input_ids,
                        dtype=torch.int32,
                        device="cpu",
                        pin_memory=True,
                    ).to(req_pool_indices_device.device, non_blocking=True)
                    mask = (decode_input_ids_tensor != -1).unsqueeze(1)
                    slot = runtime_states.future_input_map[decode_req_pool_indices, :1]
                    runtime_states.future_input_map[decode_req_pool_indices, :1] = (
                        torch.where(mask, decode_input_ids_tensor.unsqueeze(1), slot)
                    )
                decode_ids = runtime_states.future_input_map[
                    decode_req_pool_indices, :1
                ].flatten()
                self.input_ids_buf[prefill_token_count:total_tokens].copy_(
                    decode_ids,
                    non_blocking=True,
                )
                self.shifted_prefill_ids_buf[prefill_token_count:total_tokens].copy_(
                    decode_ids,
                    non_blocking=True,
                )
        else:
            # If the scheduler provides explicit decode input ids (!= -1), write
            # them into future_input_map before reading, so that they take effect
            # as the input for this decode step.
            if forward_op.decode_input_ids is not None:
                decode_input_ids_tensor = torch.tensor(
                    forward_op.decode_input_ids,
                    dtype=torch.int32,
                    device="cpu",
                    pin_memory=True,
                ).to(req_pool_indices_device.device, non_blocking=True)
                mask = (decode_input_ids_tensor != -1).unsqueeze(1)  # (bs, 1)
                slot = runtime_states.future_input_map[
                    req_pool_indices_device, :1
                ]  # (bs, 1)
                runtime_states.future_input_map[req_pool_indices_device, :1] = (
                    torch.where(mask, decode_input_ids_tensor.unsqueeze(1), slot)
                )
            self.input_ids_buf[:total_tokens].copy_(
                runtime_states.future_input_map[req_pool_indices_device].flatten(),
                non_blocking=True,
            )

        self.seq_lens_buf[:batch_size].copy_(input_lengths_device + cached_prefix_lens)

        # Reset positions beyond total_tokens to the dummy KV slot so that any
        # CUDA graph replay with a larger (padded) batch size writes padding
        # tokens to the reserved dummy slot instead of corrupting real KV cache.
        if total_tokens < self.max_num_tokens:
            self.input_ids_buf[total_tokens:].fill_(1)
            self.out_cache_loc_buf[total_tokens:].fill_(self.dummy_kv_slot)
            self.req_pool_indices_buf[batch_size:].fill_(0)
            self.seq_lens_buf[batch_size:].fill_(1)
            self.positions_buf[total_tokens:].fill_(0)

        if (
            self.has_mamba
            and hasattr(forward_op, "mamba_pool_indices")
            and forward_op.mamba_pool_indices
        ):
            self._mamba_pool_indices_cpu[:batch_size].copy_(
                torch.as_tensor(forward_op.mamba_pool_indices, dtype=torch.int32)
            )
            self._mamba_cow_src_indices_cpu[:batch_size].copy_(
                torch.as_tensor(forward_op.mamba_cow_src_indices, dtype=torch.int32)
            )
            self._mamba_branching_seqlens_cpu[:batch_size].copy_(
                torch.as_tensor(forward_op.mamba_branching_seqlens, dtype=torch.int32)
            )
            self._mamba_track_pool_indices_cpu[:batch_size].copy_(
                torch.as_tensor(forward_op.mamba_track_pool_indices, dtype=torch.int32)
            )

            self.mamba_pool_indices_buf[:batch_size].copy_(
                self._mamba_pool_indices_cpu[:batch_size], non_blocking=True
            )
            self.mamba_cow_src_indices_buf[:batch_size].copy_(
                self._mamba_cow_src_indices_cpu[:batch_size], non_blocking=True
            )
            self.mamba_branching_seqlens_buf[:batch_size].copy_(
                self._mamba_branching_seqlens_cpu[:batch_size], non_blocking=True
            )
            self.mamba_track_pool_indices_buf[:batch_size].copy_(
                self._mamba_track_pool_indices_cpu[:batch_size], non_blocking=True
            )
            if batch_size < self.mamba_pool_indices_buf.shape[0]:
                self.mamba_pool_indices_buf[batch_size:].fill_(-1)
                self.mamba_cow_src_indices_buf[batch_size:].fill_(-1)
                self.mamba_branching_seqlens_buf[batch_size:].fill_(-1)
                self.mamba_track_pool_indices_buf[batch_size:].fill_(-1)

    def fill_dummy_decode_buffers(self, batch_size: int, total_tokens: int):
        """Prepare padded decode graph inputs for a rank with no real tokens."""
        if total_tokens > 0:
            self.input_ids_buf[:total_tokens].fill_(1)
            self.out_cache_loc_buf[:total_tokens].fill_(self.dummy_kv_slot)
            self.positions_buf[:total_tokens].fill_(0)
        if batch_size > 0:
            self.req_pool_indices_buf[:batch_size].fill_(0)
            self.seq_lens_buf[:batch_size].fill_(1)
