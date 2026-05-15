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
Triton MHA attention backend for TokenSpeed scheduling.
Uses custom triton kernels for decode and prefill attention.
Supports sliding window and speculative decoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl
from tokenspeed_kernel.ops.attention.triton.mha_decode import decode_attention_fwd
from tokenspeed_kernel.ops.attention.triton.mha_prefill import prefill_attention_fwd

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.utils import (
    create_flashinfer_kv_indices_triton,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.paged_attention import PagedAttention
    from tokenspeed.runtime.spec_decode.eagle import EagleDraftInput


@dataclass
class ForwardMetadata:
    attn_logits: torch.Tensor
    attn_lse: torch.Tensor
    max_extend_len: int
    num_kv_splits: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    qo_indptr: torch.Tensor
    custom_mask: torch.Tensor
    mask_indptr: torch.Tensor
    # Sliding window
    window_kv_indptr: torch.Tensor
    window_kv_indices: torch.Tensor
    window_num_kv_splits: torch.Tensor


class TritonAttnBackend(AttentionBackend):
    """Triton MHA attention backend for TokenSpeed scheduling."""

    def __init__(self, config: MHAConfig):
        super().__init__(config)

        max_bs = config.max_bs
        self.page_size = config.page_size
        self.max_context_len = config.context_len

        # These fields depend on runtime state not available in config.
        # They must be set externally after construction (e.g. by the model runner).
        self.sliding_window_size = None
        self.req_to_page = None
        self.kv_allocator = None
        self.num_draft_tokens = 0
        self.speculative_num_steps = 0

        self.kv_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=config.device
        )

        # If sliding window is enabled, we might need two sets of buffers
        # because of interleaved attention types (e.g. for Gemma3)
        self.window_kv_indptr = None

        self.qo_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=config.device
        )

        self.mask_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int64, device=config.device
        )

        self.num_head = self.num_qo_heads
        self.num_kv_head = self.num_kv_heads

        # Default max_kv_splits; can be overridden externally.
        self.max_kv_splits = 8

        # v_head_dim is set later when the kv pool is available.
        self.v_head_dim = self.head_dim

        self.forward_metadata: ForwardMetadata = None

        self.device_core_count = 0  # Set externally via configure_runtime()

    def configure_runtime(
        self,
        sliding_window_size=None,
        req_to_page=None,
        kv_allocator=None,
        num_draft_tokens=0,
        speculative_num_steps=0,
        max_kv_splits=8,
        v_head_dim=None,
        device_core_count=0,
    ):
        """Set runtime fields that are not available from the config alone."""
        self.sliding_window_size = sliding_window_size
        self.req_to_page = req_to_page
        self.kv_allocator = kv_allocator
        self.num_draft_tokens = num_draft_tokens
        self.speculative_num_steps = speculative_num_steps
        self.max_kv_splits = max_kv_splits
        if v_head_dim is not None:
            self.v_head_dim = v_head_dim
        self.device_core_count = device_core_count

        # Allocate sliding window buffers now that we know the size
        if self.sliding_window_size is not None and self.sliding_window_size > 0:
            if self.window_kv_indptr is None:
                self.window_kv_indptr = torch.zeros(
                    (self.kv_indptr.shape[0],),
                    dtype=torch.int32,
                    device=self.device,
                )

    def get_num_kv_splits(
        self,
        num_kv_splits: torch.Tensor,
        seq_lens: torch.Tensor,
    ):
        num_token, num_seq = num_kv_splits.shape[0], seq_lens.shape[0]
        num_group = num_token // num_seq

        assert (
            num_group * num_seq == num_token
        ), f"num_seq({num_seq}), num_token({num_token}), something goes wrong!"

        if self.device_core_count <= 0:
            num_kv_splits.fill_(self.max_kv_splits)
            return

        if num_seq < 256:
            SCHEDULE_SEQ = 256
        else:
            SCHEDULE_SEQ = triton.next_power_of_2(num_seq)

        get_num_kv_splits_triton[(1,)](
            num_kv_splits,
            seq_lens,
            num_seq,
            num_group,
            self.num_head,
            self.num_kv_head,
            self.max_kv_splits,
            self.device_core_count,
            MAX_NUM_SEQ=SCHEDULE_SEQ,
        )

    def init_forward_metadata(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode = None,
        req_to_page: torch.Tensor = None,
        extend_prefix_lens: torch.Tensor | None = None,
        extend_seq_lens: torch.Tensor | None = None,
        spec_info=None,
        **kwargs,
    ):
        """Init auxiliary variables for the Triton attention backend."""

        _req_to_token = req_to_page if req_to_page is not None else self.req_to_page
        assert (
            _req_to_token is not None
        ), "req_to_page must be set via configure_runtime() or passed to init_forward_metadata()"

        kv_indptr = self.kv_indptr
        window_kv_indptr = self.window_kv_indptr
        window_kv_indices = None
        window_num_kv_splits = None

        if forward_mode.is_decode_or_idle():
            if spec_info is None:
                torch.cumsum(seq_lens, dim=0, out=kv_indptr[1 : bs + 1])
                kv_indptr = kv_indptr[: bs + 1]
                _seq_lens_sum = int(seq_lens.sum().item())
                kv_indices = torch.empty(
                    _seq_lens_sum, dtype=torch.int32, device=self.device
                )
                create_flashinfer_kv_indices_triton[(bs,)](
                    _req_to_token,
                    req_pool_indices,
                    seq_lens,
                    kv_indptr,
                    None,
                    kv_indices,
                    _req_to_token.stride(0),
                )
                # Sliding window
                if (
                    self.sliding_window_size is not None
                    and self.sliding_window_size > 0
                ):
                    window_kv_indptr, window_kv_indices, window_kv_lens = (
                        update_sliding_window_buffer(
                            self.window_kv_indptr,
                            _req_to_token,
                            self.sliding_window_size,
                            seq_lens,
                            req_pool_indices,
                            bs,
                            self.device,
                            self.kv_allocator,
                        )
                    )
                    window_num_kv_splits = torch.empty(
                        (bs,), dtype=torch.int32, device=self.device
                    )
                    self.get_num_kv_splits(window_num_kv_splits, window_kv_lens)
            else:
                kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices
                bs = kv_indptr.shape[0] - 1

            attn_logits = torch.empty(
                (bs, self.num_head, self.max_kv_splits, self.v_head_dim),
                dtype=torch.float32,
                device=self.device,
            )
            attn_lse = torch.empty(
                (bs, self.num_head, self.max_kv_splits),
                dtype=torch.float32,
                device=self.device,
            )
            num_kv_splits = torch.empty((bs,), dtype=torch.int32, device=self.device)
            self.get_num_kv_splits(num_kv_splits, seq_lens)

            qo_indptr = None
            custom_mask = None
            mask_indptr = None
            max_extend_len = None
        elif forward_mode.is_target_verify():
            bs = len(req_pool_indices)
            qo_indptr = torch.arange(
                0,
                (1 + bs) * self.num_draft_tokens,
                step=self.num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            torch.cumsum(seq_lens, dim=0, out=kv_indptr[1 : bs + 1])
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.empty(
                kv_indptr[-1], dtype=torch.int32, device=self.device
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                _req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                _req_to_token.stride(0),
            )

            if self.sliding_window_size is not None and self.sliding_window_size > 0:
                window_kv_indptr, window_kv_indices, window_kv_lens = (
                    update_sliding_window_buffer(
                        self.window_kv_indptr,
                        _req_to_token,
                        self.sliding_window_size,
                        seq_lens,
                        req_pool_indices,
                        bs,
                        self.device,
                        self.kv_allocator,
                    )
                )

            custom_mask = spec_info.custom_mask
            seq_mask_len = self.num_draft_tokens * (seq_lens + self.num_draft_tokens)
            mask_indptr = self.mask_indptr
            torch.cumsum(seq_mask_len[:bs], dim=0, out=mask_indptr[1 : bs + 1])
            mask_indptr = mask_indptr[: bs + 1]
            max_extend_len = self.num_draft_tokens
            num_kv_splits = None
            attn_logits = None
            attn_lse = None

        elif forward_mode.is_draft_extend():
            kv_indices, kv_indptr, qo_indptr, custom_mask = (
                spec_info.generate_attn_arg_prefill(
                    req_pool_indices,
                    seq_lens,
                    None,
                    _req_to_token,
                )
            )
            mask_indptr = None
            max_extend_len = torch.max(spec_info.accept_length).item()
            num_kv_splits = None
            attn_logits = None
            attn_lse = None
        else:
            # Extend (prefill)
            _extend_prefix_lens = extend_prefix_lens
            if _extend_prefix_lens is None:
                _extend_prefix_lens = torch.zeros(
                    (bs,), dtype=seq_lens.dtype, device=seq_lens.device
                )
            torch.cumsum(_extend_prefix_lens, dim=0, out=kv_indptr[1 : bs + 1])
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.empty(
                _extend_prefix_lens.sum().item(),
                dtype=torch.int32,
                device=self.device,
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                _req_to_token,
                req_pool_indices,
                _extend_prefix_lens,
                kv_indptr,
                None,
                kv_indices,
                _req_to_token.stride(0),
            )
            # Sliding window
            if self.sliding_window_size is not None and self.sliding_window_size > 0:
                window_kv_indptr, window_kv_indices, _ = update_sliding_window_buffer(
                    self.window_kv_indptr,
                    _req_to_token,
                    self.sliding_window_size,
                    _extend_prefix_lens,
                    req_pool_indices,
                    bs,
                    self.device,
                    self.kv_allocator,
                )

            _extend_seq_lens = extend_seq_lens
            if _extend_seq_lens is None:
                _extend_seq_lens = seq_lens - _extend_prefix_lens

            qo_indptr = self.qo_indptr
            torch.cumsum(_extend_seq_lens, dim=0, out=qo_indptr[1 : bs + 1])
            qo_indptr = qo_indptr[: bs + 1]
            custom_mask = None
            mask_indptr = None
            attn_logits = None
            attn_lse = None
            max_extend_len = torch.max(_extend_seq_lens).item()
            num_kv_splits = None

        self.forward_metadata = ForwardMetadata(
            attn_logits,
            attn_lse,
            max_extend_len,
            num_kv_splits,
            kv_indptr,
            kv_indices,
            qo_indptr,
            custom_mask,
            mask_indptr,
            window_kv_indptr,
            window_kv_indices,
            window_num_kv_splits,
        )

    def init_cuda_graph_state(
        self,
        max_bs: int,
        seq_lens_buf: torch.Tensor,
        kv_indices_buf: torch.Tensor | None = None,
    ):
        del seq_lens_buf  # triton backend allocates its own buffers.
        self.cuda_graph_attn_logits = torch.zeros(
            (max_bs, self.num_head, self.max_kv_splits, self.v_head_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.cuda_graph_attn_lse = torch.zeros(
            (max_bs, self.num_head, self.max_kv_splits),
            dtype=torch.float32,
            device=self.device,
        )
        self.cuda_graph_num_kv_splits = torch.full(
            (max_bs,), self.max_kv_splits, dtype=torch.int32, device=self.device
        )
        if kv_indices_buf is None:
            self.cuda_graph_kv_indices = torch.zeros(
                (max_bs * self.max_context_len),
                dtype=torch.int32,
                device=self.device,
            )
        else:
            self.cuda_graph_kv_indices = kv_indices_buf

        self.cuda_graph_custom_mask = torch.zeros(
            (max_bs * self.max_context_len),
            dtype=torch.uint8,
            device=self.device,
        )

        if self.sliding_window_size is not None and self.sliding_window_size > 0:
            if kv_indices_buf is None:
                self.cuda_graph_window_kv_indices = torch.zeros(
                    (max_bs * self.sliding_window_size),
                    dtype=torch.int32,
                    device=self.device,
                )
            else:
                self.cuda_graph_window_kv_indices = torch.zeros_like(kv_indices_buf)

            self.cuda_graph_window_num_kv_splits = torch.full(
                (max_bs,),
                self.max_kv_splits,
                dtype=torch.int32,
                device=self.device,
            )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: EagleDraftInput | None = None,
    ):
        _req_to_token = self.req_to_page
        window_kv_indptr = self.window_kv_indptr
        window_kv_indices = None
        window_num_kv_splits = None

        if forward_mode.is_decode_or_idle():
            if spec_info is None:
                kv_indptr = self.kv_indptr
                torch.cumsum(seq_lens, dim=0, out=kv_indptr[1 : bs + 1])
                kv_indptr = kv_indptr[: bs + 1]
                kv_indices = self.cuda_graph_kv_indices
                create_flashinfer_kv_indices_triton[(bs,)](
                    _req_to_token,
                    req_pool_indices,
                    seq_lens,
                    kv_indptr,
                    None,
                    kv_indices,
                    _req_to_token.stride(0),
                )
                if (
                    self.sliding_window_size is not None
                    and self.sliding_window_size > 0
                ):
                    window_kv_indices = self.cuda_graph_window_kv_indices
                    window_num_kv_splits = self.cuda_graph_window_num_kv_splits
                    window_kv_indptr, window_kv_indices, _ = (
                        update_sliding_window_buffer_cuda_graph(
                            self.window_kv_indptr,
                            window_kv_indices,
                            _req_to_token,
                            self.sliding_window_size,
                            seq_lens[:bs],
                            req_pool_indices,
                            bs,
                            self.kv_allocator,
                        )
                    )
            else:
                kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices

            attn_logits = self.cuda_graph_attn_logits
            attn_lse = self.cuda_graph_attn_lse
            max_extend_len = None
            num_kv_splits = self.cuda_graph_num_kv_splits
            qo_indptr = None
            custom_mask = None
            mask_indptr = None
        elif forward_mode.is_target_verify():
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[: bs + 1] = torch.arange(
                0,
                (1 + bs) * self.num_draft_tokens,
                step=self.num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            kv_indptr = self.kv_indptr[: bs + 1]
            torch.cumsum(seq_lens, dim=0, out=kv_indptr[1 : bs + 1])
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                _req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                _req_to_token.stride(0),
            )

            if self.sliding_window_size is not None and self.sliding_window_size > 0:
                window_kv_indices = self.cuda_graph_window_kv_indices
                window_num_kv_splits = self.cuda_graph_window_num_kv_splits
                window_kv_indptr, window_kv_indices, _ = (
                    update_sliding_window_buffer_cuda_graph(
                        self.window_kv_indptr,
                        window_kv_indices,
                        _req_to_token,
                        self.sliding_window_size,
                        seq_lens,
                        req_pool_indices,
                        bs,
                        self.kv_allocator,
                    )
                )

            custom_mask = self.cuda_graph_custom_mask
            custom_mask[: spec_info.custom_mask.shape[0]] = spec_info.custom_mask
            seq_mask_len = self.num_draft_tokens * (seq_lens + self.num_draft_tokens)
            mask_indptr = self.mask_indptr[: bs + 1]
            torch.cumsum(seq_mask_len, dim=0, out=mask_indptr[1 : bs + 1])
            max_extend_len = self.num_draft_tokens
            num_kv_splits = None
            attn_logits = None
            attn_lse = None
        elif forward_mode.is_draft_extend():
            num_tokens_per_bs = self.speculative_num_steps + 1
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[: bs + 1] = torch.arange(
                0,
                bs * num_tokens_per_bs + 1,
                step=num_tokens_per_bs,
                dtype=torch.int32,
                device=self.device,
            )
            kv_indptr = self.kv_indptr[: bs + 1]
            torch.cumsum(seq_lens, dim=0, out=kv_indptr[1 : bs + 1])
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                _req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                _req_to_token.stride(0),
            )
            custom_mask = None
            mask_indptr = None
            max_extend_len = num_tokens_per_bs
            num_kv_splits = None
            attn_logits = None
            attn_lse = None
        else:
            raise ValueError(
                f"Invalid forward mode: {forward_mode=} for CUDA Graph capture."
            )

        self.forward_metadata = ForwardMetadata(
            attn_logits,
            attn_lse,
            max_extend_len,
            num_kv_splits,
            kv_indptr,
            kv_indices,
            qo_indptr,
            custom_mask,
            mask_indptr,
            window_kv_indptr,
            window_kv_indices,
            window_num_kv_splits,
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode = None,
        req_to_page: torch.Tensor = None,
        spec_info: EagleDraftInput | None = None,
    ):
        _req_to_token = self.req_to_page

        if forward_mode.is_decode_or_idle():
            # Update kv_indptr, kv_indices
            kv_indptr = self.kv_indptr
            kv_indices = self.cuda_graph_kv_indices
            num_kv_splits = self.cuda_graph_num_kv_splits
            if spec_info is None:
                torch.cumsum(seq_lens[:bs], dim=0, out=kv_indptr[1 : bs + 1])
                kv_indptr = kv_indptr[: bs + 1]
                create_flashinfer_kv_indices_triton[(bs,)](
                    _req_to_token,
                    req_pool_indices[:bs],
                    seq_lens[:bs],
                    kv_indptr,
                    None,
                    kv_indices,
                    _req_to_token.stride(0),
                )
                num_token = bs
                if (
                    self.sliding_window_size is not None
                    and self.sliding_window_size > 0
                ):
                    window_num_kv_splits = self.cuda_graph_window_num_kv_splits
                    window_kv_indices = self.cuda_graph_window_kv_indices
                    _, _, window_kv_lens = update_sliding_window_buffer_cuda_graph(
                        self.window_kv_indptr,
                        window_kv_indices,
                        _req_to_token,
                        self.sliding_window_size,
                        seq_lens[:bs],
                        req_pool_indices[:bs],
                        bs,
                        self.kv_allocator,
                    )
                    self.get_num_kv_splits(
                        window_num_kv_splits[:num_token], window_kv_lens[:bs]
                    )

            else:
                kv_indptr[: spec_info.kv_indptr.shape[0]] = spec_info.kv_indptr
                kv_indices[: spec_info.kv_indices.shape[0]] = spec_info.kv_indices
                num_token = spec_info.kv_indptr.shape[0] - 1
            self.get_num_kv_splits(num_kv_splits[:num_token], seq_lens[:bs])

        elif forward_mode.is_target_verify():
            # Update qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr
            bs = len(req_pool_indices)
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[: bs + 1] = torch.arange(
                0,
                (1 + bs) * self.num_draft_tokens,
                step=self.num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            kv_indptr = self.kv_indptr[: bs + 1]
            torch.cumsum(seq_lens, dim=0, out=kv_indptr[1 : bs + 1])
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                _req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                _req_to_token.stride(0),
            )
            if self.sliding_window_size is not None and self.sliding_window_size > 0:
                window_num_kv_splits = self.cuda_graph_window_num_kv_splits
                window_kv_indices = self.cuda_graph_window_kv_indices
                _, _, window_kv_lens = update_sliding_window_buffer_cuda_graph(
                    self.window_kv_indptr,
                    window_kv_indices,
                    _req_to_token,
                    self.sliding_window_size,
                    seq_lens,
                    req_pool_indices,
                    bs,
                    self.kv_allocator,
                )
            custom_mask = self.cuda_graph_custom_mask
            custom_mask[: spec_info.custom_mask.shape[0]] = spec_info.custom_mask
            seq_mask_len = self.num_draft_tokens * (seq_lens + self.num_draft_tokens)
            mask_indptr = self.mask_indptr[: bs + 1]
            torch.cumsum(seq_mask_len, dim=0, out=mask_indptr[1 : bs + 1])
        elif forward_mode.is_draft_extend():
            seq_lens = seq_lens[:bs]
            accept_lens = spec_info.accept_length[:bs]
            qo_indptr = self.qo_indptr[: bs + 1]
            torch.cumsum(accept_lens, dim=0, out=qo_indptr[1 : bs + 1])
            kv_indptr = self.kv_indptr[: bs + 1]
            torch.cumsum(seq_lens, dim=0, out=kv_indptr[1 : bs + 1])
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                _req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                _req_to_token.stride(0),
            )
        else:
            raise ValueError(
                f"Invalid forward mode: {forward_mode=} for CUDA Graph replay."
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if save_kv_cache:
            token_to_kv_pool.set_kv_buffer(layer, out_cache_loc, k, v)

        sinks = kwargs.get("sinks", None)

        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            sliding_window_size = (
                layer.sliding_window_size
            )  # Needed for sliding window mask
            kv_indptr = self.forward_metadata.window_kv_indptr
            kv_indices = self.forward_metadata.window_kv_indices
        else:
            sliding_window_size = -1
            kv_indptr = self.forward_metadata.kv_indptr
            kv_indices = self.forward_metadata.kv_indices

        prefill_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k.contiguous(),
            v.contiguous(),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            token_to_kv_pool.get_key_buffer(layer.layer_id),
            token_to_kv_pool.get_value_buffer(layer.layer_id),
            self.forward_metadata.qo_indptr,
            kv_indptr,
            kv_indices,
            None,
            True,
            self.forward_metadata.mask_indptr,
            self.forward_metadata.max_extend_len,
            layer.scaling,
            layer.logit_cap,
            sliding_window_size=sliding_window_size,
            sinks=sinks,
        )
        return o

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        # During torch.compile, there is a bug in rotary_emb that causes the
        # output value to have a 3D tensor shape. This reshapes the output correctly.
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if save_kv_cache:
            token_to_kv_pool.set_kv_buffer(layer, out_cache_loc, k, v)

        sinks = kwargs.get("sinks", None)

        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            kv_indptr = self.forward_metadata.window_kv_indptr
            kv_indices = self.forward_metadata.window_kv_indices
        else:
            kv_indptr = self.forward_metadata.kv_indptr
            kv_indices = self.forward_metadata.kv_indices

        decode_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            token_to_kv_pool.get_key_buffer(layer.layer_id),
            token_to_kv_pool.get_value_buffer(layer.layer_id),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            kv_indptr,
            kv_indices,
            self.forward_metadata.attn_logits,
            self.forward_metadata.attn_lse,
            self.forward_metadata.num_kv_splits,
            self.max_kv_splits,
            layer.scaling,
            layer.logit_cap,
            sinks=sinks,
        )
        return o


# ---------------------------------------------------------------------------
# Triton kernels and helper functions
# ---------------------------------------------------------------------------


@triton.jit
def get_num_kv_splits_triton(
    num_kv_splits_ptr,
    seq_lens_ptr,
    num_seq,
    num_group,
    num_head,
    num_kv_head,
    max_kv_splits,
    device_core_count,
    MAX_NUM_SEQ: tl.constexpr,
):
    offs_seq = tl.arange(0, MAX_NUM_SEQ)
    mask_seq = offs_seq < num_seq

    seq_lens = tl.load(seq_lens_ptr + offs_seq, mask=mask_seq, other=0)
    max_seq_len = tl.max(seq_lens)
    seq_lens = tl.load(seq_lens_ptr + offs_seq, mask=mask_seq, other=max_seq_len)
    min_seq_len = tl.min(seq_lens)
    if max_seq_len * 8 < min_seq_len * 10:
        min_seq_len = max_seq_len
    max_kv_splits_1 = tl.minimum(tl.cdiv(max_seq_len, min_seq_len), max_kv_splits)
    kv_chunk_size_1 = tl.cdiv(max_seq_len, max_kv_splits_1)

    # Scale the split budget gradually with sequence length so short requests
    # avoid excessive split overhead while long requests expose more parallelism.
    ext_seq_len = tl.cast(max_seq_len, tl.float32) / 64.0
    ext_device_core_count = tl.cast(
        device_core_count * tl.maximum(tl.log2(ext_seq_len), 1.0), tl.int32
    )
    block_h, num_kv_group = 16, num_head // num_kv_head
    if num_kv_group == 1:
        token_grid = num_seq * num_group * num_head
    else:
        block_h = tl.minimum(block_h, num_kv_group)
        token_grid = num_seq * num_group * tl.cdiv(num_head, block_h)
    max_kv_splits_2 = tl.minimum(
        tl.cdiv(ext_device_core_count, token_grid), max_kv_splits
    )
    kv_chunk_size_2 = tl.cdiv(max_seq_len, max_kv_splits_2)

    num_kv_splits = tl.maximum(
        tl.cdiv(seq_lens, kv_chunk_size_1), tl.cdiv(seq_lens, kv_chunk_size_2)
    )

    offs_token = offs_seq * num_group
    mask_token = offs_token < num_seq * num_group
    for i in range(0, num_group):
        tl.store(num_kv_splits_ptr + i + offs_token, num_kv_splits, mask=mask_token)


def update_sliding_window_buffer(
    window_kv_indptr,
    req_to_page,
    sliding_window_size,
    seq_lens,
    req_pool_indices,
    bs,
    device,
    kv_allocator=None,
):
    window_kv_lens = torch.minimum(
        seq_lens,
        torch.tensor(sliding_window_size),
    )
    torch.cumsum(window_kv_lens, dim=0, out=window_kv_indptr[1 : bs + 1])
    window_kv_indptr = window_kv_indptr[: bs + 1]
    window_kv_indices = torch.empty(
        window_kv_indptr[-1], dtype=torch.int32, device=device
    )
    window_kv_start_idx = seq_lens - window_kv_lens
    create_flashinfer_kv_indices_triton[(bs,)](
        req_to_page,
        req_pool_indices,
        window_kv_lens,
        window_kv_indptr,
        window_kv_start_idx,
        window_kv_indices,
        req_to_page.stride(0),
    )
    # full to swa index mapping
    if hasattr(kv_allocator, "translate_loc_from_full_to_swa"):
        kv_last_index = window_kv_indptr[-1]
        window_kv_indices[:kv_last_index] = kv_allocator.translate_loc_from_full_to_swa(
            window_kv_indices[:kv_last_index]
        )
    return window_kv_indptr, window_kv_indices, window_kv_lens


def update_sliding_window_buffer_cuda_graph(
    window_kv_indptr,
    window_kv_indices,
    req_to_page,
    sliding_window_size,
    seq_lens,
    req_pool_indices,
    bs,
    kv_allocator=None,
):
    window_kv_lens = torch.minimum(
        seq_lens,
        torch.tensor(sliding_window_size),
    )
    torch.cumsum(window_kv_lens, dim=0, out=window_kv_indptr[1 : bs + 1])
    window_kv_indptr = window_kv_indptr[: bs + 1]
    window_kv_start_idx = seq_lens - window_kv_lens
    create_flashinfer_kv_indices_triton[(bs,)](
        req_to_page,
        req_pool_indices,
        window_kv_lens,
        window_kv_indptr,
        window_kv_start_idx,
        window_kv_indices,
        req_to_page.stride(0),
    )
    # full to swa index mapping
    if hasattr(kv_allocator, "translate_loc_from_full_to_swa"):
        kv_last_index = window_kv_indptr[-1]
        window_kv_indices[:kv_last_index] = kv_allocator.translate_loc_from_full_to_swa(
            window_kv_indices[:kv_last_index]
        )
    return window_kv_indptr, window_kv_indices, window_kv_lens
