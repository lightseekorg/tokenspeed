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

import torch

from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


class RuntimeStates:
    """Own runtime state tensors keyed by request-pool index."""

    def __init__(
        self,
        req_pool_size: int,
        context_len: int,
        vocab_size: int,
        output_length: int,
        device: str = "cuda",
        mamba_pool=None,
    ):
        self.device = device
        self.vocab_size = vocab_size

        self.valid_cache_lengths = torch.zeros(
            req_pool_size + 1, dtype=torch.int32, device=device
        )
        # Resolve input ids from here when overlap scheduling.
        self.future_input_map = torch.empty(
            (req_pool_size + 1, output_length), dtype=torch.int32, device=device
        )
        self.linear_penalties = torch.zeros(
            (req_pool_size + 1, vocab_size), dtype=torch.float32, device=device
        )
        self.scaling_penalties = torch.ones(
            (req_pool_size + 1, vocab_size), dtype=torch.float32, device=device
        )
        self.mamba_pool = mamba_pool

    def update_valid_cache_length(
        self, req_pool_indices: torch.Tensor, increment_lengths: torch.Tensor
    ) -> None:
        self.valid_cache_lengths[req_pool_indices] += increment_lengths

    def reset_states(
        self,
        extend_request_pool_indices: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
    ) -> None:
        self.valid_cache_lengths[extend_request_pool_indices] = extend_prefix_lens
        self.linear_penalties.index_fill_(0, extend_request_pool_indices, 0.0)
        self.scaling_penalties.index_fill_(0, extend_request_pool_indices, 1.0)

    def copy_mamba_states(
        self,
        mamba_pool_indices: torch.Tensor,
        mamba_cow_src_indices: torch.Tensor,
        bs: int,
    ) -> None:
        """Copy Mamba states for copy-on-write requests."""
        if self.mamba_pool is None:
            return
        if mamba_cow_src_indices is None:
            return
        valid_mask = mamba_cow_src_indices[:bs] != -1
        if not valid_mask.any():
            return
        if (mamba_pool_indices[:bs][valid_mask] == -1).any():
            raise RuntimeError(
                f"mamba_pool_indices contains -1 for requests needing COW copy: "
                f"mamba_pool_indices={mamba_pool_indices[:bs].tolist()}, "
                f"mamba_cow_src_indices={mamba_cow_src_indices[:bs].tolist()}"
            )
        src_indices = mamba_cow_src_indices[:bs][valid_mask].long()
        dst_indices = mamba_pool_indices[:bs][valid_mask].long()
        self.mamba_pool.conv_state[:, dst_indices] = self.mamba_pool.conv_state[
            :, src_indices
        ]
        self.mamba_pool.ssm_state[:, dst_indices] = self.mamba_pool.ssm_state[
            :, src_indices
        ]

    def snapshot_mamba_checkpoints(
        self,
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
        cache_lengths: torch.Tensor,
        page_size: int,
        n: int,
    ) -> None:
        """Copy current working Mamba states into checkpoint slots.

        src_indices/dst_indices are pre-filtered on CPU (only valid entries).
        Only the page_size condition is checked on GPU.
        """
        if self.mamba_pool is None or n == 0:
            return
        src = src_indices[:n]
        dst = dst_indices[:n]
        if page_size > 0:
            # Use torch.where to keep fixed-size output (no boolean indexing sync).
            # Invalid entries become self-copy (dst = src), harmless no-op.
            page_mask = cache_lengths[:n] % page_size == 0
            dst = torch.where(page_mask, dst, src)
        self.mamba_pool.conv_state[:, dst] = self.mamba_pool.conv_state[
            :, src
        ]
        self.mamba_pool.ssm_state[:, dst] = self.mamba_pool.ssm_state[
            :, src
        ]

    def zero_mamba_states(
        self,
        mamba_pool_indices: torch.Tensor,
        mamba_cow_src_indices: torch.Tensor | None,
        extend_prefix_lens: torch.Tensor | None,
        bs: int,
    ) -> None:
        """Clear Mamba states for newly allocated slots without prefix state."""
        if self.mamba_pool is None:
            return
        valid_pool = mamba_pool_indices[:bs] != -1
        no_cow = (
            (mamba_cow_src_indices[:bs] == -1)
            if mamba_cow_src_indices is not None
            else torch.ones(bs, dtype=torch.bool, device=mamba_pool_indices.device)
        )
        no_prefix = (
            (extend_prefix_lens[:bs] == 0)
            if extend_prefix_lens is not None
            else torch.ones(bs, dtype=torch.bool, device=mamba_pool_indices.device)
        )
        zero_mask = valid_pool & no_cow & no_prefix
        if not zero_mask.any():
            return
        indices = mamba_pool_indices[:bs][zero_mask].long()
        self.mamba_pool.conv_state[:, indices] = 0
        self.mamba_pool.ssm_state[:, indices] = 0
