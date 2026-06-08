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

"""TRT-LLM MoE kernels exposed via the numerics registry."""

from __future__ import annotations

import torch
from tokenspeed_kernel.numerics.moe import (
    canonicalize_align_block_size,
    compute_align_block_size_buffer_dims,
)
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel_nvidia.registration import register_kernel
from tokenspeed_kernel_nvidia.thirdparty.trtllm import (
    moe_align_block_size as _trtllm_moe_align_block_size,
)


@register_kernel(
    "moe",
    "align_block_size",
    name="trtllm_moe_align_block_size",
    solution="trtllm",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(9, 0),
        vendors=frozenset({"nvidia"}),
    ),
    signatures=format_signatures("indices", "dense", {torch.int32}),
    traits={},
    priority=12,
    tags={"latency"},
)
def trtllm_moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> torch.Tensor:
    """Call ``torch.ops.trtllm.moe_align_block_size`` and return the canonical
    packed tensor matching ``torch_moe_align_block_size``.

    Output buffer sizing mirrors the production wrapper at
    ``tokenspeed_kernel.ops.moe.triton.moe_align_block_size``.
    """
    pad_id = topk_ids.numel()
    max_num_m_blocks, sorted_ids_size = compute_align_block_size_buffer_dims(
        pad_id, num_experts, block_size
    )

    sorted_ids = torch.empty(
        (sorted_ids_size,), dtype=torch.int32, device=topk_ids.device
    )
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty(1, dtype=torch.int32, device=topk_ids.device)
    cumsum_buffer = torch.empty(
        (num_experts + 2,), dtype=torch.int32, device=topk_ids.device
    )

    # Pre-fill the sorted_ids tail with the PAD token so any block-tail slots
    # the kernel doesn't write are still PAD (matches reference). Above the
    # 4096-element threshold the wrapper does this; do it unconditionally for
    # the test path so behaviour is identical regardless of input size.
    sorted_ids.fill_(pad_id)

    _trtllm_moe_align_block_size(
        topk_ids,
        num_experts + 1,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        False,  # pad_sorted_token_ids — we already pre-filled
    )
    return canonicalize_align_block_size(
        sorted_ids, expert_ids, num_tokens_post_pad, block_size
    )
