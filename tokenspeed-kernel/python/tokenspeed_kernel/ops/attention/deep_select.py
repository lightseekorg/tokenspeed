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

"""Optional FP32 DeepSelect selection, with fused CSA2 masking and metadata."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.attention.triton.dsv41 import (
    finish_selection,
    prepare_candidate_scores,
    prepare_native_dense_scores,
    prepare_scores,
)
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from tokenspeed_kernel.thirdparty.deep_select import (
    deep_select_alignment,
    deep_select_topk,
    is_deep_select_available,
)


def eligible(logits, topk, block_size, candidates):
    """Metadata-only optional dispatch gate; unsupported cases keep Torch."""
    return (
        logits.is_cuda
        and logits.dtype == torch.float32
        and logits.ndim == 2
        and logits.shape[1] < 2**23
        and 0 < topk <= 4096
        and 0 < block_size <= 128
        and (
            candidates is None or (candidates.ndim == 2 and candidates.shape[1] <= 4096)
        )
        and is_deep_select_available()
    )


def _validate(logits, lengths, topk, block_size, candidates):
    if not eligible(logits, topk, block_size, candidates):
        raise ValueError(
            "DeepSelect CSA2 requires CUDA FP32 scores, topk<=4096 and supported geometry"
        )
    if (
        lengths.shape != logits.shape[:1]
        or lengths.dtype not in (torch.int32, torch.int64)
        or lengths.device != logits.device
    ):
        raise ValueError("valid_lengths must be integer [queries] on the logits device")
    if candidates is not None and (
        candidates.shape[0] != logits.shape[0]
        or candidates.dtype not in (torch.int32, torch.int64)
        or candidates.device != logits.device
    ):
        raise ValueError(
            "candidate_blocks must be integer [queries, blocks] on the logits device"
        )


def _native_indices(scores, lengths, topk):
    _, output_alignment = deep_select_alignment()
    alignment = output_alignment // 4
    stride = (topk + alignment - 1) // alignment * alignment
    output = torch.empty(
        (scores.shape[0], stride), dtype=torch.int32, device=scores.device
    )[:, :topk]
    return deep_select_topk(scores, lengths, topk, output)


@register_kernel(
    "attention",
    "dsv41_select_candidates",
    name="deepselect_dsv41_select_candidates",
    solution="deepselect",
    signatures=[format_signature(logits=dense_tensor_format(torch.float32))],
    traits={"deepselect_eligible": frozenset({True})},
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(10, 0),
        max_arch_version=ArchVersion(10, 3),
        vendors=frozenset({"nvidia"}),
    ),
    priority=Priority.PERFORMANT,
)
def select_candidates(logits, valid_lengths, topk_blocks, block_size):
    _validate(logits, valid_lengths, topk_blocks, block_size, None)
    if logits.shape[0] == 0 or logits.shape[1] == 0:
        return torch.full(
            (logits.shape[0], topk_blocks), -1, dtype=torch.int32, device=logits.device
        )
    input_alignment, _ = deep_select_alignment()
    # block_size=1 still needs newest pin, unlike ordinary token preparation.
    scores, ends = prepare_scores(
        logits, valid_lengths, block_size, input_alignment // 4, True
    )
    indices = _native_indices(scores, ends, topk_blocks)
    return finish_selection(
        indices, scores, ends, None, scores.shape[1], block_size, False
    )


@register_kernel(
    "attention",
    "dsv41_select_topk",
    name="deepselect_dsv41_select_topk",
    solution="deepselect",
    signatures=[format_signature(logits=dense_tensor_format(torch.float32))],
    traits={"deepselect_eligible": frozenset({True})},
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(10, 0),
        max_arch_version=ArchVersion(10, 3),
        vendors=frozenset({"nvidia"}),
    ),
    priority=Priority.PERFORMANT,
)
def select_topk(logits, valid_lengths, candidate_blocks, topk, block_size):
    _validate(logits, valid_lengths, topk, block_size, candidate_blocks)
    if (
        logits.shape[0] == 0
        or logits.shape[1] == 0
        or (candidate_blocks is not None and candidate_blocks.shape[1] == 0)
    ):
        return torch.full(
            (logits.shape[0], topk), -1, dtype=torch.int32, device=logits.device
        )
    input_alignment, _ = deep_select_alignment()
    if candidate_blocks is None:
        scores, ends = prepare_native_dense_scores(
            logits, valid_lengths, input_alignment // 4
        )
        normalized = None
    else:
        scores, ends, normalized = prepare_candidate_scores(
            logits, valid_lengths, candidate_blocks, block_size, input_alignment // 4
        )
    indices = _native_indices(scores, ends, topk)
    return finish_selection(
        indices, scores, ends, normalized, logits.shape[1], block_size, True
    )
