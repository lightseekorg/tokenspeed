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

"""Exact selected-row scoring on the validated NVIDIA/Torch combination."""

import torch
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel.thirdparty.cuda import full_vocab_scoring


def _supported(values):
    # The reused ATen iterator uses uint32 offsets; keep every byte offset
    # below 2^31 instead of allowing a fitting row/stride product to overflow.
    return values.ndim == 2 and values.shape[1] >= 32768 and values.numel() * 4 < 2**31


def cuda_full_vocab_row_selection_supports(logits):
    """Host geometry check only; acceptance values remain on the device."""
    return _supported(logits) and logits.shape[0] > 0


def cuda_full_vocab_topk(logits, *, k, active_rows):
    """Original sorted FP32 Top-K, skipping inactive rows in its CUDA kernels."""
    if not _supported(logits) or not logits.is_contiguous():
        return torch.topk(logits, k, dim=-1)
    active = (
        torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
        if active_rows is None
        else active_rows
    )
    return full_vocab_scoring.topk(logits, active, k)


def cuda_full_vocab_maximum(logits, topk_values):
    """Reuse the selected maximum, with exact zero/non-finite fallback."""
    if not logits.shape[0] or not _supported(logits) or not logits.is_contiguous():
        return logits.max(dim=-1).values
    return full_vocab_scoring.maximum(logits, topk_values)


def cuda_full_vocab_tail(logits, ids, log_z, *, row_maximum, active_rows):
    """Mask raw logits during ATen loads; fold row_maximum + log_z at writeback."""
    if not _supported(logits) or not logits.is_contiguous():
        from tokenspeed_kernel.ops.sampling.torch import torch_full_vocab_tail

        return torch_full_vocab_tail(
            logits, ids, log_z, row_maximum=row_maximum, active_rows=active_rows
        )
    active = (
        torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
        if active_rows is None
        else active_rows
    )
    if not logits.shape[0]:
        return log_z.clone()
    return full_vocab_scoring.tail(logits, ids, log_z, row_maximum, active)


if full_vocab_scoring.available():
    for mode, impl in (
        ("full_vocab_row_selection_supports", cuda_full_vocab_row_selection_supports),
        ("full_vocab_topk", cuda_full_vocab_topk),
        ("full_vocab_maximum", cuda_full_vocab_maximum),
        ("full_vocab_tail", cuda_full_vocab_tail),
    ):
        register_kernel(
            "sampling",
            mode,
            name=impl.__name__,
            solution="cuda",
            features={"accepted_rows"},
            capability=CapabilityRequirement(
                vendors=frozenset({"nvidia"}),
                min_arch_version=ArchVersion(9, 0),
                max_arch_version=ArchVersion(9, 0),
            ),
            signatures=format_signatures("values", "dense", {torch.float32}),
            priority=Priority.SPECIALIZED,
        )(impl)
