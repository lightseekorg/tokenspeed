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

"""CUDA QSA block selection registrations."""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

if current_platform().is_nvidia:
    from tokenspeed_kernel.ops.attention.dsa.cuda import (
        has_ragged_decode_topk,
        ragged_decode_topk,
    )

    if has_ragged_decode_topk():

        @register_kernel(
            "attention",
            "qsa_block_topk_select",
            name="cuda_qsa_block_topk_select",
            solution="cuda",
            capability=CapabilityRequirement(
                min_arch_version=ArchVersion(9, 0),
                vendors=frozenset({"nvidia"}),
            ),
            signatures=frozenset(
                {format_signature(logits=dense_tensor_format(torch.float32))}
            ),
            traits={"topk": frozenset({512, 1024, 2048})},
            priority=Priority.SPECIALIZED,
            features=frozenset({"logits"}),
        )
        def cuda_qsa_block_topk_select(
            logits: torch.Tensor,
            topk: int,
            *,
            lengths: torch.Tensor,
            enable_pdl: bool,
        ) -> torch.Tensor:
            """Select QSA blocks with the length-aware CUDA persistent kernel.

            Args:
                logits: FP32 scores shaped ``[rows, cols]``. Strided inputs are
                    copied using ordinary stream dependencies before selection.
                topk: Selected blocks per row: 512, 1024 or 2048.
                lengths: Per-row valid prefix lengths. Later columns are ignored.
                enable_pdl: Enable programmatic launch dependencies.

            Returns:
                Int32 column ids shaped ``[rows, topk]``, padded with ``-1``.
            """
            selected = torch.empty(
                (logits.shape[0], topk), dtype=torch.int32, device=logits.device
            )
            if logits.shape[0] == 0:
                return selected
            if logits.shape[1] == 0:
                return selected.fill_(-1)
            ragged_decode_topk(
                logits,
                selected,
                topk,
                lengths=lengths,
                max_seq_len=logits.shape[1],
                q_len_per_req=1,
                enable_pdl=enable_pdl,
            )
            return selected
