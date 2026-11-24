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

"""SM120 Full indexer with identical scoring and exact library selection."""

import torch
from tokenspeed_kernel.ops.attention.dsv41.triton import _index_topk
from tokenspeed_kernel.ops.attention.dsv41.triton_sm120 import (
    triton_sm120_dsv41_index_topk,
)
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from tokenspeed_kernel.thirdparty.flashinfer.topk import exact_topk, has_exact_topk

platform = current_platform()

if (
    platform.is_nvidia
    and platform.arch_version == ArchVersion(12, 0)
    and has_exact_topk()
):

    @register_kernel(
        "attention",
        "dsv41_index_topk",
        name="flashinfer_sm120_dsv41_index_topk",
        solution="flashinfer_sm120",
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(12, 0),
            max_arch_version=ArchVersion(12, 0),
        ),
        signatures=frozenset({format_signature(x=dense_tensor_format(torch.bfloat16))}),
        traits={"index_heads": frozenset({32}), "index_k_format": frozenset({"mxfp4"})},
        priority=Priority.SPECIALIZED + 1,
    )
    def flashinfer_sm120_dsv41_index_topk(
        index_q,
        weights,
        index_cache,
        page_table,
        visible_lens,
        candidate_blocks,
        topk,
        candidate_topk,
        candidate_block_size,
        query_chunk_size,
        score_chunk_size,
        process_group,
        out,
        index_metadata,
    ):
        """Keep the existing score rounding; replace Full Top512's streaming sort.

        A query tile materializes at most 64MiB of FP32 scores. Library
        selection retains smaller logical IDs on ties; the shared finalizer
        sorts IDs and pads invalid picks. Short/candidate/Reindex calls keep
        the existing portable implementation and never call the library.
        """
        eligible = (
            index_q.shape[0] >= 256
            and candidate_blocks is None
            and candidate_topk == 0
            and topk == 512
            and score_chunk_size >= 256
            and 512 <= page_table.shape[1] * 64 <= 65536
        )
        if not eligible:
            return triton_sm120_dsv41_index_topk(
                index_q,
                weights,
                index_cache,
                page_table,
                visible_lens,
                candidate_blocks,
                topk,
                candidate_topk,
                candidate_block_size,
                query_chunk_size,
                score_chunk_size,
                process_group,
                out,
                index_metadata,
            )
        return _index_topk(
            index_q,
            weights,
            index_cache,
            page_table,
            visible_lens,
            candidate_blocks,
            topk,
            candidate_topk,
            candidate_block_size,
            query_chunk_size,
            score_chunk_size,
            process_group,
            out,
            index_metadata,
            scan_num_warps=8,
            dense_selector=exact_topk,
        )
