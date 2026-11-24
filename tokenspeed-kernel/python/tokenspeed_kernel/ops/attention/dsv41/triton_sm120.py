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

"""SM120 launch geometry for the portable V4.1 indexer."""

import torch
from tokenspeed_kernel.ops.attention.dsv41.triton import _index_topk
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


@register_kernel(
    "attention",
    "dsv41_index_topk",
    name="triton_sm120_dsv41_index_topk",
    solution="triton_sm120",
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(12, 0),
        max_arch_version=ArchVersion(12, 1),
    ),
    signatures=frozenset({format_signature(x=dense_tensor_format(torch.bfloat16))}),
    traits={"index_heads": frozenset({32}), "index_k_format": frozenset({"mxfp4"})},
    priority=Priority.SPECIALIZED,
)
def triton_sm120_dsv41_index_topk(
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
    # Use the measured eight-warp geometry for long 32-head Full scans.
    # The scoring math and TopK contract stay the same.
    # Keep unmeasured short queries, candidate production and Reindex unchanged.
    tuned = (
        index_q.shape[0] >= 256
        and topk == 512
        and candidate_topk == 0
        and candidate_blocks is None
        and score_chunk_size >= 256
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
        scan_num_warps=8 if tuned else 4,
        dense_selector=None,
    )
