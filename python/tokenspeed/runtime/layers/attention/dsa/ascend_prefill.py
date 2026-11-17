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

"""Native DSA Prefill numerical calls over the shared QCP/KVP transport."""

import torch


def run_prefill_attention(q, layer, prefill, kv, index, *, kernels, spec, **kwargs):
    """Native Indexer and SFA on the prepared full-head QCP rows."""
    index_pages = index.view(-1, prefill.page_size, 1, spec.index_head_dim)
    if q.shape[0]:
        indices = kernels.prefill_index(
            kwargs["index_query"],
            index_pages,
            kwargs["index_weights"],
            prefill.query_cu,
            prefill.key_cu,
            prefill.page_table,
            spec.index_topk,
            kwargs["index_init_tokens"],
            kwargs["index_local_tokens"],
        )
    else:
        indices = torch.empty((0, spec.index_topk), dtype=torch.int32, device=q.device)
    kv_pages = kv.view(-1, prefill.page_size, spec.kv_cache_dim)
    if q.shape[0]:
        output = kernels.prefill_attention(
            q,
            kv_pages,
            indices,
            prefill.scan_count,
            prefill.page_table,
            prefill.query_seq,
            layer.scaling,
        )
    else:
        output = q.new_empty((q.shape[1], 0, spec.kv_lora_rank))
    return output
