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

"""Local DSA Decode: no KV ownership mapping or inter-rank communication."""

import torch
from tokenspeed_kernel.ops.attention.dsa.paged import (
    select_paged_dsa_attention,
    select_paged_dsa_indexer,
)


class AscendDSALocalDecode:
    """TP1/KVP1 execution, including causal multi-token verification."""

    def __init__(self, config, spec, *, page_size):
        self.spec = spec
        self.page_size = page_size
        self.indexer = select_paged_dsa_indexer()
        self.attention = select_paged_dsa_attention()
        self.max_rows = config.max_bs * config.speculative_num_draft_tokens
        self.query_ends = None
        self.page_table = None
        self.seq_lens = None
        self.write_slots = None
        self._query_ends = None
        self._lengths = None
        self._table = None

    def allocate_decode_buffers(self, rows, columns, device):
        # Only derived metadata needs storage. W1 borrows the host's page table.
        self._query_ends = torch.arange(1, rows + 1, dtype=torch.int32, device=device)
        self._lengths = torch.empty(rows, dtype=torch.int32, device=device)
        self._table = torch.empty((rows, columns), dtype=torch.int32, device=device)

    def refresh(self, metadata):
        start = metadata.num_extends
        table, lengths = metadata.page_table[start:], metadata.seq_lens[start:]
        width = metadata.q_len_per_req
        rows = lengths.numel() * width
        if self._query_ends is None or self._query_ends.numel() < rows:
            self.allocate_decode_buffers(
                max(self.max_rows, rows), table.shape[1], table.device
            )
        self.query_ends = self._query_ends[:rows]
        self.seq_lens = self._lengths[:rows]
        self.write_slots = None
        # The host uses a null first page for padded requests. Such rows must
        # remain empty even when verify lengths are clamped to at least one.
        active = table[:, 0] > 0
        if width == 1:
            self.page_table = table
            torch.mul(lengths, active, out=self.seq_lens)
        else:
            self.page_table = self._table[:rows]
            self.page_table.view(-1, width, table.shape[1]).copy_(table[:, None, :])
            offsets = self._query_ends[:width] - width
            causal = (lengths[:, None] + offsets).clamp_min(1)
            self.seq_lens.copy_(torch.where(active[:, None], causal, 0).reshape(-1))

    def forward(
        self,
        q,
        k,
        v,
        layer,
        locations,
        pool,
        bs,
        *,
        fused,
        metadata,
        step_counter=None,
        **kwargs
    ):
        if fused or kwargs.get("packed_query") is not None:
            raise ValueError("Local DSA Decode does not use TP/KVP fused producers")
        spec, page = self.spec, self.page_size
        if q.shape[1] != spec.num_attention_heads:
            raise ValueError("Local DSA Decode requires complete attention heads")
        index_cache = pool.get_component(layer.layer_id, "dsa_index_k").view(
            -1, page, 1, spec.index_head_dim
        )
        if self.write_slots is None:
            capacity = index_cache.shape[0] * page
            self.write_slots = torch.where(
                (locations >= page) & (locations < capacity), locations, -1
            ).to(torch.int64)
        self.indexer.store_index_keys(
            kwargs["index_key"].contiguous(), index_cache, self.write_slots
        )
        indices, chunks = self.indexer.select_indices(
            kwargs["index_query"],
            index_cache,
            kwargs["index_weights"],
            self.query_ends,
            self.seq_lens,
            self.page_table,
            spec.index_topk,
            kwargs["index_init_tokens"],
            kwargs["index_local_tokens"],
        )
        if step_counter is not None:
            step_counter.record_cache()
        cache = pool.get_key_buffer(layer.layer_id).view(-1, page, 1, spec.kv_cache_dim)
        output = self.attention.attention(
            q[..., : spec.kv_lora_rank].contiguous(),
            q[..., spec.kv_lora_rank :].contiguous(),
            cache,
            indices,
            chunks,
            self.query_ends,
            self.seq_lens,
            self.page_table,
            layer.scaling,
        )
        return (
            output.transpose(0, 1).contiguous()
            if kwargs.get("head_major_output", False)
            else output.flatten(1)
        )
