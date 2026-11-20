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

"""Local Decode preserves causal rows, padding and stable batch metadata."""

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.layers.attention.backends.paged import (
    ascend_dsa_local_decode as module,
)


@pytest.fixture
def decode(monkeypatch):
    def store(rows, cache, slots):
        valid = slots >= 0
        cache.view(-1, 128)[slots[valid]] = rows.view(-1, 128)[valid]

    def select(query, cache, weights, ends, lengths, table, topk, initial, window):
        assert query.shape[0] == ends.numel() == lengths.numel() == table.shape[0]
        return torch.zeros(query.shape[0], 1, topk, dtype=torch.int32), None

    def attention(q, qr, cache, indices, chunks, ends, lengths, table, scale):
        output = torch.zeros_like(q)
        for row, length in enumerate(lengths.tolist()):
            if length:
                data = cache[table[row].long()].view(-1, 576)[:length, :512]
                output[row] = data.mean(0)
        return output

    monkeypatch.setattr(
        module,
        "select_paged_dsa_indexer",
        lambda: SimpleNamespace(store_index_keys=store, select_indices=select),
    )
    monkeypatch.setattr(
        module,
        "select_paged_dsa_attention",
        lambda: SimpleNamespace(attention=attention),
    )
    config = SimpleNamespace(max_bs=4, speculative_num_draft_tokens=8)
    spec = SimpleNamespace(
        num_attention_heads=32,
        index_head_dim=128,
        index_topk=2048,
        kv_cache_dim=576,
        kv_lora_rank=512,
    )
    value = module.AscendDSALocalDecode(config, spec, page_size=128)
    value.allocate_decode_buffers(32, 3, "cpu")
    return value


@pytest.mark.parametrize("width", [1, 2, 4, 8])
def test_local_lengths_and_tables_refresh_in_place(decode, width):
    table = torch.tensor([[7, 6, 5], [1, 3, 2], [0, 0, 0]], dtype=torch.int32)
    lengths = torch.tensor([99, 257, 8], dtype=torch.int32)
    meta = SimpleNamespace(
        num_extends=1, seq_lens=lengths, page_table=table, q_len_per_req=width
    )
    decode.refresh(meta)
    pointers = (
        decode.query_ends.data_ptr(),
        decode.seq_lens.data_ptr(),
        decode.page_table.data_ptr(),
    )
    for end in [257, 129, 320]:
        lengths[1] = end
        table[1] = table[1].roll(1)
        decode.refresh(meta)
        assert decode.query_ends.tolist() == list(range(1, 2 * width + 1))
        assert (
            decode.seq_lens.tolist()
            == list(range(end - width + 1, end + 1)) + [0] * width
        )
        assert (
            decode.page_table.tolist()
            == [table[1].tolist()] * width + [[0, 0, 0]] * width
        )
        assert pointers == (
            decode.query_ends.data_ptr(),
            decode.seq_lens.data_ptr(),
            decode.page_table.data_ptr(),
        )
        if width == 1:
            assert decode.page_table.is_set_to(table[1:])


@pytest.mark.parametrize("width", [1, 8])
def test_local_forward_commits_keys_and_uses_each_causal_end(decode, width):
    cache = (
        torch.arange(4 * 128, dtype=torch.float32)[:, None]
        .expand(-1, 576)
        .reshape(4, 128, 1, 576)
        .contiguous()
    )
    keys = [torch.full((4, 128, 1, 128), -99.0) for _ in range(2)]
    table = torch.tensor([[2, 1, 3], [0, 0, 0]], dtype=torch.int32)
    lengths = torch.tensor([257, 8], dtype=torch.int32)
    meta = SimpleNamespace(
        num_extends=0, seq_lens=lengths, page_table=table, q_len_per_req=width
    )
    pool = SimpleNamespace(
        get_key_buffer=lambda layer: cache,
        get_component=lambda layer, name: keys[layer],
    )
    for end in [257, 259]:
        lengths[0] = end
        decode.refresh(meta)
        positions = torch.arange(end - width, end)
        slots = torch.cat(
            (
                table[0, positions // 128] * 128 + positions % 128,
                torch.full((width,), -1),
            )
        ).long()
        q = torch.zeros(2 * width, 32, 576)
        for layer_id in range(2):
            values = torch.full(
                (2 * width, 1, 128), layer_id + end, dtype=torch.float32
            )
            actual = decode.forward(
                q,
                None,
                None,
                SimpleNamespace(layer_id=layer_id, scaling=1),
                slots,
                pool,
                2,
                fused=False,
                metadata=meta,
                index_key=values,
                index_query=torch.zeros(2 * width, 16, 128),
                index_weights=torch.zeros(2 * width, 16),
                index_init_tokens=0,
                index_local_tokens=0,
            )
            for row, visible in enumerate(range(end - width + 1, end + 1)):
                expected = (
                    cache[table[0].long()]
                    .reshape(-1, 576)[:visible, :512]
                    .mean(0)
                    .repeat(32)
                )
                torch.testing.assert_close(actual[row], expected)
            assert torch.count_nonzero(actual[width:]) == 0
            assert torch.equal(
                keys[layer_id].view(-1, 128)[slots[:width]], values[:width, 0]
            )
            assert (keys[layer_id][0] == -99).all()
