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

"""GFX1250 MLA decode with causal query blocks on the query axis."""

from __future__ import annotations

import math

import pytest
import torch
from utils import assert_no_triton_compile, is_cdna5

if not is_cdna5():
    pytest.skip("AMD CDNA5 is required for Gluon MLA tests", allow_module_level=True)

from tokenspeed_kernel.ops.attention.mla import (  # noqa: E402
    mla_decode_with_kvcache,
    supports_mla_decode_query_blocks,
)
from tokenspeed_kernel_amd.ops.gfx1250.attention.mla.decode import (  # noqa: E402
    _mla_decode_fwd_kernel,
    _mla_decode_fwd_reduce_kernel,
    _select_query_block_num_kv_splits,
)

_KV_LORA_RANK = 512
_ROPE_DIM = 64
_QK_DIM = _KV_LORA_RANK + _ROPE_DIM
_PAGE_SIZE = 64
_SOFTMAX_SCALE = 192**-0.5


def _make_inputs(cache_lengths, heads, queries, dtype=torch.float8_e4m3fn, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    pages_per_request = [math.ceil(length / _PAGE_SIZE) for length in cache_lengths]
    num_pages = sum(pages_per_request)
    cache = (
        torch.randn(num_pages + 1, _PAGE_SIZE, 1, _QK_DIM, generator=gen, device="cuda")
        * 0.25
    ).to(dtype)
    permutation = torch.randperm(num_pages, generator=gen, device="cuda")
    # Unused table entries name the spare page, which no request owns.
    table = torch.full(
        (len(cache_lengths), max(pages_per_request)),
        num_pages,
        device="cuda",
        dtype=torch.int32,
    )
    offset = 0
    for request, pages in enumerate(pages_per_request):
        table[request, :pages] = permutation[offset : offset + pages]
        offset += pages
    q = (
        torch.randn(
            len(cache_lengths), queries, heads, _QK_DIM, generator=gen, device="cuda"
        )
        * 0.25
    ).to(dtype)
    lengths = torch.tensor(cache_lengths, device="cuda", dtype=torch.int32)
    return q, cache, table, lengths


def _reference(q, cache, table, lengths, softmax_scale=_SOFTMAX_SCALE):
    output = torch.zeros((*q.shape[:-1], _KV_LORA_RANK), device=q.device)
    lse = torch.full(q.shape[:-1], -float("inf"), device=q.device)
    for request, length in enumerate(lengths.tolist()):
        pages = table[request, : math.ceil(length / cache.shape[1])].long()
        kv = cache[pages].reshape(-1, _QK_DIM)
        for position in range(q.shape[1]):
            visible = length - q.shape[1] + position + 1
            values = kv[:visible].float()
            scores = q[request, position].float() @ values.T * softmax_scale
            output[request, position] = scores.softmax(-1) @ values[:, :_KV_LORA_RANK]
            lse[request, position] = torch.logsumexp(scores, -1)
    return output, lse


def _decode(q, cache, table, lengths, **kwargs):
    return mla_decode_with_kvcache(
        q=q,
        kv_cache=cache,
        page_table=table,
        cache_seqlens=lengths,
        qk_nope_head_dim=128,
        kv_lora_rank=_KV_LORA_RANK,
        qk_rope_head_dim=_ROPE_DIM,
        softmax_scale=_SOFTMAX_SCALE,
        solution="gluon",
        **kwargs,
    )


def _relative_error(actual, expected):
    return ((actual.float() - expected).norm() / expected.norm()).item()


@pytest.mark.parametrize(
    "heads,queries,cache_lengths,dtype",
    [
        # Kimi-K3 verify at TP8: one 64-row block per request, with lengths
        # on both sides of page and split boundaries.
        (12, 4, [4, 5, 63, 64, 65, 1025, 4095, 4096, 4097, 16385], torch.float8_e4m3fn),
        (12, 4, [129, 4097], torch.bfloat16),
        # 16 and 32 rows per block.
        (4, 4, [4, 1025, 4097], torch.float8_e4m3fn),
        (8, 4, [4, 2049], torch.float8_e4m3fn),
        # Several query blocks per request, the last one partial.
        (16, 15, [15, 1025], torch.float8_e4m3fn),
        # Head blocks.
        (128, 4, [1025, 4], torch.bfloat16),
    ],
)
def test_query_block_mla_matches_reference(heads, queries, cache_lengths, dtype):
    q, cache, table, lengths = _make_inputs(cache_lengths, heads, queries, dtype)
    output = torch.empty(
        *q.shape[:-1], _KV_LORA_RANK, device="cuda", dtype=torch.bfloat16
    )
    result, lse = _decode(
        q,
        cache,
        table,
        lengths,
        max_seqlen_k=max(cache_lengths),
        return_lse=True,
        out=output,
    )
    reference, reference_lse = _reference(q, cache, table, lengths)
    assert result.data_ptr() == output.data_ptr()
    assert torch.isfinite(output).all()
    assert _relative_error(output, reference) < 0.05
    torch.testing.assert_close(lse, reference_lse, atol=2e-3, rtol=2e-4)


def test_query_block_mla_graph_replay_follows_pages_and_lengths():
    batch_size, heads, queries = 8, 12, 4
    q, cache, table, lengths = _make_inputs([20_000] * batch_size, heads, queries)
    output = torch.empty(
        *q.shape[:-1], _KV_LORA_RANK, device="cuda", dtype=torch.bfloat16
    )

    def run():
        return _decode(q, cache, table, lengths, max_seqlen_k=65_536, out=output)

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run()
    for history in ([4, 4096, 4097, 20_000] * 2, [20_000, 65, 64, 4095] * 2):
        lengths.copy_(torch.tensor(history, device="cuda", dtype=torch.int32))
        table.copy_(table.roll(1, 0))
        output.fill_(float("nan"))
        graph.replay()
        reference, _ = _reference(q, cache, table, lengths)
        assert result.data_ptr() == output.data_ptr()
        assert torch.isfinite(output).all()
        assert _relative_error(output, reference) < 0.05


def test_query_block_mla_reuses_binaries_across_batch_and_table_width():
    heads, queries = 12, 4
    q, cache, table, lengths = _make_inputs([4097] * 8, heads, queries)
    spare = cache.shape[0] - 1

    def run(batch_size, table_width):
        padding = table_width - table.shape[1]
        padded = torch.nn.functional.pad(table[:batch_size], (0, padding), value=spare)
        result = _decode(
            q[:batch_size],
            cache,
            padded,
            lengths[:batch_size],
            max_seqlen_k=65_536,
        )
        reference, _ = _reference(q[:batch_size], cache, padded, lengths[:batch_size])
        assert _relative_error(result, reference) < 0.05

    sweep = ((5, 72), (6, 80), (7, 96))
    num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    # Each request is one program, so the batch size sets the KV split count.
    splits = {
        _select_query_block_num_kv_splits(
            num_sms=num_sms,
            num_q_programs=batch_size,
            max_seqlen_k=65_536,
            tile_size=_PAGE_SIZE,
        )
        for batch_size in (8, 5, 6, 7)
    }
    assert len(splits) == 1, f"batches 5 to 8 need one split count, got {splits}"

    run(8, 65)
    with assert_no_triton_compile(
        _mla_decode_fwd_kernel, _mla_decode_fwd_reduce_kernel
    ):
        for batch_size, table_width in sweep:
            run(batch_size, table_width)


def test_query_block_mla_projected_value_composes_on_query_axis():
    batch_size, heads, queries, value_dim = 4, 12, 4, 128
    q, cache, table, lengths = _make_inputs([4097] * batch_size, heads, queries)
    weights = torch.randn(
        heads, _KV_LORA_RANK, value_dim, device="cuda", dtype=torch.bfloat16
    ) / math.sqrt(_KV_LORA_RANK)
    gate = torch.randn(
        batch_size * queries, heads * value_dim, device="cuda", dtype=torch.bfloat16
    )
    output = torch.empty_like(gate)
    _decode(
        q,
        cache,
        table,
        lengths,
        max_seqlen_k=65_536,
        value_weight=weights,
        gate=gate,
        out=output,
    )
    reference, _ = _reference(q, cache, table, lengths)
    projected = torch.einsum("bqhd,hdv->bqhv", reference, weights.float())
    expected = projected.reshape_as(output) * torch.sigmoid(gate.float())
    assert _relative_error(output, expected) < 0.05


def test_query_block_support_on_cdna5():
    common = dict(
        q_dtype=torch.float8_e4m3fn,
        kv_dtype=torch.float8_e4m3fn,
        page_size=_PAGE_SIZE,
        num_q_heads=12,
        kv_lora_rank=_KV_LORA_RANK,
        qk_rope_head_dim=_ROPE_DIM,
        noncausal_block_size=1,
    )
    assert supports_mla_decode_query_blocks(q_len=4, sliding_window=False, **common)
    assert supports_mla_decode_query_blocks(q_len=16, sliding_window=False, **common)
    assert not supports_mla_decode_query_blocks(q_len=4, sliding_window=True, **common)
    assert not supports_mla_decode_query_blocks(
        q_len=17, sliding_window=False, **common
    )
    assert not supports_mla_decode_query_blocks(
        q_len=4, sliding_window=False, **{**common, "noncausal_block_size": 4}
    )
