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

"""Qwen4-Exp's request-shaped state remains live in a padded prefill graph."""

from types import SimpleNamespace

import pytest
import torch

import tokenspeed.runtime.layers.attention.qsa.indexer as qsa_indexer_module
from tokenspeed.runtime.execution.breakable_cuda_graph import (
    BreakableCapture,
    active_forward,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.specific.qwen4_exp import (
    Qwen4ExpBackend,
)
from tokenspeed.runtime.layers.attention.backends.specific.qwen4_exp_ple import (
    PLEForwardMetadata,
)
from tokenspeed.runtime.layers.attention.kv_cache.qwen4_exp import (
    qwen4_exp_ple_context_field,
    qwen4_exp_ple_conv_field,
)
from tokenspeed.runtime.layers.attention.qsa.indexer import QSAIndexer
from tokenspeed.runtime.layers.qwen4_exp_ple import Qwen4ExpPLELayer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_ple_break_replays_ragged_rows_and_live_state_pages() -> None:
    class Lookup:
        def make_layout(self, global_num_tokens, count):
            return None

        def start(self, ids, layout):
            return ids

        def finish(self, pending):
            return pending.to(torch.float32).unsqueeze(-1)

    class Embedding(torch.nn.Module):
        eos_token_id = 0
        ngram_heads = 1

        def __init__(self):
            super().__init__()
            self.lookup = Lookup()

        def _ngram_ids_flat_cuda(
            self,
            flat_ids,
            initial,
            req,
            col,
            lengths_t,
            starts,
            need_tail,
            uniform_length,
            *,
            tail_out,
            tail_block_rows,
        ):
            if uniform_length:
                positions = torch.arange(flat_ids.numel(), device=flat_ids.device)
                req.copy_(positions // uniform_length)
                col.copy_(positions % uniform_length)
                lengths_t.fill_(uniform_length)
                starts.copy_(
                    torch.arange(initial.shape[0], device=flat_ids.device)
                    * uniform_length
                )
            return flat_ids, None

    class Projection(torch.nn.Module):
        def forward(self, values):
            return values.repeat(1, 4), None

    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    torch.nn.Module.__init__(layer)
    layer.layer_id = 0
    layer.context_field_id = qwen4_exp_ple_context_field(0)
    layer.hidden_size = 2
    layer.hc_count = 1
    layer.hc_hidden_size = 2
    layer.ngram_size = 2
    layer.context_len = 1
    layer.ple_embedding = Embedding()
    layer.kv_proj = Projection()
    layer._prefetched = None
    layer._ple_backend = lambda ctx: ctx.attn_backend
    layer._metadata = lambda backend: backend.forward_metadata
    layer._gate_and_norm_cuda = lambda key, hidden, value: (value, value)

    def conv_sequences(values, initial, lengths, index, *, add_terms, **kwargs):
        return add_terms[0] + add_terms[1], initial, None

    layer._conv_sequences = conv_sequences
    context = torch.zeros((6, 1), dtype=torch.int64, device="cuda")
    conv = torch.zeros((6, 2, 1), dtype=torch.bfloat16, device="cuda")
    fields = {layer.context_field_id: context, qwen4_exp_ple_conv_field(0): conv}
    pool = SimpleNamespace(arena=SimpleNamespace(field=fields.__getitem__))

    def make_ctx(lengths, input_pages, output_pages):
        metadata = PLEForwardMetadata(
            input_blocks=torch.tensor(input_pages, dtype=torch.int32, device="cuda"),
            output_blocks=torch.tensor(output_pages, dtype=torch.int32, device="cuda"),
            query_lengths=lengths,
            verify_width=None,
        )
        return SimpleNamespace(
            bs=len(lengths),
            global_num_tokens=sum(lengths),
            forward_mode=ForwardMode.EXTEND,
            token_to_kv_pool=pool,
            attn_backend=SimpleNamespace(forward_metadata=metadata),
        )

    bucket = 8
    static_ids = torch.ones(bucket, dtype=torch.int64, device="cuda")
    static_hidden = torch.zeros((bucket, 2), dtype=torch.float32, device="cuda")
    dummy = make_ctx([bucket], [0], [1])

    def forward(ctx):
        hidden = static_hidden * 2
        return layer(hidden, static_ids, ctx) * 3

    for _ in range(2):
        forward(dummy)
    torch.cuda.synchronize()
    capture = BreakableCapture()
    with active_forward(dummy), capture:
        captured = forward(dummy)
    assert capture.num_segments == 3

    for lengths, input_pages, output_pages in (
        ([3, 2], [0, 0], [2, 3]),
        ([1], [3], [4]),
        ([2, 4], [2, 4], [1, 5]),
    ):
        live = make_ctx(lengths, input_pages, output_pages)
        count = sum(lengths)
        static_ids[:count].copy_(torch.arange(11, 11 + count, device="cuda"))
        static_hidden[:count].copy_(
            torch.arange(count * 2, device="cuda").reshape(count, 2)
        )
        context.zero_()
        conv.zero_()
        expected = forward(live).clone()
        expected_context = context.clone()
        expected_conv = conv.clone()
        context.zero_()
        conv.zero_()
        with active_forward(live):
            capture.replay(valid_rows=count)
        torch.testing.assert_close(captured[:count], expected[:count], rtol=0, atol=0)
        torch.testing.assert_close(context, expected_context, rtol=0, atol=0)
        torch.testing.assert_close(conv, expected_conv, rtol=0, atol=0)
        assert torch.count_nonzero(captured[count:]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_qsa_break_replays_live_rows_positions_and_cache_writes(monkeypatch) -> None:
    indexer = QSAIndexer.__new__(QSAIndexer)
    torch.nn.Module.__init__(indexer)
    indexer.layer_id = 0
    indexer.share_topk_for_mtp_iteration = False
    indexer.compressed_token_page_size = 256
    indexer.recent_page_size = 64
    indexer.compress_ratio = 4
    indexer._project_qk_raw = lambda hidden: (hidden[:, :1], hidden[:, :1])
    indexer._fields = lambda pool: (None, pool.compressed, None)

    def write_and_compress(
        token_k, positions, logical, requests, qsa_locs, recent_locs, pool, **kwargs
    ):
        pool.cache[pool.write_page, : token_k.shape[0]] = token_k[:, 0] + positions
        return token_k

    indexer._write_and_compress = write_and_compress
    indexer._select_slots = lambda q, logical, *args, **kwargs: torch.stack(
        (logical.to(torch.int32), q[:, 0].to(torch.int32)), dim=-1
    )
    pool = SimpleNamespace(
        cache=torch.zeros((6, 8), device="cuda"),
        compressed=torch.zeros((1, 1), device="cuda"),
        layerwise_load_tracker=None,
        write_page=1,
    )
    backend = Qwen4ExpBackend.__new__(Qwen4ExpBackend)
    backend.indexer_backend = SimpleNamespace(spec_num_tokens=1, is_draft=False)
    backend.attention_backend = SimpleNamespace(
        sparse_topk=SimpleNamespace(decode=None)
    )

    def make_ctx(lengths, write_page):
        count = sum(lengths)
        logical = torch.arange(count, device="cuda", dtype=torch.int32) + write_page
        pool.write_page = write_page
        return SimpleNamespace(
            bs=len(lengths),
            num_extends=len(lengths),
            forward_mode=ForwardMode.EXTEND,
            draft_narrowing=None,
            attn_backend=backend,
            token_to_kv_pool=pool,
            layout=SimpleNamespace(
                logical_positions=logical,
                request_indices=torch.repeat_interleave(
                    torch.arange(len(lengths), device="cuda", dtype=torch.int32),
                    torch.tensor(lengths, device="cuda"),
                ),
                qsa_locs=logical,
                recent_locs=logical,
                complete_blocks=torch.ones(count, device="cuda", dtype=torch.int32),
                qsa_page_table=pool.compressed,
                full_page_table=pool.compressed,
                full_kernel_page_size=64,
            ),
        )

    monkeypatch.setattr(
        qsa_indexer_module,
        "qsa_forward_layout",
        lambda ctx, total_tokens, **kwargs: ctx.layout,
    )
    bucket = 8
    static_hidden = torch.zeros((bucket, 2), device="cuda")
    static_positions = torch.zeros(bucket, dtype=torch.int64, device="cuda")
    dummy = make_ctx([bucket], 1)

    def forward(ctx, rows=bucket):
        return (
            indexer(static_hidden[:rows] * 2, static_positions[:rows], ctx).float() * 3
        )

    for _ in range(2):
        forward(dummy)
    torch.cuda.synchronize()
    capture = BreakableCapture()
    with active_forward(dummy), capture:
        captured = forward(dummy)
    assert capture.num_segments == 3

    for lengths, page in (([3, 2], 2), ([1], 3), ([2, 4], 4)):
        live = make_ctx(lengths, page)
        count = sum(lengths)
        static_hidden[:count].copy_(
            torch.arange(count * 2, device="cuda").reshape(count, 2)
        )
        static_positions[:count].copy_(torch.arange(10, 10 + count, device="cuda"))
        pool.cache.zero_()
        expected = forward(live, count).clone()
        expected_cache = pool.cache.clone()
        pool.cache.zero_()
        pool.write_page = page
        with active_forward(live):
            capture.replay(valid_rows=count)
        torch.testing.assert_close(captured[:count], expected, rtol=0, atol=0)
        torch.testing.assert_close(pool.cache, expected_cache, rtol=0, atol=0)
        assert torch.count_nonzero(captured[count:]) == 0
