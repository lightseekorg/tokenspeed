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

"""Verify row-sharded DSA prefill warms the unsharded local-head path at startup."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.ops.tuning import kernel_warmup

from tokenspeed.runtime.layers.attention.backends.paged import dsa as dsa_mod


def _make_row_shard_backend(
    *, num_attention_heads: int = 96, tp_size: int = 8
) -> dsa_mod.DSABackend:
    backend = object.__new__(dsa_mod.DSABackend)
    backend.data_type = torch.float8_e4m3fn
    backend.num_attention_heads = num_attention_heads
    backend.num_local_heads = num_attention_heads // tp_size
    backend.attn_tp_group = tuple(range(tp_size))
    backend.attn_tp_rank = 0
    backend.qk_nope_head_dim = 128
    backend.kv_lora_rank = 512
    backend.qk_rope_head_dim = 64
    backend.kernel_page_size = 64
    backend.kernel_solution = "flashinfer_prims"
    backend.prefill_kernel_plan = None
    backend.slot_order = "selection"
    backend._row_shard_prefill = True
    backend._row_shard_unsharded_warmed = set()
    return backend


def test_row_shard_sparse_prefill_warms_unsharded_local_heads_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _make_row_shard_backend(num_attention_heads=96, tp_size=8)
    monkeypatch.setattr(dsa_mod, "_ROW_SHARD_MIN_ROWS", 16)
    monkeypatch.setattr(dsa_mod, "trtllm_sparse_mla_query_heads", lambda h: h)

    calls: list[tuple[int, int, bool]] = []

    def _fake_exchange(q_view, group, rank, counts, core_heads):
        del group
        local_rows = counts[rank]
        return torch.zeros(
            (local_rows, core_heads, q_view.shape[-1]),
            dtype=q_view.dtype,
            device=q_view.device,
        )

    def _fake_return(out, local_heads, group, counts):
        del group
        return torch.zeros(
            (sum(counts), local_heads, out.shape[-1]),
            dtype=out.dtype,
            device=out.device,
        )

    def _fake_dsa_prefill(*, q, return_lse, **kwargs):
        del kwargs
        calls.append((q.shape[0], q.shape[1], return_lse))
        return torch.zeros(
            (q.shape[0], q.shape[1], 512), dtype=torch.bfloat16, device=q.device
        )

    monkeypatch.setattr(dsa_mod, "exchange_query_rows_padded", _fake_exchange)
    monkeypatch.setattr(dsa_mod, "return_head_rows", _fake_return)
    monkeypatch.setattr(dsa_mod, "dsa_prefill", _fake_dsa_prefill)

    layer = SimpleNamespace(layer_id=0, scaling=1.0, logit_cap=0.0)
    pool = SimpleNamespace(
        get_key_buffer=lambda _lid: torch.zeros((64, 576), dtype=torch.float8_e4m3fn)
    )
    q_view = torch.zeros((32, 12, 576), dtype=torch.float8_e4m3fn)
    topk_slots = torch.zeros((32, 512), dtype=torch.int32)
    topk_lens = torch.full((32,), 16, dtype=torch.int32)

    # 1. Outside warmup: only the row-sharded core runs (4 rows x 96 heads).
    backend._forward_row_shard_sparse_prefill(
        q_view=q_view,
        layer=layer,
        token_to_kv_pool=pool,
        kv_seq_lens=None,
        topk_slots=topk_slots,
        topk_lens=topk_lens,
        max_seq_len=64,
    )
    assert calls == [(4, 96, False)]

    # 2. First warmup layer: warms unsharded (16 rows x 12 heads), then runs
    #    row-sharded (4 rows x 96 heads).
    calls.clear()
    with kernel_warmup(enabled=True):
        backend._forward_row_shard_sparse_prefill(
            q_view=q_view,
            layer=layer,
            token_to_kv_pool=pool,
            kv_seq_lens=None,
            topk_slots=topk_slots,
            topk_lens=topk_lens,
            max_seq_len=64,
        )
        # 3. Second warmup layer with identical config: skips redundant unsharded
        #    warmup and runs only the row-sharded core.
        backend._forward_row_shard_sparse_prefill(
            q_view=q_view,
            layer=SimpleNamespace(layer_id=1, scaling=1.0, logit_cap=0.0),
            token_to_kv_pool=pool,
            kv_seq_lens=None,
            topk_slots=topk_slots,
            topk_lens=topk_lens,
            max_seq_len=64,
        )
    assert calls == [(16, 12, False), (4, 96, False), (4, 96, False)]
