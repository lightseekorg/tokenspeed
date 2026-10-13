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

"""GLM DSA decode top-k: decode-window workspaces and the MTP row gather."""

import os
import sys
from types import SimpleNamespace

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

import torch

from tokenspeed.runtime.layers.attention.dsa.utils import (
    _prepare_dsa_topk_for_mtp_decode,
)
from tokenspeed.runtime.models import glm5

TOPK = 4


def _attention() -> glm5.GlmMoeDsaAttention:
    attention = object.__new__(glm5.GlmMoeDsaAttention)
    attention.index_topk = TOPK
    attention.indexer = SimpleNamespace(weights_softmax_scale=0.5)
    attention.attn_mqa = SimpleNamespace(layer_id=0)
    attention._decode_topk_indices_buffer = None
    attention._decode_topk_lens_buffer = None
    attention._retired_decode_workspaces = []
    return attention


def _select(attention, *, num_tokens: int, decode_start: int, num_extends: int):
    ctx = SimpleNamespace(
        num_extends=num_extends,
        token_to_kv_pool=SimpleNamespace(
            get_index_k_buffer=lambda layer_id: torch.empty(0),
            arena=SimpleNamespace(kv_page_size=64),
        ),
        attn_backend=SimpleNamespace(
            forward_decode_metadata=SimpleNamespace(
                _dsa_seq_lens_2d=None, _dsa_plan=torch.zeros(2, dtype=torch.int32)
            ),
            cache_placement=lambda attn_mqa: None,
        ),
    )
    return attention._compute_decode_topk_indices_portable(
        indexer_output=glm5.GlmDsaIndexerOutput(
            query=torch.zeros(num_tokens, 1, 8),
            key=torch.empty(0),
            weights=torch.zeros(num_tokens, 1),
        ),
        ctx=ctx,
        seq_lens=torch.tensor([30, 40], dtype=torch.int32),
        page_table=torch.zeros(2, 1, dtype=torch.int32),
        q_len_per_req=1,
        decode_start=decode_start,
        num_decode_tokens=2,
        topk=TOPK,
    )


def test_mixed_round_selects_into_the_decode_round_workspace(monkeypatch):
    # Each indexer layer keeps one workspace, and a captured decode graph
    # replays into it. A mixed round's selection covers only its decode rows,
    # so a long prefill chunk neither regrows it nor retires the captured one.
    selected_rows = []

    def decode_topk(q, weights, seq_lens, page_table, *, out, lens_out, **kwargs):
        selected_rows.append(q.shape[0])
        out.fill_(len(selected_rows))
        lens_out.fill_(len(selected_rows))

    monkeypatch.setattr(glm5, "dsa_decode_topk", decode_topk)
    attention = _attention()

    decode = _select(attention, num_tokens=2, decode_start=0, num_extends=0)
    mixed = _select(attention, num_tokens=8194, decode_start=8192, num_extends=1)

    assert selected_rows == [2, 2]
    assert mixed.topk_indices.shape == (2, TOPK)
    assert mixed.topk_indices.data_ptr() == decode.topk_indices.data_ptr()
    assert attention._retired_decode_workspaces == []
    assert (decode.row_offset, mixed.row_offset) == (0, 8192)

    indices, lens = attention._slice_decode_topk(mixed, 8192, 8194)
    assert indices.data_ptr() == decode.topk_indices.data_ptr()
    assert indices.tolist() == [[2] * TOPK] * 2
    assert lens.tolist() == [2, 2]


def test_mtp_gather_reads_mixed_decode_rows_past_the_extend_rows():
    prefill = glm5.GlmDsaPrefillTopK(
        workspace_indices=torch.tensor([[0, 1], [2, -1], [3, 4]], dtype=torch.int32),
        topk_lens=torch.tensor([2, 1, 2], dtype=torch.int32),
        page_table=torch.empty((0, 0), dtype=torch.int32),
        seq_lens=torch.empty(0, dtype=torch.int32),
        kv_seq_lens=torch.empty(0, dtype=torch.int32),
        max_seq_len=0,
        kv_workspace_slots=torch.tensor([10, 11, 12, 13, 14], dtype=torch.int32),
    )
    # Three extend tokens, then two decode requests at token rows 3 and 4.
    decode = glm5.GlmDsaDecodeTopK(
        topk_indices=torch.tensor([[50, 51], [60, 61]], dtype=torch.int32),
        topk_lens=torch.tensor([2, 1], dtype=torch.int32),
        row_offset=3,
    )

    _, selected = _prepare_dsa_topk_for_mtp_decode(
        (prefill, decode), torch.tensor([2, 4, 3]), num_prefill_rows=1
    )

    assert selected.topk_indices.tolist() == [[13, 14], [60, 61], [50, 51]]
    assert selected.topk_lens.tolist() == [2, 1, 2]
    assert selected.row_offset == 0
