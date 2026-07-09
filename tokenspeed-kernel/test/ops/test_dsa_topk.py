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

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.ops.attention.flashinfer import dsa_topk


def test_decode_topk_uses_ragged_path_when_lengths_and_workspace_are_provided(
    monkeypatch,
):
    calls = {}

    monkeypatch.setattr(dsa_topk, "has_persistent_topk", lambda: True)

    def fake_persistent_topk(
        logits, lengths, output, workspace, k, max_seq_len, q_len_per_req=1
    ):
        calls["logits"] = logits
        calls["lengths"] = lengths
        calls["workspace"] = workspace
        calls["k"] = k
        calls["max_seq_len"] = max_seq_len
        calls["q_len_per_req"] = q_len_per_req
        output.fill_(7)

    monkeypatch.setattr(dsa_topk, "persistent_topk", fake_persistent_topk)

    logits = torch.randn(2, 8, dtype=torch.float32)
    lengths = torch.tensor([[3], [6]], dtype=torch.int64)
    output = torch.empty(2, 4, dtype=torch.int32)
    workspace = torch.empty(32, dtype=torch.uint8)

    dsa_topk.deterministic_decode_topk(
        logits,
        output,
        4,
        lengths=lengths,
        workspace=workspace,
        max_seq_len=8,
    )

    assert torch.equal(output, torch.full_like(output, 7))
    assert calls["logits"].is_contiguous()
    assert torch.equal(calls["lengths"], torch.tensor([3, 6], dtype=torch.int32))
    assert calls["workspace"] is workspace
    assert calls["k"] == 4
    assert calls["max_seq_len"] == 8
    assert calls["q_len_per_req"] == 1


def test_decode_topk_rejects_partial_or_unavailable_ragged_inputs(monkeypatch):
    monkeypatch.setattr(dsa_topk, "has_persistent_topk", lambda: False)

    logits = torch.randn(2, 8, dtype=torch.float32)
    output = torch.empty(2, 4, dtype=torch.int32)
    lengths = torch.tensor([3, 6], dtype=torch.int32)
    workspace = torch.empty(32, dtype=torch.uint8)

    with pytest.raises(RuntimeError, match="length-aware"):
        dsa_topk.deterministic_decode_topk(
            logits,
            output,
            4,
            lengths=lengths,
            workspace=workspace,
            max_seq_len=8,
        )

    monkeypatch.setattr(dsa_topk, "has_persistent_topk", lambda: True)
    with pytest.raises(RuntimeError, match="length-aware"):
        dsa_topk.deterministic_decode_topk(
            logits,
            output,
            4,
            lengths=lengths,
            workspace=None,
            max_seq_len=8,
        )


def test_decode_topk_falls_back_to_flashinfer_deterministic_topk(monkeypatch):
    calls = {}
    indices = torch.tensor([[1, 0, 3], [2, 4, 1]], dtype=torch.int64)

    def fake_top_k(logits, k, *, deterministic, tie_break, dsa_graph_safe):
        calls["logits"] = logits
        calls["k"] = k
        calls["deterministic"] = deterministic
        calls["tie_break"] = tie_break
        calls["dsa_graph_safe"] = dsa_graph_safe
        return None, indices

    monkeypatch.setattr(dsa_topk, "top_k", fake_top_k)
    monkeypatch.setattr(dsa_topk, "TopKTieBreak", SimpleNamespace(SMALL="small"))

    logits = torch.randn(2, 8, dtype=torch.float32)
    output = torch.empty(2, 3, dtype=torch.int32)

    dsa_topk.deterministic_decode_topk(logits, output, 3)

    assert torch.equal(output, indices.to(torch.int32))
    assert calls["logits"].is_contiguous()
    assert calls["k"] == 3
    assert calls["deterministic"] is True
    assert calls["tie_break"] == "small"
    assert calls["dsa_graph_safe"] is True
