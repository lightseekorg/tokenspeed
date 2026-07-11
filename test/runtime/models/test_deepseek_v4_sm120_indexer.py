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
from unittest.mock import patch

import pytest
import torch

import tokenspeed.runtime.models.deepseek_v4 as deepseek_v4_model


@pytest.mark.parametrize(
    ("arch_major", "expected_backend"),
    [(12, "tokenspeed"), (10, "trtllm")],
)
def test_prefill_topk_uses_tokenspeed_cuda_only_on_sm120(
    monkeypatch,
    arch_major: int,
    expected_backend: str,
) -> None:
    monkeypatch.setattr(
        deepseek_v4_model,
        "_platform",
        SimpleNamespace(
            is_nvidia=True,
            arch_version=SimpleNamespace(major=arch_major),
        ),
    )
    calls: list[str] = []

    def fake_tokenspeed_topk(*args, **kwargs) -> None:
        del args, kwargs
        calls.append("tokenspeed")

    def fake_trtllm_topk(*args, **kwargs) -> None:
        del args, kwargs
        calls.append("trtllm")

    monkeypatch.setattr(
        deepseek_v4_model,
        "has_indexer_topk_prefill",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(
        deepseek_v4_model,
        "indexer_topk_prefill",
        fake_tokenspeed_topk,
        raising=False,
    )
    logits = torch.empty((2, 4), dtype=torch.float32)
    row_starts = torch.zeros(2, dtype=torch.int32)
    row_ends = torch.full((2,), 4, dtype=torch.int32)
    output = torch.empty((2, 2), dtype=torch.int32)

    with patch.object(
        torch.ops.trtllm,
        "indexer_topk_prefill",
        fake_trtllm_topk,
        create=True,
    ):
        deepseek_v4_model._deepseek_v4_launch_indexer_topk_prefill(
            logits,
            row_starts,
            row_ends,
            output,
            2,
        )

    assert calls == [expected_backend]


@pytest.mark.parametrize(
    ("arch_major", "expects_direct_out"),
    [(12, True), (10, False)],
)
def test_prefill_indexer_writes_directly_only_on_sm120(
    monkeypatch,
    arch_major: int,
    expects_direct_out: bool,
) -> None:
    monkeypatch.setattr(
        deepseek_v4_model,
        "_platform",
        SimpleNamespace(
            is_nvidia=True,
            arch_version=SimpleNamespace(major=arch_major),
        ),
    )
    captured: dict[str, object] = {}

    def fake_prefill_topk(**kwargs):
        out = kwargs.get("out")
        captured["out"] = out
        if out is None:
            topk = torch.full((2, 4), 7, dtype=torch.int32)
        else:
            assert isinstance(out, torch.Tensor)
            out.fill_(7)
            topk = out
        return topk, None

    monkeypatch.setattr(
        deepseek_v4_model,
        "_deepseek_v4_indexer_topk_prefill_deepgemm",
        fake_prefill_topk,
    )
    topk_buffer = torch.empty((2, 4), dtype=torch.int32)

    actual = deepseek_v4_model._deepseek_v4_sparse_attn_indexer_native(
        cache_2d=torch.empty((1, 1), dtype=torch.uint8),
        positions=torch.arange(2, dtype=torch.int64),
        token_to_req_indices=torch.zeros(2, dtype=torch.int32),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        seq_lens_cpu=torch.tensor([2], dtype=torch.int32),
        query_lens_cpu=torch.tensor([2], dtype=torch.int32),
        prefill_chunk_specs=torch.tensor([[0, 2, 0, 1, 0]], dtype=torch.int64),
        prefill_chunk_offsets=torch.tensor([[0, 0, 0, 1, 1, 0, 2]], dtype=torch.int64),
        prefill_slots=torch.empty(0, dtype=torch.int64),
        prefill_cu_seq_lens=torch.tensor([0, 1], dtype=torch.int32),
        prefill_cu_seqlen_k_start=torch.tensor([0], dtype=torch.int32),
        prefill_cu_seqlen_k_end=torch.tensor([1], dtype=torch.int32),
        prefill_seq_lens_k=torch.tensor([1], dtype=torch.int32),
        packed_q_values=torch.empty((2, 1, 1), dtype=torch.int8),
        packed_q_scales=torch.empty((2, 1), dtype=torch.int32),
        packed_weights=torch.empty((2, 1), dtype=torch.float32),
        decode_schedule_metadata=None,
        decode_context_lens=None,
        decode_block_table=None,
        decode_max_context_len=0,
        topk_indices_buffer=topk_buffer,
        prefill_gather_values_workspace=torch.empty((0, 1), dtype=torch.uint8),
        prefill_gather_scales_workspace=torch.empty((0, 1), dtype=torch.uint8),
        persistent_topk_workspace=torch.empty(0, dtype=torch.uint8),
        cache_block_size=1,
        compress_ratio=4,
        topk_tokens=4,
        num_prefill_tokens=2,
        num_decode_tokens=0,
    )

    if expects_direct_out:
        assert captured["out"] is not None
        assert isinstance(captured["out"], torch.Tensor)
        assert captured["out"].data_ptr() == topk_buffer.data_ptr()
    else:
        assert captured["out"] is None
    torch.testing.assert_close(actual, torch.full_like(actual, 7))
