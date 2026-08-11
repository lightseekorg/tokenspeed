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

import tokenspeed_kernel.ops.attention as attention_ops
import tokenspeed_kernel.ops.attention.flashinfer as flashinfer_ops
import torch


class _SelectedKernel:
    name = "test_dsv4_sparse_mla"

    def __init__(self, result: torch.Tensor) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> torch.Tensor:
        self.calls.append(kwargs)
        return self.result


def _inputs() -> dict[str, torch.Tensor]:
    return {
        "q": torch.randn(2, 64, 512, dtype=torch.bfloat16),
        "swa_kv_cache": torch.empty(3, 64, 1, 584, dtype=torch.uint8),
        "swa_indices": torch.zeros(2, 128, dtype=torch.int32),
        "swa_topk_lens": torch.full((2,), 128, dtype=torch.int32),
        "compressed_kv_cache": torch.empty(4, 2, 1, 584, dtype=torch.uint8),
        "compressed_indices": torch.zeros(2, 1, 512, dtype=torch.int32),
        "compressed_topk_lens": torch.full((2,), 512, dtype=torch.int32),
        "sinks": torch.zeros(64, dtype=torch.float32),
    }


def test_dsv4_sparse_mla_decode_dispatches_dual_cache_contract(monkeypatch) -> None:
    inputs = _inputs()
    expected = torch.empty_like(inputs["q"])
    selected = _SelectedKernel(expected)
    selection: dict[str, object] = {}

    def fake_select_kernel(family, mode, signature, **kwargs):
        selection.update(family=family, mode=mode, kwargs=kwargs)
        return selected

    monkeypatch.setattr(attention_ops, "select_kernel", fake_select_kernel)
    actual = attention_ops.dsv4_sparse_mla_decode(
        inputs["q"],
        inputs["swa_kv_cache"],
        inputs["swa_indices"],
        inputs["swa_topk_lens"],
        compressed_kv_cache=inputs["compressed_kv_cache"],
        compressed_indices=inputs["compressed_indices"],
        compressed_topk_lens=inputs["compressed_topk_lens"],
        softmax_scale=512**-0.5,
        sinks=inputs["sinks"],
    )

    assert actual is expected
    assert selection == {
        "family": "attention",
        "mode": "dsv4_sparse_mla_decode",
        "kwargs": {
            "traits": {
                "head_dim": 512,
                "swa_page_size": 64,
                "compressed_page_size": 2,
                "support_sinks": True,
            },
            "solution": None,
            "override": None,
        },
    }
    assert selected.calls[0]["compressed_kv_cache"] is inputs["compressed_kv_cache"]
    assert selected.calls[0]["compressed_indices"] is inputs["compressed_indices"]


def test_flashinfer_dsv4_sparse_mla_forwards_both_cache_segments(monkeypatch) -> None:
    inputs = _inputs()
    expected = torch.empty_like(inputs["q"])
    workspace = torch.empty(1, dtype=torch.uint8)
    call: dict[str, object] = {}

    def fake_sparse_mla(**kwargs):
        call.update(kwargs)
        return expected

    monkeypatch.setattr(
        flashinfer_ops, "trtllm_batch_decode_sparse_mla_dsv4", fake_sparse_mla
    )
    monkeypatch.setattr(
        flashinfer_ops, "_get_dsa_sparse_workspace", lambda _device: workspace
    )

    actual = flashinfer_ops._flashinfer_dsv4_sparse_mla_decode(
        q=inputs["q"],
        swa_kv_cache=inputs["swa_kv_cache"],
        swa_indices=inputs["swa_indices"],
        swa_topk_lens=inputs["swa_topk_lens"],
        compressed_kv_cache=inputs["compressed_kv_cache"],
        compressed_indices=inputs["compressed_indices"],
        compressed_topk_lens=inputs["compressed_topk_lens"],
        softmax_scale=512**-0.5,
        sinks=inputs["sinks"],
        out=None,
    )

    assert actual is expected
    assert call["swa_kv_cache"] is inputs["swa_kv_cache"]
    assert call["compressed_kv_cache"] is inputs["compressed_kv_cache"]
    assert call["extra_sparse_indices"] is inputs["compressed_indices"]
    assert call["kv_layout"] == "NHD"
