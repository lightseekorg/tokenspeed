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

"""NVFP4 DeepEP must run deferred work inside its dispatch window."""

from __future__ import annotations

import pytest
import torch

nvfp4 = pytest.importorskip("tokenspeed_kernel.ops.moe.nvfp4.flashinfer_cutedsl_deepep")

pytestmark = pytest.mark.skipif(
    not hasattr(nvfp4, "flashinfer_cutedsl_deepep_nvfp4_moe_apply"),
    reason="NVFP4 DeepEP needs an NVIDIA platform with FlashInfer",
)

_NUM_LOCAL_EXPERTS = 2
_RECV_M = 4
_HIDDEN = 256
_INTERMEDIATE = 128
_TOP_K = 2


class _RecordingDispatcher:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def dispatch_a(self, x, topk_ids, topk_weights, low_latency=None) -> None:
        del x, topk_ids, topk_weights, low_latency
        self.calls.append("dispatch_a")

    def dispatch_b(self):
        self.calls.append("dispatch_b")
        recv_hidden = torch.zeros(
            (_NUM_LOCAL_EXPERTS, _RECV_M, _HIDDEN), dtype=torch.bfloat16
        )
        masked_m = torch.full((_NUM_LOCAL_EXPERTS,), _RECV_M, dtype=torch.int32)
        return recv_hidden, None, None, None, None, None, masked_m

    def combine_a(self, output, topk_ids, topk_weights, low_latency=None) -> None:
        del output, topk_ids, topk_weights, low_latency
        self.calls.append("combine_a")

    def combine_b(self):
        self.calls.append("combine_b")
        return torch.zeros((1, _HIDDEN), dtype=torch.bfloat16)


def _weights() -> torch.nn.Module:
    w = torch.nn.Module()
    w.num_local_experts = _NUM_LOCAL_EXPERTS
    w.w13_weight = torch.empty(
        (_NUM_LOCAL_EXPERTS, 4 * _INTERMEDIATE, _HIDDEN), dtype=torch.bfloat16
    )
    w.w2_weight = torch.empty(
        (_NUM_LOCAL_EXPERTS, _HIDDEN, _INTERMEDIATE), dtype=torch.bfloat16
    )
    w.w13_blockscale_swizzled = torch.empty(0)
    w.w2_blockscale_swizzled = torch.empty(0)
    w.w13_input_scale_quant = torch.ones(1)
    w.w2_input_scale_quant = torch.ones(1)
    w.g1_alphas = torch.ones(_NUM_LOCAL_EXPERTS)
    w.g2_alphas = torch.ones(_NUM_LOCAL_EXPERTS)
    return w


def test_overlap_runs_between_nvfp4_dispatch_legs(monkeypatch) -> None:
    """The shared expert callback must execute after send and before receive."""
    calls: list[str] = []
    dispatcher = _RecordingDispatcher(calls)
    empty_quantized = (torch.empty(0), torch.empty(0))
    monkeypatch.setattr(
        nvfp4, "scaled_fp4_grouped_quantize", lambda *args: empty_quantized
    )
    monkeypatch.setattr(
        nvfp4,
        "silu_and_mul_scaled_nvfp4_experts_quantize",
        lambda *args: empty_quantized,
    )
    monkeypatch.setattr(nvfp4, "grouped_gemm_nt_masked", lambda *args, **kwargs: None)

    nvfp4.flashinfer_cutedsl_deepep_nvfp4_moe_apply(
        plan={"_deepep_dispatcher": dispatcher},
        x=torch.zeros((1, _HIDDEN), dtype=torch.bfloat16),
        w=_weights(),
        router_logits=torch.zeros((1, _NUM_LOCAL_EXPERTS)),
        topk_weights=torch.zeros((1, _TOP_K)),
        topk_ids=torch.zeros((1, _TOP_K), dtype=torch.int64),
        low_latency=True,
        overlap_fn=lambda: calls.append("overlap"),
    )

    assert calls == [
        "dispatch_a",
        "overlap",
        "dispatch_b",
        "combine_a",
        "combine_b",
    ]
