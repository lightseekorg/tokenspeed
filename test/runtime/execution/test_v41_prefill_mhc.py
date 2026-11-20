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

"""Real native HC joins and their consumers across prefill graph boundaries."""

from __future__ import annotations

import gc
import os
import sys
import weakref

import pytest
import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.gemm import fp8_projection, prepare_fp8_projection
from tokenspeed_kernel.platform import current_platform

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=20, suite="runtime-1gpu")

from tokenspeed.runtime.execution.breakable_cuda_graph import (  # noqa: E402
    BreakableCapture,
    HandoffSlot,
)
from tokenspeed.runtime.layers.layernorm import RMSNorm  # noqa: E402
from tokenspeed.runtime.models import deepseek_v41 as v41  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not current_platform().is_nvidia
    or current_platform().arch_version.major != 10,
    reason="requires an SM100-family GPU and native MegaMHC",
)


class NativeCaptureProbe:
    """Observe CUDA state at a real kernel call without replacing its work."""

    def __init__(self, native, *, captured: bool, supported: bool) -> None:
        self.native = native
        self.captured = captured
        self.supported = supported
        self.outputs: list[weakref.ReferenceType[torch.Tensor]] = []

    def __call__(self, *args, **kwargs):
        assert torch.cuda.is_current_stream_capturing() == self.captured
        result = self.native(*args, **kwargs)
        assert (result is not None) == self.supported
        if result is not None:
            self.outputs.extend(weakref.ref(t) for t in result)
        return result


class HCConsumers(torch.nn.Module):
    """Two real joins with projections between them and saved first outputs.

    The snapshots consume the first join before the second can reuse its
    handoff storage. A real FP8 projection reads the packed scale layout.
    Only weights are small; HC arithmetic and graph execution are unchanged.
    """

    def __init__(self, hidden: int, *, quantize_fp8: bool) -> None:
        super().__init__()
        self.quantize_fp8 = quantize_fp8
        self.norm = RMSNorm(hidden, eps=1e-5).to("cuda", torch.bfloat16)
        self.register_buffer(
            "weight", torch.randn(24, 4 * hidden, device="cuda") * 0.03
        )
        self.register_buffer("scale", torch.tensor([0.7, 1.1, 0.5], device="cuda"))
        self.register_buffer("base", torch.randn(24, device="cuda") * 0.1)
        codes = torch.randn(128, hidden, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.full((128, hidden // 32), 121, dtype=torch.uint8, device="cuda")
        prepared = prepare_fp8_projection(codes, scales) if quantize_fp8 else None
        if quantize_fp8:
            assert prepared is not None
        self.register_buffer("projection_weight", codes)
        self.register_buffer("projection_scales", prepared)
        self.register_buffer("bf16_weight", (codes.float() / 64).bfloat16())

    def join(self, x, residual, pre, post, comb):
        residual, y, mixes, quantized = v41._v41_hc_post_pre(
            residual,
            pre,
            v41.V41HCPost(x, post, comb, None),
            hc_weights=(self.weight, self.scale, self.base),
            norm=self.norm,
            hc_norm_eps=1e-20,
            hc_eps=1e-6,
            sinkhorn_iters=20,
            prefer_overlap=False,
            quantize_fp8=self.quantize_fp8,
        )
        if y is None:
            y = v41._v41_hc_input(residual, pre, self.norm)
        if mixes is None:
            mixes = v41.v41_hc_mixes(
                residual, self.weight, self.scale, self.base, 1e-20, 1e-6, 20
            )
        projected = (
            F.linear(y, self.bf16_weight)
            if quantized is None
            else fp8_projection(
                *quantized, self.projection_weight, self.projection_scales, None
            )
        )
        return residual, y, *mixes, projected, quantized

    def forward(self, x, residual, pre, post, comb):
        first = self.join(x.sin(), residual, pre, post, comb)
        r, y, pre, post, comb, projected, _ = first
        snapshots = tuple(t.clone() for t in first[:6])
        second = self.join(
            y.tanh() + projected.mean(-1, keepdim=True), r, pre, post, comb
        )
        outputs = snapshots + tuple(t.clone() for t in second[:6])
        quantized = second[-1]
        if quantized is not None:
            # Returning the scale view also checks the handoff's physical layout.
            outputs += (quantized[0].clone(), quantized[1])
        return outputs


def _inputs(tokens: int, hidden: int):
    return (
        torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16),
        torch.randn(tokens, 4, hidden, device="cuda", dtype=torch.bfloat16),
        torch.rand(tokens, 4, device="cuda"),
        torch.rand(tokens, 4, device="cuda") * 2,
        torch.randn(tokens, 4, 4, device="cuda").softmax(-2),
    )


def _refresh(inputs) -> None:
    x, residual, pre, post, comb = inputs
    x.normal_()
    residual.normal_()
    pre.uniform_()
    post.uniform_(0, 2)
    comb.copy_(torch.randn_like(comb).softmax(-2))


def _probe(monkeypatch, *, quantize_fp8: bool, supported: bool):
    name = (
        "try_mhc_shifted_post_pre_norm_quantize_fp8"
        if quantize_fp8
        else "try_mhc_shifted_post_pre_norm"
    )
    native = (
        v41.try_mhc_shifted_post_pre_norm_quantize_fp8
        if quantize_fp8
        else v41.try_mhc_shifted_post_pre_norm
    )
    probe = NativeCaptureProbe(native, captured=False, supported=supported)
    monkeypatch.setattr(v41, name, probe)
    return probe


def _capture(
    module,
    inputs,
    *,
    pool,
    storage: dict[HandoffSlot, torch.Tensor],
):
    cap = BreakableCapture(
        pool=pool,
        handoff_storage=storage,
        run_segments_while_capturing=True,
    )
    cap.stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(cap.stream):
        for _ in range(2):
            module(*inputs)
    cap.stream.synchronize()
    with cap:
        output = module(*inputs)
    return cap, output


def _compare(actual, expected, valid_rows: int) -> None:
    for got, ref in zip(actual, expected, strict=True):
        if got.dtype == torch.float8_e4m3fn:
            got, ref = got.view(torch.uint8), ref.view(torch.uint8)
        torch.testing.assert_close(got[:valid_rows], ref[:valid_rows], rtol=0, atol=0)


@torch.inference_mode()
@pytest.mark.parametrize("tokens,hidden", [(17, 1024), (33, 5120), (257, 5120)])
@pytest.mark.parametrize("quantize_fp8", [False, True])
def test_prefill_native_mhc_stays_eager_and_refreshes_handoffs(
    monkeypatch, tokens, hidden, quantize_fp8
):
    torch.manual_seed(41)
    probe = _probe(monkeypatch, quantize_fp8=quantize_fp8, supported=True)
    module = HCConsumers(hidden, quantize_fp8=quantize_fp8)
    inputs = _inputs(tokens, hidden)
    cap, actual = _capture(module, inputs, pool=None, storage={})

    for valid_rows in (tokens, tokens - 1, 1, 0, min(8, tokens), tokens):
        _refresh(inputs)
        expected = module(*inputs)
        cap.replay(valid_rows=valid_rows)
        _compare(actual, expected, valid_rows)
        for tensor in actual:
            tail = tensor[valid_rows:]
            if tail.dtype == torch.float8_e4m3fn:
                tail = tail.view(torch.uint8)
            assert torch.count_nonzero(tail) == 0
        if quantize_fp8:
            assert actual[-1].stride() == (1, (tokens + 3) // 4 * 4)
        del expected

    gc.collect()
    assert all(ref() is None for ref in probe.outputs)


@torch.inference_mode()
@pytest.mark.parametrize("quantize_fp8", [False, True])
def test_prefill_mhc_native_decline_keeps_real_fallback(monkeypatch, quantize_fp8):
    torch.manual_seed(41)
    _probe(monkeypatch, quantize_fp8=quantize_fp8, supported=False)
    module = HCConsumers(128, quantize_fp8=quantize_fp8)
    inputs = _inputs(17, 128)
    x, residual, pre, post, comb = inputs
    completed, y, mixes, quantized = v41._v41_hc_post_pre(
        residual,
        pre,
        v41.V41HCPost(x, post, comb, None),
        hc_weights=(module.weight, module.scale, module.base),
        norm=module.norm,
        hc_norm_eps=1e-20,
        hc_eps=1e-6,
        sinkhorn_iters=20,
        prefer_overlap=False,
        quantize_fp8=quantize_fp8,
    )
    assert y is mixes is quantized is None
    torch.testing.assert_close(
        completed, v41.v41_hc_post(x, residual, post, comb), rtol=0, atol=0
    )
    cap, actual = _capture(module, inputs, pool=None, storage={})
    for valid_rows in (17, 1, 0, 9, 17):
        _refresh(inputs)
        expected = module(*inputs)
        cap.replay(valid_rows=valid_rows)
        _compare(actual, expected, valid_rows)


@torch.inference_mode()
def test_prefill_mhc_packed_handoffs_survive_other_bucket_capture(monkeypatch):
    torch.manual_seed(41)
    _probe(monkeypatch, quantize_fp8=True, supported=True)
    module = HCConsumers(1024, quantize_fp8=True)
    storage: dict[HandoffSlot, torch.Tensor] = {}
    captures = []
    pool = None
    for tokens in (33, 17):
        inputs = _inputs(tokens, 1024)
        cap, output = _capture(module, inputs, pool=pool, storage=storage)
        pool = cap.pool
        captures.append((tokens, inputs, cap, output))
    for _ in range(3):
        for tokens, inputs, cap, actual in captures:
            _refresh(inputs)
            expected = module(*inputs)
            cap.replay(valid_rows=tokens - 1)
            _compare(actual, expected, tokens - 1)
            assert actual[-1].stride() == (1, (tokens + 3) // 4 * 4)


@torch.inference_mode()
@pytest.mark.parametrize("quantize_fp8", [False, True])
def test_plain_cuda_graph_keeps_mhc_capture_transparent(monkeypatch, quantize_fp8):
    """The prefill boundary must not change ordinary full-graph capture.

    Native split-barrier replay numerics are a separate kernel contract. Here
    the CUDA state at the actual native call checks only this boundary's scope.
    """
    torch.manual_seed(41)
    probe = _probe(monkeypatch, quantize_fp8=quantize_fp8, supported=True)
    module = HCConsumers(1024, quantize_fp8=quantize_fp8)
    inputs = _inputs(17, 1024)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            module(*inputs)
    stream.synchronize()
    probe.captured = True
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = module(*inputs)
    graph.replay()
    assert all(torch.isfinite(t.float()).all() for t in actual[:12])
