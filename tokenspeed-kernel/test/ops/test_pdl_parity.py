"""PDL (programmatic dependent launch) parity for the decode Triton kernels.

Each kernel that gained an ``enable_pdl`` path must produce byte-identical
results with PDL on vs off (the gdc_wait / gdc_launch_dependents intrinsics are
memory-ordering fences, not numeric transforms), both standalone and when
captured in a CUDA graph chained after a producer that writes its input.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.activation.triton import (
    rmsnorm_gated_sigmoid,
    sigmoid_mul,
    silu_and_mul,
    situ_and_mul,
)
from tokenspeed_kernel.ops.gemm.triton_gemv import rowcta_gemv, rowcta_gemv_add3
from tokenspeed_kernel.ops.moe.triton.kimi3_sigmoid_topk import kimi3_sigmoid_bias_topk
from tokenspeed_kernel.platform import current_platform

platform = current_platform()
torch.manual_seed(0)

pytestmark = pytest.mark.skipif(
    not (platform.is_nvidia or platform.is_amd),
    reason="PDL parity tests require an NVIDIA or AMD GPU.",
)


def test_rowcta_gemv_pdl_matches(device: str) -> None:
    x = torch.randn(1, 2048, dtype=torch.bfloat16, device=device)
    w = torch.randn(2304, 2048, dtype=torch.bfloat16, device=device)
    off = rowcta_gemv(x, w)
    on = rowcta_gemv(x, w, enable_pdl=True)
    torch.testing.assert_close(on, off, atol=0, rtol=0)


def test_rowcta_gemv_add3_pdl_matches(device: str) -> None:
    n, k = 2304, 2048
    x = torch.randn(1, k, dtype=torch.bfloat16, device=device)
    w = torch.randn(n, k, dtype=torch.bfloat16, device=device)
    a = torch.randn(1, n, dtype=torch.bfloat16, device=device)
    c = torch.randn(1, n, dtype=torch.bfloat16, device=device)
    off = rowcta_gemv_add3(x, w, a, c)
    on = rowcta_gemv_add3(x, w, a, c, enable_pdl=True)
    torch.testing.assert_close(on, off, atol=0, rtol=0)


def test_silu_and_mul_pdl_matches(device: str) -> None:
    x = torch.randn(1, 4096, dtype=torch.bfloat16, device=device)
    torch.testing.assert_close(
        silu_and_mul(x, enable_pdl=True), silu_and_mul(x), atol=0, rtol=0
    )


def test_situ_and_mul_pdl_matches(device: str) -> None:
    x = torch.randn(1, 4096, dtype=torch.bfloat16, device=device)
    off = situ_and_mul(x, beta=1.0, linear_beta=2.0)
    on = situ_and_mul(x, beta=1.0, linear_beta=2.0, enable_pdl=True)
    torch.testing.assert_close(on, off, atol=0, rtol=0)


def test_sigmoid_mul_pdl_matches(device: str) -> None:
    base = torch.randn(1, 4096, dtype=torch.bfloat16, device=device)
    gate = torch.randn(1, 4096, dtype=torch.bfloat16, device=device)
    off = base.clone()
    sigmoid_mul(off, gate)
    on = base.clone()
    sigmoid_mul(on, gate, enable_pdl=True)
    torch.testing.assert_close(on, off, atol=0, rtol=0)


def test_rmsnorm_gated_pdl_matches(device: str) -> None:
    nh, hd = 8, 128
    x = torch.randn(1, nh * hd, dtype=torch.bfloat16, device=device)
    gate = torch.randn(1, nh * hd, dtype=torch.bfloat16, device=device)
    weight = torch.randn(hd, dtype=torch.bfloat16, device=device)
    off = rmsnorm_gated_sigmoid(x, gate, weight, 1e-6, nh, hd)
    on = rmsnorm_gated_sigmoid(x, gate, weight, 1e-6, nh, hd, enable_pdl=True)
    torch.testing.assert_close(on, off, atol=0, rtol=0)


def test_kimi3_sigmoid_topk_pdl_matches(device: str) -> None:
    logits = torch.randn(1, 896, dtype=torch.float32, device=device)
    bias = torch.randn(896, dtype=torch.float32, device=device)
    kw = dict(routed_scaling_factor=2.5, normalize_topk_weights=True)
    w_off, i_off = kimi3_sigmoid_bias_topk(logits, bias, **kw)
    w_on, i_on = kimi3_sigmoid_bias_topk(logits, bias, enable_pdl=True, **kw)
    torch.testing.assert_close(w_on, w_off, atol=0, rtol=0)
    torch.testing.assert_close(i_on, i_off, atol=0, rtol=0)


def test_pdl_chained_producer_consumer_cuda_graph(device: str) -> None:
    """PDL-armed consumer chained after a PDL-armed producer in one graph.

    The producer GEMV writes the packed gate|up buffer the SiLU consumer reads;
    with both PDL-armed the consumer's gdc_wait must still observe the
    producer's stores, so the replayed result must match an eager reference.
    """
    k, d = 2048, 2048
    x = torch.randn(1, k, dtype=torch.bfloat16, device=device)
    w = torch.randn(2 * d, k, dtype=torch.bfloat16, device=device)
    gate_up = torch.empty(1, 2 * d, dtype=torch.bfloat16, device=device)
    out = torch.empty(1, d, dtype=torch.bfloat16, device=device)

    def step():
        rowcta_gemv(x, w, out=gate_up, enable_pdl=True)
        return silu_and_mul(gate_up, out=out, enable_pdl=True)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        res = step()

    x.copy_(torch.randn_like(x))
    graph.replay()
    torch.cuda.synchronize()

    ref_gate_up = x @ w.t()
    gate = ref_gate_up[..., :d].float()
    up = ref_gate_up[..., d:].float()
    ref = gate * torch.sigmoid(gate) * up
    torch.testing.assert_close(res.float(), ref, atol=2e-1, rtol=2e-1)
