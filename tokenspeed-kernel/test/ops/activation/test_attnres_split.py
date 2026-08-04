"""Parity tests for the split AttnRes kernels (partial + combine).

The split factors the online-softmax candidate mix into a blocks-side partial
(precomputable on the aux stream) and a prefix-side combine.
"""

import pytest
import torch
from tokenspeed_kernel.ops.model.kimi_k3.attn_res.triton import (
    attnres_combine,
    attnres_partial,
    attnres_partial_dual,
)

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

H = 7168  # K3 hidden size; the kernels static-assert two 4096 sweeps.


def _reference(prefix, blocks, wp, eps, out_w):
    """Direct softmax over the KB+1 candidate logits, fp32."""
    cands = torch.cat([blocks.float(), prefix.float().unsqueeze(0)], dim=0)
    wp = wp.float()[None, None, :]
    rms = torch.rsqrt(cands.pow(2).mean(-1, keepdim=True) + eps)
    logits = (cands * rms * wp).sum(-1)  # [KB+1, T]
    w = torch.softmax(logits, dim=0)[..., None]
    mix = (cands * w).sum(0).to(torch.bfloat16).float()
    if out_w is None:
        return mix
    rms_o = torch.rsqrt(mix.pow(2).mean(-1, keepdim=True) + eps)
    return mix * rms_o * out_w.float()


def _scratch(T):
    return (
        torch.empty(T, dtype=torch.float32, device="cuda"),
        torch.empty(T, dtype=torch.float32, device="cuda"),
        torch.empty(T, H, dtype=torch.float32, device="cuda"),
    )


@pytest.mark.parametrize(
    "T,KB,use_norm", [(1, 8, True), (1, 8, False), (1, 3, True), (4, 8, True)]
)
def test_partial_combine_parity(T, KB, use_norm):
    torch.manual_seed(T * 10 + KB)
    prefix = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
    blocks = torch.randn(KB, T, H, dtype=torch.bfloat16, device="cuda")
    wp = torch.randn(H, dtype=torch.bfloat16, device="cuda")
    out_w = (
        (torch.rand(H, dtype=torch.bfloat16, device="cuda") + 0.5) if use_norm else None
    )
    eps = 1e-5

    scratch = _scratch(T)
    out = torch.empty(T, H, dtype=torch.bfloat16, device="cuda")
    attnres_partial(blocks, wp, eps, scratch)
    attnres_combine(prefix, wp, out_w, eps, scratch, out)

    ref = _reference(prefix, blocks, wp, eps, out_w)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("T,KB", [(1, 8), (1, 3), (4, 8)])
def test_partial_dual_matches_two_singles(T, KB):
    """One dual sweep produces both sides' partials (fp32 tolerance: the
    interleaved schedule reorders fma contraction vs the single kernel)."""
    torch.manual_seed(T * 100 + KB)
    blocks = torch.randn(KB, T, H, dtype=torch.bfloat16, device="cuda")
    wp_a = torch.randn(H, dtype=torch.bfloat16, device="cuda")
    wp_b = torch.randn(H, dtype=torch.bfloat16, device="cuda")
    sa, sb, ra, rb = _scratch(T), _scratch(T), _scratch(T), _scratch(T)
    attnres_partial_dual(blocks, wp_a, wp_b, 1e-5, sa, sb)
    attnres_partial(blocks, wp_a, 1e-5, ra)
    attnres_partial(blocks, wp_b, 1e-5, rb)
    for got, ref in ((sa, ra), (sb, rb)):
        for x, y in zip(got, ref):
            torch.testing.assert_close(x, y, atol=1e-4, rtol=1e-4)
