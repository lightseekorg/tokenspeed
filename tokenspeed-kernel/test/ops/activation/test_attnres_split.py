"""Parity tests for the split AttnRes kernels (partial + combine).

The split factors the online-softmax candidate mix into a blocks-side partial
(precomputable on the aux stream) and a prefix-side combine.
"""

import pytest
import torch
from tokenspeed_kernel.ops.activation.triton import (
    _attnres_partial_dual_kernel,
    _attnres_partial_kernel,
    attnres_combine,
    attnres_partial,
    attnres_partial_dual,
)
from tokenspeed_kernel.ops.residual.torch import torch_attn_res_fwd
from tokenspeed_kernel.platform import current_platform, pdl_enabled

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

H = 7168  # K3 hidden size; the kernels static-assert two 4096 sweeps.


@pytest.fixture(autouse=True)
def disable_pdl():
    previous = pdl_enabled()
    pdl_enabled(overwrite=False)
    yield
    pdl_enabled(overwrite=previous)


def _reference(prefix, blocks, wp, eps, out_w):
    """Direct softmax over the KB+1 candidate logits, fp32."""
    return torch_attn_res_fwd(
        layer_residual=prefix.clone(),
        block_residual=blocks,
        res_weight=wp,
        rms_weight=torch.ones_like(wp),
        eps=eps,
        out_norm_weight=out_w,
        out_norm_eps=eps,
    ).float()


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


@pytest.mark.parametrize("T,KB", [(1, 1), (1, 8), (4, 2), (32, 8)])
def test_partial_dual_probes_are_independent(T, KB):
    """Which slot a probe occupies must not change its result, bit for bit
    (the hoist re-pairs probe A with a different probe B)."""
    torch.manual_seed(T * 31 + KB)
    blocks = torch.randn(KB, T, H, dtype=torch.bfloat16, device="cuda") * 0.1
    wp_a = torch.randn(H, dtype=torch.bfloat16, device="cuda") * 0.05
    wp_b = torch.randn(H, dtype=torch.bfloat16, device="cuda") * 0.05
    ab_a, ab_b = _scratch(T), _scratch(T)
    ba_a, ba_b = _scratch(T), _scratch(T)
    attnres_partial_dual(blocks, wp_a, wp_b, 1e-5, ab_a, ab_b)
    attnres_partial_dual(blocks, wp_b, wp_a, 1e-5, ba_a, ba_b)
    for got, ref, side in ((ba_b, ab_a, "A->B"), (ba_a, ab_b, "B->A")):
        for x, y in zip(got, ref):
            assert torch.equal(x, y), f"probe {side} changed when slots swapped"


def _row_kernel_args(blocks, eps):
    return dict(
        n_blocks=blocks.shape[0],
        n_cols=H,
        stride_bk=blocks.stride(0),
        stride_bt=blocks.stride(1),
        eps=eps,
        BLOCK=4096,
        ENABLE_PDL=False,
        num_warps=8,
    )


@pytest.mark.skipif(not current_platform().is_amd, reason="64-lane waves only")
@pytest.mark.parametrize("T,KB", [(1, 1), (4, 6), (16, 11), (32, 3)])
def test_pair_tiles_match_row_kernels_bitwise(T, KB):
    """The AMD [2, 4096] tiles keep each half's summation tree."""
    torch.manual_seed(T * 7 + KB)
    blocks = torch.randn(KB, T, H, dtype=torch.bfloat16, device="cuda") * 2
    wp_a = torch.randn(H, dtype=torch.bfloat16, device="cuda") * 0.05
    wp_b = torch.randn(H, dtype=torch.bfloat16, device="cuda") * 0.05
    eps = 1e-6
    args = _row_kernel_args(blocks, eps)

    got, ref = _scratch(T), _scratch(T)
    attnres_partial(blocks, wp_a, eps, got)
    _attnres_partial_kernel[(T,)](blocks, wp_a, *ref, **args)
    got_a, got_b, ref_a, ref_b = _scratch(T), _scratch(T), _scratch(T), _scratch(T)
    attnres_partial_dual(blocks, wp_a, wp_b, eps, got_a, got_b)
    _attnres_partial_dual_kernel[(T,)](blocks, wp_a, wp_b, *ref_a, *ref_b, **args)
    for x, y in zip(got + got_a + got_b, ref + ref_a + ref_b):
        assert torch.equal(x.view(torch.int32), y.view(torch.int32))
