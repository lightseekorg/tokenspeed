"""Precondition guards for the CuTe DSL fused KDA decode entry.

These cover inputs that satisfy every documented precondition and still used
to fault inside the kernel, i.e. the shapes a caller can legally build but the
happy-path tests never construct.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention.cute_dsl.kda_fused_decode import (
    cutedsl_fused_recurrent_kda_megafuse,
)
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_nvidia, reason="CuTe DSL KDA decode is NVIDIA-only"
)

HV, K, V, D_FA = 4, 128, 128, 128  # kernel is specialised for K=V=128
P = HV * K


def _args(device: str, f_a: torch.Tensor, t: int = 1):
    g = torch.Generator(device="cpu").manual_seed(0)

    def rnd(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=g, dtype=torch.float32).to(
            device=device, dtype=dtype
        )

    pages = t + 1
    idx = torch.arange(1, pages, device=device, dtype=torch.int32)
    return dict(
        qkv_raw=rnd(t, 3 * P),
        conv_w=rnd(3 * P, 4).contiguous(),
        conv_pool=torch.zeros(pages, 3 * P, 3, dtype=torch.bfloat16, device=device),
        f_a=f_a,
        w_fb=rnd(P, D_FA).contiguous(),
        beta=rnd(t, HV),
        A_log=rnd(HV, dtype=torch.float32),
        dt_bias=rnd(P, dtype=torch.float32),
        h_pool=torch.zeros(pages, HV, K, K, dtype=torch.float32, device=device),
        read_indices=idx,
        write_indices=idx,
        num_heads=HV,
        head_dim=V,
    )


def test_rejects_f_a_at_an_odd_element_offset() -> None:
    """A column slice at an odd element offset must be rejected, not faulted.

    The kernel reads f_a as 8-byte bf16x4 vectors. Slicing a merged projection
    at an odd element offset keeps ``stride(0) % 4 == 0`` -- so the row-stride
    guard passes -- while putting the base pointer on a 2-byte boundary, which
    used to surface as a sticky cudaErrorMisalignedAddress that poisoned the
    rest of the process. Any odd local head count reaches this, because the
    natural gate offset in the merged projection is then odd.
    """
    device = "cuda"
    merged = torch.randn(1, D_FA + 8, dtype=torch.bfloat16, device=device)
    misaligned = merged[:, 1 : 1 + D_FA]
    assert misaligned.stride(-1) == 1
    assert misaligned.data_ptr() % 8 != 0, "slice should be 8B-misaligned"

    with pytest.raises(AssertionError, match="8B-aligned"):
        cutedsl_fused_recurrent_kda_megafuse(**_args(device, misaligned))


def test_accepts_f_a_at_an_aligned_offset() -> None:
    """The same slice at an aligned offset stays accepted."""
    device = "cuda"
    merged = torch.randn(1, D_FA + 8, dtype=torch.bfloat16, device=device)
    aligned = merged[:, 4 : 4 + D_FA]
    assert aligned.data_ptr() % 8 == 0
    out = cutedsl_fused_recurrent_kda_megafuse(**_args(device, aligned))
    assert out.shape == (1, HV, V)
    assert torch.isfinite(out.float()).all()
