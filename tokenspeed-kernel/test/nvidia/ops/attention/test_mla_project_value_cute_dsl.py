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

"""CuTe DSL MLA value projection against the batched GEMM and ``sigmoid_mul``.

Serving-shaped inputs: the gate is a column view of the packed projection row,
``w_vc`` is the latent-contiguous view the absorbed decode weights use, and
row counts cover a partial 16-row tile. The projection sums in its own order,
so an output may differ from the batched GEMM's by one BF16 step.
"""

from __future__ import annotations

import pytest
import torch
from utils import assert_no_cute_compile, is_nvidia

if not is_nvidia():
    pytest.skip("requires an NVIDIA GPU", allow_module_level=True)
pytest.importorskip("cutlass")
from tokenspeed_kernel.platform import ArchVersion, current_platform

if not ArchVersion(10, 0) <= current_platform().arch_version <= ArchVersion(10, 3):
    pytest.skip("requires SM100 or SM103", allow_module_level=True)
from tokenspeed_kernel.ops.activation.triton import sigmoid_mul  # noqa: E402
from tokenspeed_kernel.ops.attention.mla import mla_project_value  # noqa: E402
from tokenspeed_kernel.ops.attention.mla.cute_dsl import (  # noqa: E402
    cute_dsl_mla_project_value,
)

LATENT, VALUE = 512, 128
# Columns of the Kimi-K3.1 TP8 packed DSA projection row ahead of and after the
# gate: 12 heads make its 3808-wide pitch.
GATE_OFFSET, GATE_TAIL = 1536 + 576, 160


def _inputs(rows, heads, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, scale=1.0):
        return torch.randn(*shape, device="cuda", generator=gen) * scale

    attention = randn(rows, heads, LATENT).bfloat16()
    # The absorbed weight is stored [heads, value, latent] and viewed transposed.
    w_vc = randn(heads, VALUE, LATENT, scale=LATENT**-0.5).bfloat16().transpose(1, 2)
    packed = randn(rows, GATE_OFFSET + heads * VALUE + GATE_TAIL, scale=2.0).bfloat16()
    gate = packed[:, GATE_OFFSET : GATE_OFFSET + heads * VALUE]
    return attention, w_vc, gate


def _reference(attention, w_vc, gate):
    rows, heads, _ = attention.shape
    out = torch.empty(rows, heads * VALUE, device="cuda", dtype=torch.bfloat16)
    torch.bmm(
        attention.transpose(0, 1),
        w_vc,
        out=out.view(rows, heads, VALUE).transpose(0, 1),
    )
    if gate is not None:
        sigmoid_mul(out, gate)
    return out


def _assert_close(got, want):
    """One BF16 step, plus a floor for dot products that cancel to near zero."""
    got, want = got.float(), want.float()
    floor = want.abs().max() * 2.0**-16
    assert ((got - want).abs() <= want.abs() * 2.0**-7 + floor).all()
    assert (got != want).float().mean() < 0.02


@pytest.mark.parametrize("rows", [1, 3, 8, 16, 17, 64])
@pytest.mark.parametrize("heads", [12, 16])
@pytest.mark.parametrize("gated", [True, False])
def test_value_projection_matches_the_batched_gemm(rows, heads, gated):
    attention, w_vc, gate = _inputs(rows, heads, seed=rows * heads + gated)
    gate = gate if gated else None
    out = torch.full((rows, heads * VALUE), 7.0, device="cuda", dtype=torch.bfloat16)
    cute_dsl_mla_project_value(attention, w_vc, gate, out)
    _assert_close(out, _reference(attention, w_vc, gate))


def test_the_op_selects_it_for_the_latent_contiguous_weight():
    attention, w_vc, gate = _inputs(5, 12, seed=3)
    got = mla_project_value(attention, w_vc, gate=gate)
    want = torch.empty_like(got)
    cute_dsl_mla_project_value(attention, w_vc, gate, want)
    assert torch.equal(got, want)
    # A value-contiguous weight keeps the batched GEMM.
    dense = w_vc.contiguous()
    torch.testing.assert_close(
        mla_project_value(attention, dense, gate=gate),
        _reference(attention, dense, gate),
        rtol=0,
        atol=0,
    )


def test_row_counts_reuse_one_compiled_kernel():
    for rows in (1, 9):
        attention, w_vc, gate = _inputs(rows, 12, seed=rows)
        cute_dsl_mla_project_value(
            attention,
            w_vc,
            gate,
            torch.empty(rows, 12 * VALUE, device="cuda").bfloat16(),
        )
    with assert_no_cute_compile():
        for rows in (2, 33, 64):
            attention, w_vc, gate = _inputs(rows, 12, seed=rows)
            out = torch.empty(rows, 12 * VALUE, device="cuda", dtype=torch.bfloat16)
            cute_dsl_mla_project_value(attention, w_vc, gate, out)
    torch.cuda.synchronize()


def test_unaligned_rows_are_rejected():
    attention, w_vc, gate = _inputs(4, 12, seed=5)
    shifted = torch.empty(attention.numel() + 1, device="cuda", dtype=torch.bfloat16)
    unaligned = shifted[1:].view_as(attention).copy_(attention)
    out = torch.empty(4, 12 * VALUE, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="cannot address"):
        cute_dsl_mla_project_value(unaligned, w_vc, gate, out)


def test_an_unaligned_weight_is_rejected():
    attention, w_vc, gate = _inputs(4, 12, seed=6)
    stored = w_vc.transpose(1, 2)
    shifted = torch.empty(stored.numel() + 1, device="cuda", dtype=torch.bfloat16)
    unaligned = shifted[1:].view_as(stored).copy_(stored).transpose(1, 2)
    out = torch.empty(4, 12 * VALUE, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="cannot address"):
        cute_dsl_mla_project_value(attention, unaligned, gate, out)


def test_rows_spanning_past_int32_offsets_are_rejected():
    """The kernel's row offsets are int32; a meta view spans 2**31 elements unallocated."""
    attention, w_vc, gate = _inputs(4, 12, seed=7)
    out = torch.empty_strided(
        (4, 12 * VALUE), (2**29, 1), device="meta", dtype=torch.bfloat16
    )
    with pytest.raises(ValueError, match="cannot address"):
        cute_dsl_mla_project_value(attention, w_vc, gate, out)
