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

"""The gfx950 Gluon GQA prologue head writes the Triton prologue's bytes."""

import pytest
import torch
from tokenspeed_kernel._triton import gl, gluon
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("The Gluon prologue head targets gfx950", allow_module_level=True)

from tokenspeed_kernel.ops.attention.prologue.triton import (  # noqa: E402
    triton_gqa_prologue,
)
from tokenspeed_kernel.ops.attention.prologue.types import (  # noqa: E402
    FP8TensorScales,
    HeadKVCache,
    HeadNorm,
    RopeStyle,
    Rotary,
)
from tokenspeed_kernel_amd.ops.gfx950.attention.prologue.gqa import (  # noqa: E402
    gqa_prologue_head_gfx950,
)

_EPS = 1e-6


@gluon.jit
def _prologue_heads(
    src,
    q_out,
    q_w,
    k_w,
    cos_sin,
    positions,
    k_cache,
    v_cache,
    slots,
    k_inv,
    v_inv,
    num_rows,
    HALF_ROTARY: gl.constexpr,
    KV_SCALED: gl.constexpr,
):
    gqa_prologue_head_gfx950(
        src,
        2304,
        num_rows,
        gl.program_id(0),
        q_out,
        1024,
        q_w,
        k_w,
        cos_sin,
        2 * HALF_ROTARY if HALF_ROTARY > 0 else 1,
        positions,
        k_cache,
        v_cache,
        slots,
        k_inv,
        v_inv,
        128,
        64,
        128,
        64,
        16,
        2,
        64,
        HALF_ROTARY,
        1e-6,
        0.0,
        KV_SCALED,
        16,
        "",
    )


@pytest.mark.parametrize("rows", [1, 8, 16])
@pytest.mark.parametrize("rope", [True, False])
@pytest.mark.parametrize("scaled", [False, True])
def test_prologue_head_bytes_match_triton(rows, rope, scaled):
    generator = torch.Generator(device="cuda").manual_seed(rows * 4 + rope * 2 + scaled)
    # Rows of [16 query | 2 key | 2 value | 16 gate] heads of 64.
    qkvg = (torch.randn(rows, 2304, generator=generator, device="cuda") * 2).bfloat16()
    qkvg[0, 5] = float("nan")
    qkvg[-1, 1030] = -0.0
    positions = torch.randint(0, 1 << 18, (rows,), generator=generator, device="cuda")
    inv_freq = 1.0 / (10000 ** (torch.arange(0, 64, 2, device="cuda") / 64))
    angles = torch.arange(1 << 18, device="cuda")[:, None] * inv_freq[None, :]
    cos_sin = torch.cat([angles.cos(), angles.sin()], dim=-1).contiguous()
    q_weight = torch.full((64,), 0.125, device="cuda")
    k_weight = torch.ones(64, device="cuda")
    # Distinct slots across a page boundary; unwritten slots hold NaN bytes.
    slots = (torch.randperm(64, generator=generator, device="cuda")[:rows] + 40).long()
    caches = [
        tuple(
            torch.full((256, 2, 64), 0x7F, dtype=torch.uint8, device="cuda").view(
                torch.float8_e4m3fn
            )
            for _ in range(2)
        )
        for _ in range(2)
    ]
    scales = FP8TensorScales(0.75, 1.5) if scaled else None
    expected = triton_gqa_prologue(
        q=qkvg[:, :1024].view(rows, 16, 64),
        k=qkvg[:, 1024:1152],
        v=qkvg[:, 1152:1280],
        norm=HeadNorm(q_weight, k_weight, 0.0, _EPS),
        rotary=Rotary(cos_sin, positions, RopeStyle.GPTJ, None) if rope else None,
        cache=HeadKVCache(caches[0][0], caches[0][1], scales, slots),
        query_dtype=torch.float8_e4m3fn,
        return_kv=False,
        enable_pdl=False,
    )
    query = torch.empty(rows, 1024, dtype=torch.float8_e4m3fn, device="cuda")
    _prologue_heads[(18,)](
        qkvg,
        query,
        q_weight,
        k_weight,
        cos_sin,
        positions,
        caches[1][0],
        caches[1][1],
        slots,
        1 / 0.75 if scaled else 1.0,
        1 / 1.5 if scaled else 1.0,
        rows,
        HALF_ROTARY=32 if rope else 0,
        KV_SCALED=scaled,
        num_warps=8,
    )
    assert torch.equal(
        expected.q.reshape(rows, 1024).view(torch.uint8), query.view(torch.uint8)
    )
    for written, reference in zip(caches[1], caches[0]):
        assert torch.equal(written.view(torch.uint8), reference.view(torch.uint8))
