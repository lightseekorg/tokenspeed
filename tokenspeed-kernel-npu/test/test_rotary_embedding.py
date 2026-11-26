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

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from tokenspeed_kernel_npu.ops.rotary_embedding import apply_rope

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Ascend RoPE tests require an NPU"
)


def _reference(
    x: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    rotary_dim: int,
    is_neox: bool,
) -> torch.Tensor:
    shape = x.shape
    heads = x.reshape(x.shape[0], -1, 128)
    rotary = heads[..., :rotary_dim]
    cos, sin = cos_sin_cache[positions].chunk(2, dim=-1)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    if is_neox:
        x1, x2 = rotary.chunk(2, dim=-1)
    else:
        x1, x2 = rotary[..., ::2], rotary[..., 1::2]
    first = x1 * cos - x2 * sin
    second = x2 * cos + x1 * sin
    rotated = (
        torch.cat((first, second), dim=-1)
        if is_neox
        else torch.stack((first, second), dim=-1).flatten(-2)
    )
    return torch.cat((rotated, heads[..., rotary_dim:]), dim=-1).reshape(shape)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rotary_dim", [64, 128])
@pytest.mark.parametrize("is_neox", [True, False])
@pytest.mark.parametrize("use_output_buffers", [True, False])
def test_apply_rope(
    dtype: torch.dtype,
    rotary_dim: int,
    is_neox: bool,
    use_output_buffers: bool,
) -> None:
    positions = torch.tensor([0, 7, 3, 15, 2, 9, 1], device="npu")
    frequencies = torch.randn(32, rotary_dim // 2, device="npu", dtype=dtype)
    cos_sin_cache = torch.cat((frequencies.cos(), frequencies.sin()), dim=-1)
    packed = torch.randn(7, (16 + 8) * 128, device="npu", dtype=dtype)
    q, k = packed.split((16 * 128, 8 * 128), dim=-1)
    expected_q = _reference(q, cos_sin_cache, positions, rotary_dim, is_neox)
    expected_k = _reference(k, cos_sin_cache, positions, rotary_dim, is_neox)
    q_before, k_before = q.clone(), k.clone()
    q_out = torch.empty_like(q) if use_output_buffers else None
    k_out = torch.empty_like(k) if use_output_buffers else None

    apply_rope(
        positions=positions,
        q=q,
        k=k,
        head_size=128,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
        q_rope_out=q_out,
        k_rope_out=k_out,
    )

    if use_output_buffers:
        torch.testing.assert_close(q, q_before, atol=0, rtol=0)
        torch.testing.assert_close(k, k_before, atol=0, rtol=0)
    torch.testing.assert_close(
        q if q_out is None else q_out, expected_q, atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        k if k_out is None else k_out, expected_k, atol=2e-2, rtol=2e-2
    )
