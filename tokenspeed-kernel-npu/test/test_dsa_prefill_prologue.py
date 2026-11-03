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

"""Native BF16 Prefill prologue preserves the original cache-write bytes."""

import pytest
import torch
from tokenspeed_kernel.ops.attention.prologue import (
    LatentKVCache,
    RopeStyle,
    Rotary,
    latent_store,
    mla_prologue,
)

pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Ascend NPU required"
)


@pytest.mark.parametrize("tokens", [1, 129])
@pytest.mark.parametrize(
    "style",
    [None]
    + [
        pytest.param(
            style,
            marks=pytest.mark.xfail(
                raises=RuntimeError,
                strict=True,
                reason="Existing store-enabled Ascend composite passes 3D RoPE input to the 2D-only CANN interface (161002)",
            ),
        )
        for style in (RopeStyle.GPTJ, RopeStyle.NEOX)
    ],
)
@pytest.mark.parametrize("sanitize", [False, True])
@torch.inference_mode()
def test_storeless_prefill_matches_original_write(tokens, style, sanitize):
    torch.npu.set_device(0)
    generator = torch.Generator().manual_seed(314 + tokens)
    query = (torch.randn(tokens, 8, 576, generator=generator) * 0.1).bfloat16().npu()
    latent = (torch.randn(tokens, 576, generator=generator) * 0.1).bfloat16().npu()
    if sanitize:
        latent[0, :3] = torch.tensor(
            [float("nan"), float("inf"), -float("inf")],
            device="npu",
            dtype=latent.dtype,
        )
    positions = torch.arange(2001, 2001 + tokens, device="npu")
    angle = torch.arange(4096, dtype=torch.float32)[:, None] / (
        10000 ** (torch.arange(32) / 32)
    )
    table = torch.cat((angle.cos(), angle.sin()), dim=-1).npu()
    rotary = None if style is None else Rotary(table, positions, style, None)
    slots = torch.arange(128, 128 + tokens, device="npu", dtype=torch.int64)
    mask = torch.arange(tokens, device="npu") % 3 != 1
    old_cache = torch.full((512, 1, 576), -3, dtype=latent.dtype, device="npu")
    new_cache = old_cache.clone()
    old_query, old_latent = query.clone(), latent.clone()
    original = mla_prologue(
        old_query,
        old_query[..., 512:],
        old_latent,
        expanded=None,
        rotary=rotary,
        cache=LatentKVCache(old_cache, sanitize, slots, mask),
        solution=None,
        override=None,
    )
    new_query = query.clone()
    prepared = mla_prologue(
        new_query,
        new_query[..., 512:],
        latent.clone(),
        expanded=None,
        rotary=rotary,
        cache=None,
        solution=None,
        override=None,
    )
    live = torch.nan_to_num(prepared.latent) if sanitize else prepared.latent
    latent_store(
        live, kv_lora_rank=512, cache=LatentKVCache(new_cache, sanitize, slots, mask)
    )
    torch.npu.synchronize()
    assert torch.equal(original.query, prepared.query)
    assert torch.equal(old_cache, new_cache)
    assert torch.equal(new_cache[slots[mask], 0], live[mask])
