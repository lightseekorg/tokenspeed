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

"""Beam's one-launch route and Q/K norm-RoPE as programmatic dependent launches:
each waits for a producer that releases its dependents first and publishes the
inputs late, and returns the bytes of its serial launch."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.layernorm.triton import fused_qk_rmsnorm_rope
from tokenspeed_kernel.ops.moe._triton.router_select import (
    triton_router_route_lanepar,
)
from tokenspeed_kernel.platform import current_platform, pdl_enabled

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform().is_hopper_plus,
    reason="PDL requires an NVIDIA SM90+ device",
)


@pytest.fixture(autouse=True)
def restore_pdl():
    previous = pdl_enabled()
    pdl_enabled(overwrite=False)
    yield
    pdl_enabled(overwrite=previous)


@triton.jit
def _publish_after_release(sources, targets, sizes: tl.constexpr, BLOCK: tl.constexpr):
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    start = tl.inline_asm_elementwise(
        "mov.u64 $0, %clock64;", "=l", [], dtype=tl.uint64, is_pure=False, pack=1
    )
    now = start
    while now - start < 200000:
        now = tl.inline_asm_elementwise(
            "mov.u64 $0, %clock64;", "=l", [], dtype=tl.uint64, is_pure=False, pack=1
        )
    for tensor in tl.static_range(len(sizes)):
        for base in range(
            tl.program_id(0) * BLOCK, sizes[tensor], tl.num_programs(0) * BLOCK
        ):
            offsets = base + tl.arange(0, BLOCK)
            values = tl.load(
                sources[tensor] + offsets, offsets < sizes[tensor], other=0
            )
            tl.store(targets[tensor] + offsets, values, offsets < sizes[tensor])


def _publish(sources, targets):
    _publish_after_release[(8,)](
        tuple(t.view(torch.uint8) for t in sources),
        tuple(t.view(torch.uint8) for t in targets),
        tuple(t.numel() * t.element_size() for t in sources),
        BLOCK=256,
        launch_pdl=True,
    )


def _unpublished(source: torch.Tensor) -> torch.Tensor:
    """NaN for floats; for ids a valid but wrong expert (0), so a kernel that read
    early would return other bytes without reading out of bounds."""
    if source.is_floating_point():
        return torch.full_like(source, float("nan"))
    return torch.zeros_like(source)


def _bits(value: torch.Tensor) -> torch.Tensor:
    return value.contiguous().view(torch.int32)


def _route_inputs(rows: int):
    generator = torch.Generator(device="cuda").manual_seed(rows)
    logits = torch.randn(rows, 256, generator=generator, device="cuda") * 0.6
    bias = torch.randn(256, generator=generator, device="cuda") * 0.1
    return logits, bias


@pytest.mark.parametrize("scale", (None, 2.5))
@pytest.mark.parametrize("renormalize", (True, False))
@pytest.mark.parametrize("rows", (1, 8, 33, 512))
def test_route_waits_for_published_logits(rows, renormalize, scale):
    logits, bias = _route_inputs(rows)
    args = (8, 2, 4, 8, renormalize, scale)
    serial = triton_router_route_lanepar(logits, bias, *args)
    pdl_enabled(overwrite=True)
    triton_router_route_lanepar(logits, bias, *args)  # compile first
    target_logits, target_bias = _unpublished(logits), _unpublished(bias)
    _publish((logits, bias), (target_logits, target_bias))
    weights, ids = triton_router_route_lanepar(target_logits, target_bias, *args)
    torch.cuda.synchronize()
    assert torch.equal(_bits(weights), _bits(serial[0]))
    assert torch.equal(ids, serial[1])


@pytest.mark.parametrize("tokens", (1, 7, 256))
def test_qk_rmsnorm_rope_waits_for_published_projection(tokens):
    """Grouped heads (4 query, 2 key-value, head dim 128); Q and K are strided views
    of the packed projection."""
    q_heads, kv_heads, head_dim = 4, 2, 128
    generator = torch.Generator(device="cuda").manual_seed(tokens)
    width = (q_heads + 2 * kv_heads) * head_dim
    qkv = torch.randn(tokens, width, generator=generator, device="cuda").bfloat16()
    q_weight, k_weight = (
        (torch.rand(head_dim, generator=generator, device="cuda") + 0.5).bfloat16()
        for _ in range(2)
    )
    positions = torch.randint(0, 4096, (tokens,), generator=generator, device="cuda")
    angles = torch.outer(
        torch.arange(4096, device="cuda", dtype=torch.float32),
        1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device="cuda") / head_dim)),
    )
    cos_sin = torch.cat([angles.cos(), angles.sin()], dim=-1).contiguous()

    def run(source):
        q = source[:, : q_heads * head_dim]
        k = source[:, q_heads * head_dim : (q_heads + kv_heads) * head_dim]
        heads = (q_heads, kv_heads, head_dim)
        return fused_qk_rmsnorm_rope(
            q, k, q_weight, k_weight, cos_sin, positions, 1e-6, *heads
        )

    serial = run(qkv)
    pdl_enabled(overwrite=True)
    run(qkv)  # compile first
    target = _unpublished(qkv)
    _publish((qkv,), (target,))
    pdl = run(target)
    torch.cuda.synchronize()
    for actual, expected in zip(pdl, serial, strict=True):
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
