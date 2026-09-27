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

"""Activation PDL chains with producers that publish inputs after early release."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.activation.triton import (
    add3,
    attnres_combine,
    attnres_partial,
    attnres_partial_dual,
    fused_gate_sigmoid_mul_add,
    fused_swiglu_fp8_ue8m0,
    sigmoid_mul,
    silu_and_mul,
    situ_and_mul,
    swiglu_oai,
)
from tokenspeed_kernel.platform import current_platform, pdl_enabled

pytestmark = pytest.mark.skipif(
    not current_platform().is_hopper_plus, reason="PDL requires NVIDIA SM90+"
)

H = 7168


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


def _publish(sources: tuple[torch.Tensor, ...], targets: tuple[torch.Tensor, ...]):
    assert len(sources) == len(targets)
    _publish_after_release[(8,)](
        tuple(t.view(torch.uint8) for t in sources),
        tuple(t.view(torch.uint8) for t in targets),
        tuple(t.numel() * t.element_size() for t in sources),
        BLOCK=256,
        launch_pdl=True,
    )


def _unpublished(*sources: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return tuple(torch.full_like(t, float("nan")) for t in sources)


def _scratch(tokens: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty(tokens, dtype=torch.float32, device="cuda"),
        torch.empty(tokens, dtype=torch.float32, device="cuda"),
        torch.empty(tokens, H, dtype=torch.float32, device="cuda"),
    )


def test_sigmoid_mul_waits_for_published_inputs():
    x = torch.randn(17, 4096, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn_like(x)
    serial = sigmoid_mul(x.clone(), gate)
    target_x, target_gate = _unpublished(x, gate)

    _publish((x, gate), (target_x, target_gate))
    pdl_enabled(overwrite=True)
    pdl = sigmoid_mul(target_x, target_gate)
    assert torch.equal(pdl, serial)


@pytest.mark.parametrize("operation", ["silu", "swiglu", "situ"])
def test_gated_activation_waits_for_published_input(operation: str):
    widths = {"silu": 1024, "swiglu": 256, "situ": 2 * 3072}
    x = torch.randn(17, widths[operation], device="cuda", dtype=torch.bfloat16)
    if operation != "swiglu":
        x *= 8
    target = _unpublished(x)[0]

    def run(input_: torch.Tensor, enable_pdl: bool) -> torch.Tensor:
        pdl_enabled(overwrite=enable_pdl)
        if operation == "silu":
            return silu_and_mul(input_, limit=7.0)
        if operation == "swiglu":
            return swiglu_oai(input_, alpha=1.702, limit=7.0)
        return situ_and_mul(input_, beta=4.0, linear_beta=25.0)

    serial = run(x, False)
    _publish((x,), (target,))
    pdl = run(target, True)
    assert torch.equal(pdl, serial)


def test_fused_gate_sigmoid_mul_add_waits_for_published_inputs():
    hidden = torch.randn(17, 3584, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(3584, device="cuda", dtype=torch.bfloat16)
    shared = torch.randn_like(hidden)
    final = torch.randn_like(hidden)
    serial = fused_gate_sigmoid_mul_add(hidden, weight, shared, final.clone())
    target_hidden, target_weight, target_shared, target_final = _unpublished(
        hidden, weight, shared, final
    )

    _publish(
        (hidden, weight, shared, final),
        (target_hidden, target_weight, target_shared, target_final),
    )
    pdl_enabled(overwrite=True)
    pdl = fused_gate_sigmoid_mul_add(
        target_hidden, target_weight, target_shared, target_final
    )
    assert torch.equal(pdl, serial)


def test_add3_waits_for_published_inputs():
    a = torch.randn(17, 4096, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    c = torch.randn_like(a)
    serial = add3(a, b, c)
    targets = _unpublished(a, b, c)

    _publish((a, b, c), targets)
    pdl_enabled(overwrite=True)
    pdl = add3(*targets)
    assert torch.equal(pdl, serial)


def test_fused_swiglu_fp8_ue8m0_waits_for_published_input():
    gate_up = torch.randn(33, 1280, device="cuda", dtype=torch.bfloat16)
    serial_out, serial_scale = fused_swiglu_fp8_ue8m0(gate_up, enable_pdl=False)
    target = _unpublished(gate_up)[0]

    _publish((gate_up,), (target,))
    pdl_out, pdl_scale = fused_swiglu_fp8_ue8m0(target, enable_pdl=True)
    assert torch.equal(pdl_out, serial_out)
    assert torch.equal(pdl_scale, serial_scale)


@pytest.mark.parametrize("use_norm", [False, True])
def test_attnres_combine_waits_for_published_scratch(use_norm: bool):
    torch.manual_seed(7)
    tokens, candidates = 4, 8
    prefix = torch.randn(tokens, H, dtype=torch.bfloat16, device="cuda")
    blocks = torch.randn(candidates, tokens, H, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(H, dtype=torch.bfloat16, device="cuda")
    out_weight = (
        torch.rand(H, dtype=torch.bfloat16, device="cuda") + 0.5 if use_norm else None
    )
    source_scratch = _scratch(tokens)
    attnres_partial(blocks, weight, 1e-5, source_scratch)
    serial = attnres_combine(
        prefix,
        weight,
        out_weight,
        1e-5,
        source_scratch,
        torch.empty_like(prefix),
    )
    target_scratch = _unpublished(*source_scratch)
    out = torch.empty_like(prefix)

    _publish(source_scratch, target_scratch)
    pdl_enabled(overwrite=True)
    pdl = attnres_combine(
        prefix,
        weight,
        out_weight,
        1e-5,
        target_scratch,
        out,
    )
    assert torch.equal(pdl, serial)


@pytest.mark.parametrize("mode", ["single", "dual_a", "dual_b"])
def test_attnres_partial_to_combine_pdl_chain(mode: str):
    torch.manual_seed(11)
    tokens, candidates = 16, 32
    prefix = torch.randn(tokens, H, dtype=torch.bfloat16, device="cuda")
    blocks = torch.randn(candidates, tokens, H, dtype=torch.bfloat16, device="cuda")
    weight_a = torch.randn(H, dtype=torch.bfloat16, device="cuda")
    weight_b = torch.randn_like(weight_a)
    out = torch.empty_like(prefix)

    if mode == "single":
        serial_scratch = _scratch(tokens)
        pdl_scratch = _unpublished(*serial_scratch)
        attnres_partial(blocks, weight_a, 1e-5, serial_scratch)
        serial = attnres_combine(
            prefix,
            weight_a,
            None,
            1e-5,
            serial_scratch,
            torch.empty_like(prefix),
        )
        pdl_enabled(overwrite=True)
        attnres_partial(blocks, weight_a, 1e-5, pdl_scratch)
        pdl = attnres_combine(prefix, weight_a, None, 1e-5, pdl_scratch, out)
    else:
        serial_a, serial_b = _scratch(tokens), _scratch(tokens)
        pdl_a = _unpublished(*serial_a)
        pdl_b = _unpublished(*serial_b)
        attnres_partial_dual(blocks, weight_a, weight_b, 1e-5, serial_a, serial_b)
        weight, serial_scratch, pdl_scratch = (
            (weight_a, serial_a, pdl_a)
            if mode == "dual_a"
            else (weight_b, serial_b, pdl_b)
        )
        serial = attnres_combine(
            prefix,
            weight,
            None,
            1e-5,
            serial_scratch,
            torch.empty_like(prefix),
        )
        pdl_enabled(overwrite=True)
        attnres_partial_dual(blocks, weight_a, weight_b, 1e-5, pdl_a, pdl_b)
        pdl = attnres_combine(prefix, weight, None, 1e-5, pdl_scratch, out)

    assert torch.equal(pdl, serial)
