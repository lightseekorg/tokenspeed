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

"""Finite Triton warmup must preserve ordinary launch behavior."""

import pytest
import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.tuning import kernel_warmup
from tokenspeed_kernel.warmup import kernel_launcher, register_triton_warmup
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton warmup requires CUDA or ROCm"
)


@triton.jit
def _add_kernel(x, out, n, bias, USE_BIAS: tl.constexpr, BLOCK: tl.constexpr):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(x + rows, rows < n, 0)
    if USE_BIAS:
        value += bias
    tl.store(out + rows, value, rows < n)


register_triton_warmup(
    _add_kernel,
    variants=({"n": (1, 16, 17), "bias": (1, 16, 17), "USE_BIAS": (False, True)},),
)


def test_warmup_preserves_positional_arguments_and_does_not_execute_variants():
    torch.cuda.synchronize()
    _add_kernel.device_caches.clear()
    # Spare storage makes an accidental launch of an unused variant safe,
    # while the untouched output suffix detects its extra writes.
    storage = torch.arange(64, device="cuda", dtype=torch.float32)
    out = torch.full_like(storage, -99.0)

    def run(n, bias, use_bias):
        kernel_launcher(_add_kernel, (triton.cdiv(n, 128),))(
            storage[:n], out, n, bias, use_bias, 128, num_warps=1
        )

    with kernel_warmup(enabled=True):
        run(1, 2, True)
    torch.testing.assert_close(out[:1], storage[:1] + 2)
    torch.testing.assert_close(out[1:], torch.full_like(out[1:], -99.0))
    with assert_no_triton_compile(_add_kernel):
        for n in (1, 3, 16, 17, 32, 33):
            for bias in (1, 7, 16, 17, 32):
                for use_bias in (False, True):
                    run(n, bias, use_bias)
                    torch.testing.assert_close(
                        out[:n], storage[:n] + (bias if use_bias else 0)
                    )


@triton.jit
def _affine_kernel(x, y, out, bias, WITH_BIAS: tl.constexpr):
    value = tl.load(x) * tl.load(y)
    if WITH_BIAS:
        value += bias
    tl.store(out, value)


register_triton_warmup(_affine_kernel, variants=({"WITH_BIAS": (False, True)},))


@pytest.mark.parametrize("warming", [False, True])
@pytest.mark.parametrize("fusion", [False, True])
def test_warmup_preserves_keyword_arguments_and_compiler_options(warming, fusion):
    torch.cuda.synchronize()
    _affine_kernel.device_caches.clear()
    # The fused result is -2**-46; separate FP32 multiply/add rounds to zero.
    x = torch.tensor([1 + 2**-23], dtype=torch.float32, device="cuda")
    y = torch.tensor([1 - 2**-23], dtype=torch.float32, device="cuda")
    out = torch.empty_like(x)

    def run(with_bias):
        kernel_launcher(_affine_kernel, (1,))(
            x,
            y,
            out,
            bias=-1.0,
            WITH_BIAS=with_bias,
            num_warps=1,
            enable_fp_fusion=fusion,
        )

    with kernel_warmup(enabled=warming):
        run(True)
    expected = (x.double() * y.double() - 1).float() if fusion else x * y - 1
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    if warming:
        # Both the nondefault warp count and the fusion option must reach
        # every prepared variant, as well as the actual requested launch.
        with assert_no_triton_compile(_affine_kernel):
            run(False)
        torch.testing.assert_close(out, x * y, rtol=0, atol=0)


@pytest.mark.parametrize("warming", [False, True])
def test_warmup_forwards_unsupported_launch_options(warming):
    x = torch.ones(1, device="cuda")
    out = torch.empty_like(x)
    with kernel_warmup(enabled=warming), pytest.raises((KeyError, TypeError)):
        kernel_launcher(_add_kernel, (1,))(x, out, 1, 2, True, 128, unexpected_option=1)
