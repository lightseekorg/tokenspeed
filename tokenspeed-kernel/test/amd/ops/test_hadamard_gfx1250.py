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

"""GFX1250 length-128 Hadamard coverage."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.transform import hadamard_transform
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from utils import assert_no_triton_compile, compiled_kernels, is_cdna5

if not is_cdna5():
    pytest.skip("AMD CDNA5 is required", allow_module_level=True)

from tokenspeed_kernel_amd.ops.gfx1250.transform.hadamard import (  # noqa: E402
    gluon_hadamard_transform_128_gfx1250,
)


def _sylvester(device: torch.device) -> torch.Tensor:
    hadamard = torch.ones(1, 1, device=device)
    for _ in range(7):
        hadamard = torch.cat(
            (
                torch.cat((hadamard, hadamard), 1),
                torch.cat((hadamard, -hadamard), 1),
            )
        )
    return hadamard


def _assert_matches(x: torch.Tensor, scale: float, *, atol: float, rtol: float) -> None:
    out = hadamard_transform(x, scale=scale)
    expected = (x.float().reshape(-1, 128) @ _sylvester(x.device)) * scale
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    torch.testing.assert_close(
        out.float(), expected.reshape_as(x), atol=atol, rtol=rtol
    )


def test_gfx1250_hadamard_selects_gluon_for_last_dim_128() -> None:
    selected = select_kernel(
        "transform",
        "hadamard_transform",
        format_signature(x=dense_tensor_format(torch.bfloat16)),
        traits={"last_dim": 128},
    )
    portable = select_kernel(
        "transform",
        "hadamard_transform",
        format_signature(x=dense_tensor_format(torch.bfloat16)),
        traits={"last_dim": 128},
        solution="triton",
    )
    assert selected.name == "gluon_hadamard_transform_128_gfx1250"
    assert portable.name == "triton_hadamard_transform_128"


@pytest.mark.parametrize(
    "dtype,atol,rtol",
    [
        (torch.bfloat16, 2e-2, 2e-2),
        (torch.float16, 1e-3, 1e-3),
        (torch.float32, 1e-4, 1e-4),
    ],
)
def test_gfx1250_hadamard_matches_sylvester(
    dtype: torch.dtype, atol: float, rtol: float
) -> None:
    scale = 128**-0.5
    generator = torch.Generator(device="cuda").manual_seed(7)
    rows = torch.randn(300, 128, device="cuda", generator=generator).to(dtype)
    _assert_matches(rows, scale, atol=atol, rtol=rtol)
    small = torch.randn(3, 5, 128, device="cuda", generator=generator).to(dtype)
    _assert_matches(small, scale, atol=atol, rtol=rtol)


def test_gfx1250_hadamard_row_count_shares_one_binary() -> None:
    scale = 128**-0.5
    hadamard = _sylvester(torch.device("cuda"))

    def run(rows: int) -> None:
        x = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
        out = hadamard_transform(x, scale=scale)
        torch.testing.assert_close(
            out.float(), x.float() @ hadamard * scale, atol=2e-2, rtol=2e-2
        )

    kernel = gluon_hadamard_transform_128_gfx1250
    kernel.device_caches.clear()
    run(256)
    assert len(compiled_kernels(kernel)) == 1
    with assert_no_triton_compile(kernel):
        run(4099)
    assert len(compiled_kernels(kernel)) == 1


def test_solution_triton_stays_on_the_portable_kernel() -> None:
    x = torch.randn(300, 128, device="cuda", dtype=torch.bfloat16)
    scale = 0.125
    out = hadamard_transform(x, scale=scale, solution="triton")
    torch.testing.assert_close(
        out.float(), x.float() @ _sylvester(x.device) * scale, atol=2e-2, rtol=2e-2
    )
