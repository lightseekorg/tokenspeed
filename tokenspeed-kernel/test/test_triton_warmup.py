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


def test_finish_write_selection_warmup_compiles_sparse_reindex_and_full_variants():
    from tokenspeed_kernel.ops.attention import dsv41
    from tokenspeed_kernel.ops.attention.dsv41.triton import (
        _finish_write_selection,
        finish_write_selection,
    )

    torch.cuda.synchronize()
    _finish_write_selection.device_caches.clear()
    device = "cuda"
    n, k, width = 8, 16, 96
    candidates = torch.randint(
        -1, 15, (n, width // 8), dtype=torch.int32, device=device
    )
    candidates[:, 0] = -1
    ends = torch.tensor([0, 1, 96, 70, 17, 40, 15, 3], dtype=torch.int32, device=device)
    indices = torch.randint(-1, width + 8, (n, k), dtype=torch.int32, device=device)
    scores_bf16 = torch.randn(n, width, dtype=torch.bfloat16, device=device)
    scores_bf16[:, ::7] = -torch.inf
    scores_fp32 = scores_bf16.float()

    def make_outputs(pool_k: int = 0):
        return (
            torch.full((n, k), -9, dtype=torch.int32, device=device),
            torch.full((n,), -9, dtype=torch.int32, device=device),
            torch.full((n, pool_k), -9, dtype=torch.int32, device=device),
            torch.full((n,), -9, dtype=torch.int32, device=device),
        )

    def make_slots(ratio: int):
        positions = torch.tensor([-1, 0, 200, 160, 639, 700, 255, 64], device=device)
        requests = torch.tensor([-1, 0, 0, 1, 2, 3, 1, 2], device=device)
        table = torch.tensor(
            [[1, 0, 5, 9, 2, 3], [2, 9, 4, 8, 1, 7], [7, 6, 1, 2, 0, 5]],
            dtype=torch.int32,
            device=device,
        )
        return dsv41.SelectionSlots(
            torch.full((n, k), -9, dtype=torch.int32, device=device),
            positions,
            requests,
            table,
            ratio,
            8,
        )

    # One BF16 sparse Reindex launch with slots warms both SHORT_ROWS_UNSCORED
    # and RATIO=(1, 2) variants (as well as SLOTS=False).
    with kernel_warmup(enabled=True):
        out_warm = make_outputs()
        slots_warm = make_slots(ratio=1)
        finish_write_selection(
            (indices, scores_bf16, ends, candidates, 100, 8),
            None,
            out_warm,
            slots_warm,
            False,
        )
        # Warm FP32 Full (POOL=False/True across _SELECTION_BLOCKS) and Dense Reindex.
        finish_write_selection(
            (indices, scores_fp32, ends, None, width, 1),
            None,
            make_outputs(),
            None,
            False,
        )

    with assert_no_triton_compile(_finish_write_selection):
        for short_rows_unscored in (False, True):
            for ratio in (1, 2):
                out_actual = make_outputs()
                slots_actual = make_slots(ratio=ratio)
                finish_write_selection(
                    (indices, scores_bf16, ends, candidates, 100, 8),
                    None,
                    out_actual,
                    slots_actual,
                    short_rows_unscored,
                )
                assert (out_actual[1] >= 0).all()
                assert (out_actual[3] == 0).all()
        for pool_k in (8, 32):
            block_picks = torch.randint(
                -1, 70, (n, pool_k), dtype=torch.int32, device=device
            )
            block_scores = torch.randn(n, 64, device=device)
            block_ends = torch.tensor(
                [0, 1, 7, 20, 33, 64, 64, 2], dtype=torch.int32, device=device
            )
            out_pool = make_outputs(pool_k=pool_k)
            finish_write_selection(
                (indices, scores_fp32, ends, None, width, 1),
                (block_picks, block_scores, block_ends),
                out_pool,
                None,
                False,
            )
            assert (out_pool[3] >= 0).all()
