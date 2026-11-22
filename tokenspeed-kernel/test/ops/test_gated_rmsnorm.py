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

"""Gated RMSNorm: several rows per program against a float32 reference."""

from __future__ import annotations

from typing import Any

import pytest
import tokenspeed_kernel.ops.layernorm.triton as layernorm_triton
import torch
from tokenspeed_kernel.numerics.torch.layernorm import torch_gated_rmsnorm
from tokenspeed_kernel.ops.layernorm import gated_rmsnorm
from tokenspeed_kernel.ops.layernorm._triton.gated_rmsnorm import gated_rmsnorm_kernel
from tokenspeed_kernel.ops.layernorm.triton import GATED_RMSNORM_ROWS_PER_PROGRAM
from tokenspeed_kernel.ops.tuning import kernel_warmup
from tokenspeed_kernel.platform import current_platform
from utils import assert_warmup_covers

platform = current_platform()

pytestmark = pytest.mark.skipif(
    not (platform.is_nvidia or platform.is_amd),
    reason="Triton layernorm tests require an NVIDIA or AMD GPU.",
)

ROWS_PER_PROGRAM = GATED_RMSNORM_ROWS_PER_PROGRAM


# Row counts around the per-program row block, including partial last blocks.
@pytest.mark.parametrize(
    "rows", [1, ROWS_PER_PROGRAM - 1, ROWS_PER_PROGRAM + 1, 527, 8449]
)
@pytest.mark.parametrize("width,group_size", [(128, 128), (512, 128), (1024, 1024)])
@pytest.mark.parametrize("norm_before_gate", [True, False])
@pytest.mark.parametrize("sigmoid_gate", [False, True])
def test_gated_rmsnorm_matches_reference(
    rows, width, group_size, norm_before_gate, sigmoid_gate, device
):
    torch.manual_seed(rows + width)
    x = torch.randn(rows, width, device=device, dtype=torch.bfloat16)
    z = torch.randn(rows, width, device=device, dtype=torch.bfloat16)
    weight = torch.randn(width, device=device, dtype=torch.bfloat16)
    out = gated_rmsnorm(
        x,
        weight,
        z,
        eps=1e-6,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        sigmoid_gate=sigmoid_gate,
        fp8_scale=None,
        weights_independent=True,
    )
    expected = torch_gated_rmsnorm(
        x, weight, z, 1e-6, group_size, norm_before_gate, sigmoid_gate, None, True
    )
    # Two bf16 ulps: the reference rounds once, the kernel may differ in the last bit.
    torch.testing.assert_close(out.float(), expected.float(), rtol=2**-7, atol=2**-7)


def test_gated_rmsnorm_rows_are_independent_of_their_neighbours(device):
    """A row's output must not depend on which other rows share its program."""
    torch.manual_seed(0)
    x = torch.randn(4 * ROWS_PER_PROGRAM + 3, 128, device=device, dtype=torch.bfloat16)
    z = torch.randn_like(x)
    weight = torch.randn(128, device=device, dtype=torch.bfloat16)
    kwargs = dict(
        eps=1e-6,
        group_size=None,
        norm_before_gate=True,
        sigmoid_gate=False,
        fp8_scale=None,
        weights_independent=True,
    )
    full = gated_rmsnorm(x, weight, z, **kwargs)
    for row in (0, ROWS_PER_PROGRAM - 1, ROWS_PER_PROGRAM, x.shape[0] - 1):
        alone = gated_rmsnorm(x[row : row + 1], weight, z[row : row + 1], **kwargs)
        torch.testing.assert_close(full[row : row + 1], alone, rtol=0, atol=0)


@pytest.mark.parametrize("norm_before_gate", [True, False])
def test_gated_rmsnorm_row_blocks_match_one_row_programs(
    norm_before_gate: bool, device: str
) -> None:
    """Every row of a row block equals a one-row, one-warp program bit for bit."""
    torch.manual_seed(2)
    x = torch.randn(
        2048 * ROWS_PER_PROGRAM + 5, 128, device=device, dtype=torch.bfloat16
    )
    z = torch.randn_like(x)
    weight = torch.randn(128, device=device, dtype=torch.bfloat16)
    out = gated_rmsnorm(
        x,
        weight,
        z,
        eps=1e-6,
        group_size=None,
        norm_before_gate=norm_before_gate,
        sigmoid_gate=False,
        fp8_scale=None,
        weights_independent=True,
    )
    single = torch.empty_like(x)
    gated_rmsnorm_kernel[(x.shape[0], 1)](
        x,
        single,
        weight,
        z,
        None,
        x.stride(0),
        single.stride(0),
        z.stride(0),
        x.shape[0],
        128,
        1e-6,
        BLOCK_M=1,
        BLOCK_N=128,
        ROW_BLOCKS=1,
        NORM_BEFORE_GATE=norm_before_gate,
        SIGMOID_GATE=False,
        ENABLE_PDL=False,
        WEIGHTS_INDEPENDENT=True,
        OUT_FP8=False,
        SHARED_WEIGHT=False,
        num_warps=1,
    )
    torch.testing.assert_close(out, single, rtol=0, atol=0)


def _model_norm_inputs(
    rows: int, device: str, groups: int = 32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(rows)
    x = torch.randn(rows, groups * 128, device=device, generator=generator)
    # z is the gate slice of a wider projection row, as in the linear-attention block.
    z = torch.randn(rows, 3 * groups * 128, device=device, generator=generator)
    z = z[:, 2 * groups * 128 :]
    weight = 1.0 + 0.1 * torch.randn(128, device=device, generator=generator)
    return x.bfloat16(), z.bfloat16(), weight.bfloat16()


def _model_norm(x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return gated_rmsnorm(
        x,
        weight,
        z,
        eps=1e-6,
        group_size=128,
        norm_before_gate=True,
        sigmoid_gate=True,
        fp8_scale=None,
        weights_independent=True,
    )


@pytest.mark.parametrize("rows", [2048, 2049, 2111, 4097, 8449])
@pytest.mark.parametrize(
    "norm_before_gate,sigmoid_gate,fp8", [(True, True, False), (False, False, True)]
)
def test_gated_rmsnorm_row_loops_match_single_row_blocks(
    rows: int,
    norm_before_gate: bool,
    sigmoid_gate: bool,
    fp8: bool,
    device: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Looping over row blocks writes the same bits as one row block per program."""
    x, z, weight = _model_norm_inputs(rows, device)
    kwargs = dict(
        eps=1e-6,
        group_size=128,
        norm_before_gate=norm_before_gate,
        sigmoid_gate=sigmoid_gate,
        fp8_scale=torch.tensor(0.05, device=device) if fp8 else None,
        weights_independent=True,
    )
    monkeypatch.setattr(layernorm_triton, "GATED_RMSNORM_LOOP_MIN_PROGRAMS_PER_SM", 0)
    looped = gated_rmsnorm(x, weight, z, **kwargs)
    monkeypatch.setattr(
        layernorm_triton, "GATED_RMSNORM_LOOP_MIN_PROGRAMS_PER_SM", 2**31
    )
    single = gated_rmsnorm(x, weight, z, **kwargs)
    assert torch.equal(looped.view(torch.uint8), single.view(torch.uint8))


@pytest.mark.parametrize("groups", [1, 4, 32])
def test_gated_rmsnorm_loops_only_while_the_looped_grid_fills_the_gpu(
    groups: int, device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    launches: list[tuple[int, tuple[int, ...]]] = []
    launch = layernorm_triton.kernel_launcher

    def recording_launcher(kernel: Any, grid: tuple[int, ...]) -> Any:
        inner = launch(kernel, grid)

        def call(*args: Any, **kwargs: Any) -> Any:
            launches.append((kwargs["ROW_BLOCKS"], tuple(grid)))
            return inner(*args, **kwargs)

        return call

    monkeypatch.setattr(layernorm_triton, "kernel_launcher", recording_launcher)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    floor = layernorm_triton.GATED_RMSNORM_LOOP_MIN_PROGRAMS_PER_SM * sms
    blocks = layernorm_triton.GATED_RMSNORM_ROW_BLOCKS
    rows_per_loop = ROWS_PER_PROGRAM * blocks
    # The last row count whose looped grid stays below the floor, and the next one.
    below = ((floor - 1) // groups) * rows_per_loop
    for rows in (8, below, below + 1, 4 * below):
        _model_norm(*_model_norm_inputs(rows, device, groups))
    for row_blocks, grid in launches:
        programs = (grid[0] * row_blocks + blocks - 1) // blocks * grid[1]
        assert row_blocks == (blocks if programs >= floor else 1)
    assert [row_blocks for row_blocks, _ in launches] == [1, 1, blocks, blocks]


def test_gated_rmsnorm_warmup_covers_looping_and_single_block_launches(
    device: str,
) -> None:
    def run(rows: int) -> None:
        _model_norm(*_model_norm_inputs(rows, device))

    def warm() -> None:
        with kernel_warmup(enabled=True):
            run(4)

    # Decode and prefill row counts, both alignment classes of each.
    assert_warmup_covers(
        (gated_rmsnorm_kernel.fn,), warm, run, (1, 17, 512, 3001, 16384, 16389)
    )


@pytest.mark.parametrize("rows", [65, 1000, 4097])
def test_gated_rmsnorm_row_loops_leave_rows_past_the_last_one(
    rows: int, device: str
) -> None:
    """A partial last loop writes its valid rows and nothing after them."""
    x, z, weight = _model_norm_inputs(rows, device)
    blocks = layernorm_triton.GATED_RMSNORM_ROW_BLOCKS
    padded = rows + 3 * ROWS_PER_PROGRAM * blocks
    outputs = []
    for row_blocks in (1, blocks):
        out = torch.full((padded, x.shape[1]), 7.0, device=device, dtype=torch.bfloat16)
        rows_per_program = ROWS_PER_PROGRAM * row_blocks
        gated_rmsnorm_kernel[((rows + rows_per_program - 1) // rows_per_program, 32)](
            x,
            out,
            weight,
            z,
            None,
            x.stride(0),
            out.stride(0),
            z.stride(0),
            rows,
            128,
            1e-6,
            BLOCK_M=ROWS_PER_PROGRAM,
            BLOCK_N=128,
            ROW_BLOCKS=row_blocks,
            NORM_BEFORE_GATE=True,
            SIGMOID_GATE=True,
            ENABLE_PDL=False,
            WEIGHTS_INDEPENDENT=True,
            OUT_FP8=False,
            SHARED_WEIGHT=True,
            num_warps=ROWS_PER_PROGRAM,
        )
        outputs.append(out)
    single, looped = outputs
    assert torch.equal(looped[:rows], single[:rows])
    assert torch.equal(looped[rows:], torch.full_like(looped[rows:], 7.0))


def test_one_group_weight_on_whole_rows_matches_one_row_per_group(device):
    """A ``group_size`` weight shared by every group of whole rows, with z a
    row-strided view, normalizes exactly like one row per group."""
    torch.manual_seed(0)
    tokens, heads, dim = 37, 12, 128
    x = torch.randn(tokens, heads * dim, device=device, dtype=torch.bfloat16)
    wide = torch.randn(tokens, (heads + 3) * dim, device=device, dtype=torch.bfloat16)
    z = wide[:, 3 * dim :]
    weight = torch.randn(dim, device=device, dtype=torch.bfloat16)
    kwargs = dict(
        eps=1e-6,
        group_size=dim,
        norm_before_gate=True,
        sigmoid_gate=False,
        fp8_scale=None,
        weights_independent=True,
    )
    rows = gated_rmsnorm(x, weight, z, **kwargs)
    per_group = gated_rmsnorm(
        x.view(-1, dim), weight, z.contiguous().view(-1, dim), **kwargs
    )
    torch.testing.assert_close(rows.view(-1, dim), per_group, rtol=0, atol=0)


def test_gated_rmsnorm_fp8_output_matches_quantized_bf16(device):
    """The FP8 output equals quantizing the activation-dtype output with the same scale."""
    torch.manual_seed(1)
    x = torch.randn(531, 128, device=device, dtype=torch.bfloat16)
    z = torch.randn_like(x)
    weight = torch.randn(128, device=device, dtype=torch.bfloat16)
    scale = torch.tensor([0.05], device=device, dtype=torch.float32)
    kwargs = dict(
        eps=1e-6,
        group_size=None,
        norm_before_gate=True,
        sigmoid_gate=False,
        weights_independent=True,
    )
    bf16 = gated_rmsnorm(x, weight, z, fp8_scale=None, **kwargs)
    fp8 = gated_rmsnorm(x, weight, z, fp8_scale=scale, **kwargs)
    expected = (bf16.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    torch.testing.assert_close(fp8.float(), expected.float(), rtol=0, atol=0)
