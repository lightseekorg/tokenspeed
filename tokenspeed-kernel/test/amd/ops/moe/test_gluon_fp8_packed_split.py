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

"""Partition and numerical tests for packed split-K FP8 MoE.

CPU checks do not initialize the GPU. Use ``-k 'not gpu'`` for CPU-only work;
the GPU tests need an explicitly scheduled gfx950 execution window.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("tokenspeed_kernel_amd", reason="AMD kernel package is optional")


@pytest.mark.parametrize(
    ("parameters", "expected"),
    [
        ((1, 4096, 512, 8, 256, 288, 288), 8),
        ((2, 4096, 512, 8, 256, 288, 288), 4),
        ((3, 4096, 512, 8, 256, 288, 288), 2),
        ((4, 4096, 512, 8, 256, 288, 288), 2),
        ((5, 4096, 512, 8, 256, 288, 288), 1),
        ((1, 4096, 2048, 8, 256, 288, 288), 2),
        ((1, 4096, 512, 16, 256, 288, 288), 4),
        ((1, 1024, 512, 8, 256, 288, 288), 2),
        ((1, 512, 512, 8, 256, 288, 288), 1),
        ((0, 4096, 512, 8, 256, 288, 288), 1),
        ((1, 4096, 2048, 8, 256, 72, 288), 2),
        ((3, 4096, 2048, 8, 256, 72, 288), 1),
        ((1, 4096, 512, 8, 31, 1, 4), 1),
        ((1, 4096, 512, 8, 32, 1, 4), 2),
        ((2, 4096, 2048, 8, 256, 1, 2), 2),
        ((2, 4096, 2048, 8, 256, 17, 32), 1),
        ((2, 4096, 2048, 8, 256, 3, 7), 2),
        ((2, 4096, 2048, 8, 256, 4, 7), 1),
    ],
)
def test_split_count_work_budget_and_partition_boundaries(
    parameters: tuple[int, ...], expected: int
) -> None:
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp8.packed_small import (
        packed_fp8_split_count,
    )

    actual = packed_fp8_split_count(*parameters)
    assert actual == expected
    if actual > 1:
        rows, hidden, intermediate, topk, units, _, _ = parameters
        global_work = rows * topk * ((2 * intermediate + 127) // 128)
        assert actual in (2, 4, 8)
        assert global_work * actual <= (4 if actual == 2 else 2) * units
        assert hidden % (512 * actual) == 0


def test_split_count_does_not_increase_with_work() -> None:
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp8.packed_small import (
        packed_fp8_split_count,
    )

    for hidden in (512, 1024, 2048, 4096):
        factors = [
            max(1, packed_fp8_split_count(rows, hidden, 512, 8, 256, 288, 288))
            for rows in range(1, 33)
        ]
        assert factors == sorted(factors, reverse=True)


@pytest.fixture
def gfx950_only() -> None:
    from utils import is_cdna4

    if not is_cdna4():
        pytest.skip("requires AMD gfx950 GPU")


def _stage1_fixture(split_count: int, rounding: bool) -> tuple[torch.Tensor, ...]:
    hidden, intermediate = 4096, 256
    x = torch.zeros(2, hidden)
    w = torch.zeros(2, 2 * intermediate, hidden)
    sx = torch.ones(2, hidden // 128)
    sw = torch.ones(2, 2 * intermediate // 128, hidden // 128)
    length = hidden // split_count
    if rounding:
        # Each route yields [1+2^-11, -1, 0,...] before partition rounding.
        x[:, 0], x[:, length] = 1, -1
        w[:, :, 0], w[:, :, length] = 1, 1
        sx[:, 0] = 1.0 + 2.0**-11
    else:
        # Probe both sides of every K128 scale edge, including split edges.
        for group in range(hidden // 128):
            x[0, group * 128], x[0, group * 128 + 127] = 1, -2
            x[1, group * 128], x[1, group * 128 + 127] = 2, 1
            for expert in range(2):
                for nblock in range(4):
                    w[expert, nblock * 128 : (nblock + 1) * 128, group * 128] = (
                        group + nblock + expert
                    ) % 5 - 2
                    w[expert, nblock * 128 : (nblock + 1) * 128, group * 128 + 127] = (
                        group + nblock + expert + 1
                    ) % 5 - 2
                    sw[expert, nblock, group] = 2.0 ** (
                        (group + nblock + expert) % 5 - 4
                    )
            sx[:, group] = torch.tensor(
                [2.0 ** (group % 3 - 2), 2.0 ** (group % 4 - 3)]
            )
    return x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn), sx, sw


def _stage1_reference(
    x: torch.Tensor,
    w: torch.Tensor,
    sx: torch.Tensor,
    sw: torch.Tensor,
    split_count: int,
) -> torch.Tensor:
    # Routed slots 0 and 3 belong to experts 1 and 0 respectively. Double
    # arithmetic followed by float models FMA exactly for these finite fixtures.
    outputs, hidden = w.shape[1:]
    expected = torch.zeros(4, outputs)
    groups = hidden // (128 * split_count)
    for token, slot, expert in [(0, 0, 1), (1, 1, 0)]:
        for partition in range(split_count):
            acc = torch.zeros(outputs)
            for group in range(partition * groups, (partition + 1) * groups):
                k = slice(group * 128, (group + 1) * 128)
                tile = w[expert, :, k].float() @ x[token, k].float()
                scale = sx[token, group] * sw[expert, :, group].repeat_interleave(128)
                acc = (tile.double() * scale.double() + acc.double()).float()
            expected[token * 2 + slot] += acc.half().float()
    return expected


@pytest.mark.usefixtures("gfx950_only")
@pytest.mark.parametrize("split_count", [8, 4, 2])
@pytest.mark.parametrize("fixture_kind", ["boundaries", "rounding"])
def test_gpu_stage1_partition_rounding_and_scale_boundaries(
    split_count: int, fixture_kind: str
) -> None:
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp8.packed_small import (
        gluon_fp8_packed_small_stage1_gfx950,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp8.weight_preprocess import (
        pack_fp8_weight_n16k32,
    )

    x, w, sx, sw = _stage1_fixture(split_count, fixture_kind == "rounding")
    expected = _stage1_reference(x, w, sx, sw, split_count)
    assert torch.isfinite(expected).all()
    assert torch.count_nonzero(expected[1:3]) == 0
    if fixture_kind == "rounding":
        assert torch.count_nonzero(expected) == 0
    else:
        assert torch.count_nonzero(expected[[0, 3]]) > 0
    ids = torch.full((48,), 2, dtype=torch.int32, device="cuda")
    ids[0], ids[16] = (1 << 24) | 1, 0
    experts = torch.tensor([0, 1, -1], dtype=torch.int32, device="cuda")
    valid = torch.tensor([32], dtype=torch.int32, device="cuda")

    packed = pack_fp8_weight_n16k32(w).cuda()
    x, sx, sw = x.cuda(), sx.cuda(), sw.cuda()
    actual = torch.zeros_like(expected, device="cuda")
    for _ in range(2):
        actual.zero_()
        gluon_fp8_packed_small_stage1_gfx950[(4, 3, split_count)](
            x.view(torch.uint8),
            sx,
            packed.view(torch.uint8),
            sw,
            actual,
            sx,  # Scale output is unused by split partial producers.
            ids,
            experts,
            valid,
            2,
            48,
            0.0,
            K=4096,
            I=256,
            TOPK=2,
            SPLIT_COUNT=split_count,
            USE_RECIPROCAL=False,
            num_warps=4,
        )
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        actual.fill_(19.0)
