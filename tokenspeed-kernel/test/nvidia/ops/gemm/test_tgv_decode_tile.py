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

"""Which pinned TGV tile a decode-sized BF16 GEMM on a small weight gets.

The policy and the tactic lookup are plain functions of their arguments, so
these tests need no GPU. The GPU tests next to this file, in
``test_bf16_tgv_decode_gemm.py``, check that ``mm`` follows them and that the
tiles compute the product.
"""

from __future__ import annotations

import pytest
from tokenspeed_kernel.ops.gemm import flashinfer as fi
from tokenspeed_kernel.platform import ArchVersion

SM90 = ArchVersion(9, 0)
SM100 = ArchVersion(10, 0)
SM103 = ArchVersion(10, 3)
SM107 = ArchVersion(10, 7)
TILE16, TILE32, TILE64 = fi.TGV_DECODE_TILES
MIB = 1 << 20

# FlashInfer 0.7.2.post1's TGV tactic table, the release requirements/cuda.txt
# pins: (cta_m, cta_n, num_ab_stage, use_2cta) at position = tactic id.
FLASHINFER_0_7_2_TACTICS = [
    *((64, 8, stages, False) for stages in (6, 8, 10, 12)),
    *((64, 16, stages, False) for stages in (6, 8, 11)),
    *((64, 32, stages, False) for stages in (6, 9)),
    (64, 64, 7, False),
    (64, 128, 4, False),
    (128, 8, 6, False),
    (128, 16, 6, False),
    (128, 32, 5, False),
    (128, 64, 4, False),
    (128, 128, 3, False),
    *((64, 16, stages, True) for stages in (6, 8, 12)),
    *((64, 32, stages, True) for stages in (6, 8, 11)),
    *((64, 64, stages, True) for stages in (6, 9)),
    (64, 128, 7, True),
    (128, 16, 6, True),
    (128, 32, 6, True),
    (128, 64, 5, True),
    (128, 128, 4, True),
]


def select(m, n, k=2560, *, arch=SM100, sm_count=148, pdl=True):
    return fi.select_tgv_decode_tile(
        m=m, n=n, weight_bytes=2 * n * k, arch=arch, sm_count=sm_count, pdl=pdl
    )


@pytest.mark.parametrize("arch", [SM100, SM103])
def test_rows_above_the_joint_bound_up_to_512_are_routed(arch):
    # A 512-row router weight: 8 CTAs of 64 rows, so every row count fits.
    assert select(fi.BF16_GEMM_MAX_M, 512, arch=arch) is None
    assert select(fi.BF16_GEMM_MAX_M + 1, 512, arch=arch) == TILE16
    assert select(fi.TGV_DECODE_MAX_M, 512, arch=arch) == TILE32
    assert select(fi.TGV_DECODE_MAX_M + 1, 512, arch=arch) is None


def test_weights_up_to_16_mib_are_routed():
    def tile(weight_bytes):
        return fi.select_tgv_decode_tile(
            m=64,
            n=1024,
            weight_bytes=weight_bytes,
            arch=SM100,
            sm_count=148,
            pdl=True,
        )

    assert fi.TGV_DECODE_MAX_WEIGHT_BYTES == 16 * MIB
    assert tile(16 * MIB) == TILE16
    assert tile(16 * MIB + 2) is None
    # A 2560 x 6144 attention output projection takes 30 MiB.
    assert select(160, 2560, 6144) is None


@pytest.mark.parametrize(
    "arch,routed", [(SM100, True), (SM103, True), (SM107, False), (SM90, False)]
)
def test_only_sm100_and_sm103_are_routed(arch, routed):
    assert (select(48, 512, arch=arch) is not None) is routed


@pytest.mark.parametrize("arch", [SM100, SM103])
@pytest.mark.parametrize(
    "m,n,k", [(33, 512, 2560), (64, 1024, 8192), (192, 2560, 2560), (512, 640, 2560)]
)
def test_without_pdl_no_tile_is_chosen(arch, m, n, k):
    """The tiles are pinned for the overlap PDL gives them; without it every
    GEMM, including ones routed under PDL, keeps cuBLASLt."""
    assert select(m, n, k, arch=arch) is not None
    assert select(m, n, k, arch=arch, pdl=False) is None


@pytest.mark.parametrize(
    "m,n,sm_count,expected",
    [
        # MTP fc projection, 2560 x 2560: 40 CTAs per activation-row tile.
        (48, 2560, 148, TILE16),
        (49, 2560, 148, TILE32),
        (96, 2560, 148, TILE32),
        (97, 2560, 148, TILE64),
        (192, 2560, 148, TILE64),
        (193, 2560, 148, None),
        # More SMs keep the smaller activation tile for more rows.
        (64, 2560, 148, TILE32),
        (64, 2560, 160, TILE16),
        # Index projection, 640 x 2560: 10 CTAs per activation-row tile.
        (224, 640, 148, TILE16),
        (225, 640, 148, TILE32),
        (448, 640, 148, TILE32),
        (449, 640, 148, TILE64),
        (512, 640, 148, TILE64),
        # 21 tiles of 64 weight rows launch 22 CTAs, as whole 2-CTA clusters:
        # 7 activation-row tiles need 154 CTAs, which 148 SMs cannot hold.
        (400, 1344, 148, None),
        (400, 1344, 160, TILE64),
    ],
)
def test_first_tile_whose_grid_fits_in_one_wave(m, n, sm_count, expected):
    assert select(m, n, sm_count=sm_count) == expected


@pytest.mark.parametrize("sm_count", [148, 160])
@pytest.mark.parametrize("n", [128, 512, 640, 960, 1344, 2560, 4096, 9600])
def test_tile_only_grows_with_rows(n, sm_count):
    """The choice is a step function of M over the three tiles, then None."""
    order = [*fi.TGV_DECODE_TILES, None]
    steps = [
        order.index(select(m, n, sm_count=sm_count))
        for m in range(fi.BF16_GEMM_MAX_M + 1, fi.TGV_DECODE_MAX_M + 1)
    ]
    assert steps == sorted(steps)


def test_tactics_are_found_by_shape():
    assert fi.resolve_tgv_tactics(FLASHINFER_0_7_2_TACTICS) == {
        TILE16: 18,
        TILE32: 21,
        TILE64: 23,
    }
    # A release that reorders or extends the table moves the ids.
    reordered = FLASHINFER_0_7_2_TACTICS[::-1]
    assert fi.resolve_tgv_tactics(reordered) == {
        TILE16: 28 - 18,
        TILE32: 28 - 21,
        TILE64: 28 - 23,
    }
    extended = [(64, 16, 4, True), (64, 256, 2, True), *FLASHINFER_0_7_2_TACTICS]
    assert fi.resolve_tgv_tactics(extended) == {TILE16: 20, TILE32: 23, TILE64: 25}


@pytest.mark.parametrize(
    "missing,replacement",
    [
        (TILE32, None),
        # Same tile shape with fewer stages: no longer the pinned tile.
        (TILE16, (64, 16, 10, True)),
        (TILE64, (64, 64, 9, False)),
    ],
)
def test_table_without_a_pinned_tile_resolves_to_nothing(missing, replacement):
    table = [
        replacement if entry == missing else entry for entry in FLASHINFER_0_7_2_TACTICS
    ]
    assert fi.resolve_tgv_tactics([entry for entry in table if entry]) is None
    assert fi.resolve_tgv_tactics([]) is None
