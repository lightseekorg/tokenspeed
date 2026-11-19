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
SM107_TILE16, SM107_TILE32, SM107_TILE64 = fi.TGV_DECODE_SM107_TILES
MIB = 1 << 20

# FlashInfer 0.7.3's TGV tactic table, the release requirements/cuda.txt pins:
# (cta_m, cta_n, num_ab_stage, use_2cta) at position = tactic id. Ids 0 to 28
# are those of 0.7.2.post1; 0.7.3 appends deeper-staged tiles that fit only
# SM107's larger shared memory and run only there.
FLASHINFER_0_7_3_TACTICS = [
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
    # SM107 only, from id 29.
    (64, 8, 18, False),
    (64, 16, 16, False),
    (64, 32, 13, False),
    (64, 64, 10, False),
    (64, 128, 6, False),
    (128, 8, 9, False),
    (128, 16, 9, False),
    (128, 32, 8, False),
    (128, 64, 6, False),
    (128, 128, 5, False),
    (64, 16, 18, True),
    (64, 32, 16, True),
    (64, 64, 13, True),
    (64, 128, 10, True),
    (128, 16, 9, True),
    (128, 32, 9, True),
    (128, 64, 8, True),
    (128, 128, 6, True),
]


def select(m, n, k=2560, *, arch=SM100, sm_count=148, pdl=True):
    return fi.select_tgv_decode_tile(
        m=m, n=n, k=k, weight_bytes=2 * n * k, arch=arch, sm_count=sm_count, pdl=pdl
    )


@pytest.mark.parametrize("arch", [SM100, SM103])
def test_rows_above_the_joint_bound_up_to_512_are_routed(arch):
    assert (fi.TGV_DECODE_MIN_M, fi.TGV_DECODE_MAX_M) == (32, 512)
    # A 512-row router weight: 8 CTAs of 64 rows, so every row count fits.
    assert select(fi.TGV_DECODE_MIN_M, 512, arch=arch) is None
    assert select(fi.TGV_DECODE_MIN_M + 1, 512, arch=arch) == TILE16
    assert select(fi.TGV_DECODE_MAX_M, 512, arch=arch) == TILE32
    assert select(fi.TGV_DECODE_MAX_M + 1, 512, arch=arch) is None


def test_weights_up_to_16_mib_are_routed():
    def tile(weight_bytes):
        return fi.select_tgv_decode_tile(
            m=64,
            n=4096,
            k=2048,
            weight_bytes=weight_bytes,
            arch=SM100,
            sm_count=148,
            pdl=True,
        )

    assert fi.TGV_DECODE_MAX_WEIGHT_BYTES == 16 * MIB
    # A 4096 x 2048 weight takes exactly 16 MiB.
    assert tile(16 * MIB) == TILE32
    assert tile(16 * MIB + 2) is None
    # 3072 x 3072 takes 18 MiB, though its K is in the window and its grid
    # fits in one wave at 48 rows.
    assert select(48, 3072, 3072) is None


@pytest.mark.parametrize("arch", [SM100, SM103])
def test_k_from_2048_to_3072_is_routed(arch):
    assert (fi.TGV_DECODE_MIN_K, fi.TGV_DECODE_MAX_K) == (2048, 3072)
    assert select(64, 512, 2047, arch=arch) is None
    assert select(64, 512, 2048, arch=arch) == TILE16
    assert select(64, 512, 3072, arch=arch) == TILE16
    assert select(64, 512, 3073, arch=arch) is None


@pytest.mark.parametrize("m", [36, 64, 128])
@pytest.mark.parametrize(
    "n,k",
    [
        # cuBLASLt time over TGV time, measured on GB300:
        (2560, 640),  # a shared expert's down projection, 0.90 to 0.96
        (160, 7168),  # an indexer projection, 0.73 to 0.91
        (3072, 512),  # a latent-attention KV up-projection shard, 0.87 to 0.95
    ],
)
def test_k_outside_the_window_is_not_routed(m, n, k):
    """Small weights that TGV ran slower than cuBLASLt. Only K keeps them
    off the route: at K = 2560 the same rows and weight rows get a tile."""
    assert select(m, n) is not None
    assert select(m, n, k) is None


@pytest.mark.parametrize(
    "arch,routed", [(SM100, True), (SM103, True), (SM107, True), (SM90, False)]
)
def test_only_sm100_sm103_and_sm107_are_routed(arch, routed):
    assert (select(48, 512, arch=arch) is not None) is routed


def sm107(m, n, k=2560, *, weight_bytes=None):
    return fi.select_tgv_decode_tile(
        m=m,
        n=n,
        k=k,
        weight_bytes=2 * n * k if weight_bytes is None else weight_bytes,
        arch=SM107,
        sm_count=212,
        pdl=True,
    )


def test_sm107_routes_k_from_2048_to_6144_and_weights_up_to_32_mib():
    assert sm107(64, 512, 2047) is None
    assert sm107(64, 512, 2048) == SM107_TILE16
    assert sm107(64, 512, 6144) == SM107_TILE16
    assert sm107(64, 512, 6145) is None
    # An attention output projection, 2560 x 6144, takes 30 MiB.
    assert sm107(64, 2560, 6144) == SM107_TILE32
    # 4096 x 4096 takes exactly 32 MiB; its grid fits only with 64-row tiles.
    assert sm107(33, 4096, 4096) == SM107_TILE64
    assert sm107(33, 4096, 4096, weight_bytes=32 * MIB + 2) is None
    assert select(33, 4096, 4096) is None


def test_sm107_routes_only_weights_of_at_least_512_rows():
    assert sm107(64, 511) is None
    assert sm107(64, 512) == SM107_TILE16
    assert sm107(64, 128, 6144) is None
    assert select(48, 128) is not None


@pytest.mark.parametrize(
    "m,n,expected",
    [
        # 2560 weight rows: 40 CTAs per activation-row tile; half of 212 SMs is 106.
        (33, 2560, SM107_TILE32),
        (64, 2560, SM107_TILE32),
        (65, 2560, SM107_TILE64),
        (128, 2560, SM107_TILE64),
        (129, 2560, None),
        # 512 weight rows: 8 CTAs per activation-row tile.
        (208, 512, SM107_TILE16),
        (209, 512, SM107_TILE32),
        (416, 512, SM107_TILE32),
        (417, 512, SM107_TILE64),
        (512, 512, SM107_TILE64),
    ],
)
def test_sm107_takes_the_first_tile_whose_grid_fills_half_the_sms(m, n, expected):
    assert sm107(m, n) == expected


@pytest.mark.parametrize("sm_count,routed", [(211, False), (212, True), (213, True)])
def test_sm107_grid_bound_rounds_half_the_sms_down(sm_count, routed):
    # 6784 weight rows launch 106 CTAs for one 64-row activation tile.
    tile = fi.select_tgv_decode_tile(
        m=64,
        n=6784,
        k=2048,
        weight_bytes=2 * 6784 * 2048,
        arch=SM107,
        sm_count=sm_count,
        pdl=True,
    )
    assert tile == (SM107_TILE64 if routed else None)


@pytest.mark.parametrize("arch", [SM100, SM103])
@pytest.mark.parametrize(
    "m,n,k", [(33, 512, 2560), (64, 4096, 2048), (192, 2560, 2560), (512, 640, 2560)]
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


@pytest.mark.parametrize("arch", [SM100, SM107])
@pytest.mark.parametrize("sm_count", [148, 160, 212])
@pytest.mark.parametrize("n", [128, 512, 640, 960, 1344, 2560, 4096, 9600])
def test_tile_only_grows_with_rows(n, sm_count, arch):
    """The choice is a step function of M over the three tiles, then None."""
    order = [*fi.TGV_DECODE_WINDOWS[arch].tiles, None]
    steps = [
        order.index(select(m, n, arch=arch, sm_count=sm_count))
        for m in range(fi.TGV_DECODE_MIN_M + 1, fi.TGV_DECODE_MAX_M + 1)
    ]
    assert steps == sorted(steps)


def test_tactics_are_found_by_shape():
    tiles = fi.TGV_DECODE_TILES
    assert fi.resolve_tgv_tactics(FLASHINFER_0_7_3_TACTICS, tiles) == {
        TILE16: 18,
        TILE32: 21,
        TILE64: 23,
    }
    assert fi.resolve_tgv_tactics(
        FLASHINFER_0_7_3_TACTICS, fi.TGV_DECODE_SM107_TILES
    ) == {SM107_TILE16: 39, SM107_TILE32: 40, SM107_TILE64: 41}
    # A release that reorders or extends the table moves the ids. Reversed,
    # the SM107 tiles of the pinned shapes, with more stages, come first.
    reordered = FLASHINFER_0_7_3_TACTICS[::-1]
    assert fi.resolve_tgv_tactics(reordered, tiles) == {
        TILE16: 46 - 18,
        TILE32: 46 - 21,
        TILE64: 46 - 23,
    }
    extended = [(64, 16, 4, True), (64, 256, 2, True), *FLASHINFER_0_7_3_TACTICS]
    assert fi.resolve_tgv_tactics(extended, tiles) == {
        TILE16: 20,
        TILE32: 23,
        TILE64: 25,
    }


@pytest.mark.parametrize(
    "missing,replacement",
    [
        # SM107's tile of the same shape does not stand in for it.
        (TILE32, None),
        # Same tile shape with fewer stages: no longer the pinned tile.
        (TILE16, (64, 16, 10, True)),
        (TILE64, (64, 64, 9, False)),
    ],
)
def test_table_without_a_pinned_tile_resolves_to_nothing(missing, replacement):
    table = [
        replacement if entry == missing else entry for entry in FLASHINFER_0_7_3_TACTICS
    ]
    tiles = fi.TGV_DECODE_TILES
    assert fi.resolve_tgv_tactics([entry for entry in table if entry], tiles) is None
    assert fi.resolve_tgv_tactics([], tiles) is None
