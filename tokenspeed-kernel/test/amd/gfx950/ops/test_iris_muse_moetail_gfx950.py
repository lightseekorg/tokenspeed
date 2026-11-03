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

"""The Muse post-expert tail's reducer device functions on one GPU.

The 8-rank gather end to end is covered by test_iris_muse_spark.py's moetail
worker; here a mailbox is planted by hand so the slot layout, the per-tile
stages, the launch deadline and the release run without peers.
"""

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel.ops.communication._iris.muse_spark.moetail import (  # noqa: E402
    iris_muse_moetail_reduce,
    iris_muse_moetail_release,
    iris_muse_moetail_row_stages,
)
from tokenspeed_kernel_amd._triton import gl, gluon  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.persistent import (  # noqa: E402
    ERROR_ABORT,
    ERROR_RECORD,
    ERROR_WORDS,
    Launch,
)

_ABORT = gl.constexpr(ERROR_ABORT)

_SITE = gl.constexpr(91)
_RECORD = gl.constexpr(ERROR_RECORD)
_WORLD = 8
_LOCAL = 1024
_MAX = 16 * 1024
_EMPTY = -(2**15)  # BF16 negative zero: the empty half-word marker


@gluon.jit
def _record_expiry(launch, site, detail):
    # Test handler: record the site and return (kernel handlers never return).
    # Only the expired subgroups call it, and a scalar store would run on the
    # workgroup's first lane alone, so every lane of the caller stores.
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    lanes = gl.zeros([64 * gl.num_warps()], gl.int32, layout)
    gl.atomic_max(launch.error_ptr + _RECORD + lanes, lanes + site, sem="relaxed")


@gluon.jit
def _reduce_kernel(
    region,
    epochs,
    out_ptr,
    error_ptr,
    row,
    deadline_ticks,
    WORLD_SIZE: gl.constexpr,
    LOCAL_WIDTH: gl.constexpr,
    MAX_ELEMENTS: gl.constexpr,
    RELEASE: gl.constexpr,
):
    zero = gl.program_id(0) * 0
    launch = Launch(
        zero, zero, zero, zero.to(gl.int64), zero, zero, zero, error_ptr,
        error_ptr + _ABORT, deadline_ticks, error_ptr,
    )  # fmt: skip
    stages = iris_muse_moetail_row_stages(epochs, row, LOCAL_WIDTH)
    values = iris_muse_moetail_reduce(
        region, stages, row, launch, _record_expiry, _SITE, WORLD_SIZE, LOCAL_WIDTH,
        MAX_ELEMENTS, gl.num_warps(), ".cv"
    )  # fmt: skip
    columns = gl.arange(0, LOCAL_WIDTH * WORLD_SIZE, layout=values.type.layout)
    gl.amd.cdna4.buffer_store(values, out_ptr, columns)
    if RELEASE:
        iris_muse_moetail_release(
            region, epochs, stages, row, WORLD_SIZE, LOCAL_WIDTH, MAX_ELEMENTS,
            gl.num_warps(), ".wt"
        )  # fmt: skip


def _planted(row: int, seed: int):
    """A mailbox holding every slot's part of ``row`` with its two tiles on different stages."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    expected = torch.randn(_WORLD * _LOCAL, generator=generator, device="cuda")
    expected = expected.bfloat16()
    # Payloads never carry negative zero (senders send it as zero).
    expected[expected.view(torch.int16) == _EMPTY] = 0
    region = torch.full((3, _WORLD, _MAX), _EMPTY, dtype=torch.int16, device="cuda")
    epochs = torch.zeros(_MAX // 512, dtype=torch.int32, device="cuda")
    tiles = _LOCAL // 512
    epochs[row * tiles : (row + 1) * tiles] = torch.tensor([4, 2], dtype=torch.int32)
    for slot in range(_WORLD):
        for tile in range(tiles):
            stage = int(epochs[row * tiles + tile]) % 3
            start = row * _LOCAL + tile * 512
            part = expected[slot * _LOCAL + tile * 512 :][:512]
            region[stage, slot, start : start + 512] = part.view(torch.int16)
    return region, epochs, expected


def _reduce(region, epochs, row, deadline_ticks, release):
    out = torch.empty(_WORLD * _LOCAL, dtype=torch.float32, device="cuda")
    error = torch.zeros(ERROR_WORDS, dtype=torch.int32, device="cuda")
    _reduce_kernel[(1,)](
        region,
        epochs,
        out,
        error,
        row,
        deadline_ticks,
        WORLD_SIZE=_WORLD,
        LOCAL_WIDTH=_LOCAL,
        MAX_ELEMENTS=_MAX,
        RELEASE=release,
        num_warps=16,
    )
    torch.cuda.synchronize()
    return out, int(error[ERROR_RECORD].item())


@pytest.mark.parametrize("row", [0, 13])
def test_reduce_gathers_every_slot_and_release_clears_the_row(row: int) -> None:
    region, epochs, expected = _planted(row, 40 + row)
    before = region.clone()
    epochs_before = epochs.clone()
    out, site = _reduce(region, epochs, row, 200_000_000, True)
    assert site == 0
    assert torch.equal(out, expected.float())
    # The row's words now read empty: high halves 0x8000, low halves zero.
    cleared = before.clone()
    words = cleared.view(torch.int32)
    tiles = _LOCAL // 512
    for tile in range(tiles):
        stage = int(epochs_before[row * tiles + tile]) % 3
        start = (row * _LOCAL + tile * 512) // 2
        words[stage, :, start : start + 256] = -(2**31)
    assert torch.equal(region, cleared)
    advanced = epochs_before.clone()
    advanced[row * tiles : (row + 1) * tiles] = (
        epochs_before[row * tiles : (row + 1) * tiles] % 3 + 1
    ) % 3
    assert torch.equal(epochs, advanced)


def test_reduce_stops_at_the_deadline_when_a_word_is_missing() -> None:
    region, epochs, _ = _planted(3, 7)
    # Empty one word of slot 5's second tile.
    stage = int(epochs[3 * 2 + 1]) % 3
    region[stage, 5, 3 * _LOCAL + 512 + 64 : 3 * _LOCAL + 512 + 66] = _EMPTY
    _, site = _reduce(region, epochs, 3, 100_000, False)
    assert site == _SITE
