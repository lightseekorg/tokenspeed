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
stages and the release run without peers.
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

_WORLD = 8
_LOCAL = 1024
_MAX = 16 * 1024
_EMPTY = -(2**15)  # BF16 negative zero: the empty half-word marker


@gluon.jit
def _reduce_kernel(
    region,
    epochs,
    out_ptr,
    row,
    WORLD_SIZE: gl.constexpr,
    LOCAL_WIDTH: gl.constexpr,
    MAX_ELEMENTS: gl.constexpr,
    RELEASE: gl.constexpr,
):
    stages = iris_muse_moetail_row_stages(epochs, row, LOCAL_WIDTH)
    values = iris_muse_moetail_reduce(
        region, stages, row, WORLD_SIZE, LOCAL_WIDTH, MAX_ELEMENTS, gl.num_warps(), ".cv"
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


def _reduce(region, epochs, row, release):
    out = torch.empty(_WORLD * _LOCAL, dtype=torch.float32, device="cuda")
    _reduce_kernel[(1,)](
        region,
        epochs,
        out,
        row,
        WORLD_SIZE=_WORLD,
        LOCAL_WIDTH=_LOCAL,
        MAX_ELEMENTS=_MAX,
        RELEASE=release,
        num_warps=16,
    )
    torch.cuda.synchronize()
    return out


@pytest.mark.parametrize("row", [0, 13])
def test_reduce_gathers_every_slot_and_release_clears_the_row(row: int) -> None:
    region, epochs, expected = _planted(row, 40 + row)
    before = region.clone()
    epochs_before = epochs.clone()
    out = _reduce(region, epochs, row, True)
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
