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


"""Registry selection of the gfx950 MXFP4 MoE stream (no GPU needed): only
the expert-TP slice it tiles, on a device with one CU per workgroup."""

import dataclasses

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("The MXFP4 MoE stream registers on gfx950", allow_module_level=True)

from tokenspeed_kernel.ops.moe import prepare_moe_stream_decode  # noqa: E402
from tokenspeed_kernel.selection import (  # noqa: E402
    NoKernelFoundError,
    select_kernel,
)
from tokenspeed_kernel.signature import dense_format_signature  # noqa: E402

GEOMETRY = dict(
    weight_dtype="mxfp4", hidden=4096, ispp=512, num_experts=256, top_k=8, ep_size=1
)


@pytest.fixture
def mi355_platform(mi350_platform):
    return dataclasses.replace(mi350_platform, sm_count=256)


def _select(platform, **traits):
    return select_kernel(
        "moe",
        "stream_decode",
        dense_format_signature(("x",), (torch.uint8,)),
        platform=platform,
        traits=traits,
    )


def test_selects_the_stream_for_the_expert_tp_slice(mi355_platform):
    kernel = _select(mi355_platform, **GEOMETRY, num_cus=256)
    assert kernel.name == "gluon_mxfp4_moe_stream_gfx950"


@pytest.mark.parametrize(
    "change",
    [
        dict(ep_size=8),  # EP8 keeps the planned kernel
        dict(weight_dtype="nvfp4"),  # another format with the same geometry
        dict(num_cus=304),  # not every workgroup resident at once
    ],
    ids=lambda c: "-".join(f"{k}{v}" for k, v in c.items()),
)
def test_no_stream_for_other_layouts_or_devices(mi355_platform, change):
    traits = {**GEOMETRY, "num_cus": 256, **change}
    with pytest.raises(NoKernelFoundError):
        _select(mi355_platform, **traits)


def test_prepare_returns_none_without_a_kernel():
    # No kernel serves EP8 slices, so no state is created (no device needed).
    stream = prepare_moe_stream_decode(
        torch.device("cpu"), **{**GEOMETRY, "ep_size": 8}, num_cus=256
    )
    assert stream is None
