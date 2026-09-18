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


"""Ready-group NVFP4 bytes, graph replay, and live mailbox-row ownership."""

import pytest
import torch
from flashinfer import fp4_quantize
from tokenspeed_kernel.ops.moe.latent_down_nvfp4 import fusion_mode
from tokenspeed_kernel.platform import current_platform, pdl_enabled
from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.nvfp4_input_grouped import (
    compile_kernel,
    launch,
)

pytestmark = pytest.mark.skipif(
    not current_platform().is_nvidia or current_platform().arch_version.major != 10,
    reason="requires Blackwell",
)


@pytest.mark.parametrize(
    "configured,expected", [(None, "auto"), ("auto", "auto"), ("off", "off")]
)
def test_fusion_mode_default_and_override(monkeypatch, configured, expected):
    if configured is None:
        monkeypatch.delenv("TOKENSPEED_K3_DOWN_NVFP4_FUSION", raising=False)
    else:
        monkeypatch.setenv("TOKENSPEED_K3_DOWN_NVFP4_FUSION", configured)
    assert fusion_mode() == expected


def _fixture(m, hidden):
    torch.manual_seed(9421 + m)
    source = torch.randn((m + 1, hidden), device="cuda", dtype=torch.bfloat16)
    pattern = torch.tensor(
        [
            0.0,
            -0.0,
            1e-30,
            -1e-30,
            0.25,
            -0.25,
            0.75,
            -0.75,
            1.25,
            -1.25,
            2.5,
            -2.5,
            5.0,
            6.0,
            1e10,
            -1e10,
        ],
        device="cuda",
        dtype=torch.bfloat16,
    )
    source[0] = pattern.repeat(hidden // 16)
    # A valid mailbox producer never publishes the reserved negative-zero pair.
    source.view(torch.int32).masked_fill_(source.view(torch.int32) == -2147450880, 0)
    return source


def _check(mailbox, source, data, scales, multiplier, m, pdl):
    expected, expected_scales = fp4_quantize(
        source[:m], multiplier, is_sf_swizzled_layout=False, enable_pdl=pdl
    )
    torch.testing.assert_close(data[:m], expected, rtol=0, atol=0)
    torch.testing.assert_close(
        scales[:m].flatten(),
        expected_scales.view(torch.uint8).flatten(),
        rtol=0,
        atol=0,
    )
    assert bool((mailbox[:m].view(torch.int32) == -2147450880).all())
    torch.testing.assert_close(mailbox[m], source[m], rtol=0, atol=0)
    assert bool((data[m] == 193).all() and (scales[m] == 193).all())


@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("hidden", [64, 3584])
@pytest.mark.parametrize(
    "m",
    [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        32,
        33,
        64,
        65,
        73,
        74,
        95,
        96,
        97,
        111,
        127,
        128,
        129,
        1279,
        1280,
    ],
)
def test_grouped_bytes_and_live_row_reset(pdl, hidden, m):
    pdl_enabled(pdl)
    source = _fixture(m, hidden)
    mailbox = source.clone()
    data = torch.full((m + 1, hidden // 2), 193, device="cuda", dtype=torch.uint8)
    scales = torch.full((m + 1, hidden // 16), 193, device="cuda", dtype=torch.uint8)
    for value in (0.13721, 1.0, 128.0, 1433.6):
        multiplier = torch.tensor(value, device="cuda", dtype=torch.float32)
        mailbox.copy_(source)
        launch(
            mailbox,
            data[:m],
            scales[:m],
            multiplier,
            hidden=hidden,
            m=m,
            use_pdl=pdl,
        )
        _check(mailbox, source, data, scales, multiplier, m, pdl)


@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("m", [8, 9, 32, 64, 95, 96, 1280])
def test_grouped_graph_replay(pdl, m):
    pdl_enabled(pdl)
    source = _fixture(m, 3584)
    mailbox = source.clone()
    multiplier = torch.tensor(128.0, device="cuda", dtype=torch.float32)
    data = torch.full((m + 1, 1792), 193, device="cuda", dtype=torch.uint8)
    scales = torch.full((m + 1, 224), 193, device="cuda", dtype=torch.uint8)
    # Compiling once covers every M/grid width; capture must not compile again.
    compile_kernel(3584, mailbox.device.index, pdl)
    before = compile_kernel.cache_info().misses
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mailbox.copy_(source)
        launch(
            mailbox,
            data[:m],
            scales[:m],
            multiplier,
            hidden=3584,
            m=m,
            use_pdl=pdl,
        )
    assert compile_kernel.cache_info().misses == before
    for phase in range(2):
        if phase:
            # Change the next generation without allowing sentinel collisions.
            source.copy_(_fixture(m, 3584).roll(64, dims=1))
        for _ in range(1000):
            graph.replay()
        torch.cuda.synchronize()
        _check(mailbox, source, data, scales, multiplier, m, pdl)


@pytest.mark.parametrize("pdl", [False, True])
def test_grouped_scale_rounding_boundaries(pdl):
    pdl_enabled(pdl)
    m, hidden = 17, 64
    source = _fixture(m, hidden)
    source[1].zero_()
    source[2] = source[0].abs().clamp(max=1e-30)
    source[3] = torch.nextafter(source[0], torch.full_like(source[0], float("inf")))
    source[4] = torch.nextafter(source[0], torch.full_like(source[0], -float("inf")))
    source[5:13] = (
        torch.linspace(-1.375, 1.375, 16, device="cuda").to(torch.bfloat16).repeat(4)
    )
    mailbox = source.clone()
    data = torch.full((m + 1, hidden // 2), 193, device="cuda", dtype=torch.uint8)
    scales = torch.full((m + 1, hidden // 16), 193, device="cuda", dtype=torch.uint8)
    for midpoint in (1.0625, 2.125, 4.25, 8.5, 17.0, 34.0, 68.0, 136.0):
        center = torch.tensor(midpoint * 6 / 1.375, dtype=torch.float32)
        for value in (
            torch.nextafter(center, torch.tensor(-float("inf"))).item(),
            center.item(),
            torch.nextafter(center, torch.tensor(float("inf"))).item(),
        ):
            multiplier = torch.tensor(value, device="cuda", dtype=torch.float32)
            mailbox.copy_(source)
            launch(
                mailbox,
                data[:m],
                scales[:m],
                multiplier,
                hidden=hidden,
                m=m,
                use_pdl=pdl,
            )
            _check(mailbox, source, data, scales, multiplier, m, pdl)


@pytest.mark.parametrize("m", [0, 1281])
def test_grouped_rejects_unsupported_width(m):
    source = torch.empty((1, 3584), device="cuda", dtype=torch.bfloat16)
    data = torch.empty((1, 1792), device="cuda", dtype=torch.uint8)
    scales = torch.empty((1, 224), device="cuda", dtype=torch.uint8)
    multiplier = torch.ones((), device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="unsupported grouped NVFP4 geometry"):
        launch(source, data, scales, multiplier, hidden=3584, m=m, use_pdl=True)
