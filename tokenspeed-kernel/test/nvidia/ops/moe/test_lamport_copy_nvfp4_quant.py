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


"""Lamport copy/NVFP4 bytes, graph replay, and live mailbox-row ownership."""

import pytest
import torch
from utils import is_nvidia

if not is_nvidia():
    pytest.skip("NVIDIA GPU required", allow_module_level=True)

from flashinfer import fp4_quantize  # noqa: E402
from tokenspeed_kernel.platform import current_platform, pdl_enabled  # noqa: E402
from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.lamport_copy_nvfp4_quant import (  # noqa: E402
    LamportCopyNvfp4QuantKernel,
    compile_kernel,
    launch,
)

pytestmark = pytest.mark.skipif(
    not current_platform().is_nvidia or current_platform().arch_version.major != 10,
    reason="requires Blackwell",
)


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
@pytest.mark.parametrize(
    "hidden,m",
    [
        (64, 1),
        (64, 16),
        (64, 17),
        (3584, 1),
        (3584, 8),
        (3584, 9),
        # 73/74 rows cross the 256-CTA minimum at width 3584.
        (3584, 73),
        (3584, 74),
        (3584, 1279),
        (3584, 1280),
    ],
)
def test_copy_bytes_and_live_row_reset(pdl, hidden, m):
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
def test_copy_scale_rounding_boundaries(pdl):
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


@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("m", [1, 9, 1280])
def test_gather_allocated_outputs_and_graph_replay(pdl, m):
    pdl_enabled(pdl)
    source = _fixture(m, 3584)
    mailbox = source.clone()
    multiplier = torch.tensor(128.0, device="cuda", dtype=torch.float32)
    gather = LamportCopyNvfp4QuantKernel(hidden_dim=3584, device=source.device)

    def gather_pair():
        outputs = []
        for scale in (multiplier, multiplier * 2):
            mailbox.copy_(source)
            outputs.append(gather(mailbox, scale, m=m))
        return outputs

    def check(outputs):
        assert outputs[0][0].data_ptr() != outputs[1][0].data_ptr()
        assert outputs[0][1].data_ptr() != outputs[1][1].data_ptr()
        for (data, scales), scale in zip(outputs, (multiplier, multiplier * 2)):
            expected, expected_scales = fp4_quantize(
                source[:m], scale, is_sf_swizzled_layout=False, enable_pdl=pdl
            )
            assert data.dtype == torch.uint8
            assert scales.dtype == torch.float8_e4m3fn
            torch.testing.assert_close(data, expected, rtol=0, atol=0)
            torch.testing.assert_close(
                scales.view(torch.uint8).flatten(),
                expected_scales.view(torch.uint8).flatten(),
                rtol=0,
                atol=0,
            )
        assert bool((mailbox[:m].view(torch.int32) == -2147450880).all())
        torch.testing.assert_close(mailbox[m], source[m], rtol=0, atol=0)

    check(gather_pair())
    before = compile_kernel.cache_info().misses
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = gather_pair()
    assert compile_kernel.cache_info().misses == before
    for _ in range(2):
        source.copy_(source.roll(64, dims=1))
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize()
        check(outputs)
