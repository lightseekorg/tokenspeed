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


"""Cooperative scale ownership and mailbox lifetime, including partial warps."""

import pytest
import torch
from flashinfer import fp4_quantize
from tokenspeed_kernel.platform import current_platform, pdl_enabled
from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.nvfp4_input_cooperative import (
    launch,
)

pytestmark = pytest.mark.skipif(
    not current_platform().is_nvidia or current_platform().arch_version.major != 10,
    reason="requires Blackwell",
)


def run(source, scale, values, mailbox, pdl, m, ctas):
    h = source.shape[1]
    payload = torch.full((m + 1, h // 2), 193, device="cuda", dtype=torch.uint8)
    sf = torch.full((m + 1, h // 16), 193, device="cuda", dtype=torch.uint8)
    launch(
        source,
        payload[:m],
        sf[:m],
        scale,
        hidden=h,
        m=m,
        values=values,
        ctas=ctas,
        threads=128,
        mailbox=mailbox,
        use_pdl=pdl,
    )
    return payload, sf


@pytest.mark.parametrize("values", [2, 4, 8])
@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("h", [64, 3584])
@pytest.mark.parametrize(
    "m", [1, 3, 4, 5, 7, 8, 9, 15, 17, 33, 64, 1279, 1280, 1281, 8192]
)
def test_plain_bytes(values, pdl, h, m):
    pdl_enabled(pdl)
    torch.manual_seed(3842)
    x = torch.randn((m, h), device="cuda", dtype=torch.bfloat16)
    for multiplier in (0.13721, 1.0, 128.0, 1433.6):
        scale = torch.tensor(multiplier, device="cuda", dtype=torch.float32)
        expected, esf = fp4_quantize(
            x, scale, is_sf_swizzled_layout=False, enable_pdl=pdl
        )
        data, sf = run(x, scale, values, False, pdl, m, 16)
        torch.testing.assert_close(data[:m], expected, rtol=0, atol=0)
        torch.testing.assert_close(sf[:m].flatten(), esf.flatten(), rtol=0, atol=0)
        assert bool((data[m] == 193).all() and (sf[m] == 193).all())


@pytest.mark.parametrize("values", [2, 4, 8])
@pytest.mark.parametrize("pdl", [False, True])
def test_special_values_and_multiplier_boundaries(values, pdl):
    pdl_enabled(pdl)
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
    ).repeat(4)
    x = pattern.repeat(17, 1)
    x[0].zero_()
    x[1] = pattern.abs().clamp(max=1e-30)
    x[2] = torch.nextafter(pattern, torch.full_like(pattern, float("inf")))
    x[3] = torch.nextafter(pattern, torch.full_like(pattern, -float("inf")))
    # Fixed amax makes the multiplier midpoints below exercise actual E4M3
    # scale rounding, independently of the saturation/tiny-value rows.
    x[4:12] = (
        torch.linspace(-1.375, 1.375, 16, device="cuda").to(torch.bfloat16).repeat(4)
    )
    multipliers = [0.13721, 1.0, 128.0, 1433.6]
    for midpoint in (1.0625, 2.125, 4.25, 8.5, 17.0, 34.0, 68.0, 136.0):
        center = torch.tensor(midpoint * 6 / 1.375, dtype=torch.float32)
        multipliers += [
            torch.nextafter(center, torch.tensor(-float("inf"))).item(),
            center.item(),
            torch.nextafter(center, torch.tensor(float("inf"))).item(),
        ]
    for mult in multipliers:
        scale = torch.tensor(mult, device="cuda", dtype=torch.float32)
        expected, esf = fp4_quantize(
            x, scale, is_sf_swizzled_layout=False, enable_pdl=pdl
        )
        data, sf = run(x, scale, values, False, pdl, 17, 16)
        torch.testing.assert_close(data[:17], expected, rtol=0, atol=0)
        torch.testing.assert_close(sf[:17].flatten(), esf.flatten(), rtol=0, atol=0)


@pytest.mark.parametrize("values", [2, 4, 8])
@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("ctas", [16, 128, 608])
def test_mailbox_replay_and_live_row_reset(values, pdl, ctas):
    pdl_enabled(pdl)
    torch.manual_seed(9524)
    x = torch.randn((5, 3584), device="cuda", dtype=torch.bfloat16)
    mailbox = x.clone()
    scale = torch.tensor(128.0, device="cuda", dtype=torch.float32)
    data, sf = run(mailbox, scale, values, True, pdl, 4, ctas)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mailbox.copy_(x)
        launch(
            mailbox,
            data[:4],
            sf[:4],
            scale,
            hidden=3584,
            m=4,
            values=values,
            ctas=ctas,
            threads=128,
            mailbox=True,
            use_pdl=pdl,
        )
    for sign in (1, -1):
        x[:4].mul_(sign)
        for _ in range(1000):
            graph.replay()
        torch.cuda.synchronize()
        expected, esf = fp4_quantize(
            x[:4], scale, is_sf_swizzled_layout=False, enable_pdl=pdl
        )
        torch.testing.assert_close(data[:4], expected, rtol=0, atol=0)
        torch.testing.assert_close(sf[:4].flatten(), esf.flatten(), rtol=0, atol=0)
        assert bool((mailbox[:4].view(torch.int32) == -2147450880).all())
        torch.testing.assert_close(mailbox[4], x[4], rtol=0, atol=0)
        assert bool((data[4] == 193).all() and (sf[4] == 193).all())
