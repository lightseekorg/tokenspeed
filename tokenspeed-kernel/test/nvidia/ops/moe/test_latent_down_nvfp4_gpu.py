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

"""Bitwise NVFP4 and mailbox lifecycle regression gates on Blackwell."""

import pytest
import torch
from tokenspeed_kernel.ops.moe.activation import Nvfp4Activation
from tokenspeed_kernel.platform import current_platform, pdl_enabled

pytestmark = pytest.mark.skipif(
    not current_platform().is_nvidia or current_platform().arch_version.major != 10,
    reason="requires Blackwell",
)


def _run(source, multiplier, mode, pdl):
    from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.nvfp4_input import launch

    m, h = source.shape
    data = torch.empty((m, h // 2), dtype=torch.uint8, device=source.device)
    scales = torch.empty(
        (((m + 15) // 16 * 16), h // 16), dtype=torch.uint8, device=source.device
    )[:m]
    signals = torch.empty((1,), dtype=torch.int64, device=source.device)
    pdl_enabled(pdl)
    launch(source, data, scales, multiplier, signals, hidden=h, m=m, mode=mode)
    return data, scales


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
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        1279,
        1280,
        1281,
        2048,
        4096,
        8192,
    ],
)
@pytest.mark.parametrize("multiplier", [0.125, 1.0, 128.0, 1792.0])
def test_quantization_is_byte_identical(m, multiplier):
    from flashinfer import fp4_quantize
    from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.nvfp4_input import PLAIN

    torch.manual_seed(73)
    source = torch.randn((m, 3584), device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor(multiplier, device="cuda", dtype=torch.float32)
    ref_data, ref_sf = fp4_quantize(source, scale, is_sf_swizzled_layout=False)
    data, sf = _run(source, scale, PLAIN, False)
    torch.testing.assert_close(data, ref_data, rtol=0, atol=0)
    torch.testing.assert_close(sf.flatten(), ref_sf.flatten(), rtol=0, atol=0)


@pytest.mark.parametrize("pdl", [False, True])
def test_quantization_zero_tiny_and_saturation(pdl):
    from flashinfer import fp4_quantize
    from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.nvfp4_input import PLAIN

    values = torch.tensor(
        [
            0.0,
            -0.0,
            1e-30,
            -1e-30,
            0.25,
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
            2.5,
            3.5,
            5.0,
            6.0,
            1e10,
            -1e10,
        ],
        device="cuda",
        dtype=torch.bfloat16,
    )
    source = values.repeat(4).reshape(1, 64).repeat(17, 1)
    # Include all-zero groups and subnormal scale-producing groups.
    source[0].zero_()
    source[1] = values.abs().clamp(max=1e-30).repeat(4)
    ties = torch.tensor(
        [
            0.0,
            -0.0,
            0.25,
            -0.25,
            0.75,
            -0.75,
            1.25,
            -1.25,
            1.75,
            -1.75,
            2.5,
            -2.5,
            3.5,
            -3.5,
            5.0,
            6.0,
        ],
        device="cuda",
        dtype=torch.bfloat16,
    ).repeat(4)
    source[2] = ties
    source[3] = torch.nextafter(ties, torch.full_like(ties, float("inf")))
    source[4] = torch.nextafter(ties, torch.full_like(ties, -float("inf")))
    scale = torch.tensor(128.0, device="cuda", dtype=torch.float32)
    expected, expected_sf = fp4_quantize(source, scale, is_sf_swizzled_layout=False)
    actual, actual_sf = _run(source, scale, PLAIN, pdl)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        actual_sf.flatten(), expected_sf.flatten(), rtol=0, atol=0
    )


@pytest.mark.parametrize("pdl", [False, True])
def test_quantization_arbitrary_encoding_multipliers(pdl):
    from flashinfer import fp4_quantize
    from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.nvfp4_input import PLAIN

    generator = torch.Generator().manual_seed(31847)
    multipliers = [0.13721, 3.1415927, 358.4, 1433.6]
    multipliers += torch.empty(32).uniform_(-3, 12, generator=generator).exp2().tolist()
    # Neighbours of scales that put an amax=1.375 group at an E4M3 midpoint.
    # Keep the multiplier arithmetic FP32, as in the processed expert weights.
    for midpoint in (1.0625, 2.125, 4.25, 8.5, 17.0, 34.0, 68.0, 136.0):
        center = torch.tensor(midpoint * 6 / 1.375, dtype=torch.float32)
        multipliers.extend(
            [
                torch.nextafter(center, torch.tensor(-float("inf"))).item(),
                center.item(),
                torch.nextafter(center, torch.tensor(float("inf"))).item(),
            ]
        )
    torch.manual_seed(31847)
    source = torch.randn((128, 3584), device="cuda", dtype=torch.bfloat16)
    source.clamp_(-1.375, 1.375)
    source[:, ::16] = 1.375
    for multiplier in multipliers:
        scale = torch.tensor(multiplier, device="cuda", dtype=torch.float32)
        expected, expected_sf = fp4_quantize(
            source, scale, is_sf_swizzled_layout=False, enable_pdl=pdl
        )
        actual, actual_sf = _run(source, scale, PLAIN, pdl)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            actual_sf.flatten(), expected_sf.flatten(), rtol=0, atol=0
        )


@pytest.mark.parametrize("pdl", [False, True])
def test_mailbox_replay_consumes_then_rearms(pdl):
    from flashinfer import fp4_quantize
    from tokenspeed_kernel.ops.moe.latent_down import arm_mailbox
    from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.nvfp4_input import (
        MAILBOX,
        launch,
    )

    m, h = 9, 3584
    source = torch.randn((m, h), device="cuda", dtype=torch.bfloat16)
    mailbox = torch.empty_like(source)
    scale = torch.tensor(128.0, device="cuda", dtype=torch.float32)
    mailbox.copy_(source)
    data, sf = _run(mailbox, scale, MAILBOX, pdl)
    signals = torch.empty((1,), dtype=torch.int64, device="cuda")
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mailbox.copy_(source)
        launch(mailbox, data, sf, scale, signals, hidden=h, m=m, mode=MAILBOX)
    for _ in range(1000):
        graph.replay()
    torch.cuda.synchronize()
    expected, expected_sf = fp4_quantize(source, scale, is_sf_swizzled_layout=False)
    torch.testing.assert_close(data, expected, rtol=0, atol=0)
    torch.testing.assert_close(sf.flatten(), expected_sf.flatten(), rtol=0, atol=0)
    empty = torch.empty_like(mailbox)
    arm_mailbox(empty)
    torch.testing.assert_close(
        mailbox.view(torch.int32), empty.view(torch.int32), rtol=0, atol=0
    )


def test_activation_rejects_wrong_receiver_and_packed_shape():
    multiplier = torch.ones((), device="cuda")
    data = torch.empty((2, 32), dtype=torch.uint8, device="cuda")
    scales = torch.empty((2, 4), dtype=torch.float8_e4m3fn, device="cuda")
    activation = Nvfp4Activation(data, scales, (2, 64), multiplier)
    assert activation.shape == (2, 64)
    activation.validate_receiver(multiplier)
    with pytest.raises(ValueError, match="different MoE"):
        activation.validate_receiver(multiplier.clone())
    with pytest.raises(ValueError, match="data"):
        Nvfp4Activation(data, scales, (2, 128), multiplier)


@pytest.mark.parametrize("mode", ["off", "mailbox", "all", "auto"])
def test_startup_mode_and_dispatch_boundaries(monkeypatch, mode):
    from tokenspeed_kernel.ops.moe.latent_down_nvfp4 import (
        KimiK3Nvfp4DownOp,
        fusion_mode,
    )

    monkeypatch.setenv("TOKENSPEED_K3_DOWN_NVFP4_FUSION", mode)
    assert fusion_mode() == mode
    op = object.__new__(KimiK3Nvfp4DownOp)
    op.mode, op.max_m = mode, 8192
    for m in (0, 1, 8, 9, 256, 257, 1023, 1024, 1280, 1281, 4095, 4096, 8192, 8193):
        expected = 1 <= m <= 8192
        if mode == "auto":
            expected = expected and m <= 1280
        assert op.handles(m) == expected
    # Unequal TP coefficients reduce prepared capacity to mailbox rows.
    op.mode, op.max_m = "auto", 1280
    assert not op.handles(4096)
    monkeypatch.setenv("TOKENSPEED_K3_DOWN_NVFP4_FUSION", "typo")
    with pytest.raises(ValueError):
        fusion_mode()


@pytest.mark.parametrize("unequal_scales", [False, True])
def test_auto_prepares_mailbox_only(monkeypatch, unequal_scales):
    from types import SimpleNamespace

    from tokenspeed_kernel.ops.moe import latent_down_nvfp4
    from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail import (
        nvfp4_input,
        nvfp4_input_cooperative,
    )

    op_type = latent_down_nvfp4.KimiK3Nvfp4DownOp
    group = SimpleNamespace(group_name="nvfp4_auto_policy_test")
    mailbox = SimpleNamespace(shard_dim=448, max_m=1280, _slot=object())
    scale = torch.ones((1,), dtype=torch.float32, device="cuda")
    allocations, compilations, cooperative = [], [], []
    for name in (
        "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH",
        "FLASHINFER_NVFP4_4OVER6",
    ):
        monkeypatch.delenv(name, raising=False)

    def gather(outputs, value, group):
        for rank, output in enumerate(outputs):
            output.copy_(value)
            if unequal_scales and value.dtype == torch.float32:
                output.mul_(rank + 1)

    def build_workspace(group, device, max_m, hidden, mode, ctas):
        allocations.append((max_m, hidden, mode))
        return object()

    monkeypatch.setattr(latent_down_nvfp4.dist, "get_world_size", lambda group: 8)
    monkeypatch.setattr(latent_down_nvfp4.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(latent_down_nvfp4.dist, "all_gather", gather)
    monkeypatch.setattr(latent_down_nvfp4.dist, "all_reduce", lambda *a, **k: None)
    monkeypatch.setattr(latent_down_nvfp4, "select_kernel", lambda *a, **k: None)
    monkeypatch.setattr(op_type, "_workspaces", {})
    monkeypatch.setattr(op_type, "_build_workspace", staticmethod(build_workspace))
    monkeypatch.setattr(
        nvfp4_input, "compile_kernel", lambda *args: compilations.append(args[1])
    )
    monkeypatch.setattr(
        nvfp4_input_cooperative,
        "compile_kernel",
        lambda *args: cooperative.append(args[1:5]),
    )
    op = op_type.initialize(mailbox, scale, group=group, max_m=8192, mode="auto")
    assert op is not None and op.mode == "auto" and op.max_m == 1280
    assert allocations == [(1280, 3584, "mailbox")]
    assert compilations == [nvfp4_input.PLAIN]
    assert cooperative == [
        (8, 256, 128, False),
        (2, 608, 128, True),
        (4, 608, 128, True),
        (8, 608, 128, True),
    ]
    assert op.handles(32) and op.handles(1024)
    assert not op.handles(4096) and not op.handles(8192)


def test_auto_three_band_geometry():
    from tokenspeed_kernel.ops.moe.latent_down_nvfp4 import _auto_geometry

    for m in range(1, 1281):
        fused, values, ctas, threads = _auto_geometry(m)
        assert fused == (8 < m <= 128)
        if m <= 8:
            assert (values, ctas, threads) == (8, 256, 128)
        elif m <= 128:
            assert values == (2 if m <= 32 else 4 if m <= 64 else 8)
            assert (ctas, threads) == (608, 128)
        else:
            assert (values, ctas, threads) == (16, 608, 256)
    for m in (0, 1281, 8192):
        with pytest.raises(ValueError, match="M=1..1280"):
            _auto_geometry(m)
