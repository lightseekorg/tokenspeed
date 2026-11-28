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

from __future__ import annotations

import torch
from tokenspeed_kernel.thirdparty.cute_dsl.ll_bf16 import (
    LLBf16Router,
    splitk_config_for,
    splitk_configs,
)


def test_splitk_configs_covers_every_m_threshold():
    configs = splitk_configs()
    assert len(configs) >= 2
    assert splitk_config_for(8) in configs
    assert splitk_config_for(16) in configs
    assert splitk_config_for(17) in configs
    assert splitk_config_for(32) in configs
    assert splitk_config_for(16) != splitk_config_for(17)


def test_warm_splitk_compiles_all_stage_variants_once(monkeypatch):
    router = LLBf16Router()
    compiled: list[tuple[tuple[int, int], int, torch.dtype, bool, bool]] = []

    def fake_compile_splitk(config, device, out_dtype, has_bias, enable_pdl):
        key = (device.index or 0, *config, out_dtype, has_bias, enable_pdl)
        router._splitk[key] = object()
        compiled.append((config, device.index or 0, out_dtype, has_bias, enable_pdl))

    monkeypatch.setattr(router, "is_available", lambda: True)
    monkeypatch.setattr(router, "_compile_splitk", fake_compile_splitk)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _dev: (10, 3))

    dev = torch.device("cuda", 0)
    router.warm_splitk(dev, out_dtype=torch.float32, has_bias=False, enable_pdl=True)
    expected = [(cfg, 0, torch.float32, False, True) for cfg in splitk_configs()]
    assert compiled == expected

    # Second call is a no-op once every threshold variant is cached.
    router.warm_splitk(dev, out_dtype=torch.float32, has_bias=False, enable_pdl=True)
    assert compiled == expected


def test_supports_splitk_k_rejects_narrow_projections():
    assert not LLBf16Router.supports_splitk_k(320)
    assert not LLBf16Router.supports_splitk_k(512)
    assert LLBf16Router.supports_splitk_k(1024)
    assert LLBf16Router.supports_splitk_k(7168)


def test_warm_splitk_skips_pre_hopper_and_cpu(monkeypatch):
    router = LLBf16Router()
    calls: list[object] = []
    monkeypatch.setattr(router, "is_available", lambda: True)
    monkeypatch.setattr(
        router, "_compile_splitk", lambda *args, **kwargs: calls.append(args)
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _dev: (8, 0))

    router.warm_splitk(
        torch.device("cpu"), out_dtype=torch.float32, has_bias=False, enable_pdl=False
    )
    router.warm_splitk(
        torch.device("cuda", 0),
        out_dtype=torch.float32,
        has_bias=False,
        enable_pdl=False,
    )
    assert calls == []


def test_warm_ll_bf16_splitk_is_noop_on_cpu_and_narrow_weights(monkeypatch):
    from tokenspeed_kernel.ops.gemm.cute_dsl import warm_ll_bf16_splitk
    from tokenspeed_kernel.thirdparty.cute_dsl.ll_bf16 import ll_bf16_router

    called = False

    def fail_if_called(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(ll_bf16_router, "warm_splitk", fail_if_called)
    x = torch.zeros((64, 7168), dtype=torch.bfloat16)
    w = torch.zeros((896, 7168), dtype=torch.bfloat16)
    warm_ll_bf16_splitk(x, w, out_dtype=torch.float32, has_bias=False, enable_pdl=False)
    assert not called


def test_router_call_sites_warm_splitk_under_kernel_warmup(monkeypatch):
    """Real call sites execute their is_warming_up() paths cleanly without GPU."""
    from tokenspeed_kernel.ops import gemm
    from tokenspeed_kernel.ops.moe import block_scaled
    from tokenspeed_kernel.ops.tuning import kernel_warmup
    from tokenspeed_kernel.platform import pdl_enabled

    expected_pdl = pdl_enabled()
    recorded: list[tuple[str, torch.dtype, bool, bool]] = []

    def make_recorder(tag: str):
        def _record(_x, _w, *, out_dtype, has_bias, enable_pdl):
            recorded.append((tag, out_dtype, has_bias, enable_pdl))

        return _record

    class _FakeCudaTensor(torch.Tensor):
        @property
        def is_cuda(self) -> bool:
            return True

    # 1. block_scaled_router_mm (uses real block_scaled.pdl_enabled import)
    monkeypatch.setattr(
        block_scaled, "warm_ll_bf16_splitk", make_recorder("block_scaled")
    )
    monkeypatch.setattr(block_scaled, "ll_bf16_router_supported", lambda *_a: False)
    monkeypatch.setattr(block_scaled, "skinny_mm_supported", lambda *_a: False)
    monkeypatch.setattr(
        block_scaled, "mm", lambda x, w, *, out_dtype: x.float() @ w.float().T
    )
    x_bs = torch.zeros((64, 1024), dtype=torch.bfloat16).as_subclass(_FakeCudaTensor)
    w_bs = torch.zeros((32, 1024), dtype=torch.bfloat16).as_subclass(_FakeCudaTensor)
    with kernel_warmup(enabled=True):
        block_scaled.block_scaled_router_mm(x_bs, w_bs, solution=None)

    # 2. dsv4_linear_fp32 (uses real gemm.pdl_enabled import; decline GPU kernel
    # dispatch so this test stays device-independent on GPU hosts too).
    monkeypatch.setattr(gemm, "warm_ll_bf16_splitk", make_recorder("dsv4"))

    def _no_gpu_kernel(*_args, **_kwargs):
        raise gemm.NoKernelFoundError("cpu test")

    monkeypatch.setattr(gemm, "select_kernel", _no_gpu_kernel)
    with kernel_warmup(enabled=True):
        gemm.dsv4_linear_fp32(
            torch.zeros((64, 1024), dtype=torch.bfloat16),
            torch.zeros((32, 1024), dtype=torch.bfloat16),
        )

    assert recorded == [
        ("block_scaled", torch.float32, False, expected_pdl),
        ("dsv4", torch.float32, False, expected_pdl),
    ]
