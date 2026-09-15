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

import sys
from types import ModuleType, SimpleNamespace

import torch
from tokenspeed_kernel.ops.moe.flashinfer import moe_tactic_sweep as sweep


def test_parser_preserves_mxfp_defaults_and_accepts_nvfp4() -> None:
    defaults = sweep._build_parser().parse_args([])
    assert defaults.quant_format == "mxfp"
    assert defaults.finalize_modes == "both"

    nvfp4 = sweep._build_parser().parse_args(["--quant-format", "nvfp4"])
    assert nvfp4.quant_format == "nvfp4"
    assert nvfp4.finalize_modes == "both"


def test_weight_scale_shapes_follow_quant_format() -> None:
    common = dict(local_experts=2, hidden=64, ispp=32, device="cpu", seed=1)
    mxfp = sweep._make_weights(**common, quant_format="mxfp")
    nvfp4 = sweep._make_weights(**common, quant_format="nvfp4")

    assert mxfp[0].shape == nvfp4[0].shape == (2, 64, 32)
    assert mxfp[2].shape == nvfp4[2].shape == (2, 64, 16)
    assert mxfp[1].shape == (2, 64, 2)
    assert mxfp[3].shape == (2, 64, 1)
    assert nvfp4[1].shape == (2, 64, 4)
    assert nvfp4[3].shape == (2, 64, 2)


def test_nvfp4_builds_required_per_expert_output_scales() -> None:
    assert sweep._make_output_scales("mxfp", 2, "cpu") == (None, None, None)

    scales = sweep._make_output_scales("nvfp4", 2, "cpu")
    assert all(scale is not None for scale in scales)
    assert all(scale.shape == (2,) for scale in scales if scale is not None)
    assert all(scale.tolist() == [1.0, 1.0] for scale in scales if scale is not None)


def test_nvfp4_tokens_use_fp4_quantize_and_exact_cache_key_shapes(
    monkeypatch,
) -> None:
    fake_flashinfer = ModuleType("flashinfer")
    calls = []

    def fp4_quantize(x, global_scale, **kwargs):
        calls.append((x.shape, global_scale, kwargs))
        return (
            torch.zeros((x.shape[0], x.shape[1] // 2), dtype=torch.uint8),
            torch.zeros((x.shape[0], x.shape[1] // 16), dtype=torch.uint8),
        )

    fake_flashinfer.fp4_quantize = fp4_quantize
    monkeypatch.setitem(sys.modules, "flashinfer", fake_flashinfer)

    x_q, x_scale, topk_ids, topk_weights = sweep._make_tokens(
        2,
        3584,
        896,
        16,
        torch.device("cpu"),
        seed=2,
        quant_format="nvfp4",
    )

    assert x_q.shape == (2, 1792)
    assert x_scale.shape == (2, 224)
    assert topk_ids.shape == topk_weights.shape == (2, 16)
    assert len(calls) == 1
    _, global_scale, kwargs = calls[0]
    assert global_scale.shape == ()
    assert global_scale.item() == 1.0
    assert kwargs == {"sf_vec_size": 16, "is_sf_swizzled_layout": False}


def test_mxfp_tokens_keep_legacy_quantizer_call(monkeypatch) -> None:
    fake_flashinfer = ModuleType("flashinfer")
    calls = []

    def mxfp8_quantize(x, use_pow2_scale, *, alignment):
        calls.append((x.shape, use_pow2_scale, alignment))
        return (
            torch.zeros(x.shape, dtype=torch.uint8),
            torch.zeros((x.shape[0], x.shape[1] // 32), dtype=torch.uint8),
        )

    fake_flashinfer.mxfp8_quantize = mxfp8_quantize
    monkeypatch.setitem(sys.modules, "flashinfer", fake_flashinfer)

    x_q, x_scale, _, _ = sweep._make_tokens(
        2, 64, 8, 2, torch.device("cpu"), seed=2, quant_format="mxfp"
    )

    assert x_q.shape == (2, 64)
    assert x_scale.shape == (2, 2)
    assert calls == [(torch.Size([2, 64]), False, 64)]


def _install_fake_tactic_modules(monkeypatch):
    calls = []
    dtypes = SimpleNamespace(MxE4m3=11, MxE2m1=12, E2m1=13)

    class FakeMoeOp:
        def trtllm_get_valid_moe_configs(self, *args):
            calls.append(args)
            return [(64, 4)]

    class FakeBuilder:
        @staticmethod
        def build_and_load():
            return FakeMoeOp()

    flashinfer = ModuleType("flashinfer")
    fused_moe = ModuleType("flashinfer.fused_moe")
    core = ModuleType("flashinfer.fused_moe.core")
    enums = ModuleType("flashinfer.tllm_enums")
    core.gen_trtllm_gen_fused_moe_sm100_module = lambda: FakeBuilder()
    enums.ActivationType = SimpleNamespace(Situ=SimpleNamespace(value=10))
    enums.DtypeTrtllmGen = dtypes
    enums.Fp8QuantizationType = SimpleNamespace(NoneFp8=20)
    enums.WeightLayout = SimpleNamespace(MajorK=SimpleNamespace(value=30))
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.fused_moe", fused_moe)
    monkeypatch.setitem(sys.modules, "flashinfer.fused_moe.core", core)
    monkeypatch.setitem(sys.modules, "flashinfer.tllm_enums", enums)
    return calls, dtypes


def test_candidate_tactics_use_format_specific_dtype_pair(monkeypatch) -> None:
    calls, dtypes = _install_fake_tactic_modules(monkeypatch)
    args = SimpleNamespace(
        quant_format="nvfp4",
        top_k=16,
        hidden_size=3584,
        intermediate_size=384,
        local_experts=896,
    )
    assert sweep._candidate_tactics(args) == [(64, 4)]
    assert len(calls) == 4
    assert all(call[:2] == (dtypes.E2m1, dtypes.E2m1) for call in calls)

    calls.clear()
    args.quant_format = "mxfp"
    assert sweep._candidate_tactics(args) == [(64, 4)]
    assert len(calls) == 4
    assert all(call[:2] == (dtypes.MxE4m3, dtypes.MxE2m1) for call in calls)
