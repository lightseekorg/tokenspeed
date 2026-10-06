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

"""Qwen3.5 layers pick the fused norm only where it can hand its copy to the next projection."""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.layernorm import GemmaRMSNorm
from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config
from tokenspeed.runtime.layers.quantization.nvfp4 import Nvfp4Config
from tokenspeed.runtime.models.qwen3_5 import _input_norm, _post_attn_norm
from tokenspeed.runtime.models.qwen3_5_moe import Qwen3_5MoeMLP

register_cuda_ci(
    est_time=30,
    suite="runtime-1gpu",
    disabled_on_runners=["amd-*"],
    disabled_on_runners_reason="the quantized copies are produced on NVIDIA only",
)

_IS_BLACKWELL = current_platform().is_blackwell
_HIDDEN = 5120


def _norm() -> GemmaRMSNorm:
    norm = GemmaRMSNorm(_HIDDEN, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
    norm.weight.data.normal_(0, 0.3)
    return norm


def _comm_manager(norm, calls, *, attn_tp, unfused, dense_tp):
    def input_reduce_norm(hidden_states, residual):
        calls.append("input_reduce_norm")
        if residual is None:
            return norm(hidden_states), hidden_states
        return norm(hidden_states, residual)

    def post_attn_reduce_norm(hidden_states, residual, ctx):
        calls.append("post_attn_reduce_norm")
        return norm(hidden_states, residual)

    return SimpleNamespace(
        mapping=SimpleNamespace(
            has_attn_tp=attn_tp, dense=SimpleNamespace(has_tp=dense_tp)
        ),
        layer_boundary_norm="unfused" if unfused else "fused",
        input_reduce_norm=input_reduce_norm,
        post_attn_reduce_norm=post_attn_reduce_norm,
    )


@pytest.mark.parametrize("attn_tp", [False, True])
@pytest.mark.parametrize("unfused", [False, True])
@pytest.mark.parametrize("dense_tp", [False, True])
@pytest.mark.parametrize("first_layer", [False, True])
@pytest.mark.parametrize("with_scales", [False, True])
def test_layer_norms_pick_the_fused_copy(
    attn_tp: bool, unfused: bool, dense_tp: bool, first_layer: bool, with_scales: bool
) -> None:
    torch.manual_seed(0)
    norm = _norm()
    calls = []
    comm_manager = _comm_manager(
        norm, calls, attn_tp=attn_tp, unfused=unfused, dense_tp=dense_tp
    )
    hidden_states = torch.randn(129, _HIDDEN, device="cuda", dtype=torch.bfloat16)
    residual = None if first_layer else torch.randn_like(hidden_states) * 3
    fp8_scale = torch.tensor([0.02], device="cuda") if with_scales else None
    fp4_scale = torch.tensor([7.5], device="cuda") if with_scales else None

    hidden_states, hidden_fp8, residual = _input_norm(
        comm_manager, norm, hidden_states, residual, fp8_scale
    )
    fused_input = not (first_layer or attn_tp or unfused)
    assert calls == ([] if fused_input else ["input_reduce_norm"])
    assert (hidden_fp8 is not None) == (fused_input and with_scales)

    calls.clear()
    hidden_states, hidden_fp4, residual = _post_attn_norm(
        comm_manager, norm, hidden_states, residual, fp4_scale, None
    )
    assert calls == (["post_attn_reduce_norm"] if attn_tp else [])
    copy_fp4 = not (attn_tp or dense_tp) and with_scales and _IS_BLACKWELL
    assert (hidden_fp4 is not None) == copy_fp4


@pytest.mark.parametrize("scheme", ["static", "dynamic", "block", "unquantized"])
def test_static_fp8_input_scale_hook(scheme: str) -> None:
    config = None
    if scheme != "unquantized":
        config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic" if scheme == "block" else scheme,
            weight_block_size=[128, 128] if scheme == "block" else None,
        )
    layer = ReplicatedLinear(
        128,
        128,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=config,
        prefix="model.proj",
    )

    scale = layer.quant_method.static_fp8_input_scale(layer)

    if scheme == "static":
        assert scale is layer.input_scale
    else:
        assert scale is None


def _fill_nvfp4(layer, generator: torch.Generator) -> None:
    layer.weight.data.copy_(
        torch.randint(
            0, 256, layer.weight.shape, dtype=torch.uint8, generator=generator
        )
    )
    scales = torch.rand(layer.weight_scale.shape, generator=generator) * 2 + 0.25
    layer.weight_scale.data.copy_(scales.to(torch.float8_e4m3fn))
    layer.input_scale.data.fill_(0.05)
    layer.weight_scale_2.data.fill_(0.002)
    layer.quant_method.process_weights_after_loading(layer)


@pytest.mark.skipif(not _IS_BLACKWELL, reason="NVFP4 GEMMs need Blackwell")
@pytest.mark.parametrize("fused_swiglu", ["0", "1"])
def test_mlp_takes_the_nvfp4_copy_as_its_own_quant(
    monkeypatch, fused_swiglu: str
) -> None:
    monkeypatch.setenv("TOKENSPEED_NVFP4_GEMM_SWIGLU_NVFP4_QUANT", fused_swiglu)
    mlp = Qwen3_5MoeMLP(
        hidden_size=_HIDDEN,
        intermediate_size=1024,
        hidden_act="silu",
        mapping=Mapping(rank=0, world_size=1),
        quant_config=Nvfp4Config(),
        reduce_results=False,
        parallelism="dense",
    ).cuda()
    assert mlp._use_nvfp4_gemm_swiglu_nvfp4_quant == (fused_swiglu == "1")
    generator = torch.Generator().manual_seed(0)
    _fill_nvfp4(mlp.gate_up_proj, generator)
    _fill_nvfp4(mlp.down_proj, generator)
    norm = _norm()

    for rows in (0, 1, 129):
        torch.manual_seed(rows)
        x = torch.randn(rows, _HIDDEN, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn_like(x) * 4
        normed, normed_fp4, _ = norm.add_norm_with_fp4(
            x, residual, mlp.input_fp4_scale()
        )

        assert normed_fp4 is not None
        torch.testing.assert_close(
            mlp.forward_prequantized(normed, normed_fp4), mlp(normed), atol=0, rtol=0
        )
