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

"""Dynamic-activation MXFP4 dense linears: load, prepare, fused SwiGLU MLP."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.platform import current_platform

if not current_platform().is_amd:
    pytest.skip("Dynamic MXFP4 activations are AMD-only", allow_module_level=True)

from tokenspeed_kernel.ops.activation.triton import silu_and_mul  # noqa: E402
from tokenspeed_kernel.ops.gemm.triton import triton_mm_mxfp4  # noqa: E402
from tokenspeed_kernel.ops.quantization.triton import mxfp4_quantize  # noqa: E402

from tokenspeed.runtime.layers.dense.mxfp4 import Mxfp4LinearMethod  # noqa: E402
from tokenspeed.runtime.layers.linear import ReplicatedLinear  # noqa: E402
from tokenspeed.runtime.layers.quantization.mxfp4 import Mxfp4Config  # noqa: E402

_HIDDEN = 2560
_INTERMEDIATE = 320


def _linear(input_size: int, output_size: int, prefix: str) -> ReplicatedLinear:
    config = Mxfp4Config(
        is_checkpoint_mxfp4_serialized=True,
        use_dynamic_mxfp4_activations=True,
    )
    with torch.device("cuda"):
        layer = ReplicatedLinear(
            input_size,
            output_size,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=config,
            prefix=prefix,
        )
    assert isinstance(layer.quant_method, Mxfp4LinearMethod)
    return layer


def _load(layer: ReplicatedLinear) -> tuple[torch.Tensor, torch.Tensor]:
    n, k_packed = layer.weight.shape
    weight = torch.randint(0, 256, (n, k_packed), device="cuda", dtype=torch.uint8)
    scale = torch.randint(
        118, 124, tuple(layer.weight_scale.shape), device="cuda", dtype=torch.uint8
    )
    layer.weight_loader(layer.weight, weight)
    layer.weight_loader(layer.weight_scale, scale)
    return weight, scale


def _unfused_linear(x, weight, scale) -> torch.Tensor:
    x_fp4, x_scale = mxfp4_quantize(x)
    return triton_mm_mxfp4(x_fp4, weight, x_scale, scale, torch.bfloat16)


@pytest.mark.parametrize("tokens", [3, 64, 700])
def test_shared_expert_mlp_matches_unfused_path(tokens: int) -> None:
    torch.manual_seed(0)
    gate_up = _linear(_HIDDEN, 2 * _INTERMEDIATE, "mlp.shared_expert.gate_up_proj")
    down = _linear(_INTERMEDIATE, _HIDDEN, "mlp.shared_expert.down_proj")
    gate_up.quant_method.enable_swiglu(gate_up)
    gate_up_weight, gate_up_scale = _load(gate_up)
    down_weight, down_scale = _load(down)
    for layer in (gate_up, down):
        layer.quant_method.process_weights_after_loading(layer)

    x = torch.randn((tokens, _HIDDEN), device="cuda", dtype=torch.bfloat16)
    act = gate_up.quant_method.apply_swiglu(gate_up, x)
    out, _ = down(act)

    gate_up_ref = _unfused_linear(x, gate_up_weight, gate_up_scale)
    act_ref = silu_and_mul(
        gate_up_ref,
        torch.empty((tokens, _INTERMEDIATE), device="cuda", dtype=torch.bfloat16),
    )
    out_ref = _unfused_linear(act_ref, down_weight, down_scale)
    # Bit-identical activation quantization; FP32 summation order may differ.
    torch.testing.assert_close(act.float(), act_ref.float(), atol=1e-3, rtol=2**-8)
    torch.testing.assert_close(out.float(), out_ref.float(), atol=1e-3, rtol=2**-8)

    # Plain projection through the same prepared gate_up weight.
    gate_up_out, _ = gate_up(x)
    torch.testing.assert_close(
        gate_up_out.float(), gate_up_ref.float(), atol=1e-3, rtol=2**-8
    )


def test_enable_swiglu_after_processing_is_rejected() -> None:
    layer = _linear(_HIDDEN, 2 * _INTERMEDIATE, "mlp.gate_up_proj")
    _load(layer)
    layer.quant_method.process_weights_after_loading(layer)
    with pytest.raises(RuntimeError, match="precede weight processing"):
        layer.quant_method.enable_swiglu(layer)


def test_qwen_shared_expert_mlp_fuses_swiglu() -> None:
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.models.qwen3_5_moe import Qwen3_5MoeMLP

    torch.manual_seed(0)
    config = Mxfp4Config(
        is_checkpoint_mxfp4_serialized=True,
        use_dynamic_mxfp4_activations=True,
    )
    with torch.device("cuda"):
        mlp = Qwen3_5MoeMLP(
            hidden_size=_HIDDEN,
            intermediate_size=_INTERMEDIATE,
            hidden_act="silu",
            mapping=Mapping(rank=0, world_size=1),
            quant_config=config,
            reduce_results=False,
            prefix="model.layers.0.mlp.shared_expert",
            parallelism="moe_shared",
        )
    assert mlp.gate_up_proj.mxfp4_swiglu and not mlp.down_proj.mxfp4_swiglu
    gate_up_weight, gate_up_scale = _load(mlp.gate_up_proj)
    down_weight, down_scale = _load(mlp.down_proj)
    for layer in (mlp.gate_up_proj, mlp.down_proj):
        layer.quant_method.process_weights_after_loading(layer)

    x = torch.randn((40, _HIDDEN), device="cuda", dtype=torch.bfloat16)
    act_ref = silu_and_mul(
        _unfused_linear(x, gate_up_weight, gate_up_scale),
        torch.empty((40, _INTERMEDIATE), device="cuda", dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        mlp(x).float(),
        _unfused_linear(act_ref, down_weight, down_scale).float(),
        atol=1e-3,
        rtol=2**-8,
    )
