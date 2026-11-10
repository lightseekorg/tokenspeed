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

"""GatedMLP runs separate kernels unless the model lists a fused one."""

import os
import sys

import pytest
import torch
import torch.nn.functional as F
from tokenspeed_kernel.numerics.torch.activation import torch_swiglu_oai
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.activation import SiluAndMul, SwigluOAI
from tokenspeed.runtime.layers.mlp import GatedMLP, GatedMLPFusion
from tokenspeed.runtime.layers.quantization.nvfp4 import Nvfp4Config

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

_HIDDEN = 1024
_INTERMEDIATE = 512
_NVFP4 = GatedMLPFusion.NVFP4_GEMM_SWIGLU_QUANT


def _mlp(fusions, quant_config, activation, device) -> GatedMLP:
    dense = Mapping(rank=0, world_size=1).dense
    with torch.device(device):
        return GatedMLP.tensor_parallel(
            hidden_size=_HIDDEN,
            intermediate_size=_INTERMEDIATE,
            activation=activation,
            tp_rank=dense.tp_rank,
            tp_size=dense.tp_size,
            tp_group=dense.tp_group,
            reduce_results=False,
            fusions=fusions,
            quant_config=quant_config,
            prefix="model.layers.0.mlp",
        )


def _bf16_mlp(fusions, activation):
    mlp = _mlp(fusions, None, activation, "cuda").to(torch.bfloat16)
    gate, up = (torch.randn(_INTERMEDIATE, _HIDDEN) * 0.05 for _ in range(2))
    down = torch.randn(_HIDDEN, _INTERMEDIATE) * 0.05
    mlp.gate_up_proj.weight_loader(mlp.gate_up_proj.weight, gate, 0)
    mlp.gate_up_proj.weight_loader(mlp.gate_up_proj.weight, up, 1)
    mlp.down_proj.weight_loader(mlp.down_proj.weight, down)
    return mlp, *(w.to("cuda", torch.bfloat16) for w in (gate, up, down))


@torch.no_grad()
@pytest.mark.parametrize("rows", [0, 1, 77])
def test_bf16_matches_swiglu_reference(rows: int) -> None:
    torch.manual_seed(0)
    # A listed fusion the bf16 projections cannot run leaves the separate kernels.
    mlp, gate, up, down = _bf16_mlp((_NVFP4,), SiluAndMul())
    x = torch.randn(rows, _HIDDEN, device="cuda", dtype=torch.bfloat16)

    out = mlp(x)

    # SwiGLU in FP32 over the BF16 GEMM outputs, rounded once, as the kernel does.
    act = F.silu(F.linear(x, gate).float()) * F.linear(x, up).float()
    expected = F.linear(act.to(torch.bfloat16), down)
    assert out.shape == (rows, _HIDDEN)
    torch.testing.assert_close(out, expected, atol=1e-3, rtol=2**-7)


@torch.no_grad()
def test_swiglu_oai_matches_reference() -> None:
    torch.manual_seed(0)
    mlp, gate, up, down = _bf16_mlp((), SwigluOAI(alpha=1.702, limit=2.0))
    x = torch.randn(77, _HIDDEN, device="cuda", dtype=torch.bfloat16)

    out = mlp(x)

    gate_up = torch.cat([F.linear(x, gate), F.linear(x, up)], dim=-1)
    expected = F.linear(torch_swiglu_oai(gate_up, alpha=1.702, limit=2.0), down)
    torch.testing.assert_close(out, expected, atol=1e-3, rtol=2**-7)


def test_fusions_need_unclamped_swiglu() -> None:
    with pytest.raises(ValueError, match="compute SwiGLU"):
        _mlp((_NVFP4,), None, SiluAndMul(swiglu_limit=7.0), "cpu")


def _nvfp4_mlp(fusions) -> GatedMLP:
    mlp = _mlp(fusions, Nvfp4Config(), SiluAndMul(), "cuda")
    generator = torch.Generator().manual_seed(0)
    for proj in (mlp.gate_up_proj, mlp.down_proj):
        proj.weight.data.copy_(
            torch.randint(0, 256, proj.weight.shape, generator=generator)
        )
        proj.weight_scale.data.copy_(
            torch.rand(proj.weight_scale.shape, generator=generator) + 0.25
        )
        proj.weight_scale_2.data.fill_(0.002)
        proj.input_scale.data.fill_(0.05)
        proj.quant_method.process_weights_after_loading(proj)
    return mlp


@pytest.mark.skipif(
    not current_platform().is_blackwell, reason="NVFP4 GEMMs need Blackwell"
)
@torch.no_grad()
def test_nvfp4_fusion_runs_only_when_listed(monkeypatch) -> None:
    monkeypatch.setenv("TOKENSPEED_NVFP4_GEMM_SWIGLU_NVFP4_QUANT", "1")
    separate, fused = _nvfp4_mlp(()), _nvfp4_mlp((_NVFP4,))
    torch.manual_seed(0)
    x = torch.randn(33, _HIDDEN, device="cuda", dtype=torch.bfloat16)

    gate_up, _ = separate.gate_up_proj(x)
    expected, _ = separate.down_proj(SiluAndMul()(gate_up))
    assert torch.equal(separate(x), expected)
    # The fused GEMM skips the BF16 rounding of gate/up, so only close.
    out = fused(x)
    assert (out - expected).float().norm() < 0.05 * expected.float().norm()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
