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

"""MLA weight absorption: ``kv_b_proj`` to the absorbed ``w_kc``/``w_vc`` pair."""

import os
import sys

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig

# CPU-only test scheduled in runtime-1gpu because it imports the full runtime.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.layers.linear import ReplicatedLinear  # noqa: E402
from tokenspeed.runtime.layers.quantization.fp8 import (  # noqa: E402
    Fp8Config,
    Mxfp8Config,
)
from tokenspeed.runtime.models.deepseek_v3 import (  # noqa: E402
    DeepseekV3ForCausalLM,
    absorb_mla_kv_b_proj,
)

HEADS, NOPE, VDIM, LATENT, BLOCK = 2, 128, 128, 256, 128


class _Attention(nn.Module):
    """The MLA fields absorption reads, around a real ``kv_b_proj``."""

    def __init__(self, quant_config: Fp8Config | None) -> None:
        super().__init__()
        self.kv_b_proj = ReplicatedLinear(
            LATENT,
            HEADS * (NOPE + VDIM),
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=quant_config,
            prefix="kv_b_proj",
        )
        self.qk_nope_head_dim = NOPE
        self.v_head_dim = VDIM
        self.w_kc: torch.Tensor | None = None
        self.w_vc: torch.Tensor | None = None


def _fp8_config(block: list[int] | None) -> Fp8Config:
    return Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=block,
    )


def _assert_absorbed(attn: _Attention, weight: torch.Tensor) -> None:
    kc, vc = weight.unflatten(0, (HEADS, NOPE + VDIM)).split([NOPE, VDIM], dim=1)
    torch.testing.assert_close(attn.w_kc, kc)
    torch.testing.assert_close(attn.w_vc, vc.transpose(1, 2))


def test_unquantized_kv_b_is_split_as_loaded():
    attn = _Attention(None)
    attn.kv_b_proj.weight.data.normal_()

    absorb_mla_kv_b_proj(attn)

    _assert_absorbed(attn, attn.kv_b_proj.weight)


def test_block_fp8_kv_b_is_dequantized_with_its_own_tiles():
    attn = _Attention(_fp8_config([BLOCK, BLOCK]))
    gen = torch.Generator().manual_seed(0)
    weight = torch.randn(HEADS * (NOPE + VDIM), LATENT, generator=gen)
    attn.kv_b_proj.weight.data.copy_(weight.to(torch.float8_e4m3fn))
    scale = torch.rand(HEADS * (NOPE + VDIM) // BLOCK, LATENT // BLOCK) + 0.5
    attn.kv_b_proj.weight_scale_inv.data.copy_(scale)

    absorb_mla_kv_b_proj(attn)

    tiles = scale.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
    dequantized = attn.kv_b_proj.weight.float() * tiles
    # The layer's activation dtype, whatever the ambient default dtype is.
    _assert_absorbed(attn, dequantized.to(torch.bfloat16))


def test_fp8_kv_b_without_block_scales_is_refused():
    attn = _Attention(_fp8_config(None))

    with pytest.raises(NotImplementedError, match="block-FP8 kv_b_proj"):
        absorb_mla_kv_b_proj(attn)
    assert attn.w_kc is None and attn.w_vc is None


def test_ue8m0_scaled_kv_b_is_refused():
    attn = _Attention(
        Mxfp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[1, 32],
            scale_fmt="ue8m0",
        )
    )

    with pytest.raises(NotImplementedError, match="float scales"):
        absorb_mla_kv_b_proj(attn)


def test_deepseek_v3_post_load_refuses_a_per_tensor_fp8_kv_b():
    lm = object.__new__(DeepseekV3ForCausalLM)
    nn.Module.__init__(lm)
    lm.config = PretrainedConfig(num_hidden_layers=1)
    lm.quant_config = _fp8_config(None)
    layer = nn.Module()
    layer.self_attn = _Attention(lm.quant_config)
    lm.model = nn.Module()
    lm.model.layers = nn.ModuleList([layer])

    with pytest.raises(NotImplementedError, match="block-FP8 kv_b_proj"):
        lm.post_load_weights()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
