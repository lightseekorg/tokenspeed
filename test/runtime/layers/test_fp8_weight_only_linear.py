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

"""Weight-only FP8 config: which linears it quantizes, and their outputs."""

from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.layers.dense.fp8_weight_only import (
    Fp8WeightOnlyLinearMethod,
    quantize_fp8_weight_rows,
)
from tokenspeed.runtime.layers.dense.unquant import UnquantizedLinearMethod
from tokenspeed.runtime.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.quantization import get_quantization_config
from tokenspeed.runtime.layers.quantization.fp8_weight_only import (
    Fp8WeightOnlyConfig,
)
from tokenspeed.runtime.utils.server_args import ServerArgs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


def test_the_flag_name_selects_the_config():
    assert get_quantization_config("fp8_weight_only") is Fp8WeightOnlyConfig


@pytest.mark.parametrize(
    "prefix, quantized",
    [
        ("layers.0.mlp.gate_up_proj", True),
        ("model.layers.4.mlp.down_proj", True),
        ("layers.0.self_attn.qkv_proj", False),
        ("layers.0.self_attn.o_proj", True),
        ("layers.0.mlp_conv.kernel_projection", False),
        ("fc", False),
    ],
)
def test_only_dense_mlp_projections_take_fp8(prefix, quantized):
    layer = ReplicatedLinear(
        256, 512, bias=False, quant_config=Fp8WeightOnlyConfig(), prefix=prefix
    )
    expected = Fp8WeightOnlyLinearMethod if quantized else UnquantizedLinearMethod
    assert type(layer.quant_method) is expected


def test_row_quantization_bounds_the_error_and_keeps_zero_rows():
    torch.manual_seed(0)
    weight = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
    weight[3] = 0
    q, scale = quantize_fp8_weight_rows(weight)
    assert q.dtype == torch.float8_e4m3fn and scale.dtype == torch.float32
    assert scale[3] == 1 and torch.count_nonzero(q[3].float()) == 0
    # Row maxima map to E4M3's largest value; E4M3 keeps 3 mantissa bits.
    assert torch.equal(
        q.float().abs().amax(dim=1)[weight.abs().amax(dim=1) > 0],
        torch.full((63,), 448.0, device="cuda"),
    )
    error = (q.float() * scale[:, None] - weight.float()).abs()
    assert torch.all(error <= weight.float().abs() * 2**-4 + scale[:, None] * 2**-9)


def _mlp(intermediate: int, hidden: int):
    config = Fp8WeightOnlyConfig()
    gate_up = MergedColumnParallelLinear(
        hidden,
        [intermediate] * 2,
        bias=False,
        quant_config=config,
        prefix="layers.0.mlp.gate_up_proj",
        tp_rank=0,
        tp_size=1,
    ).cuda()
    down = RowParallelLinear(
        intermediate,
        hidden,
        bias=False,
        quant_config=config,
        prefix="layers.0.mlp.down_proj",
        reduce_results=False,
        tp_rank=0,
        tp_size=1,
    ).cuda()
    return gate_up, down


@pytest.mark.parametrize("rows", [8, 16, 200])
def test_loaded_mlp_projections_match_their_dequantized_weights(rows):
    hidden, intermediate = 1024, 512
    gate_up, down = _mlp(intermediate, hidden)
    torch.manual_seed(rows)
    gate = torch.randn(intermediate, hidden, dtype=torch.bfloat16) / hidden**0.5
    up = torch.randn(intermediate, hidden, dtype=torch.bfloat16) / hidden**0.5
    down_w = torch.randn(hidden, intermediate, dtype=torch.bfloat16) / intermediate**0.5
    gate_up.weight_loader(gate_up.weight, gate, 0)
    gate_up.weight_loader(gate_up.weight, up, 1)
    down.weight_loader(down.weight, down_w)
    for layer in (gate_up, down):
        layer.quant_method.process_weights_after_loading(layer)
        assert layer.weight.dtype == torch.float8_e4m3fn

    # Gate and up quantize per row after both shards loaded.
    q, scale = quantize_fp8_weight_rows(torch.cat([gate, up]).cuda())
    assert torch.equal(gate_up.weight.view(torch.uint8), q.view(torch.uint8))
    assert torch.equal(gate_up.weight_scale, scale)

    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    out, _ = gate_up(x)
    dequant = gate_up.weight.double() * gate_up.weight_scale.double()[:, None]
    torch.testing.assert_close(
        out, (x.double() @ dequant.T).to(x.dtype), rtol=2**-7, atol=1e-3
    )
    y, _ = down(out[:, :intermediate])
    assert y.shape == (rows, hidden)


def test_targets_refuse_fp8_weight_only():
    with pytest.raises(ValueError, match="--speculative-draft-model-quantization"):
        ServerArgs(model="x", max_num_seqs=8, quantization="fp8_weight_only")
