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

"""Serialized W8A16 weights remain the only persistent model weights."""

import pytest
import torch
from tokenspeed_kernel.ops.gemm import plan_dense_gemm
from tokenspeed_kernel.selection import NoKernelFoundError

from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.layers.quantization.w8a16_fp8_block import W8A16Fp8BlockConfig
from tokenspeed.runtime.utils.env import global_server_args_dict


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("numerics", ["auto", "rl-bitwise"])
def test_linear_preserves_storage_across_updates(monkeypatch, numerics):
    monkeypatch.setitem(global_server_args_dict, "numerics", numerics)
    # A device without a batch-invariant BF16 GEMM must reject this envelope.
    requires_unavailable_plan = False
    if numerics == "rl-bitwise":
        try:
            plan_dense_gemm(
                mode="mm",
                input_size=256,
                output_size=256,
                dtype=torch.bfloat16,
                batch_size=None,
                batch_invariant=True,
            )
        except NoKernelFoundError:
            requires_unavailable_plan = True
    with torch.device("cuda"):
        if requires_unavailable_plan:
            with pytest.raises(NoKernelFoundError, match="batch_invariant"):
                ReplicatedLinear(
                    256,
                    256,
                    bias=False,
                    params_dtype=torch.bfloat16,
                    quant_config=W8A16Fp8BlockConfig(ignored_layers=[]),
                    prefix="ffn",
                )
            return
        layer = ReplicatedLinear(
            256,
            256,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=W8A16Fp8BlockConfig(ignored_layers=[]),
            prefix="ffn",
        )
        x = torch.randn(7, 256, dtype=torch.bfloat16)
    addresses = [p.data_ptr() for p in layer.parameters()]
    for version in range(3):
        q = (torch.randn((256, 256), device="cuda") * (version + 1)).to(
            torch.float8_e4m3fn
        )
        s = torch.rand((2, 2), device="cuda") * 0.1 + 0.01
        layer.weight.data.copy_(q)
        layer.weight_scale_inv.data.copy_(s)
        layer.quant_method.process_weights_after_loading(layer)
        expected_weight = (
            q.float() * s.repeat_interleave(128, 0).repeat_interleave(128, 1)
        ).bfloat16()
        from tokenspeed_kernel.ops.gemm import mm

        expected = mm(x, expected_weight, plan=layer.quant_method.kernel_plan)
        actual, _ = layer(x)
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        if numerics == "rl-bitwise":
            single, _ = layer(x[:1])
            assert torch.equal(single.view(torch.int16), actual[:1].view(torch.int16))
        assert [p.data_ptr() for p in layer.parameters()] == addresses
        assert all(p.dtype != torch.bfloat16 for p in layer.parameters())
        assert torch.equal(layer.weight.view(torch.uint8), q.view(torch.uint8))
    layer.weight_scale_inv.data.fill_(float("nan"))
    with pytest.raises(ValueError, match="finite and positive"):
        layer.quant_method.process_weights_after_loading(layer)


def test_checkpoint_interpretation_requires_explicit_weight_only_mode():
    config = {
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": ["lm_head"],
    }
    assert W8A16Fp8BlockConfig.override_quantization_method(config, None) is None
    assert (
        W8A16Fp8BlockConfig.override_quantization_method(config, "w8a16_fp8_block")
        == "w8a16_fp8_block"
    )
    assert W8A16Fp8BlockConfig.from_config(config).ignored_layers == ["lm_head"]
    with pytest.raises(ValueError, match="128x128"):
        W8A16Fp8BlockConfig.from_config(config | {"weight_block_size": [64, 128]})


def test_launch_parser_accepts_weight_only_block_mode():
    import argparse

    from tokenspeed.runtime.utils.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(
        ["--model", "/checkpoint", "--quantization", "w8a16_fp8_block"]
    )
    assert args.quantization == "w8a16_fp8_block"
