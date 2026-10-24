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

"""CPU checkpoint FP8 -> Ascend BF16: allocation, exact scales and GEMM."""

import unittest
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401

from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config
from tokenspeed.runtime.models.deepseek_v41 import (
    DeepseekV41ForCausalLM,
    _replicated,
    v41_mxfp8_config,
)


class DenseLoadingTest(unittest.TestCase):
    def test_grouped_output_projection_loader_decodes_before_transfer(self):
        torch.npu.set_device(0)
        codes = ((torch.arange(128 * 128).reshape(128, 128) % 17 - 8) / 4).to(
            torch.float8_e4m3fn
        )
        scales = (torch.arange(16).reshape(4, 4) % 5 + 123).to(torch.uint8)
        expected = (
            codes.float()
            * scales.view(torch.float8_e8m0fnu)
            .float()
            .repeat_interleave(32, 0)
            .repeat_interleave(32, 1)
        ).bfloat16()
        for rank in (0, 1):
            linear = SimpleNamespace(
                tp_rank=rank,
                output_size=128,
                input_size=128,
                weight=torch.nn.Parameter(
                    torch.empty(64, 128, dtype=torch.bfloat16, device="npu"),
                    requires_grad=False,
                ),
            )
            owner = SimpleNamespace(
                mapping=SimpleNamespace(attn=SimpleNamespace(tp_rank=rank)),
                model=SimpleNamespace(
                    layers=[SimpleNamespace(attn=SimpleNamespace(wo_a=linear))]
                ),
            )
            for order in (
                (("weight", codes), ("scale", scales)),
                (("scale", scales), ("weight", codes)),
            ):
                pending = {}
                for field, tensor in order:
                    DeepseekV41ForCausalLM._load_wo_a(
                        owner, "layers.0.attn.wo_a." + field, tensor, pending
                    )
                self.assertFalse(pending)
                torch.testing.assert_close(
                    linear.weight.cpu(),
                    expected[rank * 64 : (rank + 1) * 64],
                    rtol=0,
                    atol=0,
                )

    def test_fp8_checkpoint_loads_to_exact_bf16_weights(self):
        torch.set_num_threads(4)
        torch.npu.set_device(0)
        torch.npu.matmul.allow_hf32 = False
        quant = v41_mxfp8_config(
            Fp8Config(
                is_checkpoint_fp8_serialized=True,
                activation_scheme="dynamic",
                weight_block_size=[32, 32],
                scale_fmt="ue8m0",
            )
        )
        with torch.device("npu"):
            layer = _replicated(5120, 128, torch.bfloat16, quant, "model.test")
        values = ((torch.arange(128 * 5120).reshape(128, 5120) % 17) - 8).to(
            torch.float8_e4m3fn
        )
        scales = (torch.arange(4 * 160).reshape(4, 160) % 7 + 120).to(torch.uint8)
        layer.weight.weight_loader(layer.weight, values)
        layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, scales)
        layer.quant_method.process_weights_after_loading(layer)
        expected = (
            values.float()
            * scales.view(torch.float8_e8m0fnu)
            .float()
            .repeat_interleave(32, 0)
            .repeat_interleave(32, 1)
        ).bfloat16()
        self.assertEqual(layer.weight.dtype, torch.bfloat16)
        self.assertEqual(layer.weight.device.type, "npu")
        torch.testing.assert_close(layer.weight.cpu(), expected, rtol=0, atol=0)
        torch.manual_seed(12)
        x = torch.randn(7, 5120).bfloat16()
        actual, _ = layer(x.npu())
        oracle = (x.float() @ expected.float().T).bfloat16()
        torch.testing.assert_close(actual.cpu(), oracle, rtol=0.008, atol=0.0625)


if __name__ == "__main__":
    unittest.main(verbosity=2)
