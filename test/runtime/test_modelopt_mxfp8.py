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

"""ModelOpt MXFP8 checkpoints: config detection and load-time dense dequant."""

from __future__ import annotations

import os
import sys
import unittest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

import torch

from tokenspeed.runtime.layers.quantization import QUANTIZATION_METHODS
from tokenspeed.runtime.layers.quantization.modelopt_mxfp8 import (
    ModelOptMxfp8Config,
    dequantize_mxfp8_weight,
    preprocess_modelopt_mxfp8_weights,
)

_SECTION = {
    "quant_algo": "MXFP8",
    "kv_cache_quant_algo": None,
    "exclude_modules": ["lm_head", "model.layers.*.mlp.gate"],
}
_NESTED = {"quant_method": "modelopt", "quantization": dict(_SECTION)}


def _detect(hf_quant_cfg) -> str | None:
    for method in QUANTIZATION_METHODS.values():
        found = method.override_quantization_method(hf_quant_cfg, None)
        if found:
            return found
    return None


def _mxfp8(n: int, k: int, generator: torch.Generator):
    weight = (torch.randn(n, k, generator=generator) * 64).clamp(-448, 448)
    weight = weight.to(torch.float8_e4m3fn)
    scale = torch.randint(110, 140, (n, k // 32), generator=generator)
    return weight, scale.to(torch.uint8)


def _reference(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    factor = torch.exp2(scale.double() - 127).repeat_interleave(32, dim=1)
    return (weight.double() * factor).to(torch.bfloat16)


class ModelOptMxfp8ConfigTests(unittest.TestCase):
    def test_detects_nested_and_flat_sections(self):
        self.assertEqual(_detect(_NESTED), "modelopt_mxfp8")
        self.assertEqual(
            _detect({"quant_method": "modelopt", **_SECTION}), "modelopt_mxfp8"
        )

    def test_ignores_other_formats(self):
        self.assertIsNone(
            ModelOptMxfp8Config.override_quantization_method(
                {"quant_method": "fp8", "quant_algo": "MXFP8"}, None
            )
        )
        self.assertEqual(
            _detect(
                {
                    "quant_method": "modelopt",
                    "quantization": {"quant_algo": "MIXED_PRECISION"},
                }
            ),
            "modelopt_mixed",
        )

    def test_routes_experts_to_mxfp8_and_dense_to_bf16(self):
        config = ModelOptMxfp8Config.from_config(_NESTED)
        self.assertEqual(config.exclude_modules, _SECTION["exclude_modules"])
        moe = config.get_moe_quant_config("model.layers.3.mlp")
        self.assertEqual(config.moe_weight_dtype("model.layers.3.mlp"), "fp8")
        self.assertEqual(moe.weight_block_size, [1, 32])
        self.assertEqual(moe.weight_scale_dtype, torch.uint8)
        self.assertEqual(
            type(config.get_quant_method(None, "model.layers.0.o_proj")).__name__,
            "UnquantizedLinearMethod",
        )

    def test_rejects_kv_cache_quantization(self):
        with self.assertRaisesRegex(ValueError, "KV-cache"):
            ModelOptMxfp8Config.from_config(
                {"quant_algo": "MXFP8", "kv_cache_quant_algo": "FP8"}
            )


@unittest.skipUnless(torch.cuda.is_available(), "dequantizes on CUDA")
class ModelOptMxfp8PreprocessTests(unittest.TestCase):
    def setUp(self):
        self.config = ModelOptMxfp8Config.from_config(_NESTED)
        self.generator = torch.Generator().manual_seed(0)

    def _run(self, weights):
        return list(
            preprocess_modelopt_mxfp8_weights(
                weights,
                self.config,
                keeps_mxfp8=lambda name: ".mlp.experts." in name,
            )
        )

    def test_dequant_is_exact(self):
        weight, scale = _mxfp8(8, 128, self.generator)
        out = dequantize_mxfp8_weight(weight, scale)
        self.assertEqual(out.dtype, torch.bfloat16)
        torch.testing.assert_close(out.cpu(), _reference(weight, scale), rtol=0, atol=0)

    def test_pairs_dense_weights_in_any_order(self):
        w0, s0 = _mxfp8(4, 64, self.generator)
        w1, s1 = _mxfp8(4, 96, self.generator)
        norm = torch.ones(4, dtype=torch.bfloat16)
        out = self._run(
            [
                ("a.o_proj.weight", w0),
                ("a.norm.weight", norm),
                ("a.q_a_proj.weight_scale", s1),
                ("a.o_proj.weight_scale", s0),
                ("a.q_a_proj.weight", w1),
            ]
        )
        self.assertEqual(
            [name for name, _ in out],
            ["a.norm.weight", "a.o_proj.weight", "a.q_a_proj.weight"],
        )
        self.assertIs(out[0][1], norm)
        torch.testing.assert_close(out[1][1].cpu(), _reference(w0, s0), rtol=0, atol=0)
        torch.testing.assert_close(out[2][1].cpu(), _reference(w1, s1), rtol=0, atol=0)

    def test_experts_and_bf16_refits_pass_through(self):
        weight, scale = _mxfp8(4, 64, self.generator)
        refit = torch.randn(4, 64, dtype=torch.bfloat16)
        stream = [
            ("model.layers.3.mlp.experts.gate_up_proj", weight),
            ("model.layers.3.mlp.experts.gate_up_proj_scale", scale),
            ("model.layers.0.self_attn.o_proj.weight", refit),
        ]
        out = self._run(stream)
        self.assertEqual([name for name, _ in out], [name for name, _ in stream])
        for (_, got), (_, sent) in zip(out, stream):
            self.assertIs(got, sent)

    def test_unpaired_module_raises(self):
        weight, _ = _mxfp8(4, 64, self.generator)
        with self.assertRaisesRegex(RuntimeError, "a.o_proj"):
            self._run([("a.o_proj.weight", weight)])

    def test_rejects_nan_scale_and_bad_shape(self):
        weight, scale = _mxfp8(4, 64, self.generator)
        scale[1, 1] = 0xFF
        with self.assertRaisesRegex(ValueError, "NaN"):
            dequantize_mxfp8_weight(weight, scale)
        with self.assertRaisesRegex(ValueError, "1x32"):
            dequantize_mxfp8_weight(weight, scale[:, :1])

    def test_other_configs_pass_through(self):
        weight, scale = _mxfp8(4, 64, self.generator)
        stream = [("a.weight", weight), ("a.weight_scale", scale)]
        out = list(
            preprocess_modelopt_mxfp8_weights(
                iter(stream), None, keeps_mxfp8=lambda name: False
            )
        )
        self.assertEqual(out, stream)


if __name__ == "__main__":
    unittest.main()
