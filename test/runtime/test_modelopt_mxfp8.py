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

"""ModelOpt MXFP8 checkpoints: config detection and quantized dense loading."""

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

    def test_expert_scale_geometry(self):
        config = ModelOptMxfp8Config.from_config(_NESTED)
        self.assertEqual(config.exclude_modules, _SECTION["exclude_modules"])
        moe = config.get_moe_quant_config("model.layers.3.mlp")
        self.assertEqual(config.moe_weight_dtype("model.layers.3.mlp"), "fp8")
        self.assertEqual(moe.weight_block_size, [1, 32])
        self.assertEqual(moe.weight_scale_dtype, torch.uint8)

    def test_rejects_kv_cache_quantization(self):
        with self.assertRaisesRegex(ValueError, "KV-cache"):
            ModelOptMxfp8Config.from_config(
                {"quant_algo": "MXFP8", "kv_cache_quant_algo": "FP8"}
            )


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
            [
                "a.o_proj.weight",
                "a.norm.weight",
                "a.q_a_proj.weight_scale",
                "a.o_proj.weight_scale",
                "a.q_a_proj.weight",
            ],
        )
        for (_, got), expected in zip(out, (w0, norm, s1, s0, w1)):
            self.assertIs(got, expected)

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

    def test_e8m0_zero_is_valid_for_plain_dequantization(self):
        # E8M0 byte zero is 2**-127, not zero or NaN. Marlin's separate
        # restriction must not change the raw format's meaning.
        weight = torch.full((1, 32), 1.0).to(torch.float8_e4m3fn)
        scale = torch.zeros((1, 1), dtype=torch.uint8)
        expected = torch.full((1, 32), 2.0**-127, dtype=torch.bfloat16)
        torch.testing.assert_close(
            dequantize_mxfp8_weight(weight, scale), expected, rtol=0, atol=0
        )

    def test_rejects_nan_weight(self):
        weight, scale = _mxfp8(4, 64, self.generator)
        weight.view(torch.uint8)[0, 0] = 127
        with self.assertRaisesRegex(ValueError, "NaN"):
            dequantize_mxfp8_weight(weight, scale)

    def test_other_configs_pass_through(self):
        weight, scale = _mxfp8(4, 64, self.generator)
        stream = [("a.weight", weight), ("a.weight_scale", scale)]
        out = list(
            preprocess_modelopt_mxfp8_weights(
                iter(stream), None, keeps_mxfp8=lambda name: False
            )
        )
        self.assertEqual(out, stream)


class ModelOptMxfp8DenseLoaderTests(unittest.TestCase):
    def _config(self):
        return ModelOptMxfp8Config.from_config(_NESTED)

    def _raw(self, n, k):
        w = (torch.arange(n * k).reshape(n, k) % 16).float().to(torch.float8_e4m3fn)
        scale = (torch.arange(n * (k // 32)).reshape(n, k // 32) % 20 + 110).to(
            torch.uint8
        )
        return w, scale

    def test_excluded_linear_keeps_bf16_weight_values(self):
        from tokenspeed.runtime.layers.linear import ReplicatedLinear

        layer = ReplicatedLinear(
            64,
            32,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=self._config(),
            prefix="lm_head",
        )
        weight = torch.arange(32 * 64, dtype=torch.float32).reshape(32, 64).bfloat16()
        layer.weight.weight_loader(layer.weight, weight)
        torch.testing.assert_close(layer.weight, weight, rtol=0, atol=0)

    def test_column_parallel_loads_weight_and_scale_rows(self):
        from tokenspeed.runtime.layers.linear import ColumnParallelLinear

        layer = ColumnParallelLinear(
            128,
            128,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=self._config(),
            prefix="layer.o_proj",
            tp_rank=1,
            tp_size=2,
            tp_group=(0, 1),
            gather_output=False,
            use_presharded_weights=False,
        )
        w, scale = self._raw(128, 128)
        layer.weight.weight_loader(layer.weight, w)
        layer.weight_scale.weight_loader(
            layer.weight_scale, scale.view(torch.float8_e8m0fnu)
        )
        self.assertTrue(
            torch.equal(layer.weight.view(torch.uint8), w[64:].view(torch.uint8))
        )
        self.assertTrue(torch.equal(layer.weight_scale, scale[64:]))

    def test_row_parallel_scales_shard_in_block_units(self):
        from tokenspeed.runtime.layers.linear import RowParallelLinear

        layer = RowParallelLinear(
            256,
            64,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=self._config(),
            prefix="layer.o_proj",
            tp_rank=1,
            tp_size=2,
            tp_group=(0, 1),
            input_is_parallel=True,
            reduce_results=False,
            use_presharded_weights=False,
        )
        w, scale = self._raw(64, 256)
        layer.weight.weight_loader(layer.weight, w)
        layer.weight_scale.weight_loader(layer.weight_scale, scale)
        self.assertTrue(
            torch.equal(layer.weight.view(torch.uint8), w[:, 128:].view(torch.uint8))
        )
        self.assertTrue(torch.equal(layer.weight_scale, scale[:, 4:]))

    def test_replicated_fused_qkv_scales_keep_row_offsets(self):
        from tokenspeed.runtime.layers.linear import ReplicatedLinear

        layer = ReplicatedLinear(
            64,
            192,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=self._config(),
            prefix="layer.fused_qkv_a_proj_with_mqa",
        )
        w0, s0 = self._raw(128, 64)
        w1, s1 = self._raw(64, 64)
        for name, x0, x1 in (("weight", w0, w1), ("weight_scale", s0, s1)):
            param = layer.get_parameter(name)
            param.weight_loader(param, x1, begin_size=128)
            param.weight_loader(param, x0, begin_size=0)
            self.assertTrue(
                torch.equal(
                    param.view(torch.uint8), torch.cat([x0, x1]).view(torch.uint8)
                )
            )

    def test_merged_tp_keeps_both_scale_shards(self):
        from tokenspeed.runtime.layers.linear import MergedColumnParallelLinear

        layer = MergedColumnParallelLinear(
            64,
            [128, 64],
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=self._config(),
            prefix="layer.gate_up_proj",
            tp_rank=1,
            tp_size=2,
            tp_group=(0, 1),
            gather_output=False,
            use_presharded_weights=False,
        )
        w0, s0 = self._raw(128, 64)
        w1, s1 = self._raw(64, 64)
        for name, x0, x1 in (("weight", w0, w1), ("weight_scale", s0, s1)):
            param = layer.get_parameter(name)
            param.weight_loader(param, x0, 0)
            param.weight_loader(param, x1, 1)
            expected = torch.cat([x0[64:], x1[32:]])
            self.assertTrue(
                torch.equal(param.view(torch.uint8), expected.view(torch.uint8))
            )

    def test_dense_loader_rejects_silent_bf16_refit(self):
        from tokenspeed.runtime.layers.linear import ReplicatedLinear

        layer = ReplicatedLinear(
            64,
            64,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=self._config(),
            prefix="layer.q_b_proj",
        )
        with self.assertRaisesRegex(TypeError, "BF16 refits"):
            layer.weight.weight_loader(
                layer.weight, torch.zeros(64, 64, dtype=torch.bfloat16)
            )
        with self.assertRaisesRegex(RuntimeError, "prepared"):
            layer.quant_method.apply(layer, torch.zeros(1, 64, dtype=torch.bfloat16))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for Marlin")
    def test_prepared_linear_forward_absorption_and_refit_rejection(self):
        from tokenspeed.runtime.layers.linear import ReplicatedLinear

        with torch.device("cuda"):
            layer = ReplicatedLinear(
                64,
                70,
                bias=False,
                params_dtype=torch.bfloat16,
                quant_config=self._config(),
                prefix="layer.kv_b_proj",
            )
        weight, scale = self._raw(70, 64)
        layer.weight.weight_loader(layer.weight, weight)
        layer.weight_scale.weight_loader(layer.weight_scale, scale)
        expected_weight = _reference(weight, scale).cuda()
        layer.quant_method.process_weights_after_loading(layer)
        materialized = layer.quant_method.weight_for_absorption(layer)
        torch.testing.assert_close(materialized, expected_weight, rtol=0, atol=0)
        x = torch.zeros(2, 64, dtype=torch.bfloat16, device="cuda")
        x[:, 3] = 1
        actual, _ = layer(x)
        torch.testing.assert_close(
            actual, expected_weight[:, 3].expand(2, -1), rtol=0, atol=0
        )
        for name, value in (("weight", weight), ("weight_scale", scale)):
            parameter = layer.get_parameter(name)
            with self.assertRaisesRegex(RuntimeError, "fresh model load"):
                parameter.weight_loader(parameter, value)

    def test_preprocessor_rejects_duplicate_and_mismatched_scales(self):
        w, scale = self._raw(64, 128)
        config = self._config()
        for tensors, pattern in (
            ([("p.weight", w), ("p.weight", w)], "Duplicate"),
            ([("p.weight", w), ("p.weight_scale", scale[:, :2])], "1x32"),
        ):
            with self.assertRaisesRegex((RuntimeError, ValueError), pattern):
                list(
                    preprocess_modelopt_mxfp8_weights(
                        tensors, config, keeps_mxfp8=lambda _: False
                    )
                )


if __name__ == "__main__":
    unittest.main()
