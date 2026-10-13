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

"""HY V4 config, weight-loading and iHC numerics tests (cheap, no GPU)."""

from __future__ import annotations

import os
import re
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=20, suite="runtime-1gpu")

import torch
from tokenspeed_kernel.ops.residual.torch import (
    torch_ihc_head,
    torch_ihc_post,
    torch_ihc_pre,
)

from tokenspeed.runtime.configs.hy_v4_config import HYV4Config
from tokenspeed.runtime.configs.model_config import (
    AttentionArch,
    _resolve_attention_family,
    configure_dsa_attention,
)
from tokenspeed.runtime.execution.model_runner import ModelRunner
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config, Mxfp8Config
from tokenspeed.runtime.layers.quantization.modelopt_mixed import ModelOptMixedConfig
from tokenspeed.runtime.layers.quantization.modelopt_mxfp8 import ModelOptMxfp8Config
from tokenspeed.runtime.layers.quantization.w8a8_fp8 import W8A8Fp8Config
from tokenspeed.runtime.model_loader import loader as model_loader
from tokenspeed.runtime.models import hy_v4, hy_v4_nextn
from tokenspeed.runtime.models.base.weight_update import weight_update_session
from tokenspeed.runtime.models.hy_v4 import (
    HYV4Attention,
    HYV4ForCausalLM,
    HYV4Model,
    check_hyv4_quant_config,
    check_hyv4_speculative_algorithm,
    load_hyv4_weights,
    normalize_hyv4_checkpoint_name,
)
from tokenspeed.runtime.models.hy_v4_nextn import HYV4ForCausalLMNextN, HYV4ModelNextN
from tokenspeed.runtime.utils.server_args import ServerArgs

# Per-layer vectors of the released ``tencent/Hy4-preview`` checkpoint.
_RELEASED_FULL_INDEXERS = [0, 1] + list(range(5, 78, 4))


def _reference_ihc_pre(x, fn, scale, base, hc, magnitude, hc_eps, rms_eps):
    """Line-by-line copy of the reference ``HYV4HCPreLayer.forward``."""
    shape = x.size()
    flat = x.flatten(1).float()
    rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + rms_eps)
    mixes = torch.nn.functional.linear(flat, fn) * rsqrt
    pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + hc_eps
    post = (
        magnitude * torch.sigmoid(mixes[..., hc : 2 * hc] * scale[1] + base[hc:])
        + hc_eps
    )
    return torch.sum(pre.unsqueeze(-1) * x.reshape(shape), dim=1).to(x.dtype), post


def _reference_ihc_head(x, fn, scale, base, hc_eps, rms_eps):
    """Line-by-line copy of the reference ``HYV4HCHeadLayer.forward``."""
    shape, dtype = x.size(), x.dtype
    flat = x.flatten(1).float()
    rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + rms_eps)
    mixes = torch.nn.functional.linear(flat, fn) * rsqrt
    pre = torch.sigmoid(mixes * scale + base) + hc_eps
    return torch.sum(pre.unsqueeze(-1) * flat.reshape(shape), dim=1).to(dtype)


class HYV4ConfigTests(unittest.TestCase):
    def test_defaults_reproduce_released_layer_vectors(self):
        config = HYV4Config()
        self.assertEqual(config.num_hidden_layers, 78)
        self.assertEqual(config.mlp_layer_types[0], "dense")
        self.assertTrue(all(t == "sparse" for t in config.mlp_layer_types[1:]))
        self.assertEqual(config.first_k_dense_replace, 1)
        full = [i for i, t in enumerate(config.indexer_types) if t == "full"]
        self.assertEqual(full, _RELEASED_FULL_INDEXERS)

    def test_derived_attention_geometry(self):
        config = HYV4Config()
        self.assertEqual(config.qk_head_dim, 256)
        # RoPE covers only the rope slice.
        self.assertEqual(config.head_dim, 64)
        self.assertEqual(config.rope_theta, 10000000.0)
        self.assertIsNone(config.rope_scaling)
        self.assertEqual(config.head_dtype, "float32")
        self.assertEqual(config.topk_method, "noaux_tc")
        self.assertTrue(config.indexer_rope_interleave)

    def test_sparse_layer_aliases_fold_to_dsa(self):
        config = HYV4Config(
            num_hidden_layers=2,
            layer_types=["sparse", "sparse_attention"],
            mlp_layer_types=["dense", "sparse"],
            indexer_types=["full", "shared"],
        )
        self.assertEqual(config.layer_types, ["deepseek_sparse_attention"] * 2)

    def test_rejects_mismatched_layer_vectors(self):
        with self.assertRaises(ValueError):
            HYV4Config(num_hidden_layers=4, indexer_types=["full"])
        with self.assertRaises(ValueError):
            HYV4Config(num_hidden_layers=4, mlp_layer_types=["dense"])

    def test_rejects_dense_attention_layers(self):
        with self.assertRaisesRegex(ValueError, "full_attention"):
            HYV4Config(
                num_hidden_layers=2,
                layer_types=["sparse", "full_attention"],
                mlp_layer_types=["dense", "sparse"],
                indexer_types=["full", "shared"],
            )

    def test_rejects_yarn(self):
        with self.assertRaises(ValueError):
            HYV4Config(rope_parameters={"rope_theta": 1e7, "rope_type": "yarn"})

    def test_dense_layers_must_lead_the_stack(self):
        for mlp_layer_types in (
            ["dense", "sparse", "dense"],
            ["sparse", "dense", "dense"],
            ["dense", "moe", "sparse"],
        ):
            with self.subTest(mlp_layer_types=mlp_layer_types):
                with self.assertRaisesRegex(ValueError, "mlp_layer_types"):
                    HYV4Config(num_hidden_layers=3, mlp_layer_types=mlp_layer_types)
        config = HYV4Config(num_hidden_layers=3, mlp_layer_types=["sparse"] * 3)
        self.assertEqual(config.first_k_dense_replace, 0)

    def test_legacy_rope_scaling_must_be_plain_rope(self):
        for rope_scaling in (
            {"type": "yarn", "factor": 4.0},
            {"rope_type": "linear", "factor": 2.0},
        ):
            with self.subTest(rope_scaling=rope_scaling):
                with self.assertRaisesRegex(ValueError, "plain RoPE"):
                    HYV4Config(rope_scaling=rope_scaling)

    def test_legacy_rope_scaling_folds_into_rope_parameters(self):
        config = HYV4Config(rope_scaling={"type": "default", "rope_theta": 5e6})
        self.assertEqual(config.rope_theta, 5e6)
        self.assertEqual(config.rope_parameters["rope_type"], "default")
        self.assertIsNone(config.rope_scaling)
        for rope_scaling in (None, {}):
            with self.subTest(rope_scaling=rope_scaling):
                self.assertEqual(HYV4Config(rope_scaling=rope_scaling).rope_theta, 1e7)

    def test_rope_scaling_must_agree_with_rope_parameters(self):
        rope_parameters = {"rope_theta": 1e7, "rope_type": "default"}
        with self.assertRaisesRegex(ValueError, "disagree"):
            HYV4Config(
                rope_parameters=rope_parameters, rope_scaling={"rope_theta": 5e6}
            )
        config = HYV4Config(
            rope_parameters=rope_parameters,
            rope_scaling={"type": "default", "rope_theta": 10000000},
        )
        self.assertEqual(config.rope_theta, 1e7)

    def test_explicit_head_dim_must_be_the_rope_slice(self):
        self.assertEqual(HYV4Config(head_dim=64).head_dim, 64)
        with self.assertRaisesRegex(ValueError, "head_dim"):
            HYV4Config(head_dim=128)

    def test_serialized_config_parses_again(self):
        config = HYV4Config.from_dict(HYV4Config().to_dict())
        self.assertEqual(config.head_dim, 64)
        self.assertEqual(config.rope_theta, 1e7)
        self.assertEqual(config.first_k_dense_replace, 1)

    def test_runtime_attention_config_is_dsa(self):
        config = HYV4Config()
        model_config = SimpleNamespace(hf_text_config=config, hf_config=config)
        configure_dsa_attention(model_config, ServerArgs(model="x"))
        self.assertEqual(model_config.attention_arch, AttentionArch.DSA)
        self.assertEqual(model_config.head_dim, 256)
        self.assertEqual(model_config.kv_lora_rank, 512)
        self.assertEqual(model_config.index_topk, 2048)
        self.assertAlmostEqual(model_config.scaling, 256**-0.5)

    def test_target_and_draft_declare_padded_prefill_replay(self):
        for arch, expected in (
            ("HYV4ForCausalLM", True),
            ("HYV4ForCausalLMNextN", True),
            ("GlmMoeDsaForCausalLM", False),
        ):
            config = HYV4Config(architectures=[arch])
            model_config = SimpleNamespace(hf_text_config=config, hf_config=config)
            family = _resolve_attention_family(config, config)
            family.configure(model_config, ServerArgs(model="x"))
            self.assertIs(model_config.dsa_padded_prefill_replay, expected)

    def test_architectures_resolve_to_the_hy_v4_dsa_family(self):
        for arch in ("HYV4ForCausalLM", "HYV4ForCausalLMNextN"):
            config = HYV4Config(architectures=[arch])
            family = _resolve_attention_family(config, config)
            self.assertEqual((family.name, family.default_backend), ("HY V4", "dsa"))


class HYV4CheckpointNameTests(unittest.TestCase):
    def test_ihc_boundary_names_flatten(self):
        self.assertEqual(
            normalize_hyv4_checkpoint_name("model.layers.3.hc_attn_layer.hc_pre.hc_fn"),
            "model.layers.3.hc_attn_layer.hc_fn",
        )
        self.assertEqual(
            normalize_hyv4_checkpoint_name(
                "model.layers.3.hc_mlp_layer.hc_pre.hc_scale"
            ),
            "model.layers.3.hc_mlp_layer.hc_scale",
        )

    def test_head_names_drop_prefix(self):
        for suffix in ("fn", "scale", "base"):
            self.assertEqual(
                normalize_hyv4_checkpoint_name(f"model.hc_head.hc_head_{suffix}"),
                f"model.hc_head.hc_{suffix}",
            )

    def test_other_names_untouched(self):
        name = "model.layers.1.self_attn.learnable_sink_param"
        self.assertEqual(normalize_hyv4_checkpoint_name(name), name)


class HYV4IHCNumericsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.hc, self.hidden = 4, 96
        self.x = (torch.randn(7, self.hc, self.hidden) * 3).bfloat16()

    def test_pre_matches_reference(self):
        fn = torch.randn(2 * self.hc, self.hc * self.hidden) * 0.02
        scale = torch.tensor([0.7, 1.3])
        base = torch.randn(2 * self.hc)
        expected, expected_post = _reference_ihc_pre(
            self.x, fn, scale, base, self.hc, 2.0, 1e-6, 1e-5
        )
        actual, actual_post = torch_ihc_pre(
            self.x, fn, scale, base, 2.0, 1e-5, 1e-6, None, None
        )
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actual_post, expected_post))

    def test_fused_norm_equals_separate_norm(self):
        fn = torch.randn(2 * self.hc, self.hc * self.hidden) * 0.02
        scale = torch.tensor([0.7, 1.3])
        base = torch.randn(2 * self.hc)
        weight = torch.randn(self.hidden).bfloat16()
        reduced, _ = torch_ihc_pre(self.x, fn, scale, base, 2.0, 1e-5, 1e-6, None, None)
        fused, _ = torch_ihc_pre(self.x, fn, scale, base, 2.0, 1e-5, 1e-6, weight, 1e-5)
        separate = torch.nn.functional.rms_norm(reduced, (self.hidden,), weight, 1e-5)
        self.assertTrue(torch.equal(fused, separate))

    def test_post_scatters_with_gates(self):
        block = torch.randn(7, self.hidden).bfloat16()
        gates = torch.rand(7, self.hc) * 2
        expected = (
            gates.unsqueeze(-1) * block.float().unsqueeze(-2) + self.x.float()
        ).to(torch.bfloat16)
        self.assertTrue(torch.equal(torch_ihc_post(block, self.x, gates), expected))

    def test_head_matches_reference(self):
        fn = torch.randn(self.hc, self.hc * self.hidden) * 0.02
        scale = torch.tensor([0.9])
        base = torch.randn(self.hc)
        expected = _reference_ihc_head(self.x, fn, scale, base, 1e-6, 1e-5)
        actual = torch_ihc_head(self.x, fn, scale, base, 1e-5, 1e-6)
        self.assertTrue(torch.equal(actual, expected))

    def test_rejects_wrong_projection_shape(self):
        fn = torch.randn(self.hc, self.hc * self.hidden)  # head shape, not pre
        with self.assertRaises(ValueError):
            torch_ihc_pre(
                self.x,
                fn,
                torch.ones(2),
                torch.zeros(self.hc),
                2.0,
                1e-5,
                1e-6,
                None,
                None,
            )


def _owner_with_params(names: list[str]) -> tuple[torch.nn.Module, list]:
    """Build a module tree holding ``names`` whose loaders record each call."""
    owner = torch.nn.Module()
    owner.config = SimpleNamespace(n_routed_experts=2, q_lora_rank=4)
    owner.mapping = SimpleNamespace(moe=SimpleNamespace(ep_rank=0, ep_size=1))
    owner.quant_config = None
    calls = []
    for name in names:
        module = owner
        *parents, leaf = name.split(".")
        for part in parents:
            if not hasattr(module, part):
                module.add_module(part, torch.nn.Module())
            module = getattr(module, part)
        param = torch.nn.Parameter(torch.empty(1), requires_grad=False)
        param.weight_loader = (
            lambda _param, _weight, *args, _name=name, **kwargs: calls.append(
                (_name, args, kwargs)
            )
        )
        module.register_parameter(leaf, param)
    return owner, calls


class HYV4WeightLoaderTests(unittest.TestCase):
    _PARAMS = [
        "model.layers.0.self_attn.fused_qkv_a_proj_with_mqa.weight",
        "model.layers.0.self_attn.indexer.wk_weights_proj.weight",
        "model.layers.0.mlp.gate_up_proj.weight",
        "model.layers.0.hc_attn_layer.hc_fn",
    ]
    _CHECKPOINT = [
        "model.layers.0.self_attn.q_a_proj.weight",
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
        "model.layers.0.self_attn.indexer.wk.weight",
        "model.layers.0.self_attn.indexer.weights_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.hc_attn_layer.hc_pre.hc_fn",
        "model.layers.0.self_attn.rotary_emb.inv_freq",
    ]
    _EXPERTS = "model.layers.0.mlp.experts."

    def _load(self, owner, names, weights=(), **kwargs):
        load_hyv4_weights(
            owner,
            [(name, torch.empty(1)) for name in names] + list(weights),
            module_prefix="model",
            checkpoint_prefix="model",
            **kwargs,
        )

    def test_routes_checkpoint_names_onto_fused_params(self):
        owner, calls = _owner_with_params(self._PARAMS)
        self._load(owner, self._CHECKPOINT)
        qkv_a, indexer, gate_up = self._PARAMS[:3]
        self.assertEqual(
            calls,
            [
                (qkv_a, (), {"begin_size": 0}),
                (qkv_a, (), {"begin_size": 4}),
                (indexer, (0,), {}),
                (indexer, (1,), {}),
                (gate_up, (0,), {}),
                (gate_up, (1,), {}),
                ("model.layers.0.hc_attn_layer.hc_fn", (), {}),
            ],
        )

    def test_unloaded_parameter_raises(self):
        owner, _ = _owner_with_params(self._PARAMS)
        with self.assertRaisesRegex(RuntimeError, "hc_attn_layer.hc_fn"):
            self._load(owner, self._CHECKPOINT[:-2])

    def test_fused_parameter_missing_a_shard_raises(self):
        qkv_a, indexer, gate_up = self._PARAMS[:3]
        for dropped, fused in zip(
            self._CHECKPOINT[:6], (qkv_a, qkv_a, indexer, indexer, gate_up, gate_up)
        ):
            with self.subTest(dropped=dropped):
                owner, _ = _owner_with_params(self._PARAMS)
                checkpoint = [name for name in self._CHECKPOINT if name != dropped]
                with self.assertRaisesRegex(RuntimeError, re.escape(fused)):
                    self._load(owner, checkpoint)

    def test_shared_parameters_may_stay_unloaded(self):
        owner, _ = _owner_with_params(self._PARAMS + ["lm_head.weight"])
        self._load(owner, self._CHECKPOINT, shared_params=("lm_head.",))
        with self.assertRaisesRegex(RuntimeError, "lm_head.weight"):
            self._load(owner, self._CHECKPOINT)

    def test_unmatched_checkpoint_tensor_warns(self):
        owner, _ = _owner_with_params(self._PARAMS)
        with self.assertLogs("tokenspeed.runtime.models.hy_v4", "WARNING") as logs:
            self._load(owner, self._CHECKPOINT + ["model.layers.0.extra.weight"])
        self.assertIn("model.layers.0.extra.weight", logs.output[0])

    def test_packed_experts_fill_every_local_expert(self):
        owner, calls = _owner_with_params(
            [self._EXPERTS + "w13_weight", self._EXPERTS + "w2_weight"]
        )
        packed = torch.zeros(2, 4, 8)
        self._load(
            owner,
            [],
            weights=[
                (self._EXPERTS + "gate_up_proj", packed),
                (self._EXPERTS + "down_proj", packed),
            ],
        )
        self.assertEqual(
            sorted(
                (name, kwargs["shard_id"], kwargs["local_expert_id"])
                for name, _, kwargs in calls
            ),
            sorted(
                (self._EXPERTS + param, shard, expert)
                for param, shards in (
                    ("w13_weight", ("w1", "w3")),
                    ("w2_weight", ("w2",)),
                )
                for shard in shards
                for expert in range(2)
            ),
        )

    def test_per_expert_tensors_leave_the_experts_unloaded(self):
        owner, calls = _owner_with_params(
            [self._EXPERTS + "w13_weight", self._EXPERTS + "w2_weight"]
        )
        per_expert = [
            f"{self._EXPERTS}{expert}.{proj}.weight"
            for expert in range(2)
            for proj in ("gate_proj", "up_proj", "down_proj")
        ]
        with self.assertLogs("tokenspeed.runtime.models.hy_v4", "WARNING"):
            with self.assertRaisesRegex(RuntimeError, re.escape("w13_weight")):
                self._load(owner, per_expert)
        self.assertEqual(calls, [])

    @unittest.skipUnless(torch.cuda.is_available(), "MXFP8 dequantizes on CUDA")
    def test_mxfp8_checkpoint_feeds_bf16_dense_and_fp8_experts(self):
        experts = "model.layers.0.mlp.experts."
        owner, calls = _owner_with_params(
            [
                "model.layers.0.self_attn.o_proj.weight",
                experts + "w13_weight",
                experts + "w13_weight_scale_inv",
                experts + "w2_weight",
                experts + "w2_weight_scale_inv",
            ]
        )
        owner.quant_config = ModelOptMxfp8Config.from_config(
            {"quant_method": "modelopt", "quantization": {"quant_algo": "MXFP8"}}
        )
        received = {}
        owner.get_parameter("model.layers.0.self_attn.o_proj.weight").weight_loader = (
            lambda _param, weight: received.setdefault("o_proj", weight)
        )
        fp8 = torch.ones(2, 4, 64).to(torch.float8_e4m3fn)
        scale = torch.full((2, 4, 2), 127, dtype=torch.uint8)
        self._load(
            owner,
            [],
            weights=[
                ("model.layers.0.self_attn.o_proj.weight", fp8[0]),
                ("model.layers.0.self_attn.o_proj.weight_scale", scale[0]),
                (experts + "gate_up_proj", fp8),
                (experts + "gate_up_proj_scale", scale),
                (experts + "down_proj", fp8),
                (experts + "down_proj_scale", scale),
            ],
        )
        self.assertEqual(received["o_proj"].dtype, torch.bfloat16)
        self.assertTrue(bool((received["o_proj"] == 1).all()))
        loaded = {(name, kwargs["shard_id"]) for name, _, kwargs in calls}
        self.assertEqual(
            loaded,
            {
                (experts + param, shard)
                for param, shards in (
                    ("w13_weight", ("w1", "w3")),
                    ("w13_weight_scale_inv", ("w1", "w3")),
                    ("w2_weight", ("w2",)),
                    ("w2_weight_scale_inv", ("w2",)),
                )
                for shard in shards
            },
        )

    def test_nextn_maps_the_mtp_block(self):
        captured = {}

        def fake_load(owner, weights, **kwargs):
            captured["names"] = [name for name, _ in weights]
            captured["kwargs"] = kwargs

        draft = SimpleNamespace(
            post_load_weights=lambda: None, _weight_update_non_unit_kv_scales=None
        )
        checkpoint = [
            "model.layers.3.mlp.down_proj.weight",
            "model.mtp_layers.0.enorm.weight",
            "model.mtp_layers.0.eh_proj.weight",
            "model.mtp_layers.0.final_layernorm.weight",
            "model.mtp_layers.0.self_attn.q_a_proj.weight",
            "model.mtp_layers.0.hc_attn_layer.hc_pre.hc_fn",
        ]
        with mock.patch.object(hy_v4_nextn, "load_hyv4_weights", fake_load):
            HYV4ForCausalLMNextN.load_weights(
                draft, ((name, torch.empty(1)) for name in checkpoint)
            )
        self.assertEqual(
            captured["names"],
            [
                "model.enorm.weight",
                "model.eh_proj.weight",
                "model.final_layernorm.weight",
                "model.decoder.self_attn.q_a_proj.weight",
                "model.decoder.hc_attn_layer.hc_pre.hc_fn",
            ],
        )
        self.assertEqual(
            captured["kwargs"]["shared_params"], ("model.embed_tokens.", "lm_head.")
        )


class HYV4QuantizationTests(unittest.TestCase):
    def test_quantization_other_than_modelopt_mxfp8_is_rejected(self):
        for label, quant_config in (
            (
                "block-scale",
                Fp8Config(
                    is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128]
                ),
            ),
            ("per-tensor", Fp8Config(is_checkpoint_fp8_serialized=True)),
            ("online", Fp8Config()),
            (
                "mxfp8",
                Mxfp8Config(
                    is_checkpoint_fp8_serialized=True, weight_block_size=[1, 32]
                ),
            ),
            ("w8a8-fp8", W8A8Fp8Config(is_checkpoint_fp8_serialized=True)),
            (
                "modelopt-mixed",
                ModelOptMixedConfig(quantized_layers={"model.layers.1.mlp": "FP8"}),
            ),
        ):
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, "BF16 and ModelOpt MXFP8"):
                    check_hyv4_quant_config(quant_config)

    def test_bf16_and_modelopt_mxfp8_are_accepted(self):
        check_hyv4_quant_config(None)
        check_hyv4_quant_config(
            ModelOptMxfp8Config.from_config(
                {"quant_method": "modelopt", "quantization": {"quant_algo": "MXFP8"}}
            )
        )

    def test_models_reject_fp8_before_building_layers(self):
        fp8 = Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])
        for model_cls in (HYV4Model, HYV4ModelNextN):
            with self.subTest(model=model_cls.__name__):
                with self.assertRaisesRegex(ValueError, "'fp8' quantization"):
                    model_cls(HYV4Config(), mapping=SimpleNamespace(), quant_config=fp8)

    def _initialize_from_modelopt_mxfp8(self, model_cls):
        mxfp8 = ModelOptMxfp8Config.from_config(
            {"quant_method": "modelopt", "quantization": {"quant_algo": "MXFP8"}}
        )
        model_config = SimpleNamespace(
            hf_config=None, mapping=None, is_multimodal=False
        )
        with (
            mock.patch.object(
                model_loader, "get_model_architecture", return_value=(model_cls, "")
            ),
            mock.patch.object(
                model_loader, "_get_quantization_config", return_value=mxfp8
            ),
        ):
            return model_loader._initialize_model(model_config, load_config=None)

    def test_loader_builds_hyv4_from_a_modelopt_mxfp8_checkpoint(self):
        for model_cls in (HYV4ForCausalLM, HYV4ForCausalLMNextN):
            with self.subTest(model=model_cls.__name__):
                with mock.patch.object(model_cls, "__init__", return_value=None):
                    model = self._initialize_from_modelopt_mxfp8(model_cls)
                self.assertIsInstance(model, model_cls)

    def test_loader_rejects_modelopt_mxfp8_for_models_that_keep_it_quantized(self):
        class DenseModel:
            def __init__(self, **kwargs):
                pass

        with self.assertRaisesRegex(
            ValueError, "DenseModel does not support ModelOpt MXFP8"
        ):
            self._initialize_from_modelopt_mxfp8(DenseModel)


class HYV4WeightUpdateTests(unittest.TestCase):
    def test_update_group_is_refused_before_the_rendezvous(self):
        runner = SimpleNamespace(model=object.__new__(HYV4ForCausalLM))
        with mock.patch("torch.distributed.distributed_c10d.rendezvous") as rendezvous:
            ok, message = ModelRunner.init_weights_update_group(
                runner, SimpleNamespace()
            )
        self.assertFalse(ok)
        self.assertIn("HYV4ForCausalLM does not support live weight updates", message)
        rendezvous.assert_not_called()

    def test_a_weight_update_session_is_refused(self):
        for model_cls in (HYV4ForCausalLM, HYV4ForCausalLMNextN):
            with self.subTest(model=model_cls.__name__):
                model = object.__new__(model_cls)
                with self.assertRaisesRegex(
                    RuntimeError, "does not support live weight updates"
                ):
                    with weight_update_session([model]):
                        pass
                self.assertFalse(model._weight_update_active)


class HYV4SpeculativeAlgorithmTests(unittest.TestCase):
    def test_drafts_other_than_mtp_are_rejected(self):
        for algorithm in ("EAGLE3", "DFLASH", "DSPARK"):
            with self.subTest(algorithm=algorithm):
                with mock.patch.dict(
                    hy_v4.global_server_args_dict,
                    {"speculative_algorithm": algorithm},
                ):
                    with self.assertRaisesRegex(
                        ValueError, f"MTP speculative decoding only.*got {algorithm}"
                    ):
                        check_hyv4_speculative_algorithm()

    def test_mtp_and_no_speculation_are_accepted(self):
        for algorithm in (None, "MTP"):
            with self.subTest(algorithm=algorithm):
                with mock.patch.dict(
                    hy_v4.global_server_args_dict,
                    {"speculative_algorithm": algorithm},
                ):
                    check_hyv4_speculative_algorithm()

    def test_model_rejects_eagle3_before_building_layers(self):
        with mock.patch.dict(
            hy_v4.global_server_args_dict, {"speculative_algorithm": "EAGLE3"}
        ):
            with self.assertRaisesRegex(ValueError, "MTP speculative decoding only"):
                HYV4Model(HYV4Config(), mapping=SimpleNamespace())


class HYV4SpeculativeCaptureTests(unittest.TestCase):
    def test_eagle3_and_dflash_capture_are_rejected(self):
        for setter in (
            HYV4ForCausalLM.set_eagle3_layers_to_capture,
            HYV4ForCausalLM.set_dflash_layers_to_capture,
        ):
            with self.subTest(setter=setter.__name__):
                with self.assertRaisesRegex(ValueError, "speculative-algorithm MTP"):
                    setter(SimpleNamespace(), [1])


class HYV4OutputGateTests(unittest.TestCase):
    @staticmethod
    def _attention(gate: torch.Tensor, gate_size_per_head: int) -> SimpleNamespace:
        return SimpleNamespace(
            linear_gate=lambda _hidden_states: (gate,),
            gate_size_per_head=gate_size_per_head,
            v_head_dim=4,
        )

    def test_elementwise_gate_runs_the_fused_kernel_in_place(self):
        attn_output, gate = torch.randn(3, 8), torch.randn(3, 8)
        expected = attn_output * torch.sigmoid(gate)
        with mock.patch.object(
            hy_v4, "sigmoid_mul", side_effect=lambda x, g: x.mul_(torch.sigmoid(g))
        ) as fused:
            out = HYV4Attention._apply_output_gate(
                self._attention(gate, 4), attn_output, torch.empty(3, 16)
            )
        fused.assert_called_once()
        (x, g), _ = fused.call_args
        self.assertIs(x, attn_output)
        self.assertIs(g, gate)
        self.assertIs(out, attn_output)
        torch.testing.assert_close(out, expected)

    def test_headwise_gate_scales_each_head(self):
        attn_output, gate = torch.randn(3, 8), torch.randn(3, 2)
        expected = (
            attn_output.view(3, 2, 4) * torch.sigmoid(gate).unsqueeze(-1)
        ).reshape(3, 8)
        out = HYV4Attention._apply_output_gate(
            self._attention(gate, 1), attn_output, torch.empty(3, 16)
        )
        torch.testing.assert_close(out, expected)


class HYV4RuntimeCapabilityTests(unittest.TestCase):
    def test_pipeline_is_rejected_before_embeddings_or_layers(self):
        mapping = SimpleNamespace(has_pp=True)
        for model_cls in (HYV4Model, HYV4ModelNextN):
            with self.subTest(model=model_cls.__name__):
                with (
                    mock.patch.object(hy_v4, "VocabParallelEmbedding") as embedding,
                    mock.patch.object(
                        hy_v4_nextn, "VocabParallelEmbedding"
                    ) as draft_embedding,
                    mock.patch.dict(
                        hy_v4.global_server_args_dict, {"speculative_algorithm": None}
                    ),
                ):
                    with self.assertRaisesRegex(ValueError, "pipeline parallelism"):
                        model_cls(HYV4Config(), mapping=mapping)
                    embedding.assert_not_called()
                    draft_embedding.assert_not_called()

    def test_missing_ihc_is_an_actionable_model_error(self):
        with mock.patch.object(hy_v4, "residual_ops", SimpleNamespace()):
            with self.assertRaisesRegex(
                RuntimeError, "requires newer tokenspeed-kernel"
            ):
                hy_v4.check_hyv4_runtime(SimpleNamespace(has_pp=False), enable_ihc=True)

    def test_old_dsa_without_sink_is_rejected(self):
        def old_attention(q):
            return q

        old = SimpleNamespace(dsa_prefill=old_attention, dsa_decode=old_attention)
        with mock.patch.object(hy_v4, "dsa_ops", old):
            with self.assertRaisesRegex(RuntimeError, "attn_sink"):
                hy_v4.check_hyv4_runtime(
                    SimpleNamespace(has_pp=False), enable_ihc=False
                )

    def test_current_capabilities_are_accepted_without_allocating(self):
        hy_v4.check_hyv4_runtime(SimpleNamespace(has_pp=False), enable_ihc=True)

    def test_missing_fused_boundary_is_rejected_before_allocation(self):
        available = SimpleNamespace(
            ihc_pre=lambda: None, ihc_post=lambda: None, ihc_head=lambda: None
        )
        with mock.patch.object(hy_v4, "residual_ops", available):
            with self.assertRaisesRegex(RuntimeError, "ihc_fused_hc"):
                hy_v4.check_hyv4_runtime(SimpleNamespace(has_pp=False), enable_ihc=True)


if __name__ == "__main__":
    unittest.main()
