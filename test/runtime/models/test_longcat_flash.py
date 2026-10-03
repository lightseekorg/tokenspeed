"""Cheap LongCat-Flash model wiring tests."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from tokenspeed.runtime.layers.moe.topk import StandardTopKOutput, TopKOutputFormat
from tokenspeed.runtime.models.longcat_flash import (
    LongcatFlashForCausalLM,
    _ensure_longcat_config,
    _get_longcat_moe_quant_config,
    _RuntimeLongcatMoE,
)


class TestLongcatFlashRegistry(unittest.TestCase):
    def test_registered(self):
        from tokenspeed.runtime.models.registry import ModelRegistry

        cls, arch = ModelRegistry.resolve_model_cls(["LongcatFlashForCausalLM"])
        self.assertIs(cls, LongcatFlashForCausalLM)
        self.assertEqual(arch, "LongcatFlashForCausalLM")

    def test_mla_and_double_attention_metadata_registered(self):
        from tokenspeed.runtime.configs import model_config

        self.assertIn("LongcatFlashForCausalLM", model_config._MLA_ARCHITECTURES)
        self.assertIn(
            "LongcatFlashForCausalLM",
            model_config._DOUBLE_ATTENTION_LAYER_ARCHITECTURES,
        )


class TestLongcatFlashConfig(unittest.TestCase):
    def test_config_aliases_are_normalized(self):
        config = SimpleNamespace(
            num_layers=28,
            ffn_hidden_size=14336,
            expert_ffn_hidden_size=2048,
            moe_topk=8,
            hidden_size=6144,
            n_routed_experts=512,
        )

        _ensure_longcat_config(config)

        self.assertEqual(config.num_hidden_layers, 28)
        self.assertEqual(config.intermediate_size, 14336)
        self.assertEqual(config.moe_intermediate_size, 2048)
        self.assertEqual(config.num_experts_per_tok, 8)
        self.assertEqual(config.hidden_act, "silu")
        self.assertEqual(config.zero_expert_num, 0)
        self.assertFalse(config.router_bias)


class TestLongcatMixedFp8Config(unittest.TestCase):
    def test_moe_layer_uses_unquantized_backend_when_all_experts_are_ignored(self):
        config = SimpleNamespace(n_routed_experts=2)
        quant_config = SimpleNamespace(
            ignored_layers=[
                f"model.layers.0.mlp.experts.{expert_id}.{proj_name}"
                for expert_id in range(2)
                for proj_name in ("gate_proj", "up_proj", "down_proj")
            ]
        )

        self.assertIsNone(
            _get_longcat_moe_quant_config(
                config,
                quant_config,
                "model.layers.0.mlp",
            )
        )

    def test_moe_layer_keeps_quantization_when_no_experts_are_ignored(self):
        config = SimpleNamespace(n_routed_experts=2)
        quant_config = SimpleNamespace(ignored_layers=[])

        self.assertIs(
            _get_longcat_moe_quant_config(
                config,
                quant_config,
                "model.layers.0.mlp",
            ),
            quant_config,
        )

    def test_moe_layer_rejects_partially_ignored_experts(self):
        config = SimpleNamespace(n_routed_experts=2)
        quant_config = SimpleNamespace(
            ignored_layers=[
                "model.layers.0.mlp.experts.0.gate_proj",
            ]
        )

        with self.assertRaisesRegex(ValueError, "partially ignored"):
            _get_longcat_moe_quant_config(
                config,
                quant_config,
                "model.layers.0.mlp",
            )


class TestLongcatZeroExpert(unittest.TestCase):
    """The router's contract: zero experts are -1, every other id is a slot < P."""

    @staticmethod
    def _moe(num_physical_experts: int):
        moe = object.__new__(_RuntimeLongcatMoE)
        moe.zero_expert_num = 1
        moe.n_routed_experts = 3
        moe.zero_expert_type = "identity"
        moe.experts = SimpleNamespace(num_experts=num_physical_experts)
        return moe

    def test_identity_zero_expert_masks_and_adds_hidden_state(self):
        moe = self._moe(3)
        hidden_states = torch.tensor(
            [[2.0, 4.0], [6.0, 8.0]],
            dtype=torch.float32,
        )
        topk_output = StandardTopKOutput(
            topk_weights=torch.tensor([[0.25, 0.75], [0.5, 0.5]]),
            topk_ids=torch.tensor([[0, -1], [-1, 1]]),
            router_logits=torch.zeros(2, 4),
        )

        zero_output = _RuntimeLongcatMoE._apply_zero_experts(
            moe,
            hidden_states,
            topk_output,
        )

        torch.testing.assert_close(
            zero_output,
            torch.tensor([[1.5, 3.0], [3.0, 4.0]]),
        )
        torch.testing.assert_close(
            topk_output.topk_weights,
            torch.tensor([[0.25, 0.0], [0.0, 0.5]]),
        )
        torch.testing.assert_close(
            topk_output.topk_ids,
            torch.tensor([[0, 0], [0, 1]]),
        )

    def test_placed_replicas_beyond_the_routed_count_are_not_zero_experts(self):
        # With a placement the router emits physical ids: E=3 routed experts on
        # P=5 slots, so ids 3 and 4 are replicas, and only -1 is a zero expert.
        moe = self._moe(5)
        hidden_states = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
        topk_output = StandardTopKOutput(
            topk_weights=torch.tensor([[0.25, 0.75], [0.5, 0.5]]),
            topk_ids=torch.tensor([[4, -1], [3, 1]]),
            router_logits=torch.zeros(2, 4),
        )

        zero_output = _RuntimeLongcatMoE._apply_zero_experts(
            moe, hidden_states, topk_output
        )

        torch.testing.assert_close(zero_output, torch.tensor([[1.5, 3.0], [0.0, 0.0]]))
        torch.testing.assert_close(
            topk_output.topk_weights, torch.tensor([[0.25, 0.0], [0.5, 0.5]])
        )
        torch.testing.assert_close(topk_output.topk_ids, torch.tensor([[4, 0], [3, 1]]))

    def test_an_id_at_or_beyond_the_physical_experts_is_refused(self):
        # A zero expert numbered from E (the reference engine's convention) is
        # not this router's: it must have been normalized to -1 upstream.
        moe = self._moe(3)
        topk_output = StandardTopKOutput(
            topk_weights=torch.tensor([[0.25, 0.75]]),
            topk_ids=torch.tensor([[0, 3]]),
            router_logits=torch.zeros(1, 4),
        )
        with self.assertRaisesRegex(RuntimeError, "zero experts must be -1"):
            _RuntimeLongcatMoE._apply_zero_experts(moe, torch.ones(1, 2), topk_output)


def _longcat_moe_config(n_routed_experts: int = 4) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=128,
        moe_intermediate_size=128,
        n_routed_experts=n_routed_experts,
        zero_expert_num=2,
        zero_expert_type="identity",
        moe_topk=2,
        hidden_act="silu",
        routed_scaling_factor=1.0,
        norm_topk_prob=False,
        router_bias=False,
        router_dtype="float32",
    )


class TestLongcatExpertPlacementWiring(unittest.TestCase):
    """The MoE sizes its slots from the placement and routes through its view."""

    def _build(self, placement, *, all_to_all_ep: bool, algorithm: str):
        from tokenspeed.runtime.distributed.mapping import Mapping
        from tokenspeed.runtime.layers.moe import expert as expert_module
        from tokenspeed.runtime.moe import expert_location
        from tokenspeed.runtime.utils.env import global_server_args_dict

        def fake_moe_plan(weight_dtype: str, **kwargs) -> dict:
            return {
                "solution": "flashinfer_trtllm",
                "support_routing": False,
                "supports_precomputed_topk": True,
                "supports_deferred_finalize": False,
                "supports_all_to_all_ep": all_to_all_ep,
            }

        trtllm = SimpleNamespace(value="flashinfer_trtllm")
        mapping = Mapping(
            rank=1, world_size=2, attn_tp_size=2, moe_tp_size=1, moe_ep_size=2
        )
        expert_location.set_global_expert_location_metadata(placement)
        self.addCleanup(expert_location.set_global_expert_location_metadata, None)
        with (
            torch.device("cpu"),
            mock.patch.object(expert_module, "get_moe_backend", lambda: trtllm),
            mock.patch.object(
                expert_module.tokenspeed_kernel, "moe_plan", fake_moe_plan
            ),
            mock.patch.dict(
                global_server_args_dict,
                {
                    "ep_num_redundant_experts": 2,
                    "ep_dispatch_algorithm": algorithm,
                    "enable_deep_ep": False,
                },
            ),
        ):
            return _RuntimeLongcatMoE(
                config=_longcat_moe_config(),
                mapping=mapping,
                quant_config=None,
                layer_index=1,
                prefix="model.layers.1.mlp",
                alt_stream=None,
            )

    def _placement(self):
        from tokenspeed.runtime.moe.expert_location import ExpertLocationMetadata

        return ExpertLocationMetadata.from_physical_to_logical_map(
            torch.tensor([[0, 1, 2, 3, 0, 2], [3, 2, 1, 0, 1, 1]]),
            4,
            ep_size=2,
            ep_rank=1,
            ep_rank_nodes=(0, 0),
        )

    def test_without_a_placement_the_layer_keeps_its_routed_experts(self):
        moe = self._build(None, all_to_all_ep=False, algorithm=None)
        self.assertIsNone(moe.expert_placement)
        self.assertIsNone(moe.expert_dispatch_info)
        self.assertEqual(moe.experts.num_experts, 4)
        self.assertEqual(moe.topk.topk_config.layer_id, 1)

    def test_placement_sizes_slots_and_picks_the_dispatch_flavour(self):
        placement = self._placement()
        moe = self._build(
            placement, all_to_all_ep=False, algorithm="static_with_zero_expert"
        )
        self.assertIs(moe.expert_placement, placement)
        self.assertEqual(moe.experts.num_experts, 6)
        self.assertEqual(moe.experts.num_local_experts, 3)
        info = moe.expert_dispatch_info
        self.assertEqual(info.layer_id, 1)
        # Replicated-input EP: rank-agnostic replica tables, views of layer 1,
        # and no static per-rank map is ever built.
        self.assertIsNotNone(info.replica_dispatch)
        self.assertTrue(
            info.replica_dispatch.replicas.data_ptr()
            == placement.logical_to_all_physical_map[1].data_ptr()
        )
        self.assertIsNone(placement._rank_dispatch_map)
        moe = self._build(
            placement, all_to_all_ep=True, algorithm="static_with_zero_expert"
        )
        # All-to-all EP: this rank's static nearest-replica map.
        self.assertIsNone(moe.expert_dispatch_info.replica_dispatch)
        self.assertEqual(
            moe.expert_dispatch_info.partial_logical_to_rank_dispatch_physical_map.tolist(),
            placement.rank_dispatch_map()[1].tolist(),
        )

    def test_placement_geometry_must_match_the_model(self):
        from tokenspeed.runtime.moe.expert_location import ExpertLocationMetadata

        wrong_experts = ExpertLocationMetadata.from_physical_to_logical_map(
            torch.tensor([[0, 1, 2, 3, 4, 0], [0, 1, 2, 3, 4, 1]]),
            5,
            ep_size=2,
            ep_rank=1,
            ep_rank_nodes=(0, 0),
        )
        with self.assertRaisesRegex(ValueError, "logical experts"):
            self._build(
                wrong_experts, all_to_all_ep=False, algorithm="static_with_zero_expert"
            )
        with self.assertRaisesRegex(ValueError, "zero experts"):
            self._build(self._placement(), all_to_all_ep=False, algorithm="static")

    def test_longcat_opts_in_to_expert_placement(self):
        self.assertTrue(LongcatFlashForCausalLM.supports_expert_placement)


class TestLongcatMoePlan(unittest.TestCase):
    def test_blackwell_ep4_plans_accept_zero_expert_routing(self):
        from tokenspeed_kernel.platform import current_platform

        from tokenspeed.runtime.distributed.mapping import Mapping
        from tokenspeed.runtime.layers.moe import utils as moe_utils
        from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config
        from tokenspeed.runtime.utils.env import global_server_args_dict

        if not torch.cuda.is_available() or not current_platform().is_blackwell:
            self.skipTest("Native TRT-LLM MoE plans require NVIDIA Blackwell")

        config = SimpleNamespace(
            hidden_size=128,
            moe_intermediate_size=128,
            n_routed_experts=8,
            zero_expert_num=4,
            zero_expert_type="identity",
            moe_topk=2,
            hidden_act="silu",
            routed_scaling_factor=1.0,
            norm_topk_prob=False,
            router_bias=False,
            router_dtype="float32",
        )
        mapping = Mapping(
            rank=0,
            world_size=4,
            attn_tp_size=4,
            moe_tp_size=1,
            moe_ep_size=4,
        )
        fp8_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            ignored_layers=[],
            weight_block_size=[128, 128],
            scale_fmt=None,
        )
        self.addCleanup(torch.set_default_dtype, torch.get_default_dtype())
        torch.set_default_dtype(torch.bfloat16)

        # Exercise the real planner and constructor, without loading a model or
        # launching kernels. LongCat has both unquantized and block-FP8 layers:
        # each must accept its zero-expert-masked top-k IDs and weights.
        with (
            torch.device("cpu"),
            mock.patch.object(
                moe_utils, "MOE_BACKEND", moe_utils.MoeBackend.FLASHINFER_TRTLLM
            ),
            mock.patch.object(
                moe_utils, "ALL2ALL_BACKEND", moe_utils.All2AllBackend.NONE
            ),
            mock.patch.dict(
                global_server_args_dict,
                {"ep_num_redundant_experts": 0, "enable_deep_ep": False},
            ),
        ):
            for quant_config in (None, fp8_config):
                with self.subTest(quantized=quant_config is not None):
                    moe = _RuntimeLongcatMoE(
                        config=config,
                        mapping=mapping,
                        quant_config=quant_config,
                        layer_index=0,
                        prefix="model.layers.0.mlp",
                        alt_stream=None,
                    )
                    self.assertEqual(moe.experts.plan["solution"], "flashinfer_trtllm")
                    self.assertEqual(
                        moe.experts.topk_output_format, TopKOutputFormat.STANDARD
                    )
                    self.assertTrue(moe.experts.supports_precomputed_topk)
                    self.assertFalse(moe.experts.support_routing)


class TestLongcatCheckpointLoading(unittest.TestCase):
    def test_missing_kv_scale_params_are_silent(self):
        model = object.__new__(LongcatFlashForCausalLM)
        with mock.patch(
            "tokenspeed.runtime.models.longcat_flash._longcat_logger.warning"
        ) as warning:
            self.assertIsNone(model.get_param({}, "model.layers.0.self_attn.0.k_scale"))
            self.assertIsNone(model.get_param({}, "model.layers.0.self_attn.1.v_scale"))
        warning.assert_not_called()

    def test_missing_mtp_params_are_silent(self):
        model = object.__new__(LongcatFlashForCausalLM)
        with mock.patch(
            "tokenspeed.runtime.models.longcat_flash._longcat_logger.warning"
        ) as warning:
            self.assertIsNone(
                model.get_param({}, "model.mtp.layers.0.self_attn.q_proj.weight")
            )
        warning.assert_not_called()


if __name__ == "__main__":
    unittest.main()
