"""Kimi-K3 config + registration wiring tests (cheap, no GPU).

Covers the parts landed in the Kimi-K3 model-registration change: the
``KimiLinearConfig`` mixed-layer protocol (consumed by the hybrid KV-cache
layer) and the architecture-registration touchpoints (``_CONFIG_REGISTRY``,
``_MLA_ARCHITECTURES``, ``is_multimodal_model``, ``EntryClass``).
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.configs.kimi_k3_config import (  # noqa: E402
    KimiK3Config,
    KimiK3VisionConfig,
    KimiLinearConfig,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (  # noqa: E402
    FULL_ATTENTION,
    LINEAR_ATTENTION,
)

# Checkpoint-derived reference values (moonshotai/kimi-k3).
_NUM_LAYERS = 93
_NUM_KDA = 69
_NUM_MLA = 24
_KDA_HEADS = 96
_KDA_HEAD_DIM = 128
_KDA_CONV = 4


class KimiK3ConfigTests(unittest.TestCase):
    def test_top_level_wraps_text_and_vision(self):
        cfg = KimiK3Config()
        self.assertEqual(cfg.model_type, "kimi_k3")
        self.assertIsInstance(cfg.text_config, KimiLinearConfig)
        self.assertIsInstance(cfg.vision_config, KimiK3VisionConfig)
        # hidden_size / vocab_size forward to the text config.
        self.assertEqual(cfg.hidden_size, cfg.text_config.hidden_size)
        self.assertEqual(cfg.vocab_size, cfg.text_config.vocab_size)

    def test_dict_subconfigs_are_materialized(self):
        cfg = KimiK3Config(
            text_config={"hidden_size": 4096},
            vision_config={"vt_hidden_size": 512},
        )
        self.assertIsInstance(cfg.text_config, KimiLinearConfig)
        self.assertIsInstance(cfg.vision_config, KimiK3VisionConfig)
        self.assertEqual(cfg.text_config.hidden_size, 4096)
        self.assertEqual(cfg.vision_config.vt_hidden_size, 512)

    def test_vision_text_hidden_forced_to_text_hidden(self):
        # The projector must emit the text hidden size, overriding any value
        # supplied in vision_config.
        cfg = KimiK3Config(
            text_config={"hidden_size": 4096},
            vision_config={"text_hidden_size": 1234},
        )
        self.assertEqual(cfg.vision_config.text_hidden_size, 4096)

    def test_layer_partition_is_exact(self):
        la = KimiLinearConfig().linear_attn_config
        kda = set(la["kda_layers"])
        full = set(la["full_attn_layers"])
        self.assertEqual(len(kda), _NUM_KDA)
        self.assertEqual(len(full), _NUM_MLA)
        # 1-based, no overlap, exact cover of 1..N.
        self.assertEqual(kda & full, set())
        self.assertEqual(kda | full, set(range(1, _NUM_LAYERS + 1)))

    def test_is_kda_layer_matches_config(self):
        c = KimiLinearConfig()
        kda = set(c.linear_attn_config["kda_layers"])
        for i in range(c.num_hidden_layers):
            self.assertEqual(c.is_kda_layer(i), (i + 1) in kda)

    def test_layers_block_type_and_ids(self):
        c = KimiLinearConfig()
        lbt = c.layers_block_type
        self.assertEqual(len(lbt), _NUM_LAYERS)
        self.assertEqual(set(lbt), {"attention", "linear_attention"})
        self.assertEqual(len(c.linear_layer_ids), _NUM_KDA)
        self.assertEqual(len(c.full_attention_layer_ids), _NUM_MLA)
        # 0-based full-attention ids == (1-based full_attn_layers - 1).
        self.assertEqual(
            c.full_attention_layer_ids,
            sorted(x - 1 for x in c.linear_attn_config["full_attn_layers"]),
        )

    def test_layer_types_translate_full_attention(self):
        c = KimiLinearConfig()
        self.assertEqual(set(c.layer_types), {FULL_ATTENTION, LINEAR_ATTENTION})
        for block_type, cache_label in zip(c.layers_block_type, c.layer_types):
            if block_type == "attention":
                self.assertEqual(cache_label, FULL_ATTENTION)
            else:
                self.assertEqual(cache_label, block_type)

    @staticmethod
    def _kda_spec(tp_size):
        """The KDA component as boot construction builds it, at one TP width."""
        from tokenspeed.runtime.layers.attention.configs.linear_attn import (
            LinearAttnConfig,
        )

        server_args = SimpleNamespace(
            mapping=SimpleNamespace(linear_attn=SimpleNamespace(tp_size=tp_size))
        )
        model = SimpleNamespace(
            hf_config=SimpleNamespace(text_config=KimiLinearConfig())
        )
        return LinearAttnConfig.generate(server_args, model)

    def test_kda_state_shapes(self):
        spec = self._kda_spec(tp_size=1)
        # conv over q/k/v (3 * num_heads * head_dim) wide, kernel_size - 1 deep.
        self.assertEqual(
            spec.conv_state_shape, (3 * _KDA_HEADS * _KDA_HEAD_DIM, _KDA_CONV - 1)
        )
        # per-head (head_dim x head_dim) recurrent state.
        self.assertEqual(
            spec.temporal_state_shape, (_KDA_HEADS, _KDA_HEAD_DIM, _KDA_HEAD_DIM)
        )
        self.assertEqual(spec.layer_ids, tuple(KimiLinearConfig().linear_layer_ids))

    def test_kda_state_shapes_respect_tp(self):
        spec = self._kda_spec(tp_size=4)
        self.assertEqual(
            spec.conv_state_shape,
            (3 * _KDA_HEADS * _KDA_HEAD_DIM // 4, _KDA_CONV - 1),
        )
        self.assertEqual(
            spec.temporal_state_shape,
            (_KDA_HEADS // 4, _KDA_HEAD_DIM, _KDA_HEAD_DIM),
        )


class KimiK3RegistrationTests(unittest.TestCase):
    def test_hybrid_moe_precomputes_routing_only_for_decode(self):
        from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
        from tokenspeed.runtime.layers.moe.topk import TopKOutputFormat
        from tokenspeed.runtime.models.kimi_k3 import KimiLinearMoE

        layer = KimiLinearMoE.__new__(KimiLinearMoE)
        torch.nn.Module.__init__(layer)
        layer.execution_plan = SimpleNamespace(use_trtllm=True)
        layer._gather_dp_tokens_for_moe = False
        layer.experts = SimpleNamespace(
            support_routing=True,
            supports_precomputed_topk=True,
            topk_output_format=TopKOutputFormat.BYPASSED,
        )

        self.assertEqual(layer._routing_output_format(None), TopKOutputFormat.STANDARD)
        for mode, expected in (
            (ForwardMode.DECODE, TopKOutputFormat.STANDARD),
            (ForwardMode.EXTEND, TopKOutputFormat.BYPASSED),
            (ForwardMode.MIXED, TopKOutputFormat.BYPASSED),
            (ForwardMode.IDLE, TopKOutputFormat.BYPASSED),
        ):
            with self.subTest(mode=mode):
                ctx = SimpleNamespace(forward_mode=mode)
                self.assertEqual(layer._routing_output_format(ctx), expected)

    def test_mla_mixed_batch_slices_decode_gate_to_live_rows(self):
        from tokenspeed.runtime.execution.context import ForwardContext
        from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
        from tokenspeed.runtime.models.deepseek_v3 import DeepseekV3AttentionMLA

        metadata = SimpleNamespace(
            extend_seq_lens_cpu=[1],
            use_absorbed_cached_extend=False,
        )
        backend = SimpleNamespace(
            spec_num_tokens=1,
            chunked_prefill_metadata=metadata,
            write_locations=lambda layer, mode: torch.arange(4),
        )
        ctx = ForwardContext(
            attn_backend=backend,
            token_to_kv_pool=None,
            bs=3,
            num_extends=1,
            input_num_tokens=4,
            forward_mode=ForwardMode.DECODE,
        )
        attention = DeepseekV3AttentionMLA.__new__(DeepseekV3AttentionMLA)
        torch.nn.Module.__init__(attention)
        attention.num_local_heads = 2
        attention.v_head_dim = 3
        attention.attn_mha = SimpleNamespace(group_id="full_attention")
        attention.forward_normal_chunked = mock.Mock()
        attention.forward_absorb = mock.Mock()

        output_gate = torch.arange(24).reshape(4, 6)
        attention._attn(
            positions=torch.arange(4),
            q=torch.empty(4, 2, 8),
            latent_cache=torch.empty(4, 1, 8),
            ctx=ctx,
            output_gate=output_gate,
        )

        torch.testing.assert_close(
            attention.forward_absorb.call_args.kwargs["output_gate"],
            output_gate[1:3],
        )

    def test_shared_projection_preserves_direct_write_output(self):
        import tokenspeed.runtime.models.kimi_k3 as kimi_k3

        class FakeMergedLinear(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.weight = torch.empty(
                    1536, 7168, dtype=torch.bfloat16, device="meta"
                )

        class FakeRowLinear(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.weight = torch.empty(
                    7168, 768, dtype=torch.bfloat16, device="meta"
                )
                self.reduce_results = kwargs["reduce_results"]
                self.tp_size = kwargs["tp_size"]
                self.tp_group = kwargs["tp_group"]

        mapping = SimpleNamespace(
            moe=SimpleNamespace(
                tp_ep_rank=0,
                tp_ep_size=8,
                has_tp_ep=True,
                tp_ep_group=tuple(range(8)),
            )
        )
        activated = torch.empty(2, 768, dtype=torch.bfloat16)
        down_out = torch.empty(2, 7168, dtype=torch.bfloat16)
        with (
            mock.patch.object(kimi_k3, "MergedColumnParallelLinear", FakeMergedLinear),
            mock.patch.object(kimi_k3, "RowParallelLinear", FakeRowLinear),
            mock.patch.object(
                kimi_k3,
                "kimi3_shared_situ_projection",
                return_value=activated,
            ),
            mock.patch.object(
                kimi_k3,
                "kimi3_shared_down_projection",
                return_value=down_out,
            ) as shared_down,
        ):
            layer = kimi_k3.KimiLinearMLP(
                hidden_size=7168,
                intermediate_size=6144,
                mapping=mapping,
                quant_config=None,
                prefix="shared_experts",
                reduce_results=False,
                is_shared_expert=True,
            )
            actual = layer(
                torch.empty(2, 7168, dtype=torch.bfloat16),
                down_out=down_out,
            )

        self.assertIs(actual, down_out)
        shared_down.assert_called_once_with(
            activated,
            layer.down_proj.weight,
            out=down_out,
        )

    def test_kda_stacks_qkvfab_projection_weights(self):
        from tokenspeed.runtime.models.kimi_k3 import KimiLinearKDA

        config = KimiLinearConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            linear_attn_config={
                "kda_layers": [1],
                "full_attn_layers": [],
                "num_heads": 4,
                "head_dim": 16,
                "short_conv_kernel_size": 4,
                "gate_lower_bound": -5.0,
                "use_full_rank_gate": True,
            },
        )
        mapping = SimpleNamespace(
            attn=SimpleNamespace(tp_rank=0, tp_size=1, tp_group=(0,)),
            linear_attn=SimpleNamespace(tp_rank=0, tp_size=1, tp_group=(0,)),
        )
        layer = KimiLinearKDA(config, mapping, layer_id=0)

        self.assertEqual(tuple(layer.qkvgb_proj.weight.shape), (288, 64))
        for value, shard_id in enumerate(("q", "k", "v", "g"), start=1):
            loaded = torch.full((64, 64), float(value), dtype=torch.bfloat16)
            layer.qkvgb_proj.weight.weight_loader(
                layer.qkvgb_proj.weight,
                loaded,
                shard_id,
            )
        f_a_weight = torch.full((16, 64), 5.0, dtype=torch.bfloat16)
        beta_weight = torch.full((4, 64), 6.0, dtype=torch.bfloat16)
        layer.qkvgb_proj.weight.weight_loader(
            layer.qkvgb_proj.weight,
            f_a_weight,
            "f_a",
        )
        layer.qkvgb_proj.weight.weight_loader(
            layer.qkvgb_proj.weight,
            beta_weight,
            "b",
        )

        hidden_states = torch.randn(4, 64, dtype=torch.bfloat16)
        expected_qkvg = [
            torch.nn.functional.linear(
                hidden_states,
                layer.qkvgb_proj.weight[index * 64 : (index + 1) * 64],
            )
            for index in range(4)
        ]
        mixed_qkv, gate, f_a, beta = layer._project_qkvfab(hidden_states)
        self.assertTrue(torch.equal(mixed_qkv, torch.cat(expected_qkvg[:3], dim=-1)))
        self.assertTrue(torch.equal(gate, expected_qkvg[3]))
        self.assertTrue(
            torch.equal(f_a, torch.nn.functional.linear(hidden_states, f_a_weight))
        )
        torch.testing.assert_close(
            beta,
            torch.nn.functional.linear(hidden_states, beta_weight),
        )
        self.assertFalse(mixed_qkv.is_contiguous())
        self.assertEqual(
            mixed_qkv.untyped_storage().data_ptr(),
            gate.untyped_storage().data_ptr(),
        )

    def test_kda_compacts_prefill_qkv_before_backend_break(self):
        from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
        from tokenspeed.runtime.models.kimi_k3 import KimiLinearKDA

        config = KimiLinearConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            linear_attn_config={
                "kda_layers": [1],
                "full_attn_layers": [],
                "num_heads": 4,
                "head_dim": 16,
                "short_conv_kernel_size": 4,
                "gate_lower_bound": -5.0,
                "use_full_rank_gate": True,
            },
        )
        mapping = SimpleNamespace(
            attn=SimpleNamespace(tp_rank=0, tp_size=1, tp_group=(0,)),
            linear_attn=SimpleNamespace(tp_rank=0, tp_size=1, tp_group=(0,)),
        )
        layer = KimiLinearKDA(config, mapping, layer_id=0)
        rows, projection_width = 4, 64
        packed = torch.randn(rows, 288, dtype=torch.bfloat16)
        projection_outputs = (
            packed[:, : 3 * projection_width],
            packed[:, 3 * projection_width : 4 * projection_width],
            packed[:, 4 * projection_width : 4 * projection_width + 16],
            packed[:, 4 * projection_width + 16 : 4 * projection_width + 20],
        )

        class BackendCalled(Exception):
            pass

        for mode, expect_contiguous in (
            (ForwardMode.EXTEND, True),
            (ForwardMode.DECODE, False),
        ):
            with self.subTest(mode=mode):
                captured = []

                def capture_backend(**kwargs):
                    captured.append(kwargs["mixed_qkv"])
                    raise BackendCalled

                ctx = SimpleNamespace(
                    forward_mode=mode,
                    bs=rows,
                    attn_backend=SimpleNamespace(forward=capture_backend),
                    token_to_kv_pool=None,
                )
                with mock.patch.object(
                    layer, "_project_qkvfab", return_value=projection_outputs
                ):
                    with self.assertRaises(BackendCalled):
                        layer(
                            positions=torch.empty(rows, dtype=torch.int64),
                            hidden_states=torch.empty(rows, 64, dtype=torch.bfloat16),
                            ctx=ctx,
                            comm_manager=None,
                        )

                self.assertEqual(captured[0].is_contiguous(), expect_contiguous)

    def test_native_kimi_moe_uses_direct_ep_and_collective_tp_paths(self):
        import tokenspeed.runtime.models.kimi_k3 as kimi_k3
        from tokenspeed.runtime.layers.moe.topk import TopKOutputFormat

        shared_calls = []
        expert_calls = []

        class FakeLinear(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.weight = torch.empty(0)
                self.shard_group = kwargs.get("shard_group")

        class FakeExperts(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                expert_calls.append(kwargs)
                self.support_routing = False
                self.supports_precomputed_topk = True
                self.topk_output_format = TopKOutputFormat.STANDARD
                self.w13_weight = torch.empty(0)
                self.w13_weight_scale = torch.empty(0)
                self.w2_weight = torch.empty(0)
                self.w2_weight_scale = torch.empty(0)
                self.plan = {}
                # Consumed by K3MoeTailComm arming (real MoELayer exposes it
                # from the selected kernel's plan trait).
                self.supports_deferred_finalize = False

            def forward(self, *, hidden_states, **kwargs):
                return hidden_states + 1

        class FakeSharedExperts(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.gate_up_proj = FakeLinear()
                self.down_proj = FakeLinear()
                shared_calls.append(kwargs)

        class FakeLatentMoE(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.components = kwargs

        ep_group = tuple(range(8))
        mapping = SimpleNamespace(
            world_size=8,
            attn=SimpleNamespace(
                tp_size=8,
                cp_size=1,
                dp_size=1,
                dp_rank=0,
                dp_group=(0,),
            ),
            moe=SimpleNamespace(
                tp_rank=0,
                tp_size=1,
                tp_group=(0,),
                ep_rank=0,
                ep_size=8,
                ep_group=ep_group,
                tp_ep_size=8,
                has_tp_ep=True,
                tp_ep_rank=0,
                tp_ep_group=ep_group,
            ),
        )
        config = KimiLinearConfig(
            hidden_size=64,
            routed_expert_hidden_size=32,
            moe_intermediate_size=32,
            num_experts=8,
            num_experts_per_token=2,
            num_shared_experts=1,
        )

        with (
            mock.patch.object(kimi_k3, "ReplicatedLinear", FakeLinear),
            mock.patch.object(kimi_k3, "Kimi3LatentProjection", FakeLinear),
            mock.patch.object(kimi_k3, "MoELayer", FakeExperts),
            mock.patch.object(kimi_k3, "KimiLinearMLP", FakeSharedExperts),
            mock.patch.object(kimi_k3, "LatentMoELayer", FakeLatentMoE),
            mock.patch.object(
                kimi_k3.Kimi3MoEExecutionPlan,
                "build",
                return_value=kimi_k3.Kimi3MoEExecutionPlan(
                    use_native=True,
                    use_trtllm=False,
                    overlap_shared_experts=False,
                    joint_moe_reduce=True,
                ),
            ),
            mock.patch.dict(
                kimi_k3.global_server_args_dict,
                {"enforce_eager": False},
            ),
        ):
            layer = kimi_k3.KimiLinearMoE(
                config,
                mapping,
                layer_index=1,
                model_scope="model.layers",
                quant_config=None,
                prefix="model.layers.1.block_sparse_moe",
            )

        self.assertFalse(shared_calls[0]["reduce_results"])
        self.assertIsNone(expert_calls[0]["routing_mode"])
        self.assertEqual(expert_calls[0]["internal_activation_dtype_override"], "input")
        self.assertEqual(
            layer.topk.topk_config.output_format, TopKOutputFormat.STANDARD
        )
        self.assertTrue(layer.native_latent_moe.components["joint_reduce"])
        self.assertEqual(
            layer.native_latent_moe.components["expert_parallel_group"], ep_group
        )

        # TP shards each produce only a partial W2 result. They must use the
        # composed tail, whose reduction group spans TP x EP, instead of the
        # native LatentMoELayer path that only reduces across EP.
        tp_mapping = SimpleNamespace(
            world_size=8,
            attn=mapping.attn,
            moe=SimpleNamespace(
                tp_rank=0,
                tp_size=8,
                tp_group=ep_group,
                ep_rank=0,
                ep_size=1,
                ep_group=(0,),
                tp_ep_size=8,
                has_tp_ep=True,
                tp_ep_rank=0,
                tp_ep_group=ep_group,
            ),
        )
        with (
            mock.patch.object(kimi_k3, "ReplicatedLinear", FakeLinear),
            mock.patch.object(kimi_k3, "Kimi3LatentProjection", FakeLinear),
            mock.patch.object(kimi_k3, "MoELayer", FakeExperts),
            mock.patch.object(kimi_k3, "KimiLinearMLP", FakeSharedExperts),
            mock.patch.object(kimi_k3, "LatentMoELayer", FakeLatentMoE),
            mock.patch.object(
                kimi_k3.Kimi3MoEExecutionPlan,
                "build",
                return_value=kimi_k3.Kimi3MoEExecutionPlan(
                    use_native=True,
                    use_trtllm=False,
                    overlap_shared_experts=False,
                    joint_moe_reduce=False,
                ),
            ),
            mock.patch.dict(
                kimi_k3.global_server_args_dict,
                {"enforce_eager": False},
            ),
        ):
            tp_layer = kimi_k3.KimiLinearMoE(
                config,
                tp_mapping,
                layer_index=1,
                model_scope="model.layers",
                quant_config=None,
                prefix="model.layers.1.block_sparse_moe",
            )

        self.assertIsNone(tp_layer.native_latent_moe)
        self.assertEqual(tp_layer.comm.mapping.moe.tp_ep_group, ep_group)
        routed_input = torch.zeros(1, config.routed_expert_hidden_size)
        torch.testing.assert_close(
            tp_layer._routed_experts(
                routed_input,
                mock.Mock(),
                num_global_tokens=1,
                max_num_tokens_per_gpu=1,
            ),
            routed_input + 1,
        )

    def test_native_kimi_moe_zero_tokens_bypass_fused_pipeline(self):
        from tokenspeed.runtime.models.kimi_k3 import KimiLinearMoE

        hidden_states = torch.empty(0, 64)
        prefix_sum = torch.empty_like(hidden_states)
        native_latent_moe = mock.Mock(return_value=prefix_sum)
        fused_pipeline = mock.Mock(
            side_effect=AssertionError("zero tokens must bypass the fused pipeline")
        )
        layer = SimpleNamespace(
            _gather_dp_tokens_for_moe=False,
            native_latent_moe=native_latent_moe,
            _use_fused_decode_pipeline=True,
            _forward_fused_decode_pipeline=fused_pipeline,
        )

        output = KimiLinearMoE.forward(
            layer,
            hidden_states,
            prefix_sum,
            num_global_tokens=0,
            max_num_tokens_per_gpu=0,
        )

        torch.testing.assert_close(output, prefix_sum)
        self.assertEqual(tuple(output.shape), (0, 64))
        fused_pipeline.assert_not_called()
        native_latent_moe.assert_called_once_with(
            hidden_states,
            num_global_tokens=0,
            max_num_tokens_per_gpu=0,
            prefix_sum=prefix_sum,
        )

    def test_cross_dp_ep_gather_uses_dp_group_and_returns_local_offset(self):
        from tokenspeed.runtime.models.kimi_k3 import KimiLinearMoE

        layer = KimiLinearMoE.__new__(KimiLinearMoE)
        layer.mapping = SimpleNamespace(
            attn=SimpleNamespace(
                tp_size=8,
                cp_size=1,
                dp_size=4,
                dp_rank=2,
                dp_group=(2, 10, 18, 26),
            )
        )
        ctx = SimpleNamespace(
            collective_global_num_tokens=None,
            global_num_tokens=[3] * 8 + [5] * 8 + [7] * 8 + [11] * 8,
        )
        hidden = torch.arange(14, dtype=torch.float32).reshape(7, 2)
        prefix = hidden + 100
        gathered = []

        def gather(tensor, group, scattered_num_tokens):
            gathered.append((tensor, group, scattered_num_tokens))
            return torch.cat((tensor, tensor), dim=0)

        with mock.patch(
            "tokenspeed.runtime.models.kimi_k3.token_all_gather", side_effect=gather
        ):
            gathered_hidden, gathered_prefix, total, offset = layer._gather_dp_tokens(
                hidden, prefix, ctx
            )

        self.assertEqual(total, 26)
        self.assertEqual(offset, 8)
        self.assertEqual(len(gathered), 2)
        for _, group, counts in gathered:
            self.assertEqual(group, (2, 10, 18, 26))
            self.assertEqual(counts, [3, 5, 7, 11])
        torch.testing.assert_close(gathered_hidden, torch.cat((hidden, hidden)))
        torch.testing.assert_close(gathered_prefix, torch.cat((prefix, prefix)))

    def test_mla_gate_projection_uses_api_selected_layout(self):
        from tokenspeed.runtime.models.kimi_k3 import KimiLinearMLAAttention

        class FakeProjection(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.arange(50.0).reshape(10, 5))
                self.calls = 0

            def forward(self, hidden, block_scale, output_dtype):
                self.calls += 1
                return torch.nn.functional.linear(hidden, self.weight)

        class IdentityComm:
            @staticmethod
            def pre_attn_comm(value, ctx):
                return value

        class FakeFusedNorm(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight_q_a = torch.nn.Parameter(torch.ones(2))
                self.weight_kv_a = torch.nn.Parameter(torch.ones(3))

        class FakeQueryNorm(torch.nn.Module):
            variance_epsilon = 1e-6

        class IdentityQProjection(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.eye(2))

            def forward(self, value):
                return torch.nn.functional.linear(value, self.weight), None

        attention = KimiLinearMLAAttention.__new__(KimiLinearMLAAttention)
        torch.nn.Module.__init__(attention)
        attention.q_lora_rank = 2
        attention.kv_lora_rank = 3
        attention.qk_nope_head_dim = 1
        attention.qk_rope_head_dim = 1
        attention._qkv_a_width = 6
        attention._gate_width = 4
        attention.fused_qkv_a_proj_with_mqa = FakeProjection()
        attention.fused_qk_layernorm = FakeFusedNorm()
        attention.q_a_layernorm = FakeQueryNorm()
        attention.q_b_proj = IdentityQProjection()
        comm = IdentityComm()

        prefill = torch.ones(33, 5)
        with torch.no_grad():
            q, latent, gate, absorbed_query = attention._project_q_latent_gated(
                prefill, None, comm, None
            )
        expected = torch.nn.functional.linear(
            prefill, attention.fused_qkv_a_proj_with_mqa.weight
        )
        expected_q = torch.nn.functional.rms_norm(
            expected[:, :2],
            (2,),
            eps=attention.q_a_layernorm.variance_epsilon,
        )
        expected_latent = expected[:, 2:6].clone()
        expected_latent[:, :3] = torch.nn.functional.rms_norm(
            expected_latent[:, :3],
            (3,),
            eps=attention.q_a_layernorm.variance_epsilon,
        )
        torch.testing.assert_close(q, expected_q)
        torch.testing.assert_close(latent, expected_latent)
        torch.testing.assert_close(gate, expected[:, 6:])
        self.assertIsNone(absorbed_query)
        self.assertEqual(attention.fused_qkv_a_proj_with_mqa.calls, 0)

        with torch.no_grad():
            _, _, _, decode_absorbed = attention._project_q_latent_gated(
                prefill[:4], SimpleNamespace(num_extends=0), comm, None
            )
        self.assertIsNone(decode_absorbed)
        self.assertEqual(attention.fused_qkv_a_proj_with_mqa.calls, 0)

    def test_ungated_mla_does_not_select_attnres_projection_fusion(self):
        from tokenspeed.runtime.models.kimi_k3 import KimiLinearMLAAttention

        attention = KimiLinearMLAAttention.__new__(KimiLinearMLAAttention)
        torch.nn.Module.__init__(attention)

        self.assertFalse(attention.can_fuse_attnres_partials(torch.empty(1, 4), ()))

    def test_config_registry_maps_model_type(self):
        from tokenspeed.runtime.utils.hf_transformers_utils import _CONFIG_REGISTRY

        self.assertIs(_CONFIG_REGISTRY.get("kimi_k3"), KimiK3Config)

    def test_resolve_architecture_returns_registered_name(self):
        from tokenspeed.runtime.utils.hf_transformers_utils import resolve_architecture

        cfg = KimiK3Config(architectures=["KimiK3ForConditionalGeneration"])
        self.assertEqual(resolve_architecture(cfg), "KimiK3ForConditionalGeneration")

    def test_mla_and_multimodal_metadata_registered(self):
        from tokenspeed.runtime.configs import model_config

        self.assertIn("KimiK3ForConditionalGeneration", model_config._MLA_ARCHITECTURES)
        self.assertTrue(
            model_config.is_multimodal_model(["KimiK3ForConditionalGeneration"])
        )

    def test_entryclass_resolves_in_model_registry(self):
        from tokenspeed.runtime.models.kimi_k3 import (
            KimiK3ForConditionalGeneration,
        )
        from tokenspeed.runtime.models.registry import ModelRegistry

        cls, arch = ModelRegistry.resolve_model_cls(["KimiK3ForConditionalGeneration"])
        self.assertIs(cls, KimiK3ForConditionalGeneration)
        self.assertEqual(arch, "KimiK3ForConditionalGeneration")

    def test_text_only_wrapper_streams_and_discards_vision_weights(self):
        from tokenspeed.runtime.models.kimi_k3 import KimiK3ForConditionalGeneration

        class _Recorder:
            def __init__(self):
                self.weights = None

            def load_weights(self, weights):
                self.weights = list(weights)

        language_model = _Recorder()
        wrapper = SimpleNamespace(language_model=language_model, vision=None)
        weights = iter(
            (
                ("language_model.model.embed_tokens.weight", torch.ones(1)),
                ("vision_tower.blocks.0.weight", torch.ones(1)),
                ("mm_projector.weight", torch.ones(1)),
                ("language_model.lm_head.weight", torch.ones(1)),
            )
        )

        KimiK3ForConditionalGeneration.load_weights(wrapper, weights)

        self.assertEqual(
            [name for name, _ in language_model.weights],
            ["model.embed_tokens.weight", "lm_head.weight"],
        )


if __name__ == "__main__":
    unittest.main()


class KimiK3LcmPlanTests(unittest.TestCase):
    """LCM planning across attention-TP widths and reduced-layer variants."""

    @staticmethod
    def _plan(cfg, tp):
        from test.runtime.conftest import kimi_tp8_layout

        return kimi_tp8_layout(text_config=cfg, tp_size=tp)[2].bind(64)

    def test_mla_packing_scales_with_attn_tp(self):
        """The MLA plane is the smallest that covers one per-layer KDA state,
        so tp=16 -- where the KDA state page halves while the MLA latent page
        is tp-invariant -- halves the plane and the parent, and tp=1
        (attention-DP) grows them 8/tp-fold instead of failing the planner's
        exact-page-stride check."""
        cfg = KimiLinearConfig()
        plan8 = self._plan(cfg, 8)
        plan16 = self._plan(cfg, 16)
        plan1 = self._plan(cfg, 1)
        packs8 = {g.group_id: g.cache_blocks_per_lcm_block for g in plan8.groups}
        packs16 = {g.group_id: g.cache_blocks_per_lcm_block for g in plan16.groups}
        packs1 = {g.group_id: g.cache_blocks_per_lcm_block for g in plan1.groups}
        self.assertEqual(packs8[FULL_ATTENTION], 12)
        self.assertEqual(packs16[FULL_ATTENTION], 6)
        self.assertEqual(packs1[FULL_ATTENTION], 89)
        for gid in (f"{LINEAR_ATTENTION}_0", f"{LINEAR_ATTENTION}_1"):
            self.assertEqual(packs8[gid], 1)
            self.assertEqual(packs16[gid], 1)
            self.assertEqual(packs1[gid], 1)
        self.assertEqual(plan16.lcm_block_bytes * 2, plan8.lcm_block_bytes)
        for plan in (plan8, plan16, plan1):
            self.assertEqual(len(plan.planes), _NUM_MLA)

    def test_reduced_layer_variant_plans(self):
        """Layer counts derive from the config: a structurally-identical
        reduced-layer checkpoint (same per-layer specs, fewer layers) plans
        with one plane per MLA layer instead of tripping hardcoded 93/69/24
        checks."""
        base = KimiLinearConfig()
        linear = dict(base.linear_attn_config)
        num_layers = 24
        linear["kda_layers"] = [x for x in linear["kda_layers"] if x <= num_layers]
        linear["full_attn_layers"] = [
            x for x in linear["full_attn_layers"] if x <= num_layers
        ]
        cfg = KimiLinearConfig(num_hidden_layers=num_layers, linear_attn_config=linear)
        plan = self._plan(cfg, 8)
        self.assertEqual(len(plan.planes), len(linear["full_attn_layers"]))

    def test_full_size_split_is_enforced(self):
        """A 93-layer config must keep exactly 69 KDA + 24 MLA; the relaxed
        reduced-layer path must not weaken the released-checkpoint check."""
        base = KimiLinearConfig()
        linear = dict(base.linear_attn_config)
        kda = list(linear["kda_layers"])
        # 66 KDA (still /3 for the state groups) + 27 full: wrong split.
        kda.pop()
        kda.pop()
        kda.pop()
        linear["kda_layers"] = kda
        linear["full_attn_layers"] = sorted(set(range(1, _NUM_LAYERS + 1)) - set(kda))
        cfg = KimiLinearConfig(linear_attn_config=linear)
        with self.assertRaisesRegex(ValueError, "69 KDA and 24 MLA"):
            self._plan(cfg, 8)

    def test_reduced_layer_split_must_ride_inside_the_mla_planes(self):
        """A reduced-layer variant whose KDA groups need a plane of their own
        is rejected loudly: 5 MLA layers cannot host 6 KDA slots."""
        base = KimiLinearConfig()
        linear = dict(base.linear_attn_config)
        num_layers = 23  # 17 KDA layers: not divisible by 3
        linear["kda_layers"] = [x for x in linear["kda_layers"] if x <= num_layers]
        linear["full_attn_layers"] = [
            x for x in linear["full_attn_layers"] if x <= num_layers
        ]
        cfg = KimiLinearConfig(num_hidden_layers=num_layers, linear_attn_config=linear)
        with self.assertRaises(ValueError):
            self._plan(cfg, 8)
